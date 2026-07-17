"""Application entrypoint: lifespan wiring, middleware, error handling."""

from __future__ import annotations

import logging
import time
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.api.routes import documents, health, query
from app.config import get_settings
from app.core.errors import AppError
from app.core.logging import configure_logging, log_with, new_request_id, request_id_ctx
from app.db import mongo, sql

log = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    configure_logging(settings.log_level)
    Path(settings.chroma_dir).mkdir(parents=True, exist_ok=True)
    Path(settings.upload_dir).mkdir(parents=True, exist_ok=True)
    Path("./data").mkdir(exist_ok=True)

    await sql.init_models()
    try:
        await mongo.ensure_indexes()
    except Exception as exc:  # noqa: BLE001 — /ready will report this
        log.warning("mongo index setup deferred: %s", exc)

    log.info("%s started in %s mode", settings.app_name, settings.env)
    yield
    log.info("shutting down")


def create_app() -> FastAPI:
    settings = get_settings()
    app = FastAPI(
        title="RAG Docs API",
        description=(
            "Production-shaped Retrieval-Augmented Generation service: document "
            "ingestion, hybrid retrieval, and cited answers over your own files."
        ),
        version="1.0.0",
        lifespan=lifespan,
        docs_url="/docs",
        openapi_url="/openapi.json",
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"] if settings.env != "prod" else [],
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.middleware("http")
    async def request_context(request: Request, call_next):
        rid = request.headers.get("x-request-id") or new_request_id()
        request_id_ctx.set(rid)
        started = time.perf_counter()
        response = await call_next(request)
        elapsed = (time.perf_counter() - started) * 1000
        response.headers["x-request-id"] = rid
        log_with(
            log,
            logging.INFO,
            "request",
            method=request.method,
            path=request.url.path,
            status=response.status_code,
            duration_ms=round(elapsed, 2),
        )
        return response

    @app.exception_handler(AppError)
    async def app_error_handler(request: Request, exc: AppError):
        return JSONResponse(
            status_code=exc.status_code,
            content={"detail": exc.detail, "code": exc.code},
        )

    @app.exception_handler(Exception)
    async def unhandled_handler(request: Request, exc: Exception):
        log.exception("unhandled error on %s", request.url.path)
        return JSONResponse(
            status_code=500,
            content={"detail": "Internal server error", "code": "internal_error"},
        )

    app.include_router(health.router)
    app.include_router(documents.router, prefix=settings.api_prefix)
    app.include_router(query.router, prefix=settings.api_prefix)
    return app


app = create_app()
