"""Liveness and readiness probes.

/health is cheap and never touches dependencies — it answers "is the process
alive". /ready checks every backing service, which is what an orchestrator
should gate traffic on.
"""

import asyncio

from fastapi import APIRouter, Response, status

from app.db import mongo, redis_cache, sql
from app.models.schemas import HealthResponse, ReadinessResponse
from app.rag import vectorstore

router = APIRouter(tags=["health"])

VERSION = "1.0.0"


@router.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    return HealthResponse(status="ok", version=VERSION)


async def _check(name: str, coro) -> tuple[str, str]:
    try:
        await asyncio.wait_for(coro, timeout=3.0)
        return name, "ok"
    except Exception as exc:  # noqa: BLE001
        return name, f"error: {type(exc).__name__}"


@router.get("/ready", response_model=ReadinessResponse)
async def ready(response: Response) -> ReadinessResponse:
    results = await asyncio.gather(
        _check("postgres", sql.ping()),
        _check("mongo", mongo.ping()),
        _check("redis", redis_cache.ping()),
        _check("chroma", asyncio.to_thread(vectorstore.count)),
    )
    checks = dict(results)
    healthy = all(v == "ok" for v in checks.values())
    if not healthy:
        response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE
    return ReadinessResponse(status="ready" if healthy else "degraded", checks=checks)
