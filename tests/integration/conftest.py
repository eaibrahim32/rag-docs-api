"""Integration fixtures — these talk to REAL Postgres, Mongo and Redis.

Nothing here is faked. If a service is down, these tests fail loudly rather than
skipping, because a silently-skipped integration test is worse than none: it
reports green while proving nothing.

Connection strings come from the environment (CI injects service-container
hosts); the defaults match `docker compose up`.
"""

import os
import socket
import tempfile
import threading
import time
from contextlib import closing

import pytest
import pytest_asyncio

# setdefault, not update: CI's env must win. Hard-set only what is truly fixed.
os.environ["ENV"] = "test"
os.environ["CACHE_ENABLED"] = "true"
os.environ.setdefault("EMBEDDING_BACKEND", "onnx")
os.environ.setdefault("SQL_URL", "postgresql+asyncpg://rag:ragpass@localhost:5432/ragdocs")
os.environ.setdefault("MONGO_URL", "mongodb://localhost:27017")
os.environ.setdefault("REDIS_URL", "redis://localhost:6379/0")


def free_port() -> int:
    with closing(socket.socket()) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@pytest.fixture(autouse=True)
def isolated_chroma(monkeypatch):
    """Fresh vector index per test; clear the lru_caches that would pin the old one."""
    import app.config as config_mod
    from app.rag import embeddings, vectorstore

    def reset():
        config_mod.get_settings.cache_clear()
        embeddings.get_embedder.cache_clear()
        vectorstore.get_collection.cache_clear()

    with tempfile.TemporaryDirectory() as d:
        monkeypatch.setenv("CHROMA_DIR", d)
        reset()
        yield d
        reset()


@pytest_asyncio.fixture(autouse=True)
async def clean_stores():
    """Truncate every real store before each test so ordering can't matter.

    Also resets the module-level clients. Each of the three caches a connection
    pool bound to the event loop that created it, and pytest-asyncio builds a
    fresh loop per test — reuse across tests raises "Event loop is closed".
    Production never hits this: uvicorn runs one loop for the process lifetime.
    """
    import app.db.mongo as mongo_mod
    import app.db.redis_cache as redis_mod
    import app.db.sql as sql_mod
    from app.db import mongo, redis_cache

    sql_mod._engine = None
    sql_mod._sessionmaker = None
    mongo_mod._client = None
    redis_mod._client = None

    await sql_mod.init_models()
    async with sql_mod.get_sessionmaker()() as session:
        from sqlalchemy import text

        await session.execute(text("TRUNCATE TABLE documents"))
        await session.commit()

    await mongo.get_db().chunks.delete_many({})
    await mongo.get_db().raw_documents.delete_many({})
    await mongo.ensure_indexes()
    await redis_cache.get_client().flushdb()

    yield

    await sql_mod.get_engine().dispose()
    await redis_mod.get_client().aclose()
    mongo_mod.get_client().close()
    sql_mod._engine = None
    sql_mod._sessionmaker = None
    mongo_mod._client = None
    redis_mod._client = None


@pytest_asyncio.fixture
async def client():
    from httpx import ASGITransport, AsyncClient

    from app.main import create_app

    app = create_app()
    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as ac:
        yield ac


class StubOllama:
    """A real HTTP server speaking Ollama's /api/chat shape.

    Lets the genuine httpx client in app/rag/llm.py be exercised over a real
    socket — status codes, JSON parsing, timeouts — without a 1.3 GB model pull.
    """

    def __init__(self, reply: str = "Stub answer grounded in [1].", status: int = 200):
        self.reply = reply
        self.status = status
        self.requests: list[dict] = []
        self.port = free_port()
        self._server = None
        self._thread = None

    def _build_app(self):
        from starlette.applications import Starlette
        from starlette.responses import JSONResponse
        from starlette.routing import Route

        async def chat(request):
            self.requests.append(await request.json())
            if self.status != 200:
                return JSONResponse({"error": "boom"}, status_code=self.status)
            return JSONResponse({"message": {"role": "assistant", "content": self.reply}})

        return Starlette(routes=[Route("/api/chat", chat, methods=["POST"])])

    def __enter__(self):
        import uvicorn

        config = uvicorn.Config(
            self._build_app(), host="127.0.0.1", port=self.port, log_level="error"
        )
        self._server = uvicorn.Server(config)
        self._thread = threading.Thread(target=self._server.run, daemon=True)
        self._thread.start()
        for _ in range(100):  # wait for bind
            if self._server.started:
                break
            time.sleep(0.05)
        return self

    def __exit__(self, *exc):
        self._server.should_exit = True
        self._thread.join(timeout=5)

    @property
    def url(self) -> str:
        return f"http://127.0.0.1:{self.port}"
