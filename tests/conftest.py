"""Test fixtures.

Tests run against fakes for Mongo/Redis and the hash embedder, so `pytest` needs
no Docker, no network, and no model download. The RAG logic under test is the
same code that runs in production.
"""

import os
import tempfile

import pytest
import pytest_asyncio

os.environ.update(
    ENV="test",
    EMBEDDING_BACKEND="hash",
    CACHE_ENABLED="false",
    SQL_URL="sqlite+aiosqlite:///:memory:",
)


@pytest.fixture(autouse=True)
def tmp_chroma(monkeypatch):
    """Give every test an isolated, empty vector index.

    Settings, the embedder and the Chroma client are all lru_cached for
    performance in production, so the caches must be cleared here or a test
    would keep writing to the previous test's deleted temp directory.
    """
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


class FakeMongo:
    """In-memory stand-in for the Mongo chunk store."""

    def __init__(self):
        self.chunks: dict[tuple[str, int], dict] = {}
        self.raw: dict[str, dict] = {}

    async def put_raw(self, document_id, text, meta):
        self.raw[document_id] = {"text": text, "meta": meta}

    async def put_chunks(self, document_id, chunks):
        for c in chunks:
            self.chunks[(document_id, c["chunk_index"])] = {"document_id": document_id, **c}

    async def get_chunks_bulk(self, keys):
        return {k: self.chunks[k] for k in keys if k in self.chunks}

    async def keyword_search(self, query, limit=10, document_ids=None):
        terms = set(query.lower().split())
        scored = []
        for (doc_id, _idx), chunk in self.chunks.items():
            if document_ids and doc_id not in document_ids:
                continue
            overlap = len(terms & set(chunk["text"].lower().split()))
            if overlap:
                scored.append((overlap, chunk))
        scored.sort(key=lambda t: t[0], reverse=True)
        return [c for _, c in scored[:limit]]

    async def delete_document_data(self, document_id):
        for key in [k for k in self.chunks if k[0] == document_id]:
            del self.chunks[key]
        self.raw.pop(document_id, None)

    async def ping(self):
        return True

    async def ensure_indexes(self):
        return None


@pytest.fixture
def fake_mongo(monkeypatch):
    fake = FakeMongo()
    from app.db import mongo as mongo_mod

    for name in (
        "put_raw", "put_chunks", "get_chunks_bulk", "keyword_search",
        "delete_document_data", "ping", "ensure_indexes",
    ):
        monkeypatch.setattr(mongo_mod, name, getattr(fake, name))
    return fake


@pytest_asyncio.fixture
async def client(fake_mongo, monkeypatch):
    """App under test.

    Each test gets a fresh file-backed SQLite database. An in-memory SQLite URL
    would not work here: every async connection opens its own private database,
    so the schema created at startup would be invisible to request handlers.
    """
    from httpx import ASGITransport, AsyncClient

    import app.config as config_mod
    import app.db.sql as sql_mod
    from app.main import create_app

    with tempfile.TemporaryDirectory() as tmp:
        monkeypatch.setenv("SQL_URL", f"sqlite+aiosqlite:///{tmp}/test.db")
        config_mod.get_settings.cache_clear()
        sql_mod._engine = None
        sql_mod._sessionmaker = None

        app = create_app()
        await sql_mod.init_models()

        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test"
        ) as ac:
            yield ac

        await sql_mod.get_engine().dispose()
        sql_mod._engine = None
        sql_mod._sessionmaker = None
        config_mod.get_settings.cache_clear()


class FakeLLM:
    name = "fake/test-model"

    def __init__(self):
        self.calls: list[tuple[str, str]] = []

    async def complete(self, system: str, user: str) -> str:
        self.calls.append((system, user))
        return "Based on the context, HPA-503 indicates the metrics server is unreachable [1]."


@pytest.fixture
def fake_llm(monkeypatch):
    fake = FakeLLM()
    from app.rag import llm as llm_mod

    monkeypatch.setattr(llm_mod, "get_llm", lambda: fake)
    return fake


@pytest.fixture
def broken_llm(monkeypatch):
    """Simulates the LLM backend being down, to assert graceful 503 handling."""
    from app.core.errors import LLMUnavailable
    from app.rag import llm as llm_mod

    class Broken:
        name = "broken/test-model"

        async def complete(self, system, user):
            raise LLMUnavailable("backend down")

    monkeypatch.setattr(llm_mod, "get_llm", lambda: Broken())
