"""Real PostgreSQL. Unit tests run SQLite, which forgives things Postgres will not."""

import uuid

import pytest

from app.db import sql

pytestmark = pytest.mark.asyncio


def _doc(**over):
    base = {
        "id": uuid.uuid4().hex,
        "filename": "handbook.pdf",
        "content_type": "application/pdf",
        "size_bytes": 1024,
        "sha256": uuid.uuid4().hex * 2,
        "status": "pending",
    }
    base.update(over)
    return base


async def test_engine_is_actually_postgres():
    assert sql.get_engine().dialect.name == "postgresql"


async def test_create_and_read_roundtrip():
    async with sql.get_sessionmaker()() as s:
        created = await sql.create_document(s, **_doc())
        fetched = await sql.get_document(s, created.id)
    assert fetched is not None
    assert fetched.filename == "handbook.pdf"
    assert fetched.status == "pending"


async def test_timestamps_are_timezone_aware():
    """Postgres returns tz-aware datetimes; SQLite happily returns naive ones."""
    async with sql.get_sessionmaker()() as s:
        doc = await sql.create_document(s, **_doc())
    assert doc.created_at.tzinfo is not None


async def test_status_transition_persists_chunk_count():
    async with sql.get_sessionmaker()() as s:
        doc = await sql.create_document(s, **_doc())
        await sql.set_status(s, doc.id, "ready", chunk_count=17)
        again = await sql.get_document(s, doc.id)
    assert again.status == "ready"
    assert again.chunk_count == 17
    assert again.error is None


async def test_failed_status_records_the_error():
    async with sql.get_sessionmaker()() as s:
        doc = await sql.create_document(s, **_doc())
        await sql.set_status(s, doc.id, "failed", error="no extractable text")
        again = await sql.get_document(s, doc.id)
    assert again.status == "failed"
    assert "no extractable text" in again.error


async def test_find_by_sha_backs_deduplication():
    sha = uuid.uuid4().hex * 2
    async with sql.get_sessionmaker()() as s:
        await sql.create_document(s, **_doc(sha256=sha))
        found = await sql.find_by_sha(s, sha)
        missing = await sql.find_by_sha(s, "0" * 64)
    assert found is not None
    assert missing is None


async def test_pagination_and_total_are_consistent():
    async with sql.get_sessionmaker()() as s:
        for i in range(5):
            await sql.create_document(s, **_doc(filename=f"f{i}.txt"))
        page, total = await sql.list_documents(s, limit=2, offset=0)
        page2, _ = await sql.list_documents(s, limit=2, offset=2)
    assert total == 5
    assert len(page) == 2 and len(page2) == 2
    assert {d.id for d in page}.isdisjoint({d.id for d in page2})


async def test_status_filter_narrows_results():
    async with sql.get_sessionmaker()() as s:
        a = await sql.create_document(s, **_doc())
        await sql.create_document(s, **_doc())
        await sql.set_status(s, a.id, "ready", chunk_count=1)
        ready, total = await sql.list_documents(s, status="ready")
    assert total == 1
    assert ready[0].id == a.id


async def test_delete_removes_the_row():
    async with sql.get_sessionmaker()() as s:
        doc = await sql.create_document(s, **_doc())
        assert await sql.delete_document(s, doc.id) is True
        assert await sql.get_document(s, doc.id) is None
        assert await sql.delete_document(s, "nonexistent") is False
