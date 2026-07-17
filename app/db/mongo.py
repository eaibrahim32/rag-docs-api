"""MongoDB (Motor) — stores raw extracted text and chunk bodies.

Chunk text is schemaless-ish and read by key, so it belongs in a document store
rather than in SQL rows or duplicated inside the vector index.
"""

from __future__ import annotations

from typing import Any

from motor.motor_asyncio import AsyncIOMotorClient, AsyncIOMotorDatabase

from app.config import get_settings

_client: AsyncIOMotorClient | None = None


def get_client() -> AsyncIOMotorClient:
    global _client
    if _client is None:
        _client = AsyncIOMotorClient(get_settings().mongo_url, serverSelectionTimeoutMS=3000)
    return _client


def get_db() -> AsyncIOMotorDatabase:
    return get_client()[get_settings().mongo_db]


async def ensure_indexes() -> None:
    db = get_db()
    await db.chunks.create_index("document_id")
    await db.chunks.create_index([("document_id", 1), ("chunk_index", 1)], unique=True)
    await db.chunks.create_index([("text", "text")])  # keyword leg of hybrid search
    await db.raw_documents.create_index("document_id", unique=True)


async def ping() -> None:
    await get_client().admin.command("ping")


async def put_raw(document_id: str, text: str, meta: dict[str, Any]) -> None:
    await get_db().raw_documents.update_one(
        {"document_id": document_id},
        {"$set": {"text": text, "meta": meta}},
        upsert=True,
    )


async def put_chunks(document_id: str, chunks: list[dict[str, Any]]) -> None:
    if not chunks:
        return
    await get_db().chunks.delete_many({"document_id": document_id})
    await get_db().chunks.insert_many(
        [{"document_id": document_id, **c} for c in chunks], ordered=False
    )


async def get_chunk(document_id: str, chunk_index: int) -> dict[str, Any] | None:
    return await get_db().chunks.find_one(
        {"document_id": document_id, "chunk_index": chunk_index}, {"_id": 0}
    )


async def get_chunks_bulk(keys: list[tuple[str, int]]) -> dict[tuple[str, int], dict]:
    """One round trip for many (document_id, chunk_index) pairs."""
    if not keys:
        return {}
    query = {"$or": [{"document_id": d, "chunk_index": i} for d, i in keys]}
    out: dict[tuple[str, int], dict] = {}
    async for doc in get_db().chunks.find(query, {"_id": 0}):
        out[(doc["document_id"], doc["chunk_index"])] = doc
    return out


async def keyword_search(
    query: str, limit: int = 10, document_ids: list[str] | None = None
) -> list[dict[str, Any]]:
    """BM25-ish leg of hybrid retrieval, via Mongo's text index."""
    filt: dict[str, Any] = {"$text": {"$search": query}}
    if document_ids:
        filt["document_id"] = {"$in": document_ids}
    cursor = (
        get_db()
        .chunks.find(filt, {"_id": 0, "score": {"$meta": "textScore"}})
        .sort([("score", {"$meta": "textScore"})])
        .limit(limit)
    )
    return [d async for d in cursor]


async def delete_document_data(document_id: str) -> None:
    await get_db().chunks.delete_many({"document_id": document_id})
    await get_db().raw_documents.delete_one({"document_id": document_id})
