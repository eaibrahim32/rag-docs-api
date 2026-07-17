"""Real MongoDB, including the text index that powers hybrid search's keyword leg.

The unit suite replaces keyword_search with a naive set-overlap fake, so real
Mongo text-search behaviour — stemming, scoring, index requirements — is only
ever exercised here.
"""

import pytest

from app.db import mongo

pytestmark = pytest.mark.asyncio

CHUNKS = [
    {"chunk_index": 0, "text": "Kubernetes autoscaling adds replicas when CPU is high.",
     "start": 0, "end": 52, "filename": "ops.md"},
    {"chunk_index": 1, "text": "Prometheus scrapes metrics and stores them as time series.",
     "start": 52, "end": 110, "filename": "ops.md"},
    {"chunk_index": 2, "text": "Grafana renders dashboards from those time series.",
     "start": 110, "end": 160, "filename": "ops.md"},
]


async def test_ping_reaches_a_live_server():
    await mongo.ping()


async def test_put_and_get_chunks_roundtrip():
    await mongo.put_chunks("doc1", CHUNKS)
    got = await mongo.get_chunk("doc1", 1)
    assert "Prometheus" in got["text"]
    assert got["filename"] == "ops.md"


async def test_get_chunks_bulk_fetches_many_in_one_query():
    await mongo.put_chunks("doc1", CHUNKS)
    got = await mongo.get_chunks_bulk([("doc1", 0), ("doc1", 2)])
    assert set(got) == {("doc1", 0), ("doc1", 2)}


async def test_get_chunks_bulk_tolerates_missing_keys():
    await mongo.put_chunks("doc1", CHUNKS)
    got = await mongo.get_chunks_bulk([("doc1", 0), ("doc1", 99)])
    assert set(got) == {("doc1", 0)}


async def test_bulk_fetch_of_nothing_is_not_a_query():
    assert await mongo.get_chunks_bulk([]) == {}


async def test_keyword_search_uses_the_text_index():
    await mongo.put_chunks("doc1", CHUNKS)
    hits = await mongo.keyword_search("prometheus metrics", limit=5)
    assert hits
    assert hits[0]["chunk_index"] == 1


async def test_keyword_search_ranks_by_text_score():
    await mongo.put_chunks("doc1", CHUNKS)
    hits = await mongo.keyword_search("time series", limit=5)
    scores = [h["score"] for h in hits]
    assert scores == sorted(scores, reverse=True)


async def test_keyword_search_can_scope_to_documents():
    await mongo.put_chunks("doc1", CHUNKS)
    await mongo.put_chunks("doc2", CHUNKS)
    hits = await mongo.keyword_search("kubernetes", limit=10, document_ids=["doc2"])
    assert hits
    assert {h["document_id"] for h in hits} == {"doc2"}


async def test_keyword_search_misses_return_empty():
    await mongo.put_chunks("doc1", CHUNKS)
    assert await mongo.keyword_search("xylophone marzipan", limit=5) == []


async def test_put_chunks_replaces_rather_than_appends():
    await mongo.put_chunks("doc1", CHUNKS)
    await mongo.put_chunks("doc1", CHUNKS[:1])
    assert await mongo.get_chunk("doc1", 2) is None
    assert await mongo.get_chunk("doc1", 0) is not None


async def test_raw_document_roundtrip_and_upsert():
    await mongo.put_raw("doc1", "full text v1", {"filename": "ops.md", "pages": 2})
    await mongo.put_raw("doc1", "full text v2", {"filename": "ops.md", "pages": 3})
    raw = await mongo.get_db().raw_documents.find_one({"document_id": "doc1"})
    assert raw["text"] == "full text v2"
    assert raw["meta"]["pages"] == 3


async def test_delete_removes_chunks_and_raw():
    await mongo.put_chunks("doc1", CHUNKS)
    await mongo.put_raw("doc1", "full text", {})
    await mongo.delete_document_data("doc1")
    assert await mongo.get_chunk("doc1", 0) is None
    assert await mongo.get_db().raw_documents.find_one({"document_id": "doc1"}) is None


async def test_delete_is_scoped_to_one_document():
    await mongo.put_chunks("doc1", CHUNKS)
    await mongo.put_chunks("doc2", CHUNKS)
    await mongo.delete_document_data("doc1")
    assert await mongo.get_chunk("doc2", 0) is not None
