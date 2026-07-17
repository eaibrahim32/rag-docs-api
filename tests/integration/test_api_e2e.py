"""Full stack, nothing faked but the LLM: real Postgres + Mongo + Redis + Chroma.

This is the test that actually backs the claim on the README. Every unit-level
e2e test substitutes Mongo and the cache; here the bytes go all the way down.
"""

import pytest

from app.rag import llm as llm_mod

from .conftest import StubOllama

pytestmark = pytest.mark.asyncio

SAMPLE = b"""Employee Handbook - Leave Policy.

Permanent employees must give 30 days written notice before resignation.
Management grade staff must give 60 days notice.

Annual leave accrues at 1.75 days per completed month of service.
Unused annual leave may be carried into the next calendar year, capped at 10 days.

Error code LV-402 appears when a leave request overlaps an approved holiday.
"""


async def _ingest(client, name="handbook.txt", data=SAMPLE):
    resp = await client.post("/api/v1/documents", files={"file": (name, data, "text/plain")})
    assert resp.status_code == 202, resp.text
    return resp.json()["id"]


async def test_readiness_reports_every_store_healthy(client):
    resp = await client.get("/ready")
    body = resp.json()
    assert resp.status_code == 200, body
    assert body["status"] == "ready"
    assert set(body["checks"]) == {"postgres", "mongo", "redis", "chroma"}
    assert all(v == "ok" for v in body["checks"].values()), body["checks"]


async def test_ingestion_reaches_ready_through_real_stores(client):
    doc_id = await _ingest(client)
    resp = await client.get(f"/api/v1/documents/{doc_id}")
    body = resp.json()
    assert body["status"] == "ready", body.get("error")
    assert body["chunk_count"] > 0


async def test_hybrid_search_finds_content(client):
    await _ingest(client)
    resp = await client.get("/api/v1/search", params={"q": "notice period resignation", "top_k": 3})
    hits = resp.json()["hits"]
    assert hits
    assert any("30 days" in h["text"] for h in hits)


async def test_keyword_leg_retrieves_an_exact_error_code(client):
    """Dense vectors are poor at bare identifiers; the Mongo leg should catch this."""
    await _ingest(client)
    resp = await client.get("/api/v1/search", params={"q": "LV-402", "top_k": 5})
    hits = resp.json()["hits"]
    assert hits
    assert any("LV-402" in h["text"] for h in hits)


async def test_query_returns_cited_answer(client):
    with StubOllama(reply="Permanent staff give 30 days notice [1].") as stub:
        llm_mod.get_llm.cache_clear()
        llm = llm_mod.OllamaLLM(stub.url, "stub", timeout=10)
        original, llm_mod.get_llm = llm_mod.get_llm, lambda: llm
        try:
            await _ingest(client)
            resp = await client.post(
                "/api/v1/query", json={"question": "What is the notice period?"}
            )
        finally:
            llm_mod.get_llm = original
    body = resp.json()
    assert resp.status_code == 200, body
    assert body["citations"]
    assert body["cached"] is False


async def test_redis_cache_serves_the_second_identical_query(client):
    with StubOllama() as stub:
        llm = llm_mod.OllamaLLM(stub.url, "stub", timeout=10)
        original, llm_mod.get_llm = llm_mod.get_llm, lambda: llm
        try:
            await _ingest(client)
            payload = {"question": "What is the notice period?"}
            first = await client.post("/api/v1/query", json=payload)
            second = await client.post("/api/v1/query", json=payload)
        finally:
            llm_mod.get_llm = original

    assert first.json()["cached"] is False
    assert second.json()["cached"] is True
    # The cache hit must not have reached the model at all.
    assert len(stub.requests) == 1


async def test_ingesting_a_document_invalidates_cached_answers(client):
    with StubOllama() as stub:
        llm = llm_mod.OllamaLLM(stub.url, "stub", timeout=10)
        original, llm_mod.get_llm = llm_mod.get_llm, lambda: llm
        try:
            await _ingest(client)
            payload = {"question": "What is the notice period?"}
            await client.post("/api/v1/query", json=payload)
            await _ingest(client, "second.txt", SAMPLE + b"\nAddendum: interns give 7 days.")
            after = await client.post("/api/v1/query", json=payload)
        finally:
            llm_mod.get_llm = original
    assert after.json()["cached"] is False, "retrieval scope changed; answer must be recomputed"


async def test_duplicate_upload_is_deduplicated_in_postgres(client):
    first = await _ingest(client, "a.txt")
    second = await _ingest(client, "b.txt")
    assert first == second
    listing = await client.get("/api/v1/documents")
    assert listing.json()["total"] == 1


async def test_delete_purges_all_three_stores(client):
    from app.db import mongo

    doc_id = await _ingest(client)
    assert (await client.delete(f"/api/v1/documents/{doc_id}")).status_code == 204

    assert (await client.get(f"/api/v1/documents/{doc_id}")).status_code == 404
    assert await mongo.get_chunk(doc_id, 0) is None
    resp = await client.get("/api/v1/search", params={"q": "notice period resignation"})
    assert resp.json()["hits"] == []
