"""End-to-end: upload a document, ingest it, and retrieve it back.

This exercises the real chunker, the real hash embedder, the real Chroma index
and the real fusion code. Only Mongo and the LLM are faked, so a regression
anywhere in the ingestion or retrieval path fails here.
"""

import pytest

pytestmark = pytest.mark.asyncio

SAMPLE = b"""Kubernetes Horizontal Pod Autoscaler Guide.

The HorizontalPodAutoscaler automatically scales the number of pods in a
deployment based on observed CPU utilization. When average CPU crosses the
target threshold, the controller adds replicas up to maxReplicas.

Error code HPA-503 means the metrics server is unreachable. Check that
metrics-server is running in the kube-system namespace.

Prometheus scrapes metrics from instrumented targets at a fixed interval and
stores them as time series identified by metric name and key-value labels.
"""


async def _ingest(client, filename: str, data: bytes) -> str:
    resp = await client.post(
        "/api/v1/documents", files={"file": (filename, data, "text/plain")}
    )
    assert resp.status_code == 202, resp.text
    return resp.json()["id"]


async def test_upload_returns_202_and_pending_status(client):
    resp = await client.post(
        "/api/v1/documents", files={"file": ("hpa.txt", SAMPLE, "text/plain")}
    )
    assert resp.status_code == 202
    body = resp.json()
    assert body["status"] in {"pending", "processing", "ready"}
    assert body["filename"] == "hpa.txt"
    assert body["size_bytes"] == len(SAMPLE)


async def test_document_reaches_ready_with_chunks(client):
    doc_id = await _ingest(client, "hpa.txt", SAMPLE)
    resp = await client.get(f"/api/v1/documents/{doc_id}")
    body = resp.json()
    assert body["status"] == "ready", body.get("error")
    assert body["chunk_count"] > 0
    assert body["error"] is None


async def test_identical_upload_is_deduplicated(client):
    first = await _ingest(client, "hpa.txt", SAMPLE)
    second = await _ingest(client, "hpa-copy.txt", SAMPLE)
    assert first == second

    listing = await client.get("/api/v1/documents")
    assert listing.json()["total"] == 1


async def test_semantic_search_finds_the_relevant_chunk(client):
    await _ingest(client, "hpa.txt", SAMPLE)
    resp = await client.get("/api/v1/search", params={"q": "HPA-503 metrics server", "top_k": 3})
    assert resp.status_code == 200
    body = resp.json()
    assert body["strategy"] == "hybrid:vector+keyword:rrf"
    assert body["hits"], "hybrid retrieval returned nothing"
    assert "HPA-503" in " ".join(h["text"] for h in body["hits"])


async def test_search_hits_are_ranked_descending(client):
    await _ingest(client, "hpa.txt", SAMPLE)
    resp = await client.get("/api/v1/search", params={"q": "prometheus time series", "top_k": 5})
    scores = [h["score"] for h in resp.json()["hits"]]
    assert scores == sorted(scores, reverse=True)


async def test_delete_removes_document_from_retrieval(client):
    doc_id = await _ingest(client, "hpa.txt", SAMPLE)
    assert (await client.delete(f"/api/v1/documents/{doc_id}")).status_code == 204
    assert (await client.get(f"/api/v1/documents/{doc_id}")).status_code == 404

    resp = await client.get("/api/v1/search", params={"q": "HPA-503 metrics server"})
    assert resp.json()["hits"] == []


async def test_query_returns_answer_with_citations(client, fake_llm):
    await _ingest(client, "hpa.txt", SAMPLE)
    resp = await client.post(
        "/api/v1/query", json={"question": "What does error HPA-503 mean?", "use_cache": False}
    )
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["citations"], "answer arrived with no citations"
    assert body["cached"] is False
    assert body["latency_ms"] >= 0
    assert body["model"] == "fake/test-model"


async def test_query_with_no_indexed_documents_says_so(client, fake_llm):
    resp = await client.post("/api/v1/query", json={"question": "anything at all?"})
    body = resp.json()
    assert body["citations"] == []
    assert body["model"] == "none"


async def test_llm_outage_surfaces_as_503(client, broken_llm):
    await _ingest(client, "hpa.txt", SAMPLE)
    resp = await client.post(
        "/api/v1/query", json={"question": "What does error HPA-503 mean?", "use_cache": False}
    )
    assert resp.status_code == 503
    assert resp.json()["code"] == "llm_unavailable"
