"""API contract tests against the real ASGI app (fakes only for I/O backends)."""

import pytest

pytestmark = pytest.mark.asyncio


async def test_health_is_ok(client):
    resp = await client.get("/health")
    assert resp.status_code == 200
    assert resp.json()["status"] == "ok"


async def test_health_sets_request_id_header(client):
    resp = await client.get("/health")
    assert resp.headers.get("x-request-id")


async def test_request_id_is_echoed_when_supplied(client):
    resp = await client.get("/health", headers={"x-request-id": "trace-abc"})
    assert resp.headers["x-request-id"] == "trace-abc"


async def test_openapi_schema_is_served(client):
    resp = await client.get("/openapi.json")
    assert resp.status_code == 200
    assert "/api/v1/documents" in resp.json()["paths"]


async def test_upload_rejects_unsupported_type(client):
    resp = await client.post(
        "/api/v1/documents",
        files={"file": ("virus.exe", b"MZ\x90", "application/octet-stream")},
    )
    assert resp.status_code == 415
    assert resp.json()["code"] == "unsupported_file_type"


async def test_upload_rejects_empty_file(client):
    resp = await client.post(
        "/api/v1/documents", files={"file": ("empty.txt", b"", "text/plain")}
    )
    assert resp.status_code == 415


async def test_missing_document_returns_404(client):
    resp = await client.get("/api/v1/documents/does-not-exist")
    assert resp.status_code == 404
    assert resp.json()["code"] == "not_found"


async def test_query_validates_question_length(client):
    resp = await client.post("/api/v1/query", json={"question": "hi"})
    assert resp.status_code == 422


async def test_query_rejects_out_of_range_top_k(client):
    resp = await client.post(
        "/api/v1/query", json={"question": "what is in the docs?", "top_k": 500}
    )
    assert resp.status_code == 422


async def test_list_documents_pagination_defaults(client):
    resp = await client.get("/api/v1/documents")
    assert resp.status_code == 200
    body = resp.json()
    assert body["limit"] == 20 and body["offset"] == 0
