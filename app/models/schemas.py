"""Pydantic v2 schemas — the public contract of the REST API."""

from datetime import datetime
from enum import StrEnum

from pydantic import BaseModel, Field


class IngestStatus(StrEnum):
    PENDING = "pending"
    PROCESSING = "processing"
    READY = "ready"
    FAILED = "failed"


class DocumentOut(BaseModel):
    id: str
    filename: str
    content_type: str
    size_bytes: int
    status: IngestStatus
    chunk_count: int = 0
    error: str | None = None
    created_at: datetime
    updated_at: datetime


class DocumentList(BaseModel):
    items: list[DocumentOut]
    total: int
    limit: int
    offset: int


class Citation(BaseModel):
    document_id: str
    filename: str
    chunk_index: int
    score: float
    snippet: str


class QueryRequest(BaseModel):
    question: str = Field(min_length=3, max_length=2000)
    top_k: int | None = Field(default=None, ge=1, le=20)
    document_ids: list[str] | None = Field(
        default=None, description="Restrict retrieval to these documents."
    )
    use_cache: bool = True


class QueryResponse(BaseModel):
    answer: str
    citations: list[Citation]
    cached: bool = False
    latency_ms: int
    model: str


class SearchHit(BaseModel):
    document_id: str
    filename: str
    chunk_index: int
    score: float
    text: str


class SearchResponse(BaseModel):
    hits: list[SearchHit]
    strategy: str


class HealthResponse(BaseModel):
    status: str
    version: str


class ReadinessResponse(BaseModel):
    status: str
    checks: dict[str, str]


class ErrorResponse(BaseModel):
    detail: str
    code: str
