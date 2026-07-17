"""Document CRUD + ingestion.

Upload returns 202 immediately and ingests in the background: embedding a 200-page
PDF takes tens of seconds, and holding an HTTP connection open for that is a
timeout waiting to happen. Clients poll GET /documents/{id} for status.
Re-uploading identical bytes is deduplicated by SHA-256.
"""

from __future__ import annotations

import hashlib
import uuid
from pathlib import Path

from fastapi import APIRouter, BackgroundTasks, Depends, File, Query, UploadFile, status
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import get_settings
from app.core.errors import NotFoundError, PayloadTooLarge, UnsupportedFileType
from app.db import mongo, redis_cache, sql
from app.deps import get_session
from app.models.schemas import DocumentList, DocumentOut, IngestStatus
from app.rag import loaders, pipeline, vectorstore

router = APIRouter(prefix="/documents", tags=["documents"])


def _to_out(doc: sql.Document) -> DocumentOut:
    return DocumentOut(
        id=doc.id,
        filename=doc.filename,
        content_type=doc.content_type,
        size_bytes=doc.size_bytes,
        status=IngestStatus(doc.status),
        chunk_count=doc.chunk_count,
        error=doc.error,
        created_at=doc.created_at,
        updated_at=doc.updated_at,
    )


@router.post("", response_model=DocumentOut, status_code=status.HTTP_202_ACCEPTED)
async def upload_document(
    background: BackgroundTasks,
    file: UploadFile = File(...),
    session: AsyncSession = Depends(get_session),
) -> DocumentOut:
    settings = get_settings()

    ext = Path(file.filename or "").suffix.lower()
    if ext not in loaders.SUPPORTED:
        raise UnsupportedFileType(
            f"'{ext or file.filename}' is not supported. "
            f"Accepted: {', '.join(sorted(loaders.SUPPORTED))}"
        )

    data = await file.read()
    max_bytes = settings.max_upload_mb * 1024 * 1024
    if len(data) > max_bytes:
        raise PayloadTooLarge(f"File exceeds the {settings.max_upload_mb} MB limit")
    if not data:
        raise UnsupportedFileType("Uploaded file is empty")

    sha = hashlib.sha256(data).hexdigest()
    existing = await sql.find_by_sha(session, sha)
    if existing:
        return _to_out(existing)  # idempotent re-upload

    doc = await sql.create_document(
        session,
        id=uuid.uuid4().hex,
        filename=file.filename or "untitled",
        content_type=file.content_type or "application/octet-stream",
        size_bytes=len(data),
        sha256=sha,
        status="pending",
    )

    background.add_task(
        pipeline.ingest_document, sql.get_sessionmaker(), doc.id, doc.filename, data
    )
    return _to_out(doc)


@router.get("", response_model=DocumentList)
async def list_documents(
    limit: int = Query(20, ge=1, le=100),
    offset: int = Query(0, ge=0),
    status_filter: IngestStatus | None = Query(None, alias="status"),
    session: AsyncSession = Depends(get_session),
) -> DocumentList:
    items, total = await sql.list_documents(
        session, limit, offset, status_filter.value if status_filter else None
    )
    return DocumentList(
        items=[_to_out(d) for d in items], total=total, limit=limit, offset=offset
    )


@router.get("/{document_id}", response_model=DocumentOut)
async def get_document(
    document_id: str, session: AsyncSession = Depends(get_session)
) -> DocumentOut:
    doc = await sql.get_document(session, document_id)
    if doc is None:
        raise NotFoundError(f"No document with id {document_id}")
    return _to_out(doc)


@router.delete("/{document_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_document(
    document_id: str, session: AsyncSession = Depends(get_session)
) -> None:
    """Deletes across all three stores. SQL last, so a partial failure is retryable."""
    doc = await sql.get_document(session, document_id)
    if doc is None:
        raise NotFoundError(f"No document with id {document_id}")
    await mongo.delete_document_data(document_id)
    vectorstore.delete_document(document_id)
    await sql.delete_document(session, document_id)
    await redis_cache.invalidate_answers()
