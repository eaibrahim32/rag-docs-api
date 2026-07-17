"""SQL layer (SQLAlchemy 2.0 async) — document metadata and ingestion state.

SQL owns anything we need to filter, sort, or paginate: status, timestamps,
chunk counts. Chunk bodies live in Mongo, vectors live in Chroma.
"""

from __future__ import annotations

from datetime import UTC, datetime

from sqlalchemy import DateTime, Integer, String, Text, func, select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column

from app.config import get_settings


def _utcnow() -> datetime:
    return datetime.now(UTC)


class Base(DeclarativeBase):
    pass


class Document(Base):
    __tablename__ = "documents"

    id: Mapped[str] = mapped_column(String(32), primary_key=True)
    filename: Mapped[str] = mapped_column(String(512), index=True)
    content_type: Mapped[str] = mapped_column(String(128))
    size_bytes: Mapped[int] = mapped_column(Integer)
    sha256: Mapped[str] = mapped_column(String(64), index=True)
    status: Mapped[str] = mapped_column(String(16), index=True, default="pending")
    chunk_count: Mapped[int] = mapped_column(Integer, default=0)
    error: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=_utcnow, onupdate=_utcnow
    )


_engine = None
_sessionmaker: async_sessionmaker[AsyncSession] | None = None


def get_engine():
    global _engine
    if _engine is None:
        _engine = create_async_engine(get_settings().sql_url, future=True)
    return _engine


def get_sessionmaker() -> async_sessionmaker[AsyncSession]:
    global _sessionmaker
    if _sessionmaker is None:
        _sessionmaker = async_sessionmaker(get_engine(), expire_on_commit=False)
    return _sessionmaker


async def init_models() -> None:
    async with get_engine().begin() as conn:
        await conn.run_sync(Base.metadata.create_all)


async def ping() -> None:
    async with get_sessionmaker()() as session:
        await session.execute(select(func.now() if False else 1))


# --- repository helpers -------------------------------------------------


async def create_document(session: AsyncSession, **kwargs) -> Document:
    doc = Document(**kwargs)
    session.add(doc)
    await session.commit()
    return doc


async def get_document(session: AsyncSession, doc_id: str) -> Document | None:
    return await session.get(Document, doc_id)


async def find_by_sha(session: AsyncSession, sha256: str) -> Document | None:
    result = await session.execute(select(Document).where(Document.sha256 == sha256))
    return result.scalars().first()


async def list_documents(
    session: AsyncSession, limit: int = 20, offset: int = 0, status: str | None = None
) -> tuple[list[Document], int]:
    stmt = select(Document)
    count_stmt = select(func.count()).select_from(Document)
    if status:
        stmt = stmt.where(Document.status == status)
        count_stmt = count_stmt.where(Document.status == status)
    stmt = stmt.order_by(Document.created_at.desc()).limit(limit).offset(offset)
    items = (await session.execute(stmt)).scalars().all()
    total = (await session.execute(count_stmt)).scalar_one()
    return list(items), total


async def set_status(
    session: AsyncSession,
    doc_id: str,
    status: str,
    chunk_count: int | None = None,
    error: str | None = None,
) -> None:
    doc = await session.get(Document, doc_id)
    if doc is None:
        return
    doc.status = status
    doc.error = error
    if chunk_count is not None:
        doc.chunk_count = chunk_count
    doc.updated_at = _utcnow()
    await session.commit()


async def delete_document(session: AsyncSession, doc_id: str) -> bool:
    doc = await session.get(Document, doc_id)
    if doc is None:
        return False
    await session.delete(doc)
    await session.commit()
    return True
