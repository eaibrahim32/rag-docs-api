"""ChromaDB wrapper — the vector leg of retrieval.

Chroma holds vectors plus the minimum metadata needed to resolve a hit back to
its chunk in Mongo. Chunk text is not duplicated here.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

import chromadb
from chromadb.config import Settings as ChromaSettings

from app.config import get_settings


@dataclass
class VectorHit:
    document_id: str
    chunk_index: int
    score: float  # cosine similarity, higher is better


@lru_cache
def get_collection():
    settings = get_settings()
    client = chromadb.PersistentClient(
        path=settings.chroma_dir,
        settings=ChromaSettings(anonymized_telemetry=False, allow_reset=True),
    )
    return client.get_or_create_collection(
        name=settings.chroma_collection,
        metadata={"hnsw:space": "cosine"},
    )


def _vector_id(document_id: str, chunk_index: int) -> str:
    return f"{document_id}:{chunk_index}"


def upsert(document_id: str, chunk_indices: list[int], vectors: list[list[float]]) -> None:
    if not chunk_indices:
        return
    get_collection().upsert(
        ids=[_vector_id(document_id, i) for i in chunk_indices],
        embeddings=vectors,
        metadatas=[{"document_id": document_id, "chunk_index": i} for i in chunk_indices],
    )


def query(
    vector: list[float], top_k: int = 5, document_ids: list[str] | None = None
) -> list[VectorHit]:
    where = {"document_id": {"$in": document_ids}} if document_ids else None
    result = get_collection().query(
        query_embeddings=[vector],
        n_results=top_k,
        where=where,
        include=["metadatas", "distances"],
    )
    metas = result.get("metadatas", [[]])[0]
    distances = result.get("distances", [[]])[0]
    hits = []
    for meta, dist in zip(metas, distances, strict=False):
        hits.append(
            VectorHit(
                document_id=meta["document_id"],
                chunk_index=int(meta["chunk_index"]),
                score=1.0 - float(dist),  # cosine distance -> similarity
            )
        )
    return hits


def delete_document(document_id: str) -> None:
    get_collection().delete(where={"document_id": document_id})


def count() -> int:
    return get_collection().count()
