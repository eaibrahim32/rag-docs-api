"""RAG orchestration: ingestion and hybrid retrieval.

Ingestion:  bytes -> text -> chunks -> (Mongo bodies + Chroma vectors) -> SQL status
Retrieval:  vector search + Mongo keyword search, fused with Reciprocal Rank
            Fusion, then hydrated from Mongo and handed to the LLM.

Why hybrid: dense vectors handle paraphrase ("how do I cancel" vs "termination
clause") but miss exact identifiers (error codes, product SKUs, names) that a
keyword index nails. RRF merges the two rankings without needing the scores to
be on a comparable scale.
"""

from __future__ import annotations

import asyncio
import logging
import time

from app.config import get_settings
from app.core.errors import LLMUnavailable
from app.db import mongo, redis_cache
from app.db.sql import set_status
from app.models.schemas import Citation, QueryResponse, SearchHit
from app.rag import chunker, llm, loaders, vectorstore
from app.rag.embeddings import get_embedder

log = logging.getLogger(__name__)

RRF_K = 60  # damping constant; standard value from the original RRF paper


async def ingest_document(session_factory, document_id: str, filename: str, data: bytes) -> int:
    """Full ingestion pipeline. Returns chunk count. Runs as a background task."""
    settings = get_settings()
    async with session_factory() as session:
        await set_status(session, document_id, "processing")

    try:
        text, page_map = await asyncio.to_thread(loaders.load, filename, data)
        if not text.strip():
            raise ValueError("No extractable text found in document")

        chunks = chunker.chunk_text(text, settings.chunk_size, settings.chunk_overlap)
        if not chunks:
            raise ValueError("Document produced zero chunks")

        payload = [{**c.as_dict(), "filename": filename} for c in chunks]
        await mongo.put_raw(document_id, text, {"filename": filename, "pages": len(page_map)})
        await mongo.put_chunks(document_id, payload)

        embedder = get_embedder()
        vectors = await asyncio.to_thread(embedder.embed, [c.text for c in chunks])
        await asyncio.to_thread(
            vectorstore.upsert, document_id, [c.chunk_index for c in chunks], vectors
        )

        async with session_factory() as session:
            await set_status(session, document_id, "ready", chunk_count=len(chunks))
        await redis_cache.invalidate_answers()

        log.info("ingested %s (%d chunks)", filename, len(chunks))
        return len(chunks)

    except Exception as exc:  # noqa: BLE001 — status must reflect any failure
        log.exception("ingestion failed for %s", document_id)
        async with session_factory() as session:
            await set_status(session, document_id, "failed", error=str(exc)[:500])
        return 0


def _rrf_fuse(
    rankings: list[list[tuple[str, int]]], k: int = RRF_K
) -> list[tuple[str, int, float]]:
    """Reciprocal Rank Fusion: score = sum over lists of 1/(k + rank)."""
    scores: dict[tuple[str, int], float] = {}
    for ranking in rankings:
        for rank, key in enumerate(ranking, start=1):
            scores[key] = scores.get(key, 0.0) + 1.0 / (k + rank)
    ordered = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
    return [(doc_id, idx, score) for (doc_id, idx), score in ordered]


async def retrieve(
    question: str, top_k: int, document_ids: list[str] | None = None
) -> list[SearchHit]:
    embedder = get_embedder()
    vector = await asyncio.to_thread(embedder.embed_one, question)

    vector_task = asyncio.to_thread(vectorstore.query, vector, top_k * 3, document_ids)
    keyword_task = mongo.keyword_search(question, top_k * 3, document_ids)
    vector_hits, keyword_hits = await asyncio.gather(
        vector_task, keyword_task, return_exceptions=True
    )

    rankings: list[list[tuple[str, int]]] = []
    if isinstance(vector_hits, list):
        rankings.append([(h.document_id, h.chunk_index) for h in vector_hits])
    else:
        log.warning("vector leg failed: %s", vector_hits)
    if isinstance(keyword_hits, list):
        rankings.append([(h["document_id"], h["chunk_index"]) for h in keyword_hits])
    else:
        log.warning("keyword leg failed: %s", keyword_hits)

    if not rankings:
        return []

    fused = _rrf_fuse(rankings)[:top_k]
    bodies = await mongo.get_chunks_bulk([(d, i) for d, i, _ in fused])

    hits: list[SearchHit] = []
    for doc_id, idx, score in fused:
        chunk = bodies.get((doc_id, idx))
        if not chunk:
            continue
        hits.append(
            SearchHit(
                document_id=doc_id,
                filename=chunk.get("filename", "unknown"),
                chunk_index=idx,
                score=round(score, 6),
                text=chunk["text"],
            )
        )
    return hits


async def answer(
    question: str, top_k: int | None = None, document_ids: list[str] | None = None,
    use_cache: bool = True,
) -> QueryResponse:
    settings = get_settings()
    top_k = top_k or settings.top_k
    started = time.perf_counter()

    key = redis_cache.cache_key(question, top_k, document_ids)
    if use_cache:
        cached = await redis_cache.get_json(key)
        if cached:
            return QueryResponse(
                answer=cached["answer"],
                citations=[Citation(**c) for c in cached["citations"]],
                cached=True,
                latency_ms=int((time.perf_counter() - started) * 1000),
                model=cached["model"],
            )

    hits = await retrieve(question, top_k, document_ids)
    if not hits:
        return QueryResponse(
            answer="I couldn't find anything relevant in the indexed documents.",
            citations=[],
            cached=False,
            latency_ms=int((time.perf_counter() - started) * 1000),
            model="none",
        )

    client = llm.get_llm()
    prompt = llm.build_prompt(question, [h.text for h in hits])
    try:
        text = await client.complete(llm.SYSTEM_PROMPT, prompt)
    except LLMUnavailable:
        raise

    citations = [
        Citation(
            document_id=h.document_id,
            filename=h.filename,
            chunk_index=h.chunk_index,
            score=h.score,
            snippet=h.text[:280],
        )
        for h in hits
    ]
    response = QueryResponse(
        answer=text,
        citations=citations,
        cached=False,
        latency_ms=int((time.perf_counter() - started) * 1000),
        model=client.name,
    )
    if use_cache:
        await redis_cache.set_json(
            key,
            {
                "answer": response.answer,
                "citations": [c.model_dump() for c in citations],
                "model": client.name,
            },
        )
    return response
