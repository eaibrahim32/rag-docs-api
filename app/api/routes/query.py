"""Retrieval endpoints.

/search returns raw ranked chunks with no LLM call — useful for debugging
retrieval quality in isolation, which is where most RAG bugs actually live.
/query runs the full generate-with-citations path.
"""

from fastapi import APIRouter, Query

from app.models.schemas import QueryRequest, QueryResponse, SearchResponse
from app.rag import pipeline

router = APIRouter(tags=["query"])


@router.post("/query", response_model=QueryResponse)
async def query_documents(payload: QueryRequest) -> QueryResponse:
    return await pipeline.answer(
        question=payload.question,
        top_k=payload.top_k,
        document_ids=payload.document_ids,
        use_cache=payload.use_cache,
    )


@router.get("/search", response_model=SearchResponse)
async def search_chunks(
    q: str = Query(min_length=2, max_length=2000),
    top_k: int = Query(5, ge=1, le=20),
) -> SearchResponse:
    hits = await pipeline.retrieve(q, top_k)
    return SearchResponse(hits=hits, strategy="hybrid:vector+keyword:rrf")
