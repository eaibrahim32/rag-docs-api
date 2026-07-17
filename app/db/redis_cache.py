"""Redis — answer cache and a sliding-window rate limiter.

The cache key is a hash of the normalised question plus the retrieval scope, so
repeated questions skip both embedding and LLM inference. Cache failures are
logged and swallowed: a cold cache must never take the API down.
"""

from __future__ import annotations

import hashlib
import json
import logging
import time
from typing import Any

import redis.asyncio as aioredis

from app.config import get_settings

log = logging.getLogger(__name__)

_client: aioredis.Redis | None = None


def get_client() -> aioredis.Redis:
    global _client
    if _client is None:
        _client = aioredis.from_url(
            get_settings().redis_url, encoding="utf-8", decode_responses=True
        )
    return _client


async def ping() -> None:
    await get_client().ping()


def cache_key(question: str, top_k: int, document_ids: list[str] | None) -> str:
    scope = ",".join(sorted(document_ids)) if document_ids else "*"
    raw = f"{question.strip().lower()}|{top_k}|{scope}"
    return "answer:" + hashlib.sha256(raw.encode()).hexdigest()[:32]


async def get_json(key: str) -> dict[str, Any] | None:
    settings = get_settings()
    if not settings.cache_enabled:
        return None
    try:
        value = await get_client().get(key)
        return json.loads(value) if value else None
    except Exception as exc:  # noqa: BLE001 — cache must degrade, not fail
        log.warning("cache read failed: %s", exc)
        return None


async def set_json(key: str, value: dict[str, Any], ttl: int | None = None) -> None:
    settings = get_settings()
    if not settings.cache_enabled:
        return
    try:
        await get_client().set(key, json.dumps(value, default=str), ex=ttl or settings.cache_ttl_s)
    except Exception as exc:  # noqa: BLE001
        log.warning("cache write failed: %s", exc)


async def invalidate_answers() -> None:
    """Called after ingest/delete — retrieval scope changed, answers are stale."""
    try:
        client = get_client()
        async for key in client.scan_iter(match="answer:*", count=500):
            await client.delete(key)
    except Exception as exc:  # noqa: BLE001
        log.warning("cache invalidation failed: %s", exc)


async def allow_request(identity: str, limit: int, window_s: int) -> bool:
    """Sliding-window counter. Fails open if Redis is unreachable."""
    key = f"rate:{identity}:{int(time.time() // window_s)}"
    try:
        client = get_client()
        count = await client.incr(key)
        if count == 1:
            await client.expire(key, window_s)
        return count <= limit
    except Exception as exc:  # noqa: BLE001
        log.warning("rate limiter unavailable, failing open: %s", exc)
        return True
