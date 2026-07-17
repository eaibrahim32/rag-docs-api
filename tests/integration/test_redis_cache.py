"""Real Redis. Unit tests disable the cache entirely, so this is its only cover."""

import asyncio

import pytest

from app.db import redis_cache

pytestmark = pytest.mark.asyncio


async def test_ping_reaches_a_live_server():
    await redis_cache.ping()


async def test_set_then_get_roundtrips_json():
    await redis_cache.set_json("answer:t1", {"answer": "hi", "citations": [], "model": "m"})
    got = await redis_cache.get_json("answer:t1")
    assert got["answer"] == "hi"


async def test_missing_key_returns_none():
    assert await redis_cache.get_json("answer:absent") is None


async def test_ttl_is_applied_and_expires():
    await redis_cache.set_json("answer:ttl", {"a": 1}, ttl=1)
    assert await redis_cache.get_json("answer:ttl") is not None
    await asyncio.sleep(1.3)
    assert await redis_cache.get_json("answer:ttl") is None


async def test_invalidate_clears_answers_only():
    await redis_cache.set_json("answer:x", {"a": 1})
    await redis_cache.get_client().set("other:keep", "untouched")
    await redis_cache.invalidate_answers()
    assert await redis_cache.get_json("answer:x") is None
    assert await redis_cache.get_client().get("other:keep") == "untouched"


async def test_rate_limiter_allows_then_blocks():
    allowed = [await redis_cache.allow_request("ip-1", limit=3, window_s=60) for _ in range(4)]
    assert allowed == [True, True, True, False]


async def test_rate_limiter_is_per_identity():
    for _ in range(3):
        await redis_cache.allow_request("ip-a", limit=3, window_s=60)
    assert await redis_cache.allow_request("ip-a", limit=3, window_s=60) is False
    assert await redis_cache.allow_request("ip-b", limit=3, window_s=60) is True


async def test_cache_survives_unicode_and_large_payloads():
    payload = {"answer": "নমস্কার " * 500, "citations": [], "model": "m"}
    await redis_cache.set_json("answer:big", payload)
    assert (await redis_cache.get_json("answer:big"))["answer"] == payload["answer"]
