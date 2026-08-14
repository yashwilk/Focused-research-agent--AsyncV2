"""
Response cache for the Focused Research Agent.

Directly addresses the README's stated gap: "No caching — the same
question hits Tavily and Groq on every request."

Uses Redis when REDIS_URL is configured (safe for multi-instance
deployments — every app instance and the Celery worker share the same
cache), and falls back to an in-process TTL dict for local/single-instance
use so nothing breaks if Redis isn't set up yet.

Only the research use case (single-turn, stateless Q&A) uses this cache.
Chat and report use cases are intentionally NOT cached — chat is
conversation-specific by definition, and report runs are expected to be
deep, fresh, and comprehensive every time.
"""

import hashlib
import json
import logging
import os
import time
from typing import Optional

logger = logging.getLogger(__name__)

try:
    from redis.asyncio import Redis

    REDIS_AVAILABLE = True
except ImportError:
    Redis = None
    REDIS_AVAILABLE = False

from focused_research_agent.core.metrics import cache_lookups_total


def _cache_key(question: str) -> str:
    """Build a deterministic cache key from a normalized question."""
    normalized = " ".join(question.strip().lower().split())
    digest = hashlib.sha256(normalized.encode()).hexdigest()[:24]
    return f"research:{digest}"


class InMemoryResponseCache:
    """Simple in-process TTL cache. Used when REDIS_URL is not configured."""

    def __init__(self, default_ttl: int = 300):
        self._store: dict[str, tuple[float, str]] = {}
        self._default_ttl = default_ttl

    async def get(self, question: str) -> Optional[dict]:
        key = _cache_key(question)
        entry = self._store.get(key)
        if entry is None:
            cache_lookups_total.labels(result="miss").inc()
            return None
        expires_at, value = entry
        if time.monotonic() > expires_at:
            del self._store[key]
            cache_lookups_total.labels(result="miss").inc()
            return None
        cache_lookups_total.labels(result="hit").inc()
        return json.loads(value)

    async def set(self, question: str, value: dict, ttl: Optional[int] = None) -> None:
        key = _cache_key(question)
        self._store[key] = (
            time.monotonic() + (ttl or self._default_ttl),
            json.dumps(value),
        )

    async def close(self) -> None:
        self._store.clear()


class RedisResponseCache:
    """Redis-backed response cache — safe across multiple app instances."""

    def __init__(self, redis_url: str, default_ttl: int = 300):
        self._client = Redis.from_url(redis_url, decode_responses=True)
        self._default_ttl = default_ttl

    async def get(self, question: str) -> Optional[dict]:
        try:
            raw = await self._client.get(_cache_key(question))
        except Exception as e:
            logger.warning("cache_get_failed error=%s", e)
            return None
        cache_lookups_total.labels(result="hit" if raw else "miss").inc()
        return json.loads(raw) if raw else None

    async def set(self, question: str, value: dict, ttl: Optional[int] = None) -> None:
        try:
            await self._client.set(
                _cache_key(question), json.dumps(value), ex=(ttl or self._default_ttl)
            )
        except Exception as e:
            logger.warning("cache_set_failed error=%s", e)

    async def close(self) -> None:
        await self._client.aclose()


def _build_cache():
    ttl = int(os.getenv("CACHE_TTL_SECONDS", "300"))
    redis_url = os.getenv("REDIS_URL")

    if redis_url and REDIS_AVAILABLE:
        logger.info("response_cache_using_redis")
        return RedisResponseCache(redis_url, default_ttl=ttl)

    if redis_url and not REDIS_AVAILABLE:
        logger.warning("redis_url_set_but_redis_package_missing_falling_back_to_memory")

    logger.info("response_cache_using_in_memory")
    return InMemoryResponseCache(default_ttl=ttl)


response_cache = _build_cache()
