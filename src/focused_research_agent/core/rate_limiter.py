"""
Rate limiting for the Focused Research Agent.

Directly addresses the README's stated gap: "No rate limiting —
Groq/Tavily quota could be exhausted by a single caller."

Uses slowapi (Starlette-native), backed by Redis when REDIS_URL is set so
limits are enforced correctly across multiple app instances / Celery
workers rather than each process tracking its own separate counters.
Falls back to in-memory storage (per-process only) when Redis isn't
configured, so local development still works without extra setup.
"""

import os

from slowapi import Limiter
from slowapi.util import get_remote_address

_redis_url = os.getenv("REDIS_URL")

limiter = Limiter(
    key_func=get_remote_address,
    default_limits=[os.getenv("RATE_LIMIT_DEFAULT", "100/hour")],
    storage_uri=_redis_url,  # None -> slowapi's in-memory storage
)

# Per-endpoint overrides — the expensive, quota-consuming endpoints get
# tighter limits than simple reads like /health or /conversations.
RATE_LIMIT_RESEARCH = os.getenv("RATE_LIMIT_RESEARCH", "20/minute")
RATE_LIMIT_CHAT = os.getenv("RATE_LIMIT_CHAT", "20/minute")
RATE_LIMIT_REPORT = os.getenv("RATE_LIMIT_REPORT", "5/minute")
RATE_LIMIT_AUTH = os.getenv("RATE_LIMIT_AUTH", "10/minute")
