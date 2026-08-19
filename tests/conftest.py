"""
Pytest configuration for the Focused Research Agent test suite.

Sets DATABASE_URL to a shared in-memory async SQLite database before any
test module is imported. Using a shared cache URI ensures all connections
within the test session see the same database and the same tables,
avoiding the 'no such table' error that occurs when each connection gets
its own isolated in-memory database.

Updated for the async conversion: uses the aiosqlite driver scheme and
awaits init_db() via asyncio.run() during collection, since init_db is
now an async function (async SQLAlchemy engine).
"""

import asyncio
import os

os.environ.setdefault(
    "DATABASE_URL",
    "sqlite+aiosqlite:///file::memory:?cache=shared&uri=true",
)
os.environ.setdefault("AUTH_SECRET_KEY", "test-secret-key")
os.environ.setdefault("CACHE_ENABLED", "false")
os.environ.setdefault("LLM_PROVIDER", "groq")
os.environ.setdefault("LLM_MODEL", "llama-3.3-70b-versatile")
os.environ.setdefault("LLM_API_KEY", "test-key")
os.environ.setdefault("LLM_TEMPERATURE", "0.0")
os.environ.setdefault("LLM_MAX_RETRIES", "2")
os.environ.setdefault("LLM_MAX_TOKENS", "4096")
os.environ.setdefault("SEARCH_PROVIDER", "tavily")
os.environ.setdefault("SEARCH_API_KEY", "test-key")
os.environ.setdefault("SEARCH_MAX_RESULTS", "5")
os.environ.setdefault("SEARCH_DEPTH", "basic")


def pytest_configure(config):
    """Create all database tables after environment variables are set
    and before any tests run.

    Args:
        config: The pytest configuration object.
    """
    from focused_research_agent.database.database import init_db

    asyncio.run(init_db())
