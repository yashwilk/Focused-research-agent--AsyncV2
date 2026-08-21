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

import pytest
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

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

# Rate limits default tight enough (e.g. 5/minute for /report) that a single
# test file exercising an endpoint repeatedly would trip them and get
# spurious 429s — the limiter's counters are a process-wide singleton
# (core/rate_limiter.py's `limiter`), so they persist across tests in the
# same session. Raised here, generously, for the whole test run.
os.environ.setdefault("RATE_LIMIT_DEFAULT", "100000/minute")
os.environ.setdefault("RATE_LIMIT_RESEARCH", "100000/minute")
os.environ.setdefault("RATE_LIMIT_CHAT", "100000/minute")
os.environ.setdefault("RATE_LIMIT_REPORT", "100000/minute")
os.environ.setdefault("RATE_LIMIT_AUTH", "100000/minute")


def pytest_configure(config):
    """Create all database tables after environment variables are set
    and before any tests run.

    Args:
        config: The pytest configuration object.
    """
    from focused_research_agent.database.database import init_db

    asyncio.run(init_db())


@pytest.fixture
async def db() -> AsyncSession:
    """
    Fresh, isolated async in-memory SQLite session for one test.

    Uses a private (non-shared-cache) `:memory:` database via aiosqlite —
    unlike the app's real engine (a shared-cache DB set up above for the
    lifetime of the whole test session), each call to this fixture gets
    its own engine, so tests using it are isolated from each other without
    needing manual table resets between tests.

    Yields:
        AsyncSession: An active async SQLAlchemy session for the test.
    """
    from focused_research_agent.database.model import Base

    engine = create_async_engine(
        "sqlite+aiosqlite:///:memory:",
        connect_args={"check_same_thread": False},
    )
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    session_factory = async_sessionmaker(
        bind=engine,
        class_=AsyncSession,
        autocommit=False,
        autoflush=False,
        expire_on_commit=False,
    )
    async with session_factory() as session:
        yield session

    await engine.dispose()
