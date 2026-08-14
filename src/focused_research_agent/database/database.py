"""
Async SQLAlchemy engine and session management for the Focused Research
Agent.

This module creates the async database engine from configuration, provides
an async session factory, and exposes a get_db dependency generator for
use with FastAPI's dependency injection system.

Architecturally, this module belongs to the database layer. It knows
about SQLAlchemy and database configuration only. It does not import
from the application layer, graph layer, or API layer.

Async conversion note: this uses create_async_engine + AsyncSession
instead of the original sync engine. Combined with the aiosqlite (SQLite)
or asyncpg (PostgreSQL) drivers, database calls no longer occupy a worker
thread — they yield to the event loop like every other I/O in the async
conversion of this project.
"""

from typing import AsyncGenerator

from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine, async_sessionmaker

from focused_research_agent.config.database_config import get_database_settings
from focused_research_agent.database.model import Base

_connect_args = {}
_database_url = get_database_settings().database_url
if _database_url.startswith("sqlite"):
    _connect_args = {"check_same_thread": False}

engine = create_async_engine(_database_url, connect_args=_connect_args)

AsyncSessionLocal = async_sessionmaker(
    bind=engine, class_=AsyncSession, autocommit=False, autoflush=False, expire_on_commit=False
)


async def init_db() -> None:
    """
    Create all database tables if they do not already exist.

    Safe to call multiple times — SQLAlchemy checks whether each table
    exists before creating it. Called once at application startup via
    the FastAPI lifespan.

    Note: table creation via metadata.create_all is fine for local/dev
    use, but real schema evolution in production goes through Alembic
    migrations (see alembic/ at the project root) — this call becomes a
    no-op safety net once migrations own the schema.

    Returns:
        None
    """
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)


async def get_db() -> AsyncGenerator[AsyncSession, None]:
    """
    Yield an async database session and guarantee it is closed after use.

    Used as a FastAPI dependency via Depends(get_db). Yields one
    AsyncSession for the duration of a request and closes it in a
    finally block so it is always released, even if an error occurs.

    Yields:
        AsyncSession: An active SQLAlchemy async database session.
    """
    async with AsyncSessionLocal() as session:
        try:
            yield session
        finally:
            await session.close()
