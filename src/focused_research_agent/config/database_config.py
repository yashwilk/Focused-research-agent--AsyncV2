"""
Database configuration for the Focused Research Agent.

This module defines database connection settings used by the SQLAlchemy
async engine. It keeps database-specific configuration in one place,
following the same pattern as api_config.py and llm_config.py.
"""

import os
from dataclasses import dataclass

from dotenv import load_dotenv

load_dotenv()


@dataclass(frozen=True)
class DatabaseSettings:
    """
    Structured database settings used by the SQLAlchemy async engine.

    Attributes:
        database_url: SQLAlchemy async connection string for the database.
            Defaults to a local SQLite file if not set in the environment.
    """

    database_url: str


def get_database_settings() -> DatabaseSettings:
    """
    Load database settings from environment variables with a sensible
    default.

    Defaults to a local async SQLite file named research_agent.db in the
    project root if DATABASE_URL is not set in the environment.

    Returns:
        DatabaseSettings: Fully constructed database settings object.
    """
    database_url = os.getenv(
        "DATABASE_URL",
        "sqlite+aiosqlite:///./research_agent.db",
    )

    return DatabaseSettings(database_url=database_url)
