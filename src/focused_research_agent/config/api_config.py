"""
API application settings for the Focused Research Agent.

This module defines app-level configuration used by the FastAPI layer.
It keeps API-specific settings such as title, version, and debug mode in
one structured place instead of scattering them across the app factory.

Architecturally, this module belongs to the configuration layer. It provides
application settings to the FastAPI app factory while remaining separate
from routers, use-case logic, and workflow orchestration.
"""

import os
from dataclasses import dataclass

from dotenv import load_dotenv

load_dotenv()


def _parse_bool(value: str | None, default: bool) -> bool:
    """
    Parse a string environment value into a boolean.

    Args:
        value: Raw environment variable value.
        default: Default boolean value to use when the input is missing.

    Returns:
        bool: Parsed boolean value.
    """
    if value is None:
        return default

    normalized_value = value.strip().lower()
    return normalized_value in {"1", "true", "yes", "on"}


@dataclass(frozen=True)
class APISettings:
    """
    Structured API settings used by the FastAPI app factory.

    Attributes:
        title: Human-readable API title shown in FastAPI docs.
        version: API version label shown in FastAPI docs.
        debug: Flag controlling FastAPI debug behavior.
    """

    title: str
    version: str
    debug: bool


def get_api_settings() -> APISettings:
    """
    Load API settings from environment variables with sensible defaults.

    Returns:
        APISettings: Fully constructed API settings object.
    """
    title = os.getenv("API_TITLE", "Focused Research Agent API")
    version = os.getenv("API_VERSION", "1.0.0")
    debug = _parse_bool(os.getenv("API_DEBUG"), False)

    return APISettings(
        title=title,
        version=version,
        debug=debug,
    )
