"""
UI application settings for the Focused Research Agent.

This module defines configuration used by the Streamlit UI layer.
It keeps UI-specific settings such as the backend base URL and request
timeout in one structured place, following the same pattern as api_config.py.

Architecturally, this module belongs to the configuration layer. It provides
settings to the UI transport layer while remaining separate from rendering,
HTTP, and workflow concerns.
"""

import os
from dataclasses import dataclass

from dotenv import load_dotenv

load_dotenv()


@dataclass(frozen=True)
class UISettings:
    """
    Structured UI settings used by the Streamlit transport layer.

    Attributes:
        api_base_url: Base URL of the FastAPI backend the UI calls.
        request_timeout: Seconds to wait for a research response before
            timing out. Research involves LLM calls and web search so
            this should be generous.
    """

    api_base_url: str
    request_timeout: int


def get_ui_settings() -> UISettings:
    """
    Load UI settings from environment variables with sensible defaults.

    Returns:
        UISettings: Fully constructed UI settings object.
    """
    api_base_url = os.getenv("UI_API_BASE_URL", "http://localhost:8000")
    request_timeout = int(os.getenv("UI_REQUEST_TIMEOUT", "120"))

    return UISettings(
        api_base_url=api_base_url,
        request_timeout=request_timeout,
    )
