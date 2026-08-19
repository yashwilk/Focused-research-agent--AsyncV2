"""
UI transport layer exceptions for the Focused Research Agent.

This module defines exceptions specific to the Streamlit UI transport layer.
They represent UI-level failures such as the FastAPI backend being unreachable,
which are distinct from application-layer or graph-level errors.

Architecturally, this module belongs to the UI transport layer. It follows
the same pattern as application/exceptions.py — one named exception class
per layer, so each layer has its own clear error language.
"""


class BackendUnavailableError(Exception):
    """
    Raised when the FastAPI backend cannot be reached.

    This exception is raised by api_client.py when httpx throws a ConnectError,
    meaning the FastAPI server is not running or is not reachable at the
    configured UI_API_BASE_URL.

    It is caught by 1_🔍_Research.py to render a clear, user-facing message telling the
    user to start the backend before using the UI.

    Args:
        message: Human-readable description of why the backend is unavailable.
    """

    def __init__(self, message: str) -> None:
        super().__init__(message)
