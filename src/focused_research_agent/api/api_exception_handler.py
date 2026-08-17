"""
Centralized FastAPI exception handlers for the Focused Research Agent API.

This module converts shared application exceptions and unexpected runtime
exceptions into consistent HTTP JSON error responses.

Added: a handler for CircuitBreakerOpenError, which maps to 503 Service
Unavailable with a Retry-After-style detail message — the correct HTTP
semantics for "the upstream provider is currently down, try again later"
rather than letting it fall through as an opaque 500.
"""

import logging

from fastapi import FastAPI, Request, status
from fastapi.responses import JSONResponse

from focused_research_agent.application.exceptions import ApplicationError
from focused_research_agent.reliability.circuit_breaker import CircuitBreakerOpenError

logger = logging.getLogger("focused_research_agent.api.exception_handlers")


def _build_error_response(status_code: int, error: str, detail: str, path: str) -> JSONResponse:
    """Build a consistent JSON error response for the API layer."""
    return JSONResponse(
        status_code=status_code,
        content={"status_code": status_code, "error": error, "detail": detail, "path": path},
    )


def handle_application_error(request: Request, exc: Exception) -> JSONResponse:
    """Convert an application-layer error into an HTTP 400 response."""
    return _build_error_response(
        status_code=status.HTTP_400_BAD_REQUEST,
        error="application_error",
        detail=str(exc),
        path=str(request.url.path),
    )


def handle_circuit_breaker_open(request: Request, exc: Exception) -> JSONResponse:
    """Convert a tripped circuit breaker into an HTTP 503 response."""
    logger.warning("Circuit breaker open on path %s: %s", request.url.path, exc)
    return _build_error_response(
        status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
        error="provider_unavailable",
        detail=str(exc),
        path=str(request.url.path),
    )


def handle_unexpected_exception(request: Request, exc: Exception) -> JSONResponse:
    """Convert an unexpected server-side exception into an HTTP 500 response."""
    logger.exception("Unexpected API error on path %s", request.url.path)
    return _build_error_response(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        error="internal_server_error",
        detail="An unexpected internal error occurred",
        path=str(request.url.path),
    )


def register_exception_handlers(app: FastAPI) -> None:
    """Register centralized exception handlers on the FastAPI app."""
    app.add_exception_handler(ApplicationError, handle_application_error)
    app.add_exception_handler(CircuitBreakerOpenError, handle_circuit_breaker_open)
    app.add_exception_handler(Exception, handle_unexpected_exception)
