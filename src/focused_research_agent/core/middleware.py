"""ASGI middleware recording Prometheus HTTP metrics and tracking in-flight
requests for graceful shutdown."""

import time
from collections.abc import Callable

from fastapi import Request
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import Response

from focused_research_agent.core.metrics import (
    http_request_duration_seconds,
    http_requests_total,
)

_inflight_requests = 0


def get_inflight_request_count() -> int:
    """Return the number of requests currently being processed.

    Used by the lifespan shutdown handler to wait for in-flight research
    runs to finish (up to a grace period) before the process exits —
    directly addresses the README's stated gap: "No graceful shutdown for
    in-flight research runs."
    """
    return _inflight_requests


class MetricsMiddleware(BaseHTTPMiddleware):
    """Records request count/duration, and tracks in-flight request count."""

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        global _inflight_requests
        _inflight_requests += 1
        start = time.time()
        status_code = 500
        try:
            response = await call_next(request)
            status_code = response.status_code
            return response
        finally:
            _inflight_requests -= 1
            duration = time.time() - start
            http_requests_total.labels(
                method=request.method, endpoint=request.url.path, status=status_code
            ).inc()
            http_request_duration_seconds.labels(
                method=request.method, endpoint=request.url.path
            ).observe(duration)
