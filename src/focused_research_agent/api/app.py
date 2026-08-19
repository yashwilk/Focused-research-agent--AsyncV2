"""
FastAPI application entrypoint for the Focused Research Agent.

This module creates the FastAPI app instance, registers routers,
middleware, and centralized exception handlers, and manages the
application lifespan — including graceful shutdown, which directly
addresses the README's stated gap: "No graceful shutdown for in-flight
research runs."

Architecturally, this file belongs to the transport layer. It focuses on
app construction and wiring while delegating request handling to routers
and use-case execution to the application layer.
"""

import asyncio
import logging
import os
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.httpsredirect import HTTPSRedirectMiddleware
from slowapi import _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded

from focused_research_agent.api.api_exception_handler import register_exception_handlers
from focused_research_agent.api.router.health import health_router
from focused_research_agent.api.router.v1 import api_v1_router
from focused_research_agent.auth.router import auth_router
from focused_research_agent.config.api_config import get_api_settings
from focused_research_agent.config.logger_config import setup_logging
from focused_research_agent.core.metrics import metrics_asgi_app
from focused_research_agent.core.middleware import (
    MetricsMiddleware,
    get_inflight_request_count,
)
from focused_research_agent.core.rate_limiter import limiter
from focused_research_agent.core.tracing import setup_tracing
from focused_research_agent.database.database import engine, init_db

logger = logging.getLogger(__name__)

# Grace period to let in-flight research/report runs finish before the
# process exits. Research runs typically finish in ~15-30s; reports can
# run longer, which is exactly why /report/submit exists as an
# alternative — a graceful shutdown mid-report job is still handled
# correctly since Celery re-queues incomplete tasks independently of the
# API process's own lifecycle.
_SHUTDOWN_GRACE_SECONDS = int(os.getenv("SHUTDOWN_GRACE_SECONDS", "30"))


def register_routers(app: FastAPI) -> None:
    """Register all API routers on the FastAPI app."""
    app.include_router(health_router)
    app.include_router(auth_router)
    app.include_router(api_v1_router)
    app.mount("/metrics", metrics_asgi_app)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage startup and graceful shutdown for the application."""
    setup_logging()
    setup_tracing()
    await init_db()
    logger.info("Application startup complete.")

    yield

    logger.info(
        "Shutdown initiated. Waiting up to %ss for %d in-flight request(s) to finish.",
        _SHUTDOWN_GRACE_SECONDS,
        get_inflight_request_count(),
    )
    waited = 0
    while get_inflight_request_count() > 0 and waited < _SHUTDOWN_GRACE_SECONDS:
        await asyncio.sleep(1)
        waited += 1

    remaining = get_inflight_request_count()
    if remaining > 0:
        logger.warning(
            "Shutdown grace period elapsed with %d request(s) still in flight.",
            remaining,
        )
    else:
        logger.info("All in-flight requests completed cleanly.")

    await engine.dispose()
    logger.info("Application shutdown complete.")


def create_app() -> FastAPI:
    """Build and configure the FastAPI application instance."""
    settings = get_api_settings()

    app = FastAPI(
        title=settings.title,
        version=settings.version,
        debug=settings.debug,
        lifespan=lifespan,
    )

    app.state.limiter = limiter
    app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

    app.add_middleware(MetricsMiddleware)

    # Enforce HTTPS when explicitly enabled — off by default because it
    # actively breaks local development (plain http://localhost). Turn
    # this on in production behind a real TLS-terminating reverse proxy
    # (or set it there instead — see docs/production.md).
    if os.getenv("FORCE_HTTPS", "false").strip().lower() == "true":
        app.add_middleware(HTTPSRedirectMiddleware)

    register_routers(app)
    register_exception_handlers(app)

    logger.info(
        "Application configured. title=%s version=%s debug=%s",
        settings.title,
        settings.version,
        settings.debug,
    )

    return app
