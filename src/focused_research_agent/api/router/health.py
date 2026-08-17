"""
Health-check endpoints for the Focused Research Agent API.

Unauthenticated and unlimited by design — load balancers and container
orchestrators need to hit this without a token.
"""

from fastapi import APIRouter

health_router = APIRouter(tags=["health"])


@health_router.get("/health")
async def health() -> dict:
    """Return a simple health status for the API service."""
    return {"status": "ok"}
