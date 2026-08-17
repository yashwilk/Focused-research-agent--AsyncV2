"""
Versioned API router grouping for the Focused Research Agent.

This module defines the version-1 API namespace and attaches versioned
feature routers under a shared prefix.
"""

from fastapi import APIRouter

from focused_research_agent.api.router.research import research_router
from focused_research_agent.api.router.chat import chat_router
from focused_research_agent.api.router.conversations import conversations_router
from focused_research_agent.api.router.report import report_router


def create_v1_router() -> APIRouter:
    """Build the version-1 API router group."""
    router = APIRouter(prefix="/api/v1")
    router.include_router(research_router)
    router.include_router(chat_router)
    router.include_router(conversations_router)
    router.include_router(report_router)
    return router


api_v1_router = create_v1_router()
