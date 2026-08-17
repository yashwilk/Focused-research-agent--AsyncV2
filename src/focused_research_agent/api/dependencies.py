"""
FastAPI dependency providers for the Focused Research Agent API.

This module contains dependency functions used by API routers to obtain
application-layer use cases and other injectable components.

Type hints updated to Awaitable return types since every use case is now
async (see the async conversion across application/, database/, and
services/).
"""

from collections.abc import Awaitable, Callable
from focused_research_agent.application import chat_use_case, report_use_case
from focused_research_agent.application import research_use_case


def get_research_use_case() -> Callable[[str], Awaitable[dict]]:
    """Provide the application-layer research use case to API routes."""
    return research_use_case.research_question


def get_chat_use_case() -> Callable[..., Awaitable[dict]]:
    """Provide the application-layer chat use case to API routes."""
    return chat_use_case.execute_chat_turn


def get_report_use_case() -> Callable[..., Awaitable[dict]]:
    """Provide the application-layer report use case to API routes."""
    return report_use_case.execute_report
