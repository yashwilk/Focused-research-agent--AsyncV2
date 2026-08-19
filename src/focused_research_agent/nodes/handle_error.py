"""
Error handling node for the Focused Research Agent.

This module contains the terminal error node for the LangGraph workflow.
It is reached via conditional routing whenever any upstream node records
an error in state. It logs all collected errors and sets the final status
to 'error' so the application layer can return a structured error response
to the transport layer.

This node never raises — it always returns a partial state update so the
graph can finalize cleanly regardless of which node failed.
"""

import logging

from focused_research_agent.state import ResearchState

logger = logging.getLogger(__name__)


async def handle_error(state: ResearchState) -> dict:
    """
    Terminal error node. Logs all recorded errors and marks
    the run as failed. Reached via conditional routing when
    any upstream node records an error.
    """
    errors = state.get("errors") or []
    for error in errors:
        logger.error(error)
    return {"status": "error"}
