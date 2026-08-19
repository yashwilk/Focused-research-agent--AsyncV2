"""
Run finalization node for the Focused Research Agent.

This module contains the terminal success node for the LangGraph workflow.
It evaluates the final state and marks the run as either completed or
error based on whether an answer was produced and no errors were recorded.

A run is marked completed only when both conditions are true:
- The answer field is a non-empty string
- The errors list is empty

Any other combination results in an error status. This node is always
the last node to execute in a successful graph run.
"""

import logging
from focused_research_agent.state import ResearchState

logger = logging.getLogger(__name__)


async def finalize_run(state: ResearchState) -> dict:
    """Mark the run as completed or failed based on final state.

    Args:
    state: The current research state.

    Returns:
    dict: A partial state update containing the final status.
    """
    errors = state.get("errors") or []
    answer = (state.get("answer") or "").strip()
    run_id = state.get("run_id", "unknown")

    if errors or not answer:
        logger.error(
            "Run finalized with error. run_id=%s errors=%s",
            run_id,
            errors,
        )
        return {"status": "error"}
    logger.info("Run completed successfully. run_id=%s", run_id)
    return {"status": "completed"}
