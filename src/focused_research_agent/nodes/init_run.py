"""
Run initialization node for the Focused Research Agent.

This module contains the entry node for the LangGraph research workflow.
It validates that a user question is present before any other node
executes. If no question is found, an error is recorded in state and
the graph routes to handle_error.

The run_id itself is generated upstream in the application layer
(make_initial_state), not here — see research_use_case.py for why:
binding the log-correlation contextvar has to happen before the graph
is invoked for it to propagate to every node's tasks, not just this one.
This node still falls back to generating one if state somehow arrives
without it (e.g. a future direct-graph caller that skips make_initial_state).
"""

import logging
import uuid

from focused_research_agent.config.logger_config import bind_run_id
from focused_research_agent.state import ResearchState

logger = logging.getLogger(__name__)


async def initialize_state(state: ResearchState) -> dict:
    """
    Initializes a new research run. Validates that a question was
    provided and that a run_id is present (generating a fallback one
    if not).
    """
    run_id = state.get("run_id") or str(uuid.uuid4())
    bind_run_id(run_id)
    user_query = (state.get("question") or "").strip()
    errors = []

    if not user_query:
        logger.error("init_run: No question provided")
        errors.append("init_run: No question provided")
    else:
        logger.info(
            "Research run started. run_id=%s question='%s'", run_id, user_query[:50]
        )

    return {
        "run_id": run_id,
        "status": "started",
        "errors": errors,
    }
