"""
Application-layer report use case for the Focused Research Agent.

This module contains the application-level logic for executing the
report generation use case. It sits alongside research_use_case.py
and chat_use_case.py — same layer, same pattern, but configured for
deep research and structured long-form output.

Key differences from research_use_case.py:
- Sets mode='report' in the initial state
- Calls build_graph(search_depth='advanced') for deeper Tavily search
- Persists the completed report to the database
- NOT cached — report runs are expected to be deep and fresh every time

This function is used two ways:
1. Directly, awaited inline, for the synchronous-style /api/v1/report
   endpoint (now non-blocking thanks to the async conversion below).
2. Wrapped by a Celery task (see tasks/report_tasks.py) for the
   fire-and-poll pattern via /api/v1/report/submit — recommended for
   report generation specifically, since it is the slowest, deepest
   mode (advanced search + long-form synthesis) and benefits most from
   not tying up an HTTP request for its full duration.
"""

import logging
import time
import uuid

from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.ext.asyncio import AsyncSession

from focused_research_agent.application.exceptions import ApplicationError
from focused_research_agent.application.question_validation import (
    validate_and_clean_question,
)
from focused_research_agent.application.research_use_case import (
    make_initial_state,
    normalize_state,
)
from focused_research_agent.config.logger_config import bind_run_id
from focused_research_agent.core.metrics import graph_run_duration_seconds, graph_run_total
from focused_research_agent.database.repository import save_run
from focused_research_agent.graph import build_graph

logger = logging.getLogger(__name__)


async def execute_report(question: str, db: AsyncSession, user_id: int | None = None) -> dict:
    """Execute a deep research report generation run (async).

    Persistence failure does not fail the report result — the completed
    report is always returned even if saving to the database fails.

    Args:
        question: User research question for the report.
        db: Active async SQLAlchemy database session.

    Returns:
        dict: Normalized research result with structured markdown answer.

    Raises:
        ApplicationError: If the question fails validation.
    """
    try:
        user_query = validate_and_clean_question(question)
    except ValueError as exc:
        raise ApplicationError(str(exc)) from exc

    logger.info("Report use case started. question='%s'", user_query[:50])

    initial_state = make_initial_state(user_query)
    initial_state["mode"] = "report"

    graph = build_graph(search_depth="advanced")
    bind_run_id(initial_state["run_id"])

    _start = time.monotonic()
    final_state = await graph.ainvoke(initial_state)
    graph_run_duration_seconds.labels(mode="report").observe(time.monotonic() - _start)

    result = normalize_state(final_state, user_query)
    graph_run_total.labels(mode="report", status=result.get("status", "error")).inc()

    try:
        conversation_id = str(uuid.uuid4())
        await save_run(db, result, conversation_id, turn_number=1, mode="report", user_id=user_id)
    except SQLAlchemyError:
        logger.exception("Failed to save report run to database")

    logger.info(
        "Report use case completed. status=%s run_id=%s",
        result.get("status"),
        result.get("run_id"),
    )
    return result
