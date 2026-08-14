"""
Application-layer chat use case for the Focused Research Agent.

This module contains the application-level logic for executing the
conversation-aware research use case. It sits alongside
research_use_case.py — same layer, same pattern, but with conversation
threading added before and after graph execution.

Before invoking the graph, it fetches prior conversation turns from the
database and populates conversation_history in the initial state.
After the graph returns, it persists the completed run.

The graph itself is identical to the single-turn research flow.
The conversation awareness lives entirely in this layer and in
synthesize_answer's prompt building — not in the graph structure.

Async conversion: uses graph.ainvoke() and the now-async repository
functions (get_conversation_history, save_run) with an AsyncSession.
"""

import time
import uuid
import logging

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
from focused_research_agent.database.repository import (
    get_conversation_history,
    save_run,
)
from focused_research_agent.graph import build_graph
from focused_research_agent.state import ResearchState

logger = logging.getLogger(__name__)
MAX_HISTORY_TURNS = 5


def _build_chat_initial_state(
    question: str, conversation_id: str, conversation_history: list[dict] | None
) -> ResearchState:
    """Build the initial graph state for a conversation-aware research run."""
    state = make_initial_state(question)
    state["conversation_id"] = conversation_id
    state["conversation_history"] = conversation_history
    state["mode"] = "research"
    return state


async def execute_chat_turn(
    db: AsyncSession,
    conversation_id: str | None,
    question: str,
    user_id: int | None = None,
) -> dict:
    """Execute one turn of a conversation-aware research session (async).

    Persistence failure does not fail the research result — the
    completed answer is always returned even if saving to the
    database fails.

    Args:
        db: Active async SQLAlchemy database session.
        conversation_id: Existing conversation UUID to continue, or
            None to start a new conversation.
        question: User research question for this turn.

    Returns:
        dict: Normalized research result with conversation_id and
            turn_number added to the standard research result shape.

    Raises:
        ApplicationError: If the question fails validation.
    """
    try:
        user_query = validate_and_clean_question(question)
    except ValueError as exc:
        raise ApplicationError(str(exc)) from exc

    if conversation_id is None:
        conversation_id = str(uuid.uuid4())

    logger.info(
        "Chat turn started. conversation_id=%s turn question='%s'",
        conversation_id,
        user_query[:50],
    )

    conversation_history = await get_conversation_history(
        db, conversation_id, MAX_HISTORY_TURNS, user_id=user_id
    )

    history = conversation_history if conversation_history else None
    turn_number = len(history) + 1 if history is not None else 1

    graph = build_graph()
    initial_state = _build_chat_initial_state(user_query, conversation_id, history)
    bind_run_id(initial_state["run_id"])

    _start = time.monotonic()
    final_state = await graph.ainvoke(initial_state)
    graph_run_duration_seconds.labels(mode="chat").observe(time.monotonic() - _start)

    result = normalize_state(final_state, user_query)
    graph_run_total.labels(mode="chat", status=result.get("status", "error")).inc()

    try:
        await save_run(db, result, conversation_id, turn_number, user_id=user_id)
    except SQLAlchemyError:
        logger.exception("Failed to save chat run to database")

    result["conversation_id"] = conversation_id
    result["turn_number"] = turn_number

    logger.info(
        "Chat turn completed. conversation_id=%s turn=%d status=%s",
        conversation_id,
        turn_number,
        result.get("status"),
    )

    return result