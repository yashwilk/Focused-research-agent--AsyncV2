"""
Application-layer research use case for the Focused Research Agent.

This module contains the application-level logic for executing the research
use case. It sits between transport layers such as CLI, FastAPI, or
Streamlit and the underlying LangGraph workflow.

Architecturally, the application layer contains use-case/business logic.
It coordinates research execution while keeping terminal, HTTP, and other
transport concerns out of the core execution path.

Async + caching conversion: graph.ainvoke() replaces graph.invoke() so
this use case no longer occupies a worker thread for its full duration.
A cache lookup/store wraps the graph call — a repeated identical question
returns instantly instead of hitting Groq and Tavily again. Caching is
skipped for empty/invalid questions (those fail validation before reaching
the cache) and can be disabled entirely via CACHE_ENABLED=false.
"""

import logging
import os
import time
import uuid

from focused_research_agent.application.exceptions import ApplicationError
from focused_research_agent.application.question_validation import (
    validate_and_clean_question,
)
from focused_research_agent.caching.cache_service import response_cache
from focused_research_agent.config.logger_config import bind_run_id
from focused_research_agent.core.metrics import graph_run_duration_seconds, graph_run_total
from focused_research_agent.graph import build_graph
from focused_research_agent.state import ResearchState

logger = logging.getLogger(__name__)

_CACHE_ENABLED = os.getenv("CACHE_ENABLED", "true").strip().lower() != "false"


def _is_list_of_strings(value: object) -> bool:
    """Check whether a value is a list containing only strings."""
    return isinstance(value, list) and all(isinstance(item, str) for item in value)


def _is_list_of_dicts(value: object) -> bool:
    """Check whether a value is a list containing only dictionaries."""
    return isinstance(value, list) and all(isinstance(item, dict) for item in value)


def normalize_state(final_state: ResearchState, user_query: str) -> dict:
    """Normalize raw graph state into a stable transport-facing result shape."""
    normalized_state = {
        "run_id": final_state.get("run_id") or "",
        "question": final_state.get("question") or user_query,
        "status": final_state.get("status") or "error",
        "scope": final_state.get("scope"),
        "queries": None,
        "sources": None,
        "answer": final_state.get("answer"),
        "citations": None,
        "errors": [],
        "images": final_state.get("images"),
    }

    queries = final_state.get("queries")
    if _is_list_of_strings(queries):
        normalized_state["queries"] = queries

    sources = final_state.get("sources")
    if _is_list_of_dicts(sources):
        normalized_state["sources"] = sources

    citations = final_state.get("citations")
    if _is_list_of_strings(citations):
        normalized_state["citations"] = citations

    errors = final_state.get("errors")
    if _is_list_of_strings(errors):
        normalized_state["errors"] = errors

    return normalized_state


def make_initial_state(question: str) -> ResearchState:
    """Create the starting graph state for a research run.

    Generates the run_id here — in the use-case layer, before the graph
    is invoked — rather than inside the init_run node. This matters for
    log correlation: contextvars are captured by value when asyncio
    creates a new Task, so binding run_id here (the actual ancestor of
    every task LangGraph spawns for this run) makes it visible to every
    node's logs. Binding it from inside init_run, which itself runs as
    one of those spawned tasks, only affects that task's own context —
    it does not propagate to sibling node tasks the graph creates
    afterward. This was verified with a live run: only the init_run log
    line carried the correct run_id under the old approach.
    """
    initial_state: ResearchState = {
        "run_id": str(uuid.uuid4()),
        "question": question,
        "scope": None,
        "assumptions": None,
        "constraints": None,
        "queries": None,
        "sources": None,
        "answer": None,
        "citations": None,
        "status": "started",
        "errors": [],
        "debug": None,
        "conversation_id": None,
        "conversation_history": None,
        "mode": "research",
        "images": None,
        "search_retry_count": 0,
    }

    return initial_state


async def research_question(question: str) -> dict:
    """Execute the research use case for a user question (async, cached).

    Args:
        question: User research question.

    Returns:
        dict: Normalized research result produced by the workflow, or a
            cached copy of a prior identical successful result.

    Raises:
        ApplicationError: If the question is invalid for the research use case.
    """
    try:
        user_query = validate_and_clean_question(question)
    except ValueError as exc:
        raise ApplicationError(str(exc)) from exc

    if _CACHE_ENABLED:
        cached = await response_cache.get(user_query)
        if cached is not None:
            logger.info("Research cache hit. question='%s'", user_query[:50])
            return cached

    logger.info("Research use case started. question='%s'", user_query[:50])

    graph = build_graph()
    initial_state = make_initial_state(user_query)
    bind_run_id(initial_state["run_id"])

    _start = time.monotonic()
    final_state = await graph.ainvoke(initial_state)
    graph_run_duration_seconds.labels(mode="research").observe(time.monotonic() - _start)

    result = normalize_state(final_state, user_query)
    graph_run_total.labels(mode="research", status=result.get("status", "error")).inc()

    if _CACHE_ENABLED and result.get("status") == "completed":
        await response_cache.set(user_query, result)

    logger.info(
        "Research use case completed. status=%s run_id=%s",
        result.get("status"),
        result.get("run_id"),
    )
    return result
