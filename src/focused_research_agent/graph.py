# builds & compiles the StateGraph
from langgraph.graph import END, START, StateGraph

from focused_research_agent.nodes.finalize_run import finalize_run
from focused_research_agent.nodes.generate_queries import generate_queries
from focused_research_agent.nodes.handle_error import handle_error
from focused_research_agent.nodes.init_run import initialize_state
from focused_research_agent.nodes.reflect_and_refine import (
    MAX_SEARCH_RETRIES,
    MIN_SOURCES_FOR_REFLECTION,
    reflect_and_refine,
)
from focused_research_agent.nodes.scope_question import scope_question
from focused_research_agent.nodes.search_web import search_web
from focused_research_agent.nodes.synthesize_answer import synthesize_answer
from focused_research_agent.services.llm_factory import get_llm_provider
from focused_research_agent.services.search_factory import get_search_provider
from focused_research_agent.state import ResearchState


def route_after_node(state: ResearchState) -> str:
    """
    After any node runs, check if errors were recorded.
    If yes, route to the error handler. Otherwise, continue.
    """
    if state.get("errors"):
        return "handle_error"
    return "continue"


def route_after_search(state: ResearchState) -> str:
    """After search_web, decide whether results are thin enough to warrant
    one bounded reflection pass before synthesis.

    Routes to handle_error on any recorded error (same as every other
    node), to reflect_and_refine if the source count is below the
    reflection threshold and a retry budget remains, or continues to
    synthesize_answer otherwise.
    """
    if state.get("errors"):
        return "handle_error"

    sources = state.get("sources") or []
    retry_count = state.get("search_retry_count") or 0

    if len(sources) < MIN_SOURCES_FOR_REFLECTION and retry_count < MAX_SEARCH_RETRIES:
        return "reflect_and_refine"

    return "continue"


def build_graph(search_depth: str | None = None):
    """Build and compile the LangGraph workflow for the research agent.

    Args:
        search_depth: Optional override for search depth. When provided,
            overrides the SEARCH_DEPTH environment variable. Accepts
            'basic' or 'advanced'. Defaults to the configured value.
    """
    llm = get_llm_provider()
    search = get_search_provider(search_depth=search_depth)

    async def _scope_question(state):
        return await scope_question(state, llm)

    async def _generate_queries(state):
        return await generate_queries(state, llm)

    async def _synthesize_answer(state):
        return await synthesize_answer(state, llm)

    async def _search_web(state):
        return await search_web(state, search)

    async def _reflect_and_refine(state):
        return await reflect_and_refine(state, llm)

    builder = StateGraph(ResearchState)
    builder.add_node("init_run", initialize_state)
    builder.add_node("scope_question", _scope_question)
    builder.add_node("generate_queries", _generate_queries)
    builder.add_node("search_web", _search_web)
    builder.add_node("reflect_and_refine", _reflect_and_refine)
    builder.add_node("synthesize_answer", _synthesize_answer)
    builder.add_node("finalize_run", finalize_run)
    builder.add_node("handle_error", handle_error)

    builder.add_edge(START, "init_run")

    builder.add_conditional_edges(
        "init_run",
        route_after_node,
        {"continue": "scope_question", "handle_error": "handle_error"},
    )
    builder.add_conditional_edges(
        "scope_question",
        route_after_node,
        {"continue": "generate_queries", "handle_error": "handle_error"},
    )
    builder.add_conditional_edges(
        "generate_queries",
        route_after_node,
        {"continue": "search_web", "handle_error": "handle_error"},
    )
    builder.add_conditional_edges(
        "search_web",
        route_after_search,
        {
            "continue": "synthesize_answer",
            "reflect_and_refine": "reflect_and_refine",
            "handle_error": "handle_error",
        },
    )
    builder.add_conditional_edges(
        "reflect_and_refine",
        route_after_node,
        {"continue": "search_web", "handle_error": "handle_error"},
    )
    builder.add_conditional_edges(
        "synthesize_answer",
        route_after_node,
        {"continue": "finalize_run", "handle_error": "handle_error"},
    )

    builder.add_edge("finalize_run", END)
    builder.add_edge("handle_error", END)

    return builder.compile()
