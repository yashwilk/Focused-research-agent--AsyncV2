"""
Web search node for the Focused Research Agent.

This module contains the node responsible for executing web searches
using the generated queries. It receives an injected search provider
via closure from graph.py — the same dependency injection pattern
used for LLM nodes.

Search results are deduplicated and normalized by the search provider
before being stored in state. Images returned by the provider are
capped at _NUMBER_OF_IMAGES and stored separately for UI rendering.

If the search provider raises an exception, an error is recorded in
state and the graph routes to handle_error.
"""

import logging

from focused_research_agent.interfaces.search_inference import SearchProvider
from focused_research_agent.state import ResearchState

logger = logging.getLogger(__name__)

_NUMBER_OF_IMAGES = 12


async def search_web(state: ResearchState, search_provider: SearchProvider) -> dict:
    """Search the web using the generated queries.

    This node retrieves the active search provider from the factory,
    executes the queries, and stores normalized sources in state.

    Args:
        state: The current research state.
        search_provider: The active search provider instance.

    Returns:
        dict: A partial state update containing sources and status,
        or an errors field if search fails.
    """
    queries = state.get("queries")
    run_id = state.get("run_id", "unknown")

    if not isinstance(queries, list):
        logger.error("search_web: queries must be a list. run_id=%s", run_id)
        return {"errors": ["search_web: queries must be a list"]}

    if not queries:
        logger.error("search_web: No queries found. run_id=%s", run_id)
        return {"errors": ["search_web: No queries found"]}

    try:
        search_results, images = await search_provider.search(queries)
    except Exception as e:
        logger.exception("search_web failed. run_id=%s", run_id)
        return {"errors": [f"search_web failed: {e}"]}

    logger.info(
        "Search completed. run_id=%s sources=%d images=%d",
        run_id,
        len(search_results),
        len(images),
    )

    return {
        "sources": search_results,
        "images": images[:_NUMBER_OF_IMAGES],
        "status": "searched",
    }
