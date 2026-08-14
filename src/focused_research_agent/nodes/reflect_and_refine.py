"""
Search reflection node for the Focused Research Agent.

Directly implements an item from the project's own roadmap:
"Reflection loop — agent re-searches if initial results are insufficient."

This node is only reached when routing decides search results were thin
(see route_after_search in graph.py). It asks the LLM to produce a fresh
set of queries that specifically avoid repeating the queries already
tried, then hands control back to search_web for one more attempt.

Bounded by search_retry_count — the routing function in graph.py only
sends the flow here while retry_count is below MAX_SEARCH_RETRIES, so
this can never loop indefinitely regardless of how thin results stay.
"""

import logging

from focused_research_agent.core.metrics import search_reflection_triggered_total
from focused_research_agent.interfaces.llm_inference import LLMProvider
from focused_research_agent.state import ResearchState

logger = logging.getLogger(__name__)

MAX_SEARCH_RETRIES = 1
MIN_SOURCES_FOR_REFLECTION = 2


def _build_reflection_prompt(state: ResearchState) -> str:
    """Build the LLM prompt asking for refined, non-overlapping queries."""
    question = (state.get("question") or "").strip()
    scope = (state.get("scope") or "").strip()
    previous_queries = state.get("queries") or []
    sources_found = len(state.get("sources") or [])

    return f"""
    Return ONLY valid JSON. No markdown. No backticks. No extra text.
    Return EXACTLY one key: "queries".

    Context:
    The previous search attempt for this question only returned
    {sources_found} usable source(s), which is too few to answer well.

    Task:
    - Generate 3 to 6 NEW search-engine style queries.
    - Do NOT repeat any of the previously tried queries below.
    - Try different phrasing, more specific terms, or adjacent angles
      that might surface sources the first attempt missed.
    - Every query must still directly help answer the user's question.

    Previously tried queries (do not repeat these):
    {previous_queries}

    Output JSON schema:
    {{
      "queries": ["query 1", "query 2", "query 3"]
    }}

    Scope: {scope}
    User question: {question}
    """.strip()


async def reflect_and_refine(state: ResearchState, llm_provider: LLMProvider) -> dict:
    """Generate refined search queries after a thin first search attempt.

    Args:
        state: The current research state.
        llm_provider: The active LLM provider instance.

    Returns:
        dict: A partial state update with new queries and an incremented
            search_retry_count, or an errors field if refinement fails.
    """
    run_id = state.get("run_id", "unknown")
    retry_count = state.get("search_retry_count") or 0

    prompt = _build_reflection_prompt(state)

    try:
        response = await llm_provider.generate_json(prompt)
    except Exception as e:
        logger.exception("reflect_and_refine failed. run_id=%s error=%s", run_id, e)
        return {"errors": [f"reflect_and_refine failed: {e}"]}

    if not isinstance(response, dict) or "queries" not in response:
        return {"errors": ["reflect_and_refine: Invalid response received from LLM"]}

    raw_queries = response["queries"]
    if not isinstance(raw_queries, list):
        return {"errors": ["reflect_and_refine: 'queries' must be a list"]}

    cleaned = [q.strip() for q in raw_queries if isinstance(q, str) and q.strip()]

    if len(cleaned) < 3:
        return {"errors": ["reflect_and_refine: fewer than 3 valid refined queries"]}

    logger.info(
        "Search reflection triggered. run_id=%s retry=%d new_queries=%d",
        run_id,
        retry_count + 1,
        len(cleaned),
    )
    search_reflection_triggered_total.inc()

    return {
        "queries": cleaned[:6],
        "search_retry_count": retry_count + 1,
        "status": "refining",
    }
