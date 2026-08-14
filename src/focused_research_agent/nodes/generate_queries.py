"""
Query generation node for the Focused Research Agent.

This module contains the node responsible for producing focused web-search
queries from the scoped question. It uses the LLM provider to return 3 to 6
short, diverse, search-engine-style queries that directly support answering
the user's research question.
"""

import logging

from focused_research_agent.interfaces.llm_inference import LLMProvider
from focused_research_agent.state import ResearchState

logger = logging.getLogger(__name__)


def _build_generate_queries_prompt(state: ResearchState) -> str:
    """Build the LLM prompt for multi-query search planning.

    Args:
        state: The current research state containing the scoped question,
            original question, assumptions, and constraints.

    Returns:
        str: A prompt instructing the LLM to return 3 to 6 focused
            search-engine-style queries as strict JSON.
    """
    scope = (state.get("scope") or "").strip()
    user_query = (state.get("question") or "").strip()
    assumptions = state.get("assumptions") or []
    constraints = state.get("constraints") or {}

    generate_queries_system_prompt = """
        Return ONLY valid JSON. No markdown. No backticks. No extra text.
        Return EXACTLY one key: "queries".

        Task:
        - Generate 3 to 6 search-engine style queries (Google-style phrases).
        - Do NOT repeat the scope sentence verbatim as a query.
        - Queries must be diverse: each query should target a different facet of the topic.
        - Every query must directly help answer the user's specific question.
        - Do NOT generate queries about the general topic area if they do not help answer what the user actually asked.

        Facet coverage rule:
        - First, internally identify 4 to 6 key facets relevant to the scope and user question.
        - Then produce queries so each query focuses on a different facet.

        Use provided inputs:
        - If constraints include geography or time, include those terms in relevant queries.
        - Keep each query short (typically 5 to 10 words).

        Output JSON schema:
        {
          "queries": ["query 1", "query 2", "query 3"]
        }
        """.strip()

    inputs = f"SCOPE: {scope}\nASSUMPTIONS: {assumptions}\nCONSTRAINTS: {constraints}"

    question_scope = f"""
        {generate_queries_system_prompt}

        {inputs}

        User question:
        {user_query}
        """.strip()

    return question_scope


def _clean_generated_queries(llm_queries: object) -> list[str]:
    """Validate and clean the raw query list returned by the LLM.

    Ensures the LLM response is a list of non-empty strings, removes
    blank entries, enforces a minimum of 3 valid queries, and caps the
    result at 6 queries.

    Args:
        llm_queries: Raw value extracted from the LLM JSON response under
            the "queries" key.

    Returns:
        list[str]: Cleaned list of between 3 and 6 non-empty query strings.

    Raises:
        ValueError: If llm_queries is not a list, contains non-string
            items, or yields fewer than 3 valid queries after cleaning.
    """
    if not isinstance(llm_queries, list):
        raise ValueError("generate_queries: 'queries' must be a list")

    cleaned_list = []

    for item in llm_queries:
        if not isinstance(item, str):
            raise ValueError("generate_queries: Query item must be a string")

        item = item.strip()
        if item:
            cleaned_list.append(item)

    if len(cleaned_list) < 3:
        raise ValueError("generate_queries: LLM returned fewer than 3 valid queries")

    return cleaned_list[:6]


async def generate_queries(state: ResearchState, llm_provider: LLMProvider) -> dict:
    """Generate focused web-search queries from the scoped question.

    This node uses the LLM provider to produce 3 to 6 short,
    search-engine-style queries that directly support answering the
    user's question.

    Args:
        state: The current research state.
        llm_provider: The active LLM provider instance.

    Returns:
        dict: A partial state update containing generated queries and
            status, or an errors field if generation fails.
    """
    base = (state.get("scope") or state.get("question") or "").strip()
    run_id = state.get("run_id", "unknown")

    if not base:
        logger.error(
            "generate_queries: No scope or question available. run_id=%s", run_id
        )
        return {"errors": ["generate_queries: No scope or question available"]}

    question_scope = _build_generate_queries_prompt(state)

    try:
        response = await llm_provider.generate_json(question_scope)
    except Exception as e:
        logger.exception("generate_queries failed. run_id=%s error=%s", run_id, e)
        return {"errors": [f"generate_queries failed: {e}"]}

    if not isinstance(response, dict) or "queries" not in response:
        return {"errors": ["generate_queries: Invalid response received from LLM"]}

    llm_queries = response["queries"]

    try:
        cleaned_list = _clean_generated_queries(llm_queries)
    except ValueError as e:
        logger.warning(
            "generate_queries: Query validation failed. run_id=%s error=%s", run_id, e
        )
        return {"errors": [str(e)]}

    logger.info("Queries generated. run_id=%s count=%d", run_id, len(cleaned_list))
    return {
        "queries": cleaned_list,
        "status": "planned",
    }
