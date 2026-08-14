"""
Answer synthesis node for the Focused Research Agent.

This module contains the node responsible for synthesizing a final answer
from the collected and ranked web sources. It supports two modes:

- research: Produces a concise answer with 1 to 3 citations using the
  top ranked sources. Includes conversation history context when available
  for multi-turn chat sessions.

- report: Produces a structured long-form markdown report with four
  sections (Introduction, Key Findings, Analysis, Conclusion) and 3 to 5
  citations using a larger source set.

Sources are validated, normalized, and ranked by domain trust score before
being passed to the LLM. Citation URLs returned by the LLM are validated
against the allowed source set and normalized before being stored in state.
"""

import logging
from urllib.parse import urlparse

from focused_research_agent.interfaces.llm_inference import LLMProvider
from focused_research_agent.state import ResearchState

logger = logging.getLogger(__name__)

INVALID_LLM_RESPONSE_ERROR_MESSAGE = "Invalid response obtained from LLM"
_REPORT_MAX_SOURCES = 15
_RESEARCH_MAX_SOURCES = 6

# This is a lightweight heuristic for ranking, not a full trust system
_DOMAIN_BONUSES = {
    "britannica.com": 3.0,
    "timeanddate.com": 3.0,
    "metoffice.gov.uk": 3.0,
    "weather.gov": 3.0,
    "noaa.gov": 3.0,
}

_DOMAIN_PENALTIES = {
    "youtube.com": -3.0,
    "medium.com": -3.0,
    "reddit.com": -3.0,
    "quora.com": -3.0,
    "facebook.com": -3.0,
    "tiktok.com": -3.0,
    "instagram.com": -3.0,
}


def _extract_domain(url: str) -> str:
    """Extract and normalize the domain name from a URL."""
    domain = urlparse(url).netloc.lower().strip()

    if domain.startswith("www."):
        domain = domain[4:]

    return domain


def _matches_domain(domain: str, target: str) -> bool:
    """Return whether a domain matches or is a subdomain of a target."""
    return domain == target or domain.endswith("." + target)


def _get_domain_bonus(domain: str) -> float:
    """Return a lightweight ranking bonus or penalty for a domain.

    This is a simple heuristic used to slightly prefer stronger
    reference-style domains and slightly down-rank weaker ones.
    """
    if domain.endswith(".gov"):
        return 4.0

    if domain.endswith(".edu"):
        return 3.5

    for trusted_domain, bonus in _DOMAIN_BONUSES.items():
        if _matches_domain(domain, trusted_domain):
            return bonus

    for weak_domain, penalty in _DOMAIN_PENALTIES.items():
        if _matches_domain(domain, weak_domain):
            return penalty

    return 0.0


def _get_rank_score(source: dict) -> float:
    """Compute the final ranking score for a source.

    The score combines the provider score with a small domain-based
    heuristic.
    """
    domain = _extract_domain(source["url"])
    bonus = _get_domain_bonus(domain)
    return source["score"] + bonus


def _collect_valid_sources(sources: list[dict]) -> list[dict]:
    """Validate, normalize, and rank candidate sources for synthesis.

    Args:
        sources: Raw source items from the research state.

    Returns:
        list[dict]: Ranked source dictionaries containing title, url,
        snippet, source, and score.
    """

    valid_sources = []

    for item in sources:
        if not isinstance(item, dict):
            continue

        title = (item.get("title") or "").strip()
        url = (item.get("url") or "").strip()
        snippet = (item.get("snippet") or "").strip()
        source_name = (item.get("source") or "").strip()
        score = item.get("score", 0.0)

        if not title or not url or not snippet:
            continue

        try:
            score = float(score)
        except (TypeError, ValueError):
            score = 0.0

        valid_sources.append(
            {
                "title": title,
                "url": url,
                "snippet": snippet,
                "source": source_name,
                "score": score,
            }
        )

    if not valid_sources:
        return []

    return sorted(valid_sources, key=_get_rank_score, reverse=True)


def _build_synthesis_prompt(
    question: str, sources: list[dict], conversation_history: list[dict] | None
) -> str:
    """
    Build the LLM prompt for final answer synthesis.

    Includes conversation history context when available so the LLM
    can reference prior turns when answering follow-up questions. When
    conversation_history is None or empty, the prompt is identical to
    the single-turn research flow.

    Args:
        question: The original user question.
        sources: Ranked source dictionaries selected for synthesis.
        conversation_history: Prior conversation turns from the
            application layer, or None for single-turn research.
            Each item contains turn, question, answer, and scope keys.

    Returns:
        str: A prompt instructing the LLM to return a concise answer
            and 1 to 3 exact citation URLs as strict JSON.
    """

    source_blocks = []

    for index, source in enumerate(sources, start=1):
        source_block = (
            f"Source {index}\n"
            f"Title: {source['title']}\n"
            f"URL: {source['url']}\n"
            f"Snippet: {source['snippet']}\n"
        )
        source_blocks.append(source_block)

    joined_sources = "\n".join(source_blocks)

    conversation_context = ""
    if conversation_history:
        context_lines = ["CONVERSATION HISTORY:"]
        for turn in conversation_history:
            context_lines.append(f"Turn {turn['turn']} — Q: {turn['question']}")
            context_lines.append(f"         A: {turn['answer']}")
        context_lines.append(
            "\nAnswer the current question in the context of the "
            "above conversation history where relevant."
        )
        conversation_context = "\n".join(context_lines) + "\n\n"

    return f"""{conversation_context}Return ONLY valid JSON. No markdown. No backticks. No extra text.

The JSON MUST have exactly these keys:
- answer (string)
- citations (list of 1 to 3 URLs)

Rules:
- Answer the user's question directly in the first sentence.
- Then add 2 to 3 short supporting sentences.
- Keep the answer clear, natural, and concise.
- Avoid repetition.
- Prefer the strongest and most trustworthy sources.
- Prefer official, educational, scientific, or well-known reference sources when available.
- Use ONLY the sources provided below.
- Do NOT invent facts.
- Do NOT invent citations.
- Every citation URL MUST match one of the provided source URLs exactly.
- Choose the best 2 to 3 citations, not just any valid citations.
- Do NOT mention "sources", "snippets", or "citations" inside the answer.

Example JSON output:
{{
  "answer": "An equinox is when day and night are nearly equal in length, while a solstice is when the Sun reaches its highest or lowest point in the sky, creating the longest or shortest day of the year. Equinoxes mark the start of spring and autumn. Solstices mark the start of summer and winter.",
  "citations": [
    "https://example.com/source1",
    "https://example.com/source2"
  ]
}}

User question:
{question}

Sources:
{joined_sources}
""".strip()


def _build_report_prompt(question: str, sources: list[dict]) -> str:
    """
    Build the LLM prompt for structured report generation.

    Instructs the LLM to produce a long-form research report with
    four sections: Introduction, Key Findings, Analysis, and
    Conclusion. Uses all available sources for comprehensive coverage.

    Args:
        question: The original user question.
        sources: Ranked source dictionaries selected for synthesis.

    Returns:
        str: A prompt instructing the LLM to return a structured
            markdown report and 3 to 5 citation URLs as strict JSON.
    """
    source_blocks = []

    for index, source in enumerate(sources, start=1):
        source_block = (
            f"Source {index}\n"
            f"Title: {source['title']}\n"
            f"URL: {source['url']}\n"
            f"Snippet: {source['snippet']}\n"
        )
        source_blocks.append(source_block)

    joined_sources = "\n".join(source_blocks)

    return f"""Return ONLY valid JSON with exactly these keys:
    - answer (string) — a full structured report in markdown format with these sections:
    ## Introduction
    ## Key Findings
    ## Analysis
    ## Conclusion
    - citations (list of 3 to 5 URLs)

    Rules:
    - Write EXACTLY 3 substantial paragraphs per section
    - Each paragraph must be at least 4 sentences long
    - NEVER repeat the same sentence or idea across different sections
    - Each section must contain DIFFERENT information:
        * Introduction: background and context only
        * Key Findings: specific discoveries and data points only
        * Analysis: implications, comparisons, and critical examination only
        * Conclusion: summary of significance and future outlook only
    - Be comprehensive and detailed
    - Use all provided sources
    - Every claim must be supported by the sources
    - Do NOT invent facts or citations
    - Every citation URL must match a provided source URL exactly
    - Do NOT mention "sources", "snippets", or "citations" inside the answer


    Example JSON output:
    {{
      "answer": "An equinox is when day and night are nearly equal in length, while a solstice is when the Sun reaches its highest or lowest point in the sky, creating the longest or shortest day of the year. Equinoxes mark the start of spring and autumn. Solstices mark the start of summer and winter.",
      "citations": [
        "https://example.com/source1",
        "https://example.com/source2"
      ]
    }}

    User question:
    {question}

    Sources:
    {joined_sources}
    """.strip()


def _validate_synthesis_response(response: object) -> tuple[str, list]:
    """Validate the raw LLM synthesis response.

    Args:
        response: Raw object returned by the LLM provider.

    Returns:
        tuple[str, list]: The validated answer string and raw citations list.

    Raises:
        ValueError: If the response shape is invalid or required fields
            are missing/empty.
    """

    if not isinstance(response, dict):
        raise ValueError(INVALID_LLM_RESPONSE_ERROR_MESSAGE)

    answer = response.get("answer")
    citations = response.get("citations")

    if not isinstance(answer, str) or not answer.strip():
        raise ValueError(INVALID_LLM_RESPONSE_ERROR_MESSAGE)

    if not isinstance(citations, list) or not citations:
        raise ValueError(INVALID_LLM_RESPONSE_ERROR_MESSAGE)
    return (answer, citations)


def _normalize_url(url: str) -> str:
    """Normalize a URL for comparison by stripping trailing slashes."""
    return url.strip().rstrip("/").lower()


def _clean_citations(citations: list, allowed_urls: set[str]) -> list[str]:
    """Validate, deduplicate, and filter returned citations.

    Args:
        citations: Raw citations returned by the LLM.
        allowed_urls: Set of source URLs that the LLM is allowed to cite.

    Returns:
        list[str]: Cleaned citations limited to at most 3 items.

    Raises:
        ValueError: If citations are malformed, empty, or include URLs
            outside the allowed source set.
    """

    normalized_allowed = {}
    for url in allowed_urls:
        normalized_allowed[_normalize_url(url)] = url
    cleaned_citations = []
    seen_citations = set()

    for citation in citations:
        if not isinstance(citation, str):
            raise ValueError("Citation must be a string")

        citation = citation.strip()

        if not citation:
            raise ValueError("Empty citation returned by LLM")

        normalized_citation = _normalize_url(citation)
        if normalized_citation not in normalized_allowed:
            raise ValueError(
                f"synthesize_answer: LLM returned unknown citation URL: {citation}"
            )

        original_url = normalized_allowed[normalized_citation]
        if original_url not in seen_citations:
            seen_citations.add(original_url)
            cleaned_citations.append(original_url)

    if not cleaned_citations:
        raise ValueError("No valid citations found")
    return cleaned_citations


async def synthesize_answer(state: ResearchState, llm_provider: LLMProvider) -> dict:
    """Synthesize a final answer and citations from collected sources.

    This node validates the available sources, builds a synthesis prompt,
    asks the LLM for structured output, and verifies that all returned
    citations come from the provided source set.

    Args:
        state: The current research state.
        llm_provider: The active LLM provider instance.

    Returns:
        dict: A partial state update containing answer, citations, and
        status, or an errors field if synthesis fails.
    """
    mode = state.get("mode")
    question = (state.get("question") or "").strip()
    sources = state.get("sources")
    conversation_history = state.get("conversation_history")
    run_id = state.get("run_id", "unknown")

    if not question:
        logger.error("synthesize_answer: No question found. run_id=%s", run_id)
        return {"errors": ["synthesize_answer: No question found"]}

    if not isinstance(sources, list) or not sources:
        logger.error("synthesize_answer: No sources found. run_id=%s", run_id)
        return {"errors": ["synthesize_answer: No sources found"]}

    valid_sources = _collect_valid_sources(sources)

    if not valid_sources:
        logger.warning(
            "synthesize_answer: No valid sources after filtering. run_id=%s", run_id
        )
        return {"errors": ["synthesize_answer: No valid sources found"]}

    if mode == "report":
        synthesis_sources = valid_sources[:_REPORT_MAX_SOURCES]
    else:
        synthesis_sources = valid_sources[:_RESEARCH_MAX_SOURCES]

    allowed_urls = {source["url"] for source in synthesis_sources}

    if mode == "report":
        prompt = _build_report_prompt(question, synthesis_sources)
    else:
        prompt = _build_synthesis_prompt(
            question, synthesis_sources, conversation_history
        )

    try:
        response = await llm_provider.generate_json(prompt)
    except Exception as e:
        logger.exception("synthesize_answer failed. run_id=%s error=%s", run_id, e)
        return {"errors": [f"synthesize_answer failed: {e}"]}

    try:
        answer, citations = _validate_synthesis_response(response)
        cleaned_citations = _clean_citations(citations, allowed_urls)
    except ValueError as e:
        logger.exception(
            "synthesize_answer: Validation failed. run_id=%s error=%s", run_id, e
        )
        return {"errors": [str(e)]}

    logger.info(  # ← add
        "Synthesis completed. run_id=%s mode=%s citations=%d",
        run_id,
        mode,
        len(cleaned_citations),
    )
    max_citations = 5 if mode == "report" else 3
    return {
        "answer": answer.strip(),
        "citations": cleaned_citations[:max_citations],
        "status": "synthesized",
    }
