"""
Tavily-backed implementation of the search provider contract.

Async conversion: uses tavily.AsyncTavilyClient instead of the sync
TavilyClient. Combined with retry + a circuit breaker — directly
addresses the README's stated gap: "No retry logic for Tavily failures"
and "No circuit breaker for provider outages."
"""

import asyncio
import logging

from tavily import AsyncTavilyClient
from tenacity import retry, stop_after_attempt, wait_exponential

from focused_research_agent.config.search_config import get_search_config
from focused_research_agent.interfaces.search_inference import (
    SearchProvider,
    SearchResult,
)
from focused_research_agent.reliability.circuit_breaker import get_circuit_breaker

logger = logging.getLogger(__name__)


class TavilySearchClient(SearchProvider):
    """Tavily-backed implementation of the search provider contract."""

    def __init__(self, search_depth: str | None = None):
        """Initialize the async Tavily search client using validated config."""
        self.search_config = get_search_config()
        self.tavily_client = AsyncTavilyClient(api_key=self.search_config["api_key"])
        self._breaker = get_circuit_breaker("tavily")

        if search_depth is not None:
            self.search_config["search_depth"] = search_depth

    # ------------------------------------------------------------------
    # Static helpers — pure validation, no instance state.
    # ------------------------------------------------------------------

    @staticmethod
    def _validate_queries(queries: list[str]) -> list[str]:
        """Validate and clean incoming search queries."""
        if not isinstance(queries, list):
            raise ValueError("TavilySearchClient: queries must be a list")

        if len(queries) == 0:
            raise ValueError("TavilySearchClient: No queries provided")

        cleaned_queries = []

        for query in queries:
            if not isinstance(query, str):
                raise ValueError("TavilySearchClient: Query must be a string")

            cleaned_query = query.strip()
            if not cleaned_query:
                raise ValueError("TavilySearchClient: Query must not be empty")

            cleaned_queries.append(cleaned_query)

        return cleaned_queries

    @staticmethod
    def _validate_tavily_response(response: object, query: str) -> list[dict]:
        """Validate the Tavily API response shape for a single query."""
        if not isinstance(response, dict) or "results" not in response:
            raise ValueError(
                f"search_client: Tavily response missing valid results: {query}"
            )

        results = response["results"]

        if not isinstance(results, list):
            raise ValueError(
                f"search_client: Tavily response missing valid results: {query}"
            )

        return results

    # ------------------------------------------------------------------
    # Instance methods
    # ------------------------------------------------------------------

    def _normalize_result(self, item: dict, query: str) -> SearchResult:
        """Normalize one Tavily result item into the shared SearchResult shape."""
        if not isinstance(item, dict):
            raise ValueError(
                f"TavilySearchClient: Invalid result item returned for query: {query}"
            )

        title = (item.get("title") or "").strip()
        url = (item.get("url") or "").strip()
        snippet = (item.get("content") or "").strip()
        score_raw = item.get("score")

        if not title or not url:
            raise ValueError(
                f"TavilySearchClient: Result missing title or url for query: {query}"
            )

        if score_raw is None:
            raise ValueError(
                f"TavilySearchClient: Result missing score for query: {query}"
            )

        if not isinstance(score_raw, (int, float, str)):
            raise ValueError(
                f"TavilySearchClient: Invalid score type in result for query: {query}"
            )

        try:
            score: float = float(score_raw)
        except ValueError:
            raise ValueError(
                f"TavilySearchClient: Invalid score value in result for query: {query}"
            )

        return {
            "title": title,
            "url": url,
            "snippet": snippet,
            "source": self.search_config["provider"],
            "score": score,
        }

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=1, max=8))
    async def _search_call_with_retry(self, query: str) -> dict:
        return await self.tavily_client.search(
            query=query,
            search_depth=self.search_config["search_depth"],
            max_results=self.search_config["max_results"],
            include_images=True,
        )

    async def _search_single_query(
        self, query: str
    ) -> tuple[list[SearchResult], list[str]]:
        """Run Tavily search for one query and normalize the returned results."""
        response = await self._breaker.call(lambda: self._search_call_with_retry(query))

        response_results = self._validate_tavily_response(response, query)
        images = response.get("images") or []

        normalized_results: list[SearchResult] = [
            self._normalize_result(item, query) for item in response_results
        ]

        logger.debug(
            "Query completed. query='%s' results=%d images=%d",
            query[:60],
            len(normalized_results),
            len(images),
        )

        return normalized_results, images

    async def search(self, queries: list[str]) -> tuple[list[SearchResult], list[str]]:
        """Run Tavily searches concurrently and return deduplicated results.

        Queries are dispatched concurrently via asyncio.gather instead of
        sequentially — a research run with 6 queries now takes roughly as
        long as the slowest single query instead of the sum of all of them.

        Args:
            queries: A list of validated search queries.

        Returns:
            list[SearchResult]: Deduplicated and normalized search results.
            list[str]: List of images

        Raises:
            ValueError: If the query list is invalid or Tavily returns an
                unexpected response structure.
        """
        cleaned_queries = self._validate_queries(queries)

        query_results_list = await asyncio.gather(
            *(self._search_single_query(query) for query in cleaned_queries)
        )

        final_search_results: list[SearchResult] = []
        all_images: list[str] = []
        seen_urls: set[str] = set()

        for query_results, query_images in query_results_list:
            for result in query_results:
                if result["url"] in seen_urls:
                    continue
                seen_urls.add(result["url"])
                final_search_results.append(result)
            all_images.extend(query_images)

        logger.info(
            "Search completed. queries=%d total_sources=%d total_images=%d",
            len(cleaned_queries),
            len(final_search_results),
            len(all_images),
        )

        return final_search_results, all_images
