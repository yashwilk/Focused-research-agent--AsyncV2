"""Abstract contract for search providers used by the research agent.

Converted to async to match the LLM provider interface and avoid
occupying a worker thread for the duration of a web search call.
"""

from abc import ABC, abstractmethod
from typing import TypedDict


class SearchResult(TypedDict):
    """Normalized schema for a single search result."""

    title: str
    url: str
    snippet: str
    source: str
    score: float


class SearchProvider(ABC):
    """Abstract contract for search providers used by the research agent."""

    @abstractmethod
    async def search(self, queries: list[str]) -> tuple[list[SearchResult], list[str]]:
        """Run web searches and return normalized results.

        Args:
            queries: A list of search queries to execute.

        Returns:
            tuple: A tuple containing:
                - list[SearchResult]: Normalized search results
                - list[str]: Image URLs found during search
        """
        ...
