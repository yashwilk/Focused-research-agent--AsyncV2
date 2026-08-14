"""
Search provider factory for the Focused Research Agent.

This module contains the factory function responsible for instantiating
the correct search provider implementation based on the SEARCH_PROVIDER
environment variable.

Adding a new search provider requires:
- Implementing the SearchProvider interface
- Adding a new branch in get_search_provider


Architecturally, this module belongs to the services layer and implements
the Factory pattern. It keeps provider selection logic in one place and
decouples the rest of the application from concrete provider classes.
"""

from focused_research_agent.config.search_config import get_search_config
from focused_research_agent.interfaces.search_inference import SearchProvider
from focused_research_agent.services.search_provider_tavily import TavilySearchClient


def get_search_provider(search_depth: str | None = None) -> SearchProvider:
    """Return the active search provider implementation.

    Args:
        search_depth: Optional override for search depth. When provided,
            overrides the SEARCH_DEPTH environment variable. Accepts
            'basic' or 'advanced'.

    Returns:
        SearchProvider: The configured search provider instance.

    Raises:
        ValueError: If the configured provider is unsupported.
    """
    search_config = get_search_config()
    provider = search_config["provider"]

    if provider == "tavily":
        return TavilySearchClient(search_depth=search_depth)
    else:
        raise ValueError(f"Unsupported search provider: {provider}")
