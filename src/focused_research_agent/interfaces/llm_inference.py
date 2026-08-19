"""Abstract contract for LLM providers used by the research agent."""

from abc import ABC, abstractmethod


class LLMProvider(ABC):
    """Abstract contract for LLM providers used by the research agent."""

    @abstractmethod
    async def generate_json(self, prompt: str) -> dict:
        """Generate and return structured JSON for a given prompt.

        Args:
            prompt: The input prompt to send to the LLM.

        Returns:
            dict: Parsed JSON output returned by the provider.
        """
        ...
