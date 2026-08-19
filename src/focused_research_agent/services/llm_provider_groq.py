"""
Groq-backed implementation of the LLM provider contract.

Async conversion: uses the chat model's native `.ainvoke()` instead of
`.invoke()`, so a Groq call no longer occupies a worker thread for its
full duration — the event loop is free to serve other requests while
waiting on the network.

"""

import json
import logging

from langchain.chat_models import init_chat_model
from tenacity import retry, stop_after_attempt, wait_exponential

from focused_research_agent.config.llm_config import get_llm_config
from focused_research_agent.interfaces.llm_inference import LLMProvider
from focused_research_agent.reliability.circuit_breaker import get_circuit_breaker

logger = logging.getLogger(__name__)


class GroqLLMProvider(LLMProvider):
    """Groq-backed implementation of the LLM provider contract."""

    def __init__(self):
        """Initialize the Groq LLM client using validated config."""
        self.llm_config = get_llm_config()

        self.llm = init_chat_model(
            model_provider=self.llm_config["provider"],
            model=self.llm_config["model"],
            temperature=self.llm_config["temperature"],
            max_retries=self.llm_config["max_retries"],
            api_key=self.llm_config["api_key"],
            max_tokens=self.llm_config["max_tokens"],
        )
        self._breaker = get_circuit_breaker("groq")

    # ------------------------------------------------------------------
    # Static helpers — pure functions that support the provider but do
    # not read or modify any instance state.
    # ------------------------------------------------------------------

    @staticmethod
    def _build_json_only_prompt(prompt: str) -> str:
        """Validate the prompt and append a strict JSON-only instruction."""
        if not isinstance(prompt, str) or not prompt.strip():
            raise ValueError("GroqLLMProvider: No prompt provided!")

        return (
            prompt
            + "\nReturn ONLY valid JSON. No markdown. No backticks. No extra text."
        )

    @staticmethod
    def _strip_code_fences(text: str) -> str:
        """Remove surrounding triple-backtick code fences from LLM output."""
        if text.startswith("```"):
            lines = text.splitlines()

            if lines:
                lines = lines[1:]

            if lines and lines[-1].strip().startswith("```"):
                lines = lines[:-1]

            return "\n".join(lines).strip()

        return text

    @staticmethod
    def _extract_json_candidate(text: str) -> str | None:
        """Extract a likely JSON object or array substring from mixed text."""
        obj_start = text.find("{")
        obj_end = text.rfind("}")
        arr_start = text.find("[")
        arr_end = text.rfind("]")

        if obj_start != -1 and obj_end != -1 and obj_start < obj_end:
            return text[obj_start : obj_end + 1]

        if arr_start != -1 and arr_end != -1 and arr_start < arr_end:
            return text[arr_start : arr_end + 1]

        return None

    @staticmethod
    def _extract_text_from_content(content: str | list) -> str:
        """Extract a plain text string from a LangChain response content value."""
        if isinstance(content, str):
            return content

        if isinstance(content, list):
            text_parts = []
            for block in content:
                if isinstance(block, dict):
                    text_parts.append(block.get("text", ""))
                else:
                    text_parts.append(str(block))
            return "".join(text_parts)

        return ""

    # ------------------------------------------------------------------
    # Instance methods
    # ------------------------------------------------------------------

    @retry(
        stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=1, max=8)
    )
    async def _invoke_with_retry(self, updated_prompt: str):
        return await self.llm.ainvoke(updated_prompt)

    async def generate_json(self, prompt: str) -> dict:
        """Generate structured JSON from a prompt using Groq (async).

        Args:
            prompt: The prompt to send to the LLM.

        Returns:
            dict: Parsed JSON output from the LLM.

        Raises:
            ValueError: If the prompt is invalid or the provider does not
                return recoverable valid JSON.
            CircuitBreakerOpenError: If Groq has failed repeatedly and the
                circuit is currently open.
        """
        updated_prompt = self._build_json_only_prompt(prompt)
        logger.info("Invoking Groq LLM with model: %s", self.llm_config["model"])

        response = await self._breaker.call(
            lambda: self._invoke_with_retry(updated_prompt)
        )

        raw_text = self._extract_text_from_content(response.content)
        text = self._strip_code_fences(raw_text.strip())

        try:
            return json.loads(text)
        except json.JSONDecodeError:
            logger.exception("Invalid JSON from LLM")

        candidate = self._extract_json_candidate(text)

        if candidate is None:
            raise ValueError(f"LLM did not return JSON. Raw output:\n{text[:400]}")

        try:
            return json.loads(candidate)
        except json.JSONDecodeError as e:
            logger.exception("Invalid JSON from LLM\nRaw output:\n%s", candidate[:400])
            raise ValueError(
                f"Invalid JSON from LLM: {e}\nRaw output:\n{candidate[:400]}"
            )
