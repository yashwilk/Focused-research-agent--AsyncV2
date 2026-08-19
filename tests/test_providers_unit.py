"""
Unit tests for external provider implementations.

What is tested:
- GroqLLMProvider JSON parsing and validation behavior
- TavilySearchClient input validation, normalization, and deduplication

How it is tested:
- external SDK dependencies are replaced with fake classes
- provider methods are exercised with controlled fake responses
- expected exceptions are asserted for invalid cases

Why it matters:
- verifies the reliability of code that interacts with external systems
- ensures provider-level parsing and normalization are robust
"""

from types import SimpleNamespace

import pytest

import focused_research_agent.services.llm_provider_groq as llm_provider_module
import focused_research_agent.services.search_provider_tavily as search_provider_module
from focused_research_agent.services.llm_provider_groq import GroqLLMProvider
from focused_research_agent.services.search_provider_tavily import TavilySearchClient
import focused_research_agent.services.llm_provider_ollama as ollama_provider_module
from focused_research_agent.services.llm_provider_ollama import OllamaLLMProvider


class FakeLLM:
    def __init__(self, content: str):
        self._content = content

    def invoke(self, prompt: str):
        return SimpleNamespace(content=self._content)


class FakeTavilyClient:
    def __init__(self, responses: list[dict]):
        self._responses = responses
        self.calls = []

    def search(
        self,
        query: str,
        search_depth: str,
        max_results: int,
        include_images: bool = False,
    ):
        self.calls.append(
            {
                "query": query,
                "search_depth": search_depth,
                "max_results": max_results,
            }
        )
        return self._responses.pop(0)


def fake_llm_config():
    return {
        "provider": "groq",
        "model": "fake-model",
        "temperature": 0.0,
        "max_retries": 2,
        "api_key": "fake-key",
        "max_tokens": 4096,
    }


def fake_search_config():
    return {
        "provider": "tavily",
        "api_key": "fake-key",
        "max_results": 5,
        "search_depth": "basic",
    }


def fake_init_chat_model_with_valid_json(**kwargs):
    return FakeLLM('{"ok": true}')


def fake_init_chat_model_with_fenced_json(**kwargs):
    return FakeLLM('```json\n{"answer": "ok"}\n```')


def fake_init_chat_model_with_surrounding_text(**kwargs):
    return FakeLLM('Here is the result: {"answer": "ok"} Thanks!')


def fake_init_chat_model_with_no_json(**kwargs):
    return FakeLLM("plain text without any json structure")


def build_fake_tavily_client(responses: list[dict]):
    def fake_tavily_client(api_key: str):
        return FakeTavilyClient(responses)

    return fake_tavily_client


def build_shared_fake_tavily_client(fake_client: FakeTavilyClient):
    def fake_tavily_client(api_key: str):
        return fake_client

    return fake_tavily_client


def test_groq_generate_json_rejects_empty_prompt(monkeypatch):
    monkeypatch.setattr(llm_provider_module, "get_llm_config", fake_llm_config)
    monkeypatch.setattr(
        llm_provider_module,
        "init_chat_model",
        fake_init_chat_model_with_valid_json,
    )

    provider = GroqLLMProvider()

    with pytest.raises(ValueError, match="GroqLLMProvider: No prompt provided!"):
        provider.generate_json("   ")


def test_groq_generate_json_parses_markdown_fenced_json(monkeypatch):
    monkeypatch.setattr(llm_provider_module, "get_llm_config", fake_llm_config)
    monkeypatch.setattr(
        llm_provider_module,
        "init_chat_model",
        fake_init_chat_model_with_fenced_json,
    )

    provider = GroqLLMProvider()
    result = provider.generate_json("test prompt")

    assert result == {"answer": "ok"}


def test_groq_generate_json_parses_json_from_surrounding_text(monkeypatch):
    monkeypatch.setattr(llm_provider_module, "get_llm_config", fake_llm_config)
    monkeypatch.setattr(
        llm_provider_module,
        "init_chat_model",
        fake_init_chat_model_with_surrounding_text,
    )

    provider = GroqLLMProvider()
    result = provider.generate_json("test prompt")

    assert result == {"answer": "ok"}


def test_groq_generate_json_raises_when_no_json_found(monkeypatch):
    monkeypatch.setattr(llm_provider_module, "get_llm_config", fake_llm_config)
    monkeypatch.setattr(
        llm_provider_module,
        "init_chat_model",
        fake_init_chat_model_with_no_json,
    )

    provider = GroqLLMProvider()

    with pytest.raises(ValueError, match="LLM did not return JSON"):
        provider.generate_json("test prompt")


def test_tavily_search_rejects_non_list_queries(monkeypatch):
    monkeypatch.setattr(search_provider_module, "get_search_config", fake_search_config)
    monkeypatch.setattr(
        search_provider_module,
        "TavilyClient",
        build_fake_tavily_client([]),
    )

    provider = TavilySearchClient()

    with pytest.raises(ValueError, match="queries must be a list"):
        provider.search("not-a-list")


def test_tavily_search_rejects_empty_query_string(monkeypatch):
    monkeypatch.setattr(search_provider_module, "get_search_config", fake_search_config)
    monkeypatch.setattr(
        search_provider_module,
        "TavilyClient",
        build_fake_tavily_client([]),
    )

    provider = TavilySearchClient()

    with pytest.raises(ValueError, match="Query must not be empty"):
        provider.search(["valid query", "   "])


def test_tavily_search_deduplicates_urls(monkeypatch):
    responses = [
        {
            "results": [
                {
                    "title": "First result",
                    "url": "https://example.com/a",
                    "content": "Snippet A",
                    "score": 0.95,
                },
                {
                    "title": "Duplicate result",
                    "url": "https://example.com/a",
                    "content": "Duplicate snippet",
                    "score": 0.90,
                },
            ],
            "images": [],
        },
        {
            "results": [
                {
                    "title": "Second unique result",
                    "url": "https://example.com/b",
                    "content": "Snippet B",
                    "score": 0.85,
                }
            ],
            "images": [],
        },
    ]

    fake_client = FakeTavilyClient(responses)
    monkeypatch.setattr(search_provider_module, "get_search_config", fake_search_config)
    monkeypatch.setattr(
        search_provider_module,
        "TavilyClient",
        build_shared_fake_tavily_client(fake_client),
    )

    provider = TavilySearchClient()
    sources, images = provider.search(["query one", "query two"])  # ← unpack tuple

    assert len(sources) == 2
    assert sources[0]["url"] == "https://example.com/a"
    assert sources[1]["url"] == "https://example.com/b"
    assert sources[0]["source"] == "tavily"
    assert images == []


def test_tavily_search_raises_on_invalid_response_shape(monkeypatch):
    responses = [
        {"unexpected_key": []},
    ]

    monkeypatch.setattr(search_provider_module, "get_search_config", fake_search_config)
    monkeypatch.setattr(
        search_provider_module,
        "TavilyClient",
        build_fake_tavily_client(responses),
    )

    provider = TavilySearchClient()

    with pytest.raises(ValueError, match="Tavily response missing valid results"):
        provider.search(["query one"])


def test_tavily_search_raises_when_score_missing(monkeypatch):
    responses = [
        {
            "results": [
                {
                    "title": "Result without score",
                    "url": "https://example.com/a",
                    "content": "Snippet A",
                }
            ]
        }
    ]

    monkeypatch.setattr(search_provider_module, "get_search_config", fake_search_config)
    monkeypatch.setattr(
        search_provider_module,
        "TavilyClient",
        build_fake_tavily_client(responses),
    )

    provider = TavilySearchClient()

    with pytest.raises(ValueError, match="Result missing score"):
        provider.search(["query one"])


def test_tavily_client_overrides_search_depth_when_provided(monkeypatch):
    monkeypatch.setattr(
        search_provider_module,  # ← change tavily_provider_module to this
        "get_search_config",
        fake_search_config,
    )
    monkeypatch.setattr(
        search_provider_module,  # ← same here
        "TavilyClient",
        build_fake_tavily_client(
            []
        ),  # ← use build_fake_tavily_client, not FakeTavilyClient directly
    )
    client = TavilySearchClient(search_depth="advanced")
    assert client.search_config["search_depth"] == "advanced"


def fake_ollama_llm_config():
    return {
        "provider": "ollama",
        "model": "gpt-oss:20b-cloud",
        "temperature": 0.0,
        "max_retries": 2,
        "api_key": "fake-ollama-key",
        "max_tokens": 2048,
    }


def build_fake_ollama_client_class(content: str):
    """Return a fake Client class that returns controlled content."""

    class _FakeClient:
        def __init__(self, **kwargs):
            pass

        def chat(self, model: str, messages: list):
            return SimpleNamespace(message=SimpleNamespace(content=content))

    return _FakeClient


def test_ollama_generate_json_rejects_empty_prompt(monkeypatch):
    monkeypatch.setattr(
        ollama_provider_module, "get_llm_config", fake_ollama_llm_config
    )
    monkeypatch.setattr(
        ollama_provider_module, "Client", build_fake_ollama_client_class('{"ok": true}')
    )

    provider = OllamaLLMProvider()

    with pytest.raises(ValueError, match="OllamaLLMProvider: No prompt provided!"):
        provider.generate_json("   ")


def test_ollama_generate_json_parses_valid_json(monkeypatch):
    monkeypatch.setattr(
        ollama_provider_module, "get_llm_config", fake_ollama_llm_config
    )
    monkeypatch.setattr(
        ollama_provider_module,
        "Client",
        build_fake_ollama_client_class('{"answer": "ok"}'),
    )

    provider = OllamaLLMProvider()
    result = provider.generate_json("test prompt")

    assert result == {"answer": "ok"}


def test_ollama_generate_json_parses_fenced_json(monkeypatch):
    monkeypatch.setattr(
        ollama_provider_module, "get_llm_config", fake_ollama_llm_config
    )
    monkeypatch.setattr(
        ollama_provider_module,
        "Client",
        build_fake_ollama_client_class('```json\n{"answer": "ok"}\n```'),
    )

    provider = OllamaLLMProvider()
    result = provider.generate_json("test prompt")

    assert result == {"answer": "ok"}


def test_ollama_generate_json_parses_json_from_surrounding_text(monkeypatch):
    monkeypatch.setattr(
        ollama_provider_module, "get_llm_config", fake_ollama_llm_config
    )
    monkeypatch.setattr(
        ollama_provider_module,
        "Client",
        build_fake_ollama_client_class('Here is the result: {"answer": "ok"} Thanks!'),
    )

    provider = OllamaLLMProvider()
    result = provider.generate_json("test prompt")

    assert result == {"answer": "ok"}


def test_ollama_generate_json_raises_when_no_json_found(monkeypatch):
    monkeypatch.setattr(
        ollama_provider_module, "get_llm_config", fake_ollama_llm_config
    )
    monkeypatch.setattr(
        ollama_provider_module,
        "Client",
        build_fake_ollama_client_class("plain text without any json structure"),
    )

    provider = OllamaLLMProvider()

    with pytest.raises(ValueError, match="LLM did not return JSON"):
        provider.generate_json("test prompt")


def test_ollama_uses_cloud_client_when_api_key_provided(monkeypatch):
    """
    Verify that OllamaLLMProvider passes host and Authorization header
    when an API key is present in the config.
    """
    captured_kwargs = {}

    class FakeCloudClient:
        def __init__(self, **kwargs):
            captured_kwargs.update(kwargs)

        def chat(self, model: str, messages: list):
            return SimpleNamespace(message=SimpleNamespace(content='{"ok": true}'))

    monkeypatch.setattr(
        ollama_provider_module, "get_llm_config", fake_ollama_llm_config
    )
    monkeypatch.setattr(ollama_provider_module, "Client", FakeCloudClient)

    OllamaLLMProvider()

    assert "host" in captured_kwargs
    assert "headers" in captured_kwargs
    assert "Authorization" in captured_kwargs["headers"]
