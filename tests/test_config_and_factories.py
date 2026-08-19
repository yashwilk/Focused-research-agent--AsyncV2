import pytest

import focused_research_agent.config.llm_config as llm_config_module
import focused_research_agent.config.search_config as search_config_module
import focused_research_agent.services.llm_factory as llm_factory_module
import focused_research_agent.services.search_factory as search_factory_module

"""
Tests for configuration loading and provider factory selection.

What is tested:
- environment variable validation for LLM and search config
- numeric parsing and missing-value handling
- correct provider returned by LLM/search factories
- clear failure on unsupported provider names

How it is tested:
- monkeypatch is used to simulate environment variables
- factory dependencies are replaced with fake provider classes

Why it matters:
- ensures the app fails early and clearly on setup errors
- verifies provider selection logic without constructing real providers
"""


class FakeGroqLLMProvider:
    pass


class FakeTavilySearchClient:
    def __init__(self, search_depth: str | None = None):  # ← add this
        pass


def fake_llm_config_groq():
    return {
        "provider": "groq",
        "model": "fake-model",
        "temperature": 0.0,
        "max_retries": 2,
        "api_key": "fake-key",
        "max_tokens": 4096,
    }


def fake_llm_config_bad_provider():
    return {
        "provider": "not-supported",
        "model": "fake-model",
        "temperature": 0.0,
        "max_retries": 2,
        "api_key": "fake-key",
    }


def fake_search_config_tavily():
    return {
        "provider": "tavily",
        "api_key": "fake-key",
        "search_depth": "basic",
        "max_results": 5,
    }


def fake_search_config_bad_provider():
    return {
        "provider": "not-supported",
        "api_key": "fake-key",
        "max_results": 5,
        "search_depth": "basic",
    }


def test_get_llm_config_success(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "groq")
    monkeypatch.setenv("LLM_MODEL", "llama-test")
    monkeypatch.setenv("LLM_TEMPERATURE", "0.0")
    monkeypatch.setenv("LLM_MAX_RETRIES", "2")
    monkeypatch.setenv("LLM_API_KEY", "fake-key")
    monkeypatch.setenv("LLM_MAX_TOKENS", "4096")

    result = llm_config_module.get_llm_config()

    assert result == {
        "provider": "groq",
        "model": "llama-test",
        "temperature": 0.0,
        "max_retries": 2,
        "api_key": "fake-key",
        "max_tokens": 4096,
    }


def test_get_llm_config_raises_when_required_value_missing(monkeypatch):
    monkeypatch.delenv("LLM_PROVIDER", raising=False)
    monkeypatch.setenv("LLM_MODEL", "llama-test")
    monkeypatch.setenv("LLM_TEMPERATURE", "0.0")
    monkeypatch.setenv("LLM_MAX_RETRIES", "2")
    monkeypatch.setenv("LLM_API_KEY", "fake-key")

    with pytest.raises(ValueError, match="should be given in .env file"):
        llm_config_module.get_llm_config()


def test_get_llm_config_raises_when_temperature_invalid(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "groq")
    monkeypatch.setenv("LLM_MODEL", "llama-test")
    monkeypatch.setenv("LLM_TEMPERATURE", "not-a-float")
    monkeypatch.setenv("LLM_MAX_RETRIES", "2")
    monkeypatch.setenv("LLM_API_KEY", "fake-key")

    with pytest.raises(ValueError, match="LLM_TEMPERATURE must be a float"):
        llm_config_module.get_llm_config()


def test_get_llm_config_raises_when_max_retries_invalid(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "groq")
    monkeypatch.setenv("LLM_MODEL", "llama-test")
    monkeypatch.setenv("LLM_TEMPERATURE", "0.0")
    monkeypatch.setenv("LLM_MAX_RETRIES", "not-an-int")
    monkeypatch.setenv("LLM_API_KEY", "fake-key")

    with pytest.raises(ValueError, match="LLM_MAX_RETRIES must be an int"):
        llm_config_module.get_llm_config()


def test_get_search_config_success(monkeypatch):
    monkeypatch.setenv("SEARCH_PROVIDER", "tavily")
    monkeypatch.setenv("SEARCH_API_KEY", "fake-key")
    monkeypatch.setenv("SEARCH_MAX_RESULTS", "5")
    monkeypatch.setenv("SEARCH_DEPTH", "basic")

    result = search_config_module.get_search_config()

    assert result == {
        "provider": "tavily",
        "api_key": "fake-key",
        "search_depth": "basic",
        "max_results": 5,
    }


def test_get_search_config_raises_when_required_value_missing(monkeypatch):
    monkeypatch.delenv("SEARCH_PROVIDER", raising=False)
    monkeypatch.setenv("SEARCH_API_KEY", "fake-key")
    monkeypatch.setenv("SEARCH_MAX_RESULTS", "5")
    monkeypatch.setenv("SEARCH_DEPTH", "basic")

    with pytest.raises(ValueError, match="should be given in .env file"):
        search_config_module.get_search_config()


def test_get_search_config_raises_when_max_results_invalid(monkeypatch):
    monkeypatch.setenv("SEARCH_PROVIDER", "tavily")
    monkeypatch.setenv("SEARCH_API_KEY", "fake-key")
    monkeypatch.setenv("SEARCH_MAX_RESULTS", "not-an-int")
    monkeypatch.setenv("SEARCH_DEPTH", "basic")

    with pytest.raises(ValueError, match="SEARCH_MAX_RESULTS must be an int"):
        search_config_module.get_search_config()


def test_get_search_config_raises_when_max_results_not_positive(monkeypatch):
    monkeypatch.setenv("SEARCH_PROVIDER", "tavily")
    monkeypatch.setenv("SEARCH_API_KEY", "fake-key")
    monkeypatch.setenv("SEARCH_MAX_RESULTS", "0")
    monkeypatch.setenv("SEARCH_DEPTH", "basic")

    with pytest.raises(
        ValueError, match="SEARCH_MAX_RESULTS must be a positive integer"
    ):
        search_config_module.get_search_config()


def test_get_llm_provider_returns_groq_provider(monkeypatch):
    monkeypatch.setattr(llm_factory_module, "get_llm_config", fake_llm_config_groq)
    monkeypatch.setattr(llm_factory_module, "GroqLLMProvider", FakeGroqLLMProvider)

    result = llm_factory_module.get_llm_provider()

    assert isinstance(result, FakeGroqLLMProvider)


def test_get_llm_provider_raises_for_unsupported_provider(monkeypatch):
    monkeypatch.setattr(
        llm_factory_module,
        "get_llm_config",
        fake_llm_config_bad_provider,
    )

    with pytest.raises(ValueError, match="Unsupported LLM provider"):
        llm_factory_module.get_llm_provider()


def test_get_search_provider_returns_tavily_provider(monkeypatch):
    monkeypatch.setattr(
        search_factory_module,
        "get_search_config",
        fake_search_config_tavily,
    )
    monkeypatch.setattr(
        search_factory_module,
        "TavilySearchClient",
        FakeTavilySearchClient,
    )

    result = search_factory_module.get_search_provider()

    assert isinstance(result, FakeTavilySearchClient)


def test_get_search_provider_raises_for_unsupported_provider(monkeypatch):
    monkeypatch.setattr(
        search_factory_module,
        "get_search_config",
        fake_search_config_bad_provider,
    )

    with pytest.raises(ValueError, match="Unsupported search provider"):
        search_factory_module.get_search_provider()


def test_get_llm_config_raises_when_max_tokens_invalid(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "groq")
    monkeypatch.setenv("LLM_MODEL", "llama-test")
    monkeypatch.setenv("LLM_TEMPERATURE", "0.0")
    monkeypatch.setenv("LLM_MAX_RETRIES", "2")
    monkeypatch.setenv("LLM_API_KEY", "fake-key")
    monkeypatch.setenv("LLM_MAX_TOKENS", "not-an-int")

    with pytest.raises(ValueError, match="LLM_MAX_TOKENS must be an int"):
        llm_config_module.get_llm_config()


def fake_ollama_llm_config():
    return {
        "provider": "ollama",
        "model": "gpt-oss:20b-cloud",
        "temperature": 0.0,
        "max_retries": 2,
        "api_key": "fake-ollama-key",
        "max_tokens": 2048,
    }


def test_get_llm_provider_returns_ollama_provider(monkeypatch):
    import focused_research_agent.services.llm_provider_ollama as ollama_module

    class FakeOllamaClient:
        def __init__(self, **kwargs):
            pass

    monkeypatch.setattr(llm_factory_module, "get_llm_config", fake_ollama_llm_config)
    monkeypatch.setattr(ollama_module, "Client", FakeOllamaClient)
    monkeypatch.setattr(ollama_module, "get_llm_config", fake_ollama_llm_config)

    from focused_research_agent.services.llm_provider_ollama import OllamaLLMProvider

    result = llm_factory_module.get_llm_provider()

    assert isinstance(result, OllamaLLMProvider)
