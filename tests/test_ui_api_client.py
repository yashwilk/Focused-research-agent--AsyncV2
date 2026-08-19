"""
Tests for the Streamlit UI HTTP client.

What is tested:
- check_health returns True on 200 and False on ConnectError
- call_research returns correct ResearchCallResult shape for all
  response paths: 200, 400, 422, unexpected status, timeout
- call_research raises BackendUnavailableError on ConnectError

How it is tested:
- httpx is replaced with a fake object inside api_client using
  monkeypatch, following the same pattern used throughout the
  test suite for LLM and search providers
- FakeHttpx always carries ConnectError and TimeoutException as
  class attributes pointing to the real httpx exception classes,
  so api_client.py can resolve them during except clause evaluation

Why it matters:
- verifies the HTTP client handles all response paths correctly
  without making real network calls
"""

import httpx
import pytest

import focused_research_agent.ui.api_client as api_client_module
from focused_research_agent.ui.exceptions import BackendUnavailableError
from focused_research_agent.ui.api_client import (
    call_research,
    call_report,
    check_health,
)


class FakeResponse:
    """Simulates an httpx response object for testing."""

    def __init__(self, status_code: int, json_data: dict) -> None:
        self.status_code = status_code
        self._json_data = json_data

    def json(self) -> dict:
        return self._json_data


def test_check_health_returns_true_when_backend_responds_200(monkeypatch):
    class FakeHttpx:
        ConnectError = httpx.ConnectError
        TimeoutException = httpx.TimeoutException

        def get(self, url, timeout):
            return FakeResponse(status_code=200, json_data={})

    monkeypatch.setattr(api_client_module, "httpx", FakeHttpx())

    assert check_health() is True


def test_check_health_returns_false_on_connect_error(monkeypatch):
    class FakeHttpx:
        ConnectError = httpx.ConnectError
        TimeoutException = httpx.TimeoutException

        def get(self, url, timeout):
            raise httpx.ConnectError("connection refused")

    monkeypatch.setattr(api_client_module, "httpx", FakeHttpx())

    assert check_health() is False


def test_call_research_returns_success_result_on_200(monkeypatch):
    fake_data = {
        "run_id": "run-abc",
        "question": "What is AI?",
        "status": "completed",
        "scope": "Explain AI clearly",
        "queries": ["what is AI", "AI overview", "AI use cases"],
        "sources": [
            {
                "title": "AI Overview",
                "url": "https://example.com/ai",
                "snippet": "AI is a broad field.",
                "source": "tavily",
                "score": 0.95,
            }
        ],
        "answer": "AI is the simulation of human intelligence by machines.",
        "citations": ["https://example.com/ai"],
        "errors": [],
    }

    class FakeHttpx:
        ConnectError = httpx.ConnectError
        TimeoutException = httpx.TimeoutException

        def post(self, url, json, timeout):
            return FakeResponse(status_code=200, json_data=fake_data)

    monkeypatch.setattr(api_client_module, "httpx", FakeHttpx())

    result = call_research("What is AI?")

    assert result["success"] is True
    assert result["error"] is None
    assert result["data"]["run_id"] == "run-abc"
    assert (
        result["data"]["answer"]
        == "AI is the simulation of human intelligence by machines."
    )


def test_call_research_returns_error_on_400(monkeypatch):
    class FakeHttpx:
        ConnectError = httpx.ConnectError
        TimeoutException = httpx.TimeoutException

        def post(self, url, json, timeout):
            return FakeResponse(
                status_code=400,
                json_data={
                    "status_code": 400,
                    "error": "application_error",
                    "detail": "No user query provided",
                    "path": "/api/v1/research",
                },
            )

    monkeypatch.setattr(api_client_module, "httpx", FakeHttpx())

    result = call_research("What is AI?")

    assert result["success"] is False
    assert result["data"] is None
    assert result["error"] == "No user query provided"


def test_call_research_returns_error_on_422(monkeypatch):
    class FakeHttpx:
        ConnectError = httpx.ConnectError
        TimeoutException = httpx.TimeoutException

        def post(self, url, json, timeout):
            return FakeResponse(status_code=422, json_data={})

    monkeypatch.setattr(api_client_module, "httpx", FakeHttpx())

    result = call_research(".")

    assert result["success"] is False
    assert result["data"] is None
    assert result["error"] == "Invalid question submitted."


def test_call_research_returns_error_on_unexpected_status(monkeypatch):
    class FakeHttpx:
        ConnectError = httpx.ConnectError
        TimeoutException = httpx.TimeoutException

        def post(self, url, json, timeout):
            return FakeResponse(status_code=503, json_data={})

    monkeypatch.setattr(api_client_module, "httpx", FakeHttpx())

    result = call_research("What is AI?")

    assert result["success"] is False
    assert result["data"] is None
    assert result["error"] == "Unexpected error: 503"


def test_call_research_raises_backend_unavailable_on_connect_error(monkeypatch):
    class FakeHttpx:
        ConnectError = httpx.ConnectError
        TimeoutException = httpx.TimeoutException

        def post(self, url, json, timeout):
            raise httpx.ConnectError("connection refused")

    monkeypatch.setattr(api_client_module, "httpx", FakeHttpx())

    with pytest.raises(BackendUnavailableError):
        call_research("What is AI?")


def test_call_research_returns_error_on_timeout(monkeypatch):
    class FakeHttpx:
        ConnectError = httpx.ConnectError
        TimeoutException = httpx.TimeoutException

        def post(self, url, json, timeout):
            raise httpx.TimeoutException("timed out")

    monkeypatch.setattr(api_client_module, "httpx", FakeHttpx())

    result = call_research("What is AI?")

    assert result["success"] is False
    assert result["data"] is None
    assert result["error"] == "Request timed out — research is taking too long."


def test_call_report_returns_success_result_on_200(monkeypatch):
    class FakeHttpx:
        ConnectError = httpx.ConnectError
        TimeoutException = httpx.TimeoutException

        def post(self, url, json, timeout):
            return FakeResponse(
                200, {"run_id": "abc", "answer": "## Introduction\nTest"}
            )

    monkeypatch.setattr(api_client_module, "httpx", FakeHttpx())
    result = call_report("What is AI?")
    assert result["success"] is True
    assert result["data"]["run_id"] == "abc"


def test_call_report_returns_error_on_400(monkeypatch):
    class FakeHttpx:
        ConnectError = httpx.ConnectError
        TimeoutException = httpx.TimeoutException

        def post(self, url, json, timeout):
            return FakeResponse(400, {"detail": "Bad question"})

    monkeypatch.setattr(api_client_module, "httpx", FakeHttpx())
    result = call_report("What is AI?")
    assert result["success"] is False
    assert result["error"] == "Bad question"


def test_call_report_raises_backend_unavailable_on_connect_error(monkeypatch):
    class FakeHttpx:
        ConnectError = httpx.ConnectError
        TimeoutException = httpx.TimeoutException

        def post(self, url, json, timeout):
            raise httpx.ConnectError("refused")

    monkeypatch.setattr(api_client_module, "httpx", FakeHttpx())
    with pytest.raises(BackendUnavailableError):
        call_report("What is AI?")


def test_call_report_returns_error_on_timeout(monkeypatch):
    class FakeHttpx:
        ConnectError = httpx.ConnectError
        TimeoutException = httpx.TimeoutException

        def post(self, url, json, timeout):
            raise httpx.TimeoutException("timeout")

    monkeypatch.setattr(api_client_module, "httpx", FakeHttpx())
    result = call_report("What is AI?")
    assert result["success"] is False
    assert "timed out" in result["error"]


def test_call_chat_returns_success_result_on_200(monkeypatch):
    class FakeHttpx:
        ConnectError = httpx.ConnectError
        TimeoutException = httpx.TimeoutException

        def post(self, url, json, timeout):
            return FakeResponse(
                200,
                {"run_id": "chat-abc", "conversation_id": "conv-1", "turn_number": 1},
            )

    monkeypatch.setattr(api_client_module, "httpx", FakeHttpx())
    from focused_research_agent.ui.api_client import call_chat

    result = call_chat("What is AI?", None)
    assert result["success"] is True
    assert result["data"]["run_id"] == "chat-abc"


def test_call_chat_returns_error_on_400(monkeypatch):
    class FakeHttpx:
        ConnectError = httpx.ConnectError
        TimeoutException = httpx.TimeoutException

        def post(self, url, json, timeout):
            return FakeResponse(400, {"detail": "Bad question"})

    monkeypatch.setattr(api_client_module, "httpx", FakeHttpx())
    from focused_research_agent.ui.api_client import call_chat

    result = call_chat("What is AI?", None)
    assert result["success"] is False
    assert result["error"] == "Bad question"


def test_call_chat_raises_backend_unavailable_on_connect_error(monkeypatch):
    class FakeHttpx:
        ConnectError = httpx.ConnectError
        TimeoutException = httpx.TimeoutException

        def post(self, url, json, timeout):
            raise httpx.ConnectError("refused")

    monkeypatch.setattr(api_client_module, "httpx", FakeHttpx())
    from focused_research_agent.ui.api_client import call_chat

    with pytest.raises(BackendUnavailableError):
        call_chat("What is AI?", None)


def test_call_chat_returns_error_on_timeout(monkeypatch):
    class FakeHttpx:
        ConnectError = httpx.ConnectError
        TimeoutException = httpx.TimeoutException

        def post(self, url, json, timeout):
            raise httpx.TimeoutException("timeout")

    monkeypatch.setattr(api_client_module, "httpx", FakeHttpx())
    from focused_research_agent.ui.api_client import call_chat

    result = call_chat("What is AI?", None)
    assert result["success"] is False
    assert "timed out" in result["error"]


def test_get_conversations_returns_list_on_200(monkeypatch):
    class FakeHttpx:
        ConnectError = httpx.ConnectError
        TimeoutException = httpx.TimeoutException

        def get(self, url, timeout):
            return FakeResponse(200, [{"conversation_id": "conv-1", "title": "Test"}])

    monkeypatch.setattr(api_client_module, "httpx", FakeHttpx())
    from focused_research_agent.ui.api_client import get_conversations

    result = get_conversations()
    assert len(result) == 1
    assert result[0]["conversation_id"] == "conv-1"


def test_get_conversations_returns_empty_list_on_connect_error(monkeypatch):
    class FakeHttpx:
        ConnectError = httpx.ConnectError
        TimeoutException = httpx.TimeoutException

        def get(self, url, timeout):
            raise httpx.ConnectError("refused")

    monkeypatch.setattr(api_client_module, "httpx", FakeHttpx())
    from focused_research_agent.ui.api_client import get_conversations

    result = get_conversations()
    assert result == []


def test_get_conversation_returns_list_on_200(monkeypatch):
    class FakeHttpx:
        ConnectError = httpx.ConnectError
        TimeoutException = httpx.TimeoutException

        def get(self, url, timeout):
            return FakeResponse(200, [{"turn_number": 1, "question": "What is AI?"}])

    monkeypatch.setattr(api_client_module, "httpx", FakeHttpx())
    from focused_research_agent.ui.api_client import get_conversation

    result = get_conversation("conv-1")
    assert len(result) == 1
    assert result[0]["turn_number"] == 1


def test_get_conversation_returns_empty_list_on_connect_error(monkeypatch):
    class FakeHttpx:
        ConnectError = httpx.ConnectError
        TimeoutException = httpx.TimeoutException

        def get(self, url, timeout):
            raise httpx.ConnectError("refused")

    monkeypatch.setattr(api_client_module, "httpx", FakeHttpx())
    from focused_research_agent.ui.api_client import get_conversation

    result = get_conversation("conv-1")
    assert result == []


def test_get_reports_returns_list_on_200(monkeypatch):
    class FakeHttpx:
        ConnectError = httpx.ConnectError
        TimeoutException = httpx.TimeoutException

        def get(self, url, timeout):
            return FakeResponse(
                200, [{"conversation_id": "rep-1", "title": "Report on AI"}]
            )

    monkeypatch.setattr(api_client_module, "httpx", FakeHttpx())
    from focused_research_agent.ui.api_client import get_reports

    result = get_reports()
    assert len(result) == 1
    assert result[0]["conversation_id"] == "rep-1"


def test_get_reports_returns_empty_list_on_connect_error(monkeypatch):
    class FakeHttpx:
        ConnectError = httpx.ConnectError
        TimeoutException = httpx.TimeoutException

        def get(self, url, timeout):
            raise httpx.ConnectError("refused")

    monkeypatch.setattr(api_client_module, "httpx", FakeHttpx())
    from focused_research_agent.ui.api_client import get_reports

    result = get_reports()
    assert result == []
