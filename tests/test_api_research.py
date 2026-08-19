"""
Tests for the FastAPI versioned /research endpoint.

These tests verify request validation, success responses, and error-shaped
responses for the research API route. They override the FastAPI dependency
used by the route so the tests can focus on API behavior without invoking
the real research workflow.
"""

from fastapi.testclient import TestClient

from focused_research_agent.api.app import create_app
from focused_research_agent.api.dependencies import get_research_use_case
from focused_research_agent.application.exceptions import ApplicationError

app = create_app()
client = TestClient(app)


def fake_success_research_question(question: str) -> dict:
    """
    Return a successful mock research response.

    Args:
        question: User research question provided to the endpoint.

    Returns:
        dict: Mocked successful research result.
    """
    return {
        "run_id": "run-123",
        "question": question.strip(),
        "status": "completed",
        "scope": "Explain the topic clearly",
        "queries": [
            "ai agents overview",
            "latest ai agent frameworks",
            "ai agent use cases",
        ],
        "sources": [
            {
                "title": "AI Agents Overview",
                "url": "https://example.com/overview",
                "snippet": "A high-level overview of AI agents.",
                "source": "mock",
                "score": 0.95,
            },
            {
                "title": "AI Agent Frameworks",
                "url": "https://example.com/frameworks",
                "snippet": "A summary of current AI agent frameworks.",
                "source": "mock",
                "score": 0.91,
            },
        ],
        "answer": "AI agents are systems that can plan and act toward goals.",
        "citations": [
            "https://example.com/overview",
            "https://example.com/frameworks",
        ],
        "errors": [],
        "images": None,
    }


def fake_error_research_question(question: str) -> dict:
    """
    Return an error-shaped mock research response.

    Args:
        question: User research question provided to the endpoint.

    Returns:
        dict: Mocked error research result.
    """
    return {
        "run_id": "run-999",
        "question": question.strip(),
        "status": "error",
        "scope": None,
        "queries": None,
        "sources": None,
        "answer": None,
        "citations": None,
        "errors": ["search_web: Tavily request failed"],
        "images": None,
    }


def test_research_returns_structured_success_response():
    """
    Verify that the versioned research route returns the expected structured
    success response when the dependency provides a successful research
    result.
    """
    app.dependency_overrides[get_research_use_case] = lambda: (
        fake_success_research_question
    )

    try:
        response = client.post(
            "/api/v1/research",
            json={"question": "Tell me about AI agents"},
        )

        assert response.status_code == 200

        data = response.json()

        assert data["run_id"] == "run-123"
        assert data["question"] == "Tell me about AI agents"
        assert data["status"] == "completed"
        assert data["scope"] == "Explain the topic clearly"
        assert data["queries"] == [
            "ai agents overview",
            "latest ai agent frameworks",
            "ai agent use cases",
        ]
        assert len(data["sources"]) == 2
        assert data["sources"][0]["title"] == "AI Agents Overview"
        assert data["sources"][0]["url"] == "https://example.com/overview"
        assert (
            data["answer"]
            == "AI agents are systems that can plan and act toward goals."
        )
        assert data["citations"] == [
            "https://example.com/overview",
            "https://example.com/frameworks",
        ]
        assert data["errors"] == []
    finally:
        app.dependency_overrides.clear()


def test_research_returns_error_response_shape():
    """
    Verify that the versioned research route returns the expected error-shaped
    response when the dependency provides a graph-style error result.
    """
    app.dependency_overrides[get_research_use_case] = lambda: (
        fake_error_research_question
    )

    try:
        response = client.post(
            "/api/v1/research",
            json={"question": "Trigger graph error"},
        )

        assert response.status_code == 200

        data = response.json()

        assert data["run_id"] == "run-999"
        assert data["question"] == "Trigger graph error"
        assert data["status"] == "error"
        assert data["scope"] is None
        assert data["queries"] is None
        assert data["sources"] is None
        assert data["answer"] is None
        assert data["citations"] is None
        assert data["errors"] == ["search_web: Tavily request failed"]
    finally:
        app.dependency_overrides.clear()


def test_research_rejects_empty_question():
    """
    Verify that the versioned route rejects an empty question at the API
    validation layer.
    """
    response = client.post("/api/v1/research", json={"question": ""})

    assert response.status_code == 422


def test_research_rejects_whitespace_only_question():
    """
    Verify that the versioned route rejects a whitespace-only question at the
    API validation layer.
    """
    response = client.post("/api/v1/research", json={"question": "   "})

    assert response.status_code == 422


def test_research_rejects_missing_question():
    """
    Verify that the versioned route rejects a request body with no question
    field.
    """
    response = client.post("/api/v1/research", json={})

    assert response.status_code == 422


def test_research_rejects_wrong_question_type():
    """
    Verify that the versioned route rejects a request where question has the
    wrong type.
    """
    response = client.post("/api/v1/research", json={"question": 123})

    assert response.status_code == 422


def test_research_rejects_punctuation_only_question():
    """
    Verify that the versioned route rejects punctuation-only input at the API
    validation layer.
    """
    response = client.post("/api/v1/research", json={"question": "."})

    assert response.status_code == 422


def test_research_rejects_ultra_short_question():
    """
    Verify that the versioned route rejects meaningless ultra-short input at
    the API validation layer.
    """
    response = client.post("/api/v1/research", json={"question": "a"})

    assert response.status_code == 422


def test_research_returns_structured_400_for_application_error():
    """
    Verify that the versioned research route returns the centralized 400
    error JSON shape when the injected use case raises ApplicationError.
    """

    def fake_application_error_use_case(question: str) -> dict:
        raise ApplicationError("User query must not be empty")

    app.dependency_overrides[get_research_use_case] = lambda: (
        fake_application_error_use_case
    )

    try:
        response = client.post(
            "/api/v1/research",
            json={"question": "Valid looking question"},
        )

        assert response.status_code == 400
        assert response.json() == {
            "status_code": 400,
            "error": "application_error",
            "detail": "User query must not be empty",
            "path": "/api/v1/research",
        }
    finally:
        app.dependency_overrides.clear()


def test_research_returns_structured_500_for_unexpected_exception():
    """
    Verify that the versioned research route returns the centralized 500
    error JSON shape when the injected use case raises an unexpected
    exception.
    """

    def fake_unexpected_error_use_case(question: str) -> dict:
        raise RuntimeError("Unexpected test failure")

    app.dependency_overrides[get_research_use_case] = lambda: (
        fake_unexpected_error_use_case
    )

    local_client = TestClient(app, raise_server_exceptions=False)

    try:
        response = local_client.post(
            "/api/v1/research",
            json={"question": "Valid looking question"},
        )

        assert response.status_code == 500
        assert response.json() == {
            "status_code": 500,
            "error": "internal_server_error",
            "detail": "An unexpected internal error occurred",
            "path": "/api/v1/research",
        }
    finally:
        app.dependency_overrides.clear()
