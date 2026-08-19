"""
Tests for the FastAPI report generation endpoint.

What is tested:
- POST /api/v1/report returns structured success response
- POST /api/v1/report returns error response shape for graph errors
- POST /api/v1/report rejects invalid questions with 422
- POST /api/v1/report returns 400 for ApplicationError
- POST /api/v1/report returns 500 for unexpected exceptions
- POST /api/v1/report answer contains structured report sections

How it is tested:
- FastAPI dependency overrides replace get_report_use_case with fakes
- The same TestClient pattern used in test_api_research.py is used here
- No database session override needed — report use case handles db internally

Why it matters:
- Verifies the report transport layer handles all response paths correctly
- Confirms the structured markdown answer passes through the API correctly
"""

from fastapi.testclient import TestClient

from focused_research_agent.api.app import create_app
from focused_research_agent.api.dependencies import get_report_use_case
from focused_research_agent.application.exceptions import ApplicationError

app = create_app()
client = TestClient(app)


# ---------------------------------------------------------------------------
# Fake use case functions
# ---------------------------------------------------------------------------


def fake_success_report(question: str, db) -> dict:
    """Return a successful mock report response."""
    return {
        "run_id": "run-report-123",
        "question": question.strip(),
        "status": "completed",
        "scope": "Provide a comprehensive report on the topic",
        "queries": ["query one", "query two", "query three"],
        "sources": [
            {
                "title": "Test Source",
                "url": "https://example.com/source",
                "snippet": "A test source snippet.",
                "source": "mock",
                "score": 0.95,
            }
        ],
        "answer": (
            "## Introduction\nThis is the introduction.\n"
            "## Key Findings\nThese are the key findings.\n"
            "## Analysis\nThis is the analysis.\n"
            "## Conclusion\nThis is the conclusion."
        ),
        "citations": [
            "https://example.com/source",
            "https://example.com/source2",
        ],
        "errors": [],
        "images": None,
    }


def fake_error_report(question: str, db) -> dict:
    """Return an error-shaped mock report response."""
    return {
        "run_id": "run-report-error",
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


# ---------------------------------------------------------------------------
# POST /api/v1/report success tests
# ---------------------------------------------------------------------------


def test_report_returns_structured_success_response():
    """
    Verify that the report route returns the expected structured
    success response when the use case provides a successful result.
    """
    app.dependency_overrides[get_report_use_case] = lambda: fake_success_report

    try:
        response = client.post(
            "/api/v1/report",
            json={"question": "Tell me about quantum computing"},
        )

        assert response.status_code == 200

        data = response.json()

        assert data["run_id"] == "run-report-123"
        assert data["question"] == "Tell me about quantum computing"
        assert data["status"] == "completed"
        assert data["answer"] is not None
        assert data["errors"] == []
    finally:
        app.dependency_overrides.clear()


def test_report_answer_contains_structured_sections():
    """
    Verify that the report answer contains all four required
    structured section headers.
    """
    app.dependency_overrides[get_report_use_case] = lambda: fake_success_report

    try:
        response = client.post(
            "/api/v1/report",
            json={"question": "Tell me about quantum computing"},
        )

        assert response.status_code == 200

        answer = response.json()["answer"]

        assert "## Introduction" in answer
        assert "## Key Findings" in answer
        assert "## Analysis" in answer
        assert "## Conclusion" in answer
    finally:
        app.dependency_overrides.clear()


def test_report_returns_error_response_shape():
    """
    Verify that the report route returns the expected error-shaped
    response when the use case provides a graph-level error result.
    """
    app.dependency_overrides[get_report_use_case] = lambda: fake_error_report

    try:
        response = client.post(
            "/api/v1/report",
            json={"question": "Trigger graph error"},
        )

        assert response.status_code == 200

        data = response.json()

        assert data["status"] == "error"
        assert data["answer"] is None
        assert data["errors"] == ["search_web: Tavily request failed"]
    finally:
        app.dependency_overrides.clear()


# ---------------------------------------------------------------------------
# POST /api/v1/report validation tests
# ---------------------------------------------------------------------------


def test_report_rejects_empty_question():
    """Verify that the report route rejects an empty question."""
    response = client.post("/api/v1/report", json={"question": ""})

    assert response.status_code == 422


def test_report_rejects_whitespace_only_question():
    """Verify that the report route rejects a whitespace-only question."""
    response = client.post("/api/v1/report", json={"question": "   "})

    assert response.status_code == 422


def test_report_rejects_missing_question():
    """Verify that the report route rejects a request with no question."""
    response = client.post("/api/v1/report", json={})

    assert response.status_code == 422


def test_report_rejects_punctuation_only_question():
    """Verify that the report route rejects punctuation-only input."""
    response = client.post("/api/v1/report", json={"question": "..."})

    assert response.status_code == 422


# ---------------------------------------------------------------------------
# POST /api/v1/report error handling tests
# ---------------------------------------------------------------------------


def test_report_returns_structured_400_for_application_error():
    """
    Verify that the report route returns the centralized 400 error
    JSON shape when the use case raises ApplicationError.
    """

    def fake_application_error_use_case(question: str, db) -> dict:
        raise ApplicationError("User query must not be empty")

    app.dependency_overrides[get_report_use_case] = lambda: (
        fake_application_error_use_case
    )

    try:
        response = client.post(
            "/api/v1/report",
            json={"question": "Valid looking question"},
        )

        assert response.status_code == 400
        assert response.json() == {
            "status_code": 400,
            "error": "application_error",
            "detail": "User query must not be empty",
            "path": "/api/v1/report",
        }
    finally:
        app.dependency_overrides.clear()


def test_report_returns_structured_500_for_unexpected_exception():
    """
    Verify that the report route returns the centralized 500 error
    JSON shape when the use case raises an unexpected exception.
    """

    def fake_unexpected_error_use_case(question: str, db) -> dict:
        raise RuntimeError("Unexpected test failure")

    app.dependency_overrides[get_report_use_case] = lambda: (
        fake_unexpected_error_use_case
    )

    local_client = TestClient(app, raise_server_exceptions=False)

    try:
        response = local_client.post(
            "/api/v1/report",
            json={"question": "Valid looking question"},
        )

        assert response.status_code == 500
        assert response.json() == {
            "status_code": 500,
            "error": "internal_server_error",
            "detail": "An unexpected internal error occurred",
            "path": "/api/v1/report",
        }
    finally:
        app.dependency_overrides.clear()
