"""
Tests for the FastAPI chat and conversations endpoints.

What is tested:
- POST /api/v1/chat returns structured success response
- POST /api/v1/chat returns error response shape for graph errors
- POST /api/v1/chat rejects invalid questions with 422
- POST /api/v1/chat returns 400 for ApplicationError
- POST /api/v1/chat returns 500 for unexpected exceptions
- POST /api/v1/chat accepts optional conversation_id
- GET /api/v1/conversations returns list of conversations
- GET /api/v1/conversations/{id} returns turns for a conversation

How it is tested:
- FastAPI dependency overrides replace get_chat_use_case with fakes
- get_db is overridden with an in-memory SQLite session
- Repository functions are called directly to seed test data
- The same TestClient pattern used in test_api_research.py is used here

Why it matters:
- Verifies the chat transport layer handles all response paths correctly
- Confirms conversation_id and turn_number are present in responses
- Confirms the conversations endpoints return correctly shaped data
"""

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, Session

from focused_research_agent.api.app import create_app
from focused_research_agent.api.dependencies import get_chat_use_case
from focused_research_agent.application.exceptions import ApplicationError
from focused_research_agent.database.database import get_db
from focused_research_agent.database.models import Base
from focused_research_agent.database.repository import save_run

app = create_app()
client = TestClient(app)


# ---------------------------------------------------------------------------
# In-memory database fixture for conversations endpoint tests
# ---------------------------------------------------------------------------


@pytest.fixture
def db_session() -> Session:
    engine = create_engine(
        "sqlite:///file::memory:?cache=shared&uri=true",
        connect_args={"check_same_thread": False},
    )
    Base.metadata.create_all(bind=engine)
    TestingSessionLocal = sessionmaker(
        autocommit=False,
        autoflush=False,
        bind=engine,
    )
    session = TestingSessionLocal()
    try:
        yield session
    finally:
        session.close()
        # Clean up all rows after each test to prevent contamination
        Base.metadata.drop_all(bind=engine)
        Base.metadata.create_all(bind=engine)


@pytest.fixture
def sample_state() -> dict:
    """
    Return a realistic completed research state for seeding the database.

    Returns:
        dict: A normalized research result dict.
    """
    return {
        "run_id": "run-test-123",
        "question": "What is quantum computing?",
        "status": "completed",
        "scope": "Explain quantum computing clearly",
        "queries": ["quantum computing overview"],
        "sources": [
            {
                "title": "Quantum Overview",
                "url": "https://example.com/quantum",
                "snippet": "Quantum computing uses quantum mechanics.",
                "source": "tavily",
                "score": 0.95,
            }
        ],
        "answer": "Quantum computing uses quantum mechanical phenomena.",
        "citations": ["https://example.com/quantum"],
        "errors": [],
    }


# ---------------------------------------------------------------------------
# Fake use case functions for chat endpoint tests
# ---------------------------------------------------------------------------


def fake_success_chat_turn(
    db: Session,
    conversation_id: str | None,
    question: str,
) -> dict:
    """Return a successful mock chat response."""
    return {
        "run_id": "run-chat-123",
        "question": question.strip(),
        "status": "completed",
        "scope": "Explain the topic clearly",
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
        "answer": "This is a synthesized answer.",
        "citations": ["https://example.com/source"],
        "errors": [],
        "conversation_id": conversation_id or "conv-new-123",
        "turn_number": 1,
        "images": None,
    }


def fake_error_chat_turn(
    db: Session,
    conversation_id: str | None,
    question: str,
) -> dict:
    """Return an error-shaped mock chat response."""
    return {
        "run_id": "run-chat-error",
        "question": question.strip(),
        "status": "error",
        "scope": None,
        "queries": None,
        "sources": None,
        "answer": None,
        "citations": None,
        "errors": ["search_web: Tavily request failed"],
        "conversation_id": conversation_id or "conv-error-123",
        "turn_number": 1,
        "images": None,
    }


# ---------------------------------------------------------------------------
# POST /api/v1/chat tests
# ---------------------------------------------------------------------------


def test_chat_returns_structured_success_response():
    """
    Verify that the chat route returns the expected structured success
    response when the use case provides a successful result.
    """
    app.dependency_overrides[get_chat_use_case] = lambda: fake_success_chat_turn

    try:
        response = client.post(
            "/api/v1/chat",
            json={"question": "Tell me about AI"},
        )

        assert response.status_code == 200

        data = response.json()

        assert data["run_id"] == "run-chat-123"
        assert data["question"] == "Tell me about AI"
        assert data["status"] == "completed"
        assert data["conversation_id"] == "conv-new-123"
        assert data["turn_number"] == 1
        assert data["answer"] == "This is a synthesized answer."
        assert data["errors"] == []
    finally:
        app.dependency_overrides.clear()


def test_chat_returns_error_response_shape():
    """
    Verify that the chat route returns the expected error-shaped
    response when the use case provides a graph-level error result.
    """
    app.dependency_overrides[get_chat_use_case] = lambda: fake_error_chat_turn

    try:
        response = client.post(
            "/api/v1/chat",
            json={"question": "Trigger graph error"},
        )

        assert response.status_code == 200

        data = response.json()

        assert data["status"] == "error"
        assert data["answer"] is None
        assert data["errors"] == ["search_web: Tavily request failed"]
        assert "conversation_id" in data
        assert "turn_number" in data
    finally:
        app.dependency_overrides.clear()


def test_chat_accepts_conversation_id_in_request():
    """
    Verify that the chat route accepts and passes through a
    conversation_id from the request body.
    """
    app.dependency_overrides[get_chat_use_case] = lambda: fake_success_chat_turn

    try:
        response = client.post(
            "/api/v1/chat",
            json={
                "question": "Tell me about AI",
                "conversation_id": "conv-existing-abc",
            },
        )

        assert response.status_code == 200
        assert response.json()["conversation_id"] == "conv-existing-abc"
    finally:
        app.dependency_overrides.clear()


def test_chat_rejects_empty_question():
    """Verify that the chat route rejects an empty question."""
    response = client.post("/api/v1/chat", json={"question": ""})

    assert response.status_code == 422


def test_chat_rejects_whitespace_only_question():
    """Verify that the chat route rejects a whitespace-only question."""
    response = client.post("/api/v1/chat", json={"question": "   "})

    assert response.status_code == 422


def test_chat_rejects_missing_question():
    """Verify that the chat route rejects a request with no question."""
    response = client.post("/api/v1/chat", json={})

    assert response.status_code == 422


def test_chat_rejects_punctuation_only_question():
    """Verify that the chat route rejects punctuation-only input."""
    response = client.post("/api/v1/chat", json={"question": "..."})

    assert response.status_code == 422


def test_chat_returns_structured_400_for_application_error():
    """
    Verify that the chat route returns the centralized 400 error JSON
    shape when the use case raises ApplicationError.
    """

    def fake_application_error_use_case(
        db: Session,
        conversation_id: str | None,
        question: str,
    ) -> dict:
        raise ApplicationError("User query must not be empty")

    app.dependency_overrides[get_chat_use_case] = lambda: (
        fake_application_error_use_case
    )

    try:
        response = client.post(
            "/api/v1/chat",
            json={"question": "Valid looking question"},
        )

        assert response.status_code == 400
        assert response.json() == {
            "status_code": 400,
            "error": "application_error",
            "detail": "User query must not be empty",
            "path": "/api/v1/chat",
        }
    finally:
        app.dependency_overrides.clear()


def test_chat_returns_structured_500_for_unexpected_exception():
    """
    Verify that the chat route returns the centralized 500 error JSON
    shape when the use case raises an unexpected exception.
    """

    def fake_unexpected_error_use_case(
        db: Session,
        conversation_id: str | None,
        question: str,
    ) -> dict:
        raise RuntimeError("Unexpected test failure")

    app.dependency_overrides[get_chat_use_case] = lambda: fake_unexpected_error_use_case

    local_client = TestClient(app, raise_server_exceptions=False)

    try:
        response = local_client.post(
            "/api/v1/chat",
            json={"question": "Valid looking question"},
        )

        assert response.status_code == 500
        assert response.json() == {
            "status_code": 500,
            "error": "internal_server_error",
            "detail": "An unexpected internal error occurred",
            "path": "/api/v1/chat",
        }
    finally:
        app.dependency_overrides.clear()


# ---------------------------------------------------------------------------
# GET /api/v1/conversations tests
# ---------------------------------------------------------------------------


def test_get_conversations_returns_empty_list_when_no_data(db_session):
    """
    Verify that GET /conversations returns an empty list when no
    conversations exist in the database.
    """

    def override_get_db():
        yield db_session

    app.dependency_overrides[get_db] = override_get_db

    try:
        response = client.get("/api/v1/conversations")

        assert response.status_code == 200
        assert response.json() == []
    finally:
        app.dependency_overrides.clear()


def test_get_conversations_returns_seeded_conversations(db_session, sample_state):
    """
    Verify that GET /conversations returns conversations that were
    saved to the database.
    """
    save_run(db_session, sample_state, conversation_id="conv-abc", turn_number=1)

    def override_get_db():
        yield db_session

    app.dependency_overrides[get_db] = override_get_db

    try:
        response = client.get("/api/v1/conversations")

        assert response.status_code == 200

        data = response.json()

        assert len(data) == 1
        assert data[0]["conversation_id"] == "conv-abc"
        assert data[0]["title"] == "What is quantum computing?"
    finally:
        app.dependency_overrides.clear()


# ---------------------------------------------------------------------------
# GET /api/v1/conversations/{conversation_id} tests
# ---------------------------------------------------------------------------


def test_get_conversation_returns_empty_list_for_unknown_id(db_session):
    """
    Verify that GET /conversations/{id} returns an empty list when the
    conversation does not exist.
    """

    def override_get_db():
        yield db_session

    app.dependency_overrides[get_db] = override_get_db

    try:
        response = client.get("/api/v1/conversations/conv-unknown")

        assert response.status_code == 200
        assert response.json() == []
    finally:
        app.dependency_overrides.clear()


def test_get_conversation_returns_turns_for_existing_conversation(
    db_session, sample_state
):
    """
    Verify that GET /conversations/{id} returns all turns for an
    existing conversation with deserialized list fields.
    """
    save_run(db_session, sample_state, conversation_id="conv-abc", turn_number=1)

    def override_get_db():
        yield db_session

    app.dependency_overrides[get_db] = override_get_db

    try:
        response = client.get("/api/v1/conversations/conv-abc")

        assert response.status_code == 200

        data = response.json()

        assert len(data) == 1
        assert data[0]["question"] == "What is quantum computing?"
        assert (
            data[0]["answer"] == "Quantum computing uses quantum mechanical phenomena."
        )
        assert data[0]["turn_number"] == 1
        assert data[0]["queries"] == ["quantum computing overview"]
    finally:
        app.dependency_overrides.clear()


def test_get_reports_returns_empty_list_when_no_data(db_session):
    def override_get_db():
        yield db_session

    app.dependency_overrides[get_db] = override_get_db

    try:
        response = client.get("/api/v1/reports")
        assert response.status_code == 200
        assert response.json() == []
    finally:
        app.dependency_overrides.clear()


def test_get_reports_returns_seeded_reports(db_session, sample_state):
    from focused_research_agent.database.repository import save_run as repo_save_run

    repo_save_run(
        db_session,
        sample_state,
        conversation_id="conv-report",
        turn_number=1,
        mode="report",
    )

    def override_get_db():
        yield db_session

    app.dependency_overrides[get_db] = override_get_db

    try:
        response = client.get("/api/v1/reports")
        assert response.status_code == 200
        data = response.json()
        assert len(data) == 1
        assert data[0]["conversation_id"] == "conv-report"
    finally:
        app.dependency_overrides.clear()
