"""
Tests for the application-layer report use case.

What is tested:
- execute_report raises ApplicationError for invalid questions
- execute_report sets mode to 'report' in the graph state
- execute_report calls build_graph with search_depth='advanced'
- execute_report returns a normalized result dict
- execute_report persists the run to the database
- execute_report returns result even when save_run fails

How it is tested:
- An in-memory SQLite database is used for integration-style tests
- build_graph is patched with a fake graph that returns deterministic results
- The fake graph captures the initial_state so we can assert mode and search_depth

Why it matters:
- Verifies the report use case correctly configures the graph
- Confirms mode='report' is set before graph invocation
- Confirms advanced search depth is used
- Confirms non-blocking persistence works correctly
"""

import pytest

import focused_research_agent.application.report_use_case as report_use_case_module
from focused_research_agent.application.exceptions import ApplicationError
from focused_research_agent.application.report_use_case import execute_report

# ---------------------------------------------------------------------------
# Fixtures
#
# The `db` fixture (an isolated async in-memory session, fresh per test) is
# defined in tests/conftest.py and shared across the test suite.
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Fake graph for testing
# ---------------------------------------------------------------------------


class FakeGraph:
    """
    Fake LangGraph graph that captures the initial state and returns
    a deterministic completed result without making real LLM or
    search provider calls.
    """

    def __init__(self):
        self.captured_initial_state = None

    async def ainvoke(self, initial_state: dict) -> dict:
        """
        Capture the initial state and return a fixed successful result.

        Args:
            initial_state: Initial graph state passed into the workflow.

        Returns:
            dict: Mocked completed graph result.
        """
        self.captured_initial_state = initial_state
        return {
            "run_id": "run-report-fake-456",
            "question": initial_state["question"],
            "status": "completed",
            "scope": "A detailed report on the topic",
            "assumptions": [],
            "constraints": {},
            "queries": ["query one", "query two", "query three"],
            "sources": [
                {
                    "title": "Test Source",
                    "url": "https://example.com/source",
                    "snippet": "A test source snippet.",
                    "source": "mock",
                    "score": 0.9,
                }
            ],
            "answer": (
                "## Introduction\nThis is the introduction.\n"
                "## Key Findings\nThese are the key findings.\n"
                "## Analysis\nThis is the analysis.\n"
                "## Conclusion\nThis is the conclusion."
            ),
            "citations": ["https://example.com/source"],
            "errors": [],
            "debug": None,
            "conversation_id": None,
            "conversation_history": None,
            "mode": initial_state.get("mode"),
        }


_fake_graph_instance = None


def fake_build_graph(search_depth: str | None = None):
    """
    Return a FakeGraph instance and record the search_depth used.

    Args:
        search_depth: The search depth passed by the use case.

    Returns:
        FakeGraph: A fake graph instance.
    """
    global _fake_graph_instance
    _fake_graph_instance = FakeGraph()
    _fake_graph_instance.search_depth_used = search_depth
    return _fake_graph_instance


# ---------------------------------------------------------------------------
# Validation tests
# ---------------------------------------------------------------------------


async def test_execute_report_raises_for_empty_question(db, monkeypatch):
    monkeypatch.setattr(report_use_case_module, "build_graph", fake_build_graph)

    with pytest.raises(ApplicationError, match="No user query provided"):
        await execute_report(question="   ", db=db)


async def test_execute_report_raises_for_non_string_question(db, monkeypatch):
    monkeypatch.setattr(report_use_case_module, "build_graph", fake_build_graph)

    with pytest.raises(ApplicationError, match="User query must be a string"):
        await execute_report(question=123, db=db)  # type: ignore


async def test_execute_report_raises_for_punctuation_only_question(db, monkeypatch):
    monkeypatch.setattr(report_use_case_module, "build_graph", fake_build_graph)

    with pytest.raises(ApplicationError):
        await execute_report(question="...", db=db)


# ---------------------------------------------------------------------------
# Mode and search depth tests
# ---------------------------------------------------------------------------


async def test_execute_report_sets_mode_to_report(db, monkeypatch):
    """
    Verify that execute_report sets mode='report' in the initial state
    before invoking the graph.
    """
    monkeypatch.setattr(report_use_case_module, "build_graph", fake_build_graph)

    await execute_report(question="What is quantum computing?", db=db)

    assert _fake_graph_instance is not None
    assert _fake_graph_instance.captured_initial_state["mode"] == "report"


async def test_execute_report_uses_advanced_search_depth(db, monkeypatch):
    """
    Verify that execute_report calls build_graph with
    search_depth='advanced'.
    """
    monkeypatch.setattr(report_use_case_module, "build_graph", fake_build_graph)

    await execute_report(question="What is quantum computing?", db=db)

    assert _fake_graph_instance is not None
    assert _fake_graph_instance.search_depth_used == "advanced"


# ---------------------------------------------------------------------------
# Result shape tests
# ---------------------------------------------------------------------------


async def test_execute_report_returns_expected_result_shape(db, monkeypatch):
    monkeypatch.setattr(report_use_case_module, "build_graph", fake_build_graph)

    result = await execute_report(question="What is quantum computing?", db=db)

    assert "run_id" in result
    assert "question" in result
    assert "status" in result
    assert "answer" in result
    assert "citations" in result
    assert "errors" in result


async def test_execute_report_answer_contains_report_sections(db, monkeypatch):
    """
    Verify that the report answer contains the expected structured
    section headers.
    """
    monkeypatch.setattr(report_use_case_module, "build_graph", fake_build_graph)

    result = await execute_report(question="What is quantum computing?", db=db)

    assert "## Introduction" in result["answer"]
    assert "## Key Findings" in result["answer"]
    assert "## Analysis" in result["answer"]
    assert "## Conclusion" in result["answer"]


# ---------------------------------------------------------------------------
# Persistence tests
# ---------------------------------------------------------------------------


async def test_execute_report_persists_run_to_database(db, monkeypatch):
    """
    Verify that execute_report saves the completed report run to
    the database.
    """
    monkeypatch.setattr(report_use_case_module, "build_graph", fake_build_graph)

    await execute_report(question="What is quantum computing?", db=db)

    from sqlalchemy import select

    from focused_research_agent.database.model import ConversationRun

    result = await db.execute(select(ConversationRun))
    count = len(result.scalars().all())
    assert count == 1


async def test_execute_report_returns_result_even_when_save_fails(db, monkeypatch):
    """
    Verify that execute_report returns the result even when database
    persistence fails.
    """
    monkeypatch.setattr(report_use_case_module, "build_graph", fake_build_graph)

    from sqlalchemy.exc import SQLAlchemyError

    def fake_save_run(*args, **kwargs):
        raise SQLAlchemyError("Database write failed")

    monkeypatch.setattr(report_use_case_module, "save_run", fake_save_run)

    result = await execute_report(question="What is quantum computing?", db=db)

    assert result["answer"] is not None
    assert result["status"] == "completed"
