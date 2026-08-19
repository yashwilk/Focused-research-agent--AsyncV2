import importlib
from focused_research_agent.state import ResearchState


def make_initial_state(question: str) -> ResearchState:
    return {
        "run_id": "",
        "question": question,
        "scope": None,
        "assumptions": None,
        "constraints": None,
        "queries": None,
        "sources": None,
        "answer": None,
        "citations": None,
        "status": "started",
        "errors": [],
        "debug": None,
        "conversation_id": None,
        "conversation_history": None,
        "mode": "research",
        "images": None,
    }


class FakeLLMProvider:
    def generate_json(self, prompt: str) -> dict:
        return {
            "scope": "Should never be used in empty-question test",
            "assumptions": ["placeholder", "placeholder"],
            "constraints": {},
        }


class FakeSearchProvider:
    def search(self, queries: list[str]) -> tuple[list[dict], list[str]]:
        return ([], [])


def fake_get_llm_provider():
    return FakeLLMProvider()


def fake_get_search_provider(search_depth: str | None = None):
    return FakeSearchProvider()


def test_graph_empty_question_routes_to_handle_error(monkeypatch):
    import focused_research_agent.services.llm_factory as llm_factory
    import focused_research_agent.services.search_factory as search_factory_module
    import focused_research_agent.graph as graph_module

    monkeypatch.setattr(llm_factory, "get_llm_provider", fake_get_llm_provider)
    monkeypatch.setattr(
        search_factory_module, "get_search_provider", fake_get_search_provider
    )

    graph_module = importlib.reload(graph_module)

    initial_state = make_initial_state("")
    graph = graph_module.build_graph()
    final_state = graph.invoke(initial_state)

    assert final_state["run_id"]
    assert final_state["status"] == "error"
    assert final_state["errors"] == ["init_run: No question provided"]
