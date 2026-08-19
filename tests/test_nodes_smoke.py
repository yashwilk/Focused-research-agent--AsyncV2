import importlib
from focused_research_agent.state import ResearchState

"""
Happy-path smoke test for the full research graph.

What is tested:
- successful end-to-end flow from question to completed answer

How it is tested:
- fake LLM and search providers return deterministic outputs
- factories/providers are patched before graph reload
- the compiled graph is invoked and final state is validated

Why it matters:
- provides quick confidence that the full workflow wiring works
- acts as a fast integration-style test without real API calls
"""


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
        if not isinstance(prompt, str):
            raise ValueError("Prompt must be a string")

        if (
            '"scope"' in prompt
            and '"assumptions"' in prompt
            and '"constraints"' in prompt
        ):
            return {
                "scope": "Explain the test topic clearly",
                "assumptions": ["User is a beginner", "General context"],
                "constraints": {
                    "geography": "Global",
                    "time_range": "current",
                    "depth": "intro",
                },
            }

        if 'Return EXACTLY one key: "queries".' in prompt:
            return {
                "queries": [
                    "test topic overview",
                    "test topic rules",
                    "test topic examples",
                    "test topic pitfalls",
                ]
            }

        if '"answer"' in prompt and '"citations"' in prompt:
            return {
                "answer": (
                    "The test topic can be understood through its overview, "
                    "key rules, and common pitfalls. It is useful to start "
                    "with the basics and then review practical examples."
                ),
                "citations": [
                    "https://example.com/overview",
                    "https://example.com/rules",
                    "https://example.com/pitfalls",
                ],
            }

        raise ValueError("Unexpected prompt received by FakeLLMProvider")


class FakeSearchProvider:
    def search(self, queries: list[str]) -> tuple[list[dict], list[str]]:
        return (
            [
                {
                    "title": "Overview of the test topic",
                    "url": "https://example.com/overview",
                    "snippet": "A high-level overview of the test topic.",
                    "source": "mock",
                    "score": 0.95,
                },
                {
                    "title": "Rules and requirements",
                    "url": "https://example.com/rules",
                    "snippet": "Important rules and requirements for the test topic.",
                    "source": "mock",
                    "score": 0.91,
                },
                {
                    "title": "Common pitfalls and examples",
                    "url": "https://example.com/pitfalls",
                    "snippet": "Examples and common pitfalls for the test topic.",
                    "source": "mock",
                    "score": 0.89,
                },
            ],
            [],  # ← empty images list
        )


def fake_get_llm_provider():
    return FakeLLMProvider()


def fake_get_search_provider(search_depth: str | None = None):
    return FakeSearchProvider()


def test_graph_smoke_run(monkeypatch):
    import focused_research_agent.services.llm_factory as llm_factory
    import focused_research_agent.services.search_factory as search_factory_module  # ← change
    import focused_research_agent.graph as graph_module

    monkeypatch.setattr(llm_factory, "get_llm_provider", fake_get_llm_provider)
    monkeypatch.setattr(
        search_factory_module,
        "get_search_provider",
        fake_get_search_provider,  # ← change
    )

    graph_module = importlib.reload(graph_module)

    initial_state = make_initial_state("test question")
    graph = graph_module.build_graph()
    final_state = graph.invoke(initial_state)

    assert final_state["run_id"]
    assert final_state["scope"]
    assert final_state["queries"]
    assert final_state["sources"]
    assert final_state["answer"]
