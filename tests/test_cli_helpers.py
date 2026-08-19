import focused_research_agent.cli as cli_module


def test_format_queries_returns_placeholder_when_none():
    result = cli_module.format_queries(None)

    assert result == "(no queries)\n"


def test_format_queries_formats_bullet_list():
    result = cli_module.format_queries(["query one", "query two"])

    assert result == "- query one\n- query two\n"


def test_format_sources_returns_placeholder_when_none():
    result = cli_module.format_sources(None)

    assert result == "(no sources)\n"


def test_format_sources_formats_numbered_list():
    sources = [
        {
            "title": "First Source",
            "url": "https://example.com/one",
        },
        {
            "title": "Second Source",
            "url": "https://example.com/two",
        },
    ]

    result = cli_module.format_sources(sources)

    assert result == (
        "1. First Source — https://example.com/one\n"
        "2. Second Source — https://example.com/two"
    )


def test_format_citations_returns_placeholder_when_none():
    result = cli_module.format_citations(None)

    assert result == "(no citations)\n"


def test_format_citations_formats_bullet_list():
    result = cli_module.format_citations(
        ["https://example.com/one", "https://example.com/two"]
    )

    assert result == "- https://example.com/one\n- https://example.com/two\n"


def test_format_output_contains_expected_sections():
    state = {
        "question": "test question",
        "run_id": "run-123",
        "status": "completed",
        "scope": "Explain the test topic",
        "queries": ["query one", "query two"],
        "sources": [
            {
                "title": "First Source",
                "url": "https://example.com/one",
            }
        ],
        "answer": "This is the answer.",
        "citations": ["https://example.com/one"],
        "errors": [],
    }

    result = cli_module.format_output(state)

    assert "QUESTION:" in result
    assert "RUN ID:" in result
    assert "STATUS:" in result
    assert "SCOPE:" in result
    assert "QUERIES:" in result
    assert "SOURCES (title + url):" in result
    assert "ANSWER:" in result
    assert "CITATIONS:" in result
    assert "test question" in result
    assert "run-123" in result
    assert "completed" in result
    assert "Explain the test topic" in result
    assert "- query one" in result
    assert "1. First Source — https://example.com/one" in result
    assert "This is the answer." in result
    assert "- https://example.com/one" in result


def test_format_error_output_contains_error_message():
    result = cli_module.format_error_output("Something failed")

    assert "STATUS:" in result
    assert "Error" in result
    assert "ERROR:" in result
    assert "Something failed" in result
