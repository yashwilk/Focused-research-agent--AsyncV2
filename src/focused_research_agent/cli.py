"""
Command-line interface entrypoint for the Focused Research Agent.

This module contains terminal-specific interaction logic for the project.
It is responsible for reading user input from command-line arguments or
interactive prompts, handling exit commands, and formatting the final
research result for CLI display.

Architecturally, the CLI is a transport adapter. It should stay focused on
terminal input/output concerns and delegate research execution to the
application layer.
"""

import asyncio
import logging
import sys

from focused_research_agent.application import research_use_case
from focused_research_agent.application.exceptions import ApplicationError
from focused_research_agent.config.logger_config import setup_logging

logger = logging.getLogger("focused_research_agent.cli")

EXIT_COMMANDS = {"exit", "quit", "bye"}


def format_queries(queries: list[str] | None) -> str:
    """
    Format generated search queries for CLI display.

    Args:
        queries: Generated search queries from the research result.

    Returns:
        str: Human-readable CLI text for the queries section.
    """
    if not queries:
        return "(no queries)\n"

    lines = []
    for q in queries:
        lines.append("- " + q)

    return "\n".join(lines) + "\n"


def format_sources(sources: list[dict] | None) -> str:
    """
    Format collected source entries for CLI display.

    Args:
        sources: Source dictionaries returned in the research result.

    Returns:
        str: Human-readable CLI text for the sources section.
    """
    if not sources:
        return "(no sources)\n"

    result = []
    for i, source in enumerate(sources, start=1):
        title = source.get("title") or "No Title"
        url = source.get("url") or "No URL"
        result.append(f"{i}. {title} — {url}")

    return "\n".join(result)


def format_citations(citations: list[str] | None) -> str:
    """
    Format citation URLs for CLI display.

    Args:
        citations: Citation URLs returned in the research result.

    Returns:
        str: Human-readable CLI text for the citations section.
    """
    if not citations:
        return "(no citations)\n"

    lines = []
    for c in citations:
        lines.append("- " + c)

    return "\n".join(lines) + "\n"


def format_output(state: dict) -> str:
    """
    Build the final CLI output block from the normalized research result.

    Args:
        state: Normalized research result returned by the application layer.

    Returns:
        str: Formatted CLI output block.
    """
    return f"""
==============================
QUESTION:
{state.get("question")}

RUN ID:
{state.get("run_id")}

STATUS:
{state.get("status")}

SCOPE:
{state.get("scope")}

QUERIES:
{format_queries(state.get("queries"))}
SOURCES (title + url):
{format_sources(state.get("sources"))}

ANSWER:
{state.get("answer")}

CITATIONS:
{format_citations(state.get("citations"))}
==============================
""".strip()


def format_error_output(message: str) -> str:
    """
    Build the CLI error output block.

    Args:
        message: Error message to display in the terminal.

    Returns:
        str: Formatted CLI error block.
    """
    return f"""
==============================
STATUS:
Error

ERROR:
{message}
==============================
""".strip()


def get_user_question_from_command_line() -> str | None:
    """
    Read the user question from command-line arguments or interactive input.

    The function first checks whether the user provided a question as
    command-line arguments. If not, it falls back to prompting the user
    interactively. It also supports exit keywords to end the application
    cleanly.

    Returns:
        str | None: Validated user question, or None if the user chose to
        exit the application.
    """
    user_question = " ".join(sys.argv[1:]).strip()

    if user_question:
        if user_question.lower() in EXIT_COMMANDS:
            return None
        return user_question

    while True:
        typed_question = input("What is your question? ").strip()

        if not typed_question:
            print("Please enter a question.")
            continue

        if typed_question.lower() in EXIT_COMMANDS:
            return None

        return typed_question


async def _run(user_question: str) -> None:
    """
    Run the CLI entrypoint for the research agent.

    This function initializes logging, collects a user question from the
    terminal, executes the shared research use case, and prints either
    formatted output or a formatted CLI error block.

    Returns:
        None
    """
    try:
        final_state = await research_use_case.research_question(user_question)

        errors = final_state.get("errors") or []
        if errors:
            error_message = "\n".join(errors)
            print(format_error_output(error_message))
            return

        print(format_output(final_state))

    except ApplicationError as e:
        print(format_error_output(str(e)))
        logger.exception("ApplicationError occurred: %s", e)

    except Exception as e:
        print(format_error_output(f"Unexpected internal error occurred: {e}"))
        logger.exception("Unexpected error in CLI")


def main() -> None:
    """Run the CLI entrypoint for the research agent."""
    setup_logging()
    user_question = get_user_question_from_command_line()

    if user_question is None:
        return

    asyncio.run(_run(user_question))
