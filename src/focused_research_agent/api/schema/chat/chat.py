"""
Pydantic request and response schemas for the chat API.

This module defines the API contract for the chat endpoint. It
extends the research schemas with conversation threading fields.

SourceResponse is imported from the research schema rather than
duplicated — the source shape is identical across both endpoints.
"""

from typing import Annotated

from pydantic import AfterValidator, BaseModel, StringConstraints

from focused_research_agent.api.schema.research.research import SourceResponse
from focused_research_agent.application.question_validation import (
    validate_and_clean_question,
)


class ChatRequest(BaseModel):
    """
    Request schema for submitting a chat turn through the API.

    Contains the user's question and an optional conversation ID.
    When conversation_id is None, the backend starts a new conversation
    and returns a generated ID in the response. When provided, the
    backend threads the question into the existing conversation.

    Attributes:
        question: The user's research question for this turn.
        conversation_id: Existing conversation UUID to continue, or
            None to start a new conversation.
    """

    question: Annotated[
        str,
        StringConstraints(strip_whitespace=True, min_length=1, strict=True),
        AfterValidator(validate_and_clean_question),
    ]

    conversation_id: str | None = None


class ChatResponse(BaseModel):
    """
    Response schema returned by the chat API endpoint.

    Extends the standard research response with conversation metadata.
    The conversation_id must be stored by the client and sent back
    with subsequent questions to continue the conversation thread.

    Attributes:
        run_id: Unique identifier for this research run.
        question: The user's question for this turn.
        status: Final status of the research run.
        scope: Scoped interpretation of the question.
        queries: Generated web-search queries.
        sources: Normalized source entries used in synthesis.
        answer: Final synthesized answer.
        citations: Citation URLs supporting the answer.
        errors: Collected workflow errors. Always a list.
        conversation_id: UUID linking this turn to its conversation.
            Store this and send it back with follow-up questions.
        turn_number: Position of this turn within the conversation.
    """

    run_id: str
    question: str
    status: str
    scope: str | None
    queries: list[str] | None
    sources: list[SourceResponse] | None
    answer: str | None
    citations: list[str] | None
    errors: list[str]
    conversation_id: str
    turn_number: int
    images: list[str] | None
