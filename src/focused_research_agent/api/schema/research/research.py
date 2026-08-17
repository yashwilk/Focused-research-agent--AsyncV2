"""
Pydantic request and response schemas for the research API.

This module defines the API contract for the research endpoint. These schemas
describe the request body accepted by the FastAPI route and the structured
response returned to API clients.

These models belong to the API boundary and should represent transport-level
data shapes, not internal graph state or provider-specific models.
"""

from typing import Annotated

from pydantic import AfterValidator, BaseModel, StringConstraints

from focused_research_agent.application.question_validation import (
    validate_and_clean_question,
)


class ResearchRequest(BaseModel):
    """
    Request schema for submitting a research question through the API.

    This model represents the client payload required to trigger the research
    use case. It contains a validated, non-empty user question and rejects
    blank, whitespace-only, punctuation-only, or meaningless ultra-short
    input.

    Attributes:
        question: The user's research question.
    """

    question: Annotated[
        str,
        StringConstraints(strip_whitespace=True, min_length=1, strict=True),
        AfterValidator(validate_and_clean_question),
    ]


class SourceResponse(BaseModel):
    """
    Schema representing one source returned in the research response.

    This model defines the transport-level shape of a normalized source item
    included in the API response.

    Attributes:
        title: Human-readable title of the source.
        url: Source URL.
        snippet: Short excerpt or summary from the source.
        source: Name of the originating source provider.
        score: Relevance score assigned during search.
    """

    title: str
    url: str
    snippet: str
    source: str
    score: float


class ResearchResponse(BaseModel):
    """
    Response schema returned by the research API endpoint.

    This model represents the structured API response for the research use
    case. It mirrors the main graph output fields exposed through the
    application layer and provides a stable transport-level response shape
    for clients.

    The errors field is typed as list[str] — not list[str] | None — because
    the application layer's normalize_state always returns errors as a list,
    defaulting to an empty list when no errors occurred. This makes the
    contract explicit: callers can always iterate over errors safely without
    a None check.

    Attributes:
        run_id: Unique identifier for the research run.
        question: Original user question.
        status: Final status of the research run.
        scope: Scoped interpretation of the user's question.
        queries: Generated web-search queries.
        sources: Normalized source entries used in synthesis.
        answer: Final synthesized answer.
        citations: Citation URLs supporting the answer.
        errors: Collected workflow errors. Always a list; empty when none.
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
    images: list[str] | None
