"""
Pydantic request and response schemas for the report API.

This module defines the API contract for the report generation
endpoint. The response shape is identical to ResearchResponse —
the structured report content lives inside the answer field as
formatted markdown.

SourceResponse is imported from the research schema rather than
duplicated — the source shape is identical across all endpoints.
"""

from typing import Annotated

from pydantic import AfterValidator, BaseModel, StringConstraints

from focused_research_agent.api.schema.research.research import SourceResponse
from focused_research_agent.application.question_validation import (
    validate_and_clean_question,
)


class ReportRequest(BaseModel):
    """
    Request schema for generating a research report through the API.

    Contains only the user's question. Reports are single-turn —
    there is no conversation threading for report generation.

    Attributes:
        question: The user's research question for the report.
    """

    question: Annotated[
        str,
        StringConstraints(strip_whitespace=True, min_length=1, strict=True),
        AfterValidator(validate_and_clean_question),
    ]


class ReportResponse(BaseModel):
    """
    Response schema returned by the report API endpoint.

    The answer field contains a structured markdown report with
    Introduction, Key Findings, Analysis, and Conclusion sections.
    Citations contain up to 5 source URLs supporting the report.

    Attributes:
        run_id: Unique identifier for this research run.
        question: The user's question for this report.
        status: Final status of the research run.
        scope: Scoped interpretation of the question.
        queries: Generated web-search queries.
        sources: Normalized source entries used in synthesis.
        answer: Full structured markdown report.
        citations: Citation URLs supporting the report. Up to 5.
        errors: Collected workflow errors. Always a list.
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


class ReportJobResponse(BaseModel):
    """Response returned immediately after submitting a background report job."""

    job_id: str
    status: str


class ReportJobStatusResponse(BaseModel):
    """Response returned when polling a background report job's status."""

    job_id: str
    status: str
    result: dict | None = None
