from typing import TypedDict


class ResearchState(TypedDict):
    run_id: str
    question: str
    scope: str | None
    assumptions: list[str] | None
    constraints: dict | None
    queries: list[str] | None
    sources: list[dict] | None
    answer: str | None
    citations: list[str] | None
    status: str
    errors: list[str]
    debug: dict | None
    conversation_id: str | None
    conversation_history: list[dict] | None
    mode: str
    images: list[str] | None
    search_retry_count: int
