"""
Research API endpoint for the Focused Research Agent.

Async + auth + rate-limit conversion: the endpoint is now `async def` and
awaits the (now async, cached) research use case directly instead of
blocking a worker thread for the run's full duration. Requires a valid
bearer token (see focused_research_agent.auth) and is rate-limited per
caller IP to protect the underlying Groq/Tavily quota.
"""

from collections.abc import Callable
from typing import Annotated

from fastapi import APIRouter, Depends, Request, status

from focused_research_agent.api.dependencies import get_research_use_case
from focused_research_agent.api.schema.research import research as research_schema
from focused_research_agent.auth.dependencies import get_current_user
from focused_research_agent.core.rate_limiter import RATE_LIMIT_RESEARCH, limiter

research_router = APIRouter(tags=["research"])


@research_router.post(
    "/research",
    status_code=status.HTTP_200_OK,
    response_model=research_schema.ResearchResponse,
)
@limiter.limit(RATE_LIMIT_RESEARCH)
async def research(
    request: Request,
    search: research_schema.ResearchRequest,
    run_research_use_case: Annotated[Callable, Depends(get_research_use_case)],
    _current_user=Depends(get_current_user),
) -> dict:
    """Handle a research request through the API (authenticated, rate-limited).

    Args:
        request: The incoming request (required by slowapi's limiter).
        search: Validated research request payload.
        run_research_use_case: Injected callable that executes the research use case.
        _current_user: Authenticated user resolved from the bearer token.

    Returns:
        dict: Structured research response returned by the application layer.
    """
    return await run_research_use_case(search.question)
