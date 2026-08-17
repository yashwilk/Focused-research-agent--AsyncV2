"""
Chat API endpoint for the Focused Research Agent.

Async + auth + rate-limit conversion, same pattern as research.py.
"""

from typing import Annotated
from collections.abc import Callable

from fastapi import APIRouter, Depends, Request, status
from sqlalchemy.ext.asyncio import AsyncSession

from focused_research_agent.api.dependencies import get_chat_use_case
from focused_research_agent.api.schema.chat.chat import ChatRequest, ChatResponse
from focused_research_agent.auth.dependencies import get_current_user
from focused_research_agent.core.rate_limiter import RATE_LIMIT_CHAT, limiter
from focused_research_agent.database.database import get_db

chat_router = APIRouter(tags=["chat"])


@chat_router.post("/chat", status_code=status.HTTP_200_OK, response_model=ChatResponse)
@limiter.limit(RATE_LIMIT_CHAT)
async def chat(
    request: Request,
    chat_request: ChatRequest,
    db: Annotated[AsyncSession, Depends(get_db)],
    run_chat_use_case: Annotated[Callable, Depends(get_chat_use_case)],
    _current_user=Depends(get_current_user),
) -> dict:
    """Handle a chat research request through the API (authenticated, rate-limited)."""
    return await run_chat_use_case(
        db=db,
        conversation_id=chat_request.conversation_id,
        question=chat_request.question,
        user_id=_current_user.id,
    )
