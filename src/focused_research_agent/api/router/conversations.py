"""
Conversation history endpoints for the Focused Research Agent API.

Async + auth conversion: now requires a valid bearer token, and every
query is scoped to the authenticated user via the repository's user_id
filtering — one user can no longer read another user's conversation or
report history by guessing a conversation_id.
"""

from typing import Annotated

from fastapi import APIRouter, Depends, status
from sqlalchemy.ext.asyncio import AsyncSession

from focused_research_agent.auth.dependencies import get_current_user
from focused_research_agent.database.database import get_db
from focused_research_agent.database.model import User
from focused_research_agent.database.repository import (
    get_all_conversations,
    get_all_reports,
    get_conversation_turns,
)

conversations_router = APIRouter(tags=["conversations"])


@conversations_router.get("/conversations", status_code=status.HTTP_200_OK)
async def get_conversations(
    db: Annotated[AsyncSession, Depends(get_db)],
    current_user: Annotated[User, Depends(get_current_user)],
) -> list[dict]:
    """Return a summary list of the authenticated user's conversations."""
    return await get_all_conversations(db, user_id=current_user.id)


@conversations_router.get(
    "/conversations/{conversation_id}", status_code=status.HTTP_200_OK
)
async def get_conversation(
    conversation_id: str,
    db: Annotated[AsyncSession, Depends(get_db)],
    current_user: Annotated[User, Depends(get_current_user)],
) -> list[dict]:
    """Return all turns of a specific conversation, scoped to the current user."""
    return await get_conversation_turns(db, conversation_id, user_id=current_user.id)


@conversations_router.get("/reports", status_code=status.HTTP_200_OK)
async def get_reports(
    db: Annotated[AsyncSession, Depends(get_db)],
    current_user: Annotated[User, Depends(get_current_user)],
) -> list[dict]:
    """Return a summary list of the authenticated user's report runs."""
    return await get_all_reports(db, user_id=current_user.id)
