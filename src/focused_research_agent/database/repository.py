"""
Database repository for the Focused Research Agent.

This module is the only file in the project that reads from and writes
to the database. All other modules that need data access call these
functions — they never interact with SQLAlchemy sessions directly.

Architecturally, this module belongs to the database layer and
implements the Repository Pattern. It abstracts storage concerns away
from the application layer. Switching databases requires changing only
this file and database.py — no application or graph code changes.

List fields (queries, sources, citations, errors) are serialized to
JSON strings on save and deserialized back to Python lists on read.
This conversion is transparent to the rest of the application.

Async conversion note: every function is now async and uses SQLAlchemy
2.0's async select() API (await db.execute(select(...))) instead of the
legacy db.query() API, which has no async equivalent.
"""

import logging
import json
from datetime import datetime, timezone

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from focused_research_agent.database.model import ConversationRun

logger = logging.getLogger(__name__)


def _serialize(value: list | None) -> str | None:
    """Serialize a Python list to a JSON string for database storage."""
    if value is None:
        return None
    return json.dumps(value)


def _deserialize(value: str | None) -> list | None:
    """Deserialize a JSON string from the database back to a Python list."""
    if value is None:
        return None
    return json.loads(value)


async def save_run(
    db: AsyncSession,
    state: dict,
    conversation_id: str,
    turn_number: int,
    mode: str = "research",
    user_id: int | None = None,
) -> ConversationRun:
    """Save a completed research run to the database.

    Args:
        db: Active async SQLAlchemy database session.
        state: Normalized research result dict from the application layer.
        conversation_id: UUID string linking this run to its conversation.
        turn_number: Position of this run within the conversation (1-based).

    Returns:
        ConversationRun: The saved model instance with its database ID populated.
    """
    now = datetime.now(timezone.utc)

    conversation_title = None
    if turn_number == 1:
        conversation_title = state["question"][:60]

    run = ConversationRun(
        user_id=user_id,
        conversation_id=conversation_id,
        turn_number=turn_number,
        conversation_title=conversation_title,
        run_id=state["run_id"],
        question=state["question"],
        status=state["status"],
        scope=state.get("scope"),
        queries=_serialize(state.get("queries")),
        sources=_serialize(state.get("sources")),
        answer=state.get("answer"),
        citations=_serialize(state.get("citations")),
        errors=_serialize(state.get("errors")),
        images=_serialize(state.get("images")),
        created_at=now,
        updated_at=now,
        mode=mode,
    )

    db.add(run)
    await db.commit()
    await db.refresh(run)
    logger.info(
        "Run saved. conversation_id=%s turn=%d mode=%s run_id=%s",
        conversation_id, turn_number, mode, state["run_id"],
    )
    return run


async def get_conversation_history(
    db: AsyncSession, conversation_id: str, max_turns: int, user_id: int | None = None
) -> list[dict]:
    """Fetch the most recent turns of a conversation for context threading.

    When user_id is provided, results are scoped to that user only — a
    caller cannot thread context into another user's conversation even if
    they know or guess the conversation_id.
    """
    conditions = [ConversationRun.conversation_id == conversation_id]
    if user_id is not None:
        conditions.append(ConversationRun.user_id == user_id)

    stmt = (
        select(ConversationRun)
        .where(*conditions)
        .order_by(ConversationRun.turn_number.desc())
        .limit(max_turns)
    )
    result = await db.execute(stmt)
    runs = list(reversed(result.scalars().all()))

    history = [
        {"turn": run.turn_number, "question": run.question, "answer": run.answer, "scope": run.scope}
        for run in runs
    ]

    logger.debug(
        "Conversation history fetched. conversation_id=%s turns=%d", conversation_id, len(history)
    )
    return history


async def get_all_conversations(db: AsyncSession, user_id: int | None = None) -> list[dict]:
    """Fetch a summary list of all conversations for the history sidebar.

    Scoped to user_id when provided — each user only sees their own
    conversation history, not every conversation ever run on the server.
    """
    conditions = [ConversationRun.turn_number == 1, ConversationRun.mode != "report"]
    if user_id is not None:
        conditions.append(ConversationRun.user_id == user_id)

    stmt = (
        select(ConversationRun)
        .where(*conditions)
        .order_by(ConversationRun.created_at.desc())
    )
    result = await db.execute(stmt)
    runs = result.scalars().all()

    conversations = [
        {"conversation_id": run.conversation_id, "title": run.conversation_title, "created_at": run.created_at.isoformat()}
        for run in runs
    ]

    logger.debug("All conversations fetched. count=%d", len(conversations))
    return conversations


async def get_conversation_turns(
    db: AsyncSession, conversation_id: str, user_id: int | None = None
) -> list[dict]:
    """Fetch all turns of a conversation in chronological order.

    Scoped to user_id when provided — a user cannot load another user's
    conversation by guessing its conversation_id.
    """
    conditions = [ConversationRun.conversation_id == conversation_id]
    if user_id is not None:
        conditions.append(ConversationRun.user_id == user_id)

    stmt = (
        select(ConversationRun)
        .where(*conditions)
        .order_by(ConversationRun.turn_number.asc())
    )
    result = await db.execute(stmt)
    runs = result.scalars().all()

    turns = [
        {
            "turn_number": run.turn_number,
            "run_id": run.run_id,
            "question": run.question,
            "status": run.status,
            "scope": run.scope,
            "queries": _deserialize(run.queries),
            "sources": _deserialize(run.sources),
            "answer": run.answer,
            "citations": _deserialize(run.citations),
            "errors": _deserialize(run.errors),
            "created_at": run.created_at.isoformat(),
            "images": _deserialize(run.images),
        }
        for run in runs
    ]
    logger.debug("Conversation turns fetched. conversation_id=%s count=%d", conversation_id, len(turns))
    return turns


async def get_all_reports(db: AsyncSession, user_id: int | None = None) -> list[dict]:
    """Fetch a summary list of all report runs for the report history sidebar.

    Scoped to user_id when provided.
    """
    conditions = [ConversationRun.mode == "report"]
    if user_id is not None:
        conditions.append(ConversationRun.user_id == user_id)

    stmt = (
        select(ConversationRun)
        .where(*conditions)
        .order_by(ConversationRun.created_at.desc())
    )
    result = await db.execute(stmt)
    runs = result.scalars().all()

    reports = [
        {"conversation_id": run.conversation_id, "title": run.conversation_title, "created_at": run.created_at.isoformat()}
        for run in runs
    ]
    logger.debug("All reports fetched. count=%d", len(reports))
    return reports
