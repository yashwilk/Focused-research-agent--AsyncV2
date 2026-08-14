"""
SQLAlchemy models for the Focused Research Agent database.

This module defines the database table structure using SQLAlchemy's
declarative ORM. The ConversationRun model maps to the
conversation_runs table and stores the full state of each research
turn within a conversation.

List fields (queries, sources, citations, errors) are stored as JSON
strings. The repository layer handles serialization and deserialization
transparently so the rest of the application always works with Python
lists.

Architecturally, this module belongs to the database layer. It defines
the data schema and knows nothing about HTTP, graph nodes, or
application logic.
"""

from sqlalchemy import DateTime, Integer, String, Text, Column
from sqlalchemy.orm import DeclarativeBase


class Base(DeclarativeBase):
    """
    Base class for all SQLAlchemy models.

    All models in this project inherit from this class. SQLAlchemy uses
    it to track all table definitions and create them in the database.
    """

    pass


class ConversationRun(Base):
    """
    Represents one research turn within a conversation.

    Each row is one complete research run — one question asked and
    answered. Multiple rows with the same conversation_id form a
    full conversation thread. The turn_number field tracks the order
    of turns within a conversation.

    List fields (queries, sources, citations, errors) are stored as
    JSON-serialized strings. The repository layer handles conversion
    between Python lists and JSON strings.
    """

    __tablename__ = "conversation_runs"

    id = Column(Integer, primary_key=True, autoincrement=True)

    # Owning user — nullable for backward compatibility with CLI/anonymous
    # runs, which have no authenticated user. API-created conversations
    # (research/chat/report) always set this once auth is required.
    user_id = Column(Integer, nullable=True, index=True)

    # Conversation threading
    conversation_id = Column(String, nullable=False, index=True)
    turn_number = Column(Integer, nullable=False)
    conversation_title = Column(String, nullable=True)

    # Research run identity
    run_id = Column(String, nullable=False)

    # Core research fields
    question = Column(Text, nullable=False)
    status = Column(String, nullable=False)
    scope = Column(Text, nullable=True)

    # List fields — stored as JSON strings, deserialized by repository
    queries = Column(Text, nullable=True)
    sources = Column(Text, nullable=True)
    citations = Column(Text, nullable=True)
    errors = Column(Text, nullable=True)

    # Answer
    answer = Column(Text, nullable=True)

    # Timestamps
    created_at = Column(DateTime, nullable=False)
    updated_at = Column(DateTime, nullable=False)

    mode = Column(String, nullable=False, default="research")

    # Images to be used in report
    images = Column(Text, nullable=True)

    def __repr__(self) -> str:
        """
        Return a readable string representation for debugging.

        Returns:
            str: String showing the run ID, conversation ID, and turn
                number.
        """
        return (
            f"ConversationRun("
            f"id={self.id}, "
            f"conversation_id={self.conversation_id}, "
            f"turn={self.turn_number}"
            f")"
        )


class User(Base):
    """
    A registered user account.

    Added as part of the authentication layer — endpoints that consume
    Groq/Tavily quota (research, chat, report) require a valid bearer
    token tied to one of these rows. See focused_research_agent.auth for
    the JWT issuance/verification logic that uses this model.
    """

    __tablename__ = "users"

    id = Column(Integer, primary_key=True, autoincrement=True)
    email = Column(String, nullable=False, unique=True, index=True)
    hashed_password = Column(String, nullable=False)
    created_at = Column(DateTime, nullable=False)

    def __repr__(self) -> str:
        return f"User(id={self.id}, email={self.email})"
