"""Initial schema: conversation_runs, users.

Revision ID: 0001
Revises:
Create Date: 2026-08-08

"""
from typing import Sequence, Union

import sqlalchemy as sa

from alembic import op

revision: str = "0001"
down_revision: Union[str, Sequence[str], None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Create conversation_runs and users tables."""
    op.create_table(
        "users",
        sa.Column("id", sa.Integer(), nullable=False, autoincrement=True),
        sa.Column("email", sa.String(), nullable=False),
        sa.Column("hashed_password", sa.String(), nullable=False),
        sa.Column("created_at", sa.DateTime(), nullable=False),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(op.f("ix_users_email"), "users", ["email"], unique=True)

    op.create_table(
        "conversation_runs",
        sa.Column("id", sa.Integer(), nullable=False, autoincrement=True),
        sa.Column("user_id", sa.Integer(), nullable=True),
        sa.Column("conversation_id", sa.String(), nullable=False),
        sa.Column("turn_number", sa.Integer(), nullable=False),
        sa.Column("conversation_title", sa.String(), nullable=True),
        sa.Column("run_id", sa.String(), nullable=False),
        sa.Column("question", sa.Text(), nullable=False),
        sa.Column("status", sa.String(), nullable=False),
        sa.Column("scope", sa.Text(), nullable=True),
        sa.Column("queries", sa.Text(), nullable=True),
        sa.Column("sources", sa.Text(), nullable=True),
        sa.Column("citations", sa.Text(), nullable=True),
        sa.Column("errors", sa.Text(), nullable=True),
        sa.Column("answer", sa.Text(), nullable=True),
        sa.Column("created_at", sa.DateTime(), nullable=False),
        sa.Column("updated_at", sa.DateTime(), nullable=False),
        sa.Column("mode", sa.String(), nullable=False, server_default="research"),
        sa.Column("images", sa.Text(), nullable=True),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(
        op.f("ix_conversation_runs_conversation_id"),
        "conversation_runs",
        ["conversation_id"],
        unique=False,
    )
    op.create_index(
        op.f("ix_conversation_runs_user_id"), "conversation_runs", ["user_id"], unique=False
    )


def downgrade() -> None:
    """Drop conversation_runs and users tables."""
    op.drop_index(op.f("ix_conversation_runs_user_id"), table_name="conversation_runs")
    op.drop_index(op.f("ix_conversation_runs_conversation_id"), table_name="conversation_runs")
    op.drop_table("conversation_runs")
    op.drop_index(op.f("ix_users_email"), table_name="users")
    op.drop_table("users")
