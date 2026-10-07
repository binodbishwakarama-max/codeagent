"""create feedback table

Revision ID: 002_feedback
Revises: 001_initial
Create Date: 2024-01-02 00:00:00.000000

"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision: str = "002_feedback"
down_revision: str | None = "001_initial"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Create feedback table."""
    op.create_table(
        "feedback",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("review_id", sa.Integer(), nullable=True),
        sa.Column("repo", sa.String(length=255), nullable=False),
        sa.Column("pr", sa.Integer(), nullable=False),
        sa.Column("comment_id", sa.String(length=64), nullable=False),
        sa.Column("head_sha", sa.String(length=64), nullable=True),
        sa.Column("file_path", sa.String(length=512), nullable=True),
        sa.Column("line_number", sa.Integer(), nullable=True),
        sa.Column("feedback_type", sa.String(length=64), nullable=False),
        sa.Column("feedback_value", sa.String(length=64), nullable=True),
        sa.Column("author", sa.String(length=255), nullable=True),
        sa.Column("details", sa.Text(), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            nullable=False,
        ),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("ix_feedback_repo", "feedback", ["repo"], unique=False)
    op.create_index("ix_feedback_pr", "feedback", ["pr"], unique=False)
    op.create_index("ix_feedback_comment_id", "feedback", ["comment_id"], unique=False)
    op.create_index("ix_feedback_feedback_type", "feedback", ["feedback_type"], unique=False)


def downgrade() -> None:
    """Drop feedback table."""
    op.drop_index("ix_feedback_feedback_type", table_name="feedback")
    op.drop_index("ix_feedback_comment_id", table_name="feedback")
    op.drop_index("ix_feedback_pr", table_name="feedback")
    op.drop_index("ix_feedback_repo", table_name="feedback")
    op.drop_table("feedback")
