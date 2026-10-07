"""SQLAlchemy async engine, session factory, and ORM models for persistence."""

import json
import logging
from collections.abc import AsyncGenerator
from datetime import UTC, datetime
from typing import Any

from sqlalchemy import DateTime, Float, Integer, String, Text, func, select
from sqlalchemy.ext.asyncio import (
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column

logger = logging.getLogger(__name__)


class Base(DeclarativeBase):
    """Declarative base for all ORM models."""

    pass


class ReviewRecord(Base):
    """Persisted record of a completed PR review."""

    __tablename__ = "reviews"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    repo: Mapped[str] = mapped_column(String(255), nullable=False, index=True)
    pr: Mapped[int] = mapped_column(Integer, nullable=False)
    head_sha: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    model: Mapped[str] = mapped_column(String(128), nullable=False)
    prompt_version: Mapped[str] = mapped_column(String(32), nullable=False)
    status: Mapped[str] = mapped_column(String(32), nullable=False, default="completed")
    findings_json: Mapped[str] = mapped_column(Text, nullable=False, default="[]")
    tokens: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    cost: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    latency_ms: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
    )

    def set_findings(self, findings: list[dict[str, Any]]) -> None:
        """Serialize findings list to JSON string."""
        self.findings_json = json.dumps(findings)


class DeadLetterJob(Base):
    """Record of a permanently failed job for manual inspection."""

    __tablename__ = "dead_letter_jobs"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    repo: Mapped[str] = mapped_column(String(255), nullable=False)
    pr: Mapped[int] = mapped_column(Integer, nullable=False)
    head_sha: Mapped[str] = mapped_column(String(64), nullable=False)
    delivery_id: Mapped[str | None] = mapped_column(String(64), nullable=True)
    error_message: Mapped[str] = mapped_column(Text, nullable=False, default="")
    attempts: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
    )


_VectorType: Any
try:
    from pgvector.sqlalchemy import Vector

    _VectorType = Vector(384).with_variant(Text(), "sqlite")
except ImportError:
    _VectorType = Text()


class CodeChunkRecord(Base):
    """AST-level code chunk indexed for semantic and hybrid retrieval."""

    __tablename__ = "code_chunks"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    repo: Mapped[str] = mapped_column(String(255), nullable=False, index=True)
    commit_sha: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    file_path: Mapped[str] = mapped_column(String(512), nullable=False, index=True)
    file_hash: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    symbol_name: Mapped[str] = mapped_column(String(255), nullable=False, default="")
    symbol_type: Mapped[str] = mapped_column(String(64), nullable=False, default="function")
    start_line: Mapped[int] = mapped_column(Integer, nullable=False, default=1)
    end_line: Mapped[int] = mapped_column(Integer, nullable=False, default=1)
    language: Mapped[str] = mapped_column(String(64), nullable=False, default="python")
    content: Mapped[str] = mapped_column(Text, nullable=False)
    embedding: Mapped[Any] = mapped_column(_VectorType, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
    )


class FeedbackRecord(Base):
    """Developer feedback on review findings (reactions, resolutions, line touches)."""

    __tablename__ = "feedback"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    review_id: Mapped[int | None] = mapped_column(Integer, nullable=True, index=True)
    repo: Mapped[str] = mapped_column(String(255), nullable=False, index=True)
    pr: Mapped[int] = mapped_column(Integer, nullable=False, index=True)
    comment_id: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    head_sha: Mapped[str | None] = mapped_column(String(64), nullable=True)
    file_path: Mapped[str | None] = mapped_column(String(512), nullable=True)
    line_number: Mapped[int | None] = mapped_column(Integer, nullable=True)
    feedback_type: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    feedback_value: Mapped[str | None] = mapped_column(String(64), nullable=True)
    author: Mapped[str | None] = mapped_column(String(255), nullable=True)
    details: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
    )


# ---------------------------------------------------------------------------
# Engine / session helpers
# ---------------------------------------------------------------------------

_engine_cache: dict[str, Any] = {}


def get_async_engine(database_url: str) -> Any:
    """Create or return cached async SQLAlchemy engine.

    Converts postgresql:// to postgresql+asyncpg:// if needed.
    """
    if database_url in _engine_cache:
        return _engine_cache[database_url]

    url = database_url
    if url.startswith("postgresql://"):
        url = url.replace("postgresql://", "postgresql+asyncpg://", 1)

    if url.startswith("sqlite"):
        engine = create_async_engine(url, echo=False)
    else:
        engine = create_async_engine(url, pool_size=5, max_overflow=10, echo=False)

    _engine_cache[database_url] = engine
    return engine


def get_async_session_factory(database_url: str) -> async_sessionmaker[AsyncSession]:
    """Return an async session factory bound to the given database URL."""
    engine = get_async_engine(database_url)
    return async_sessionmaker(engine, expire_on_commit=False)


async def get_session(database_url: str) -> AsyncGenerator[AsyncSession, None]:
    """Async generator yielding a database session with automatic cleanup."""
    factory = get_async_session_factory(database_url)
    async with factory() as session:
        yield session


async def create_tables(database_url: str) -> None:
    """Create all ORM tables (for development/testing only; use Alembic in prod)."""
    engine = get_async_engine(database_url)
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)


async def save_review_record(
    database_url: str,
    repo: str,
    pr: int,
    head_sha: str,
    model: str,
    prompt_version: str,
    status: str,
    findings: list[dict[str, Any]],
    tokens: int,
    cost: float,
    latency_ms: float,
) -> int:
    """Persist a review record and return its ID."""
    factory = get_async_session_factory(database_url)
    async with factory() as session:
        record = ReviewRecord(
            repo=repo,
            pr=pr,
            head_sha=head_sha,
            model=model,
            prompt_version=prompt_version,
            status=status,
            findings_json=json.dumps(findings),
            tokens=tokens,
            cost=cost,
            latency_ms=latency_ms,
            created_at=datetime.now(UTC),
        )
        session.add(record)
        await session.commit()
        await session.refresh(record)
        return int(record.id)


async def save_dead_letter(
    database_url: str,
    repo: str,
    pr: int,
    head_sha: str,
    delivery_id: str | None,
    error_message: str,
    attempts: int,
) -> int:
    """Persist a dead-letter record and return its ID."""
    factory = get_async_session_factory(database_url)
    async with factory() as session:
        record = DeadLetterJob(
            repo=repo,
            pr=pr,
            head_sha=head_sha,
            delivery_id=delivery_id,
            error_message=error_message,
            attempts=attempts,
            created_at=datetime.now(UTC),
        )
        session.add(record)
        await session.commit()
        await session.refresh(record)
        return int(record.id)


async def get_review_by_sha(
    database_url: str,
    repo: str,
    pr: int,
    head_sha: str,
) -> ReviewRecord | None:
    """Fetch existing completed review record for a given repo, PR, and commit SHA."""
    factory = get_async_session_factory(database_url)
    async with factory() as session:
        result = await session.execute(
            select(ReviewRecord).where(
                ReviewRecord.repo == repo,
                ReviewRecord.pr == pr,
                ReviewRecord.head_sha == head_sha,
                ReviewRecord.status == "completed",
            )
        )
        return result.scalars().first()


async def save_feedback_record(
    database_url: str,
    repo: str,
    pr: int,
    comment_id: str,
    feedback_type: str,
    review_id: int | None = None,
    head_sha: str | None = None,
    file_path: str | None = None,
    line_number: int | None = None,
    feedback_value: str | None = None,
    author: str | None = None,
    details: str | None = None,
) -> int:
    """Persist developer feedback (reaction, resolution, line touch) and return record ID."""
    factory = get_async_session_factory(database_url)
    async with factory() as session:
        record = FeedbackRecord(
            repo=repo,
            pr=pr,
            comment_id=str(comment_id),
            feedback_type=feedback_type,
            review_id=review_id,
            head_sha=head_sha,
            file_path=file_path,
            line_number=line_number,
            feedback_value=feedback_value,
            author=author,
            details=details,
            created_at=datetime.now(UTC),
        )
        session.add(record)
        await session.commit()
        await session.refresh(record)
        return int(record.id)


async def get_feedback_records(
    database_url: str,
    repo: str | None = None,
    pr: int | None = None,
    feedback_type: str | None = None,
) -> list[FeedbackRecord]:
    """Retrieve feedback records matching optional filters."""
    factory = get_async_session_factory(database_url)
    async with factory() as session:
        query = select(FeedbackRecord)
        if repo is not None:
            query = query.where(FeedbackRecord.repo == repo)
        if pr is not None:
            query = query.where(FeedbackRecord.pr == pr)
        if feedback_type is not None:
            query = query.where(FeedbackRecord.feedback_type == feedback_type)
        query = query.order_by(FeedbackRecord.created_at.desc())
        result = await session.execute(query)
        return list(result.scalars().all())


async def compute_acceptance_rate(
    database_url: str,
    repo: str | None = None,
) -> float:
    """Compute live developer acceptance rate.

    Accepted findings: feedback_type in ('reaction_up', 'resolved', 'line_touched')
    Rejected findings: feedback_type in ('reaction_down')
    Returns: float between 0.0 and 1.0 (defaults to 1.0 if no negative feedback recorded).
    """
    records = await get_feedback_records(database_url=database_url, repo=repo)
    if not records:
        return 1.0

    accepted = 0
    rejected = 0
    for r in records:
        if r.feedback_type in ("reaction_up", "resolved", "line_touched"):
            accepted += 1
        elif r.feedback_type in ("reaction_down", "rejected"):
            rejected += 1

    total = accepted + rejected
    if total == 0:
        return 1.0
    return round(float(accepted) / float(total), 4)
