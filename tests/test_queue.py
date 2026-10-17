"""Tests for queue module: rate limiter, cache, worker jobs, and idempotency."""

import asyncio
import hashlib
import json
from typing import Any
from unittest.mock import ANY, AsyncMock, MagicMock, patch

import pytest

from prism.config import Settings

# Re-use models for building test data
from prism.models import (
    DiffLine,
    FileDiff,
    Hunk,
    IssueCategory,
    ReviewFinding,
    ReviewResult,
    Severity,
)
from prism.queue.cache import (
    _build_cache_key,
    compute_standards_hash,
    get_cache_stats,
    get_cached_result,
    set_cached_result,
)
from prism.queue.jobs import _is_trivial_diff, run_review
from prism.queue.rate_limiter import RateLimiter
from prism.review.llm import LLMUsage

# ---------------------------------------------------------------------------
# Rate limiter tests
# ---------------------------------------------------------------------------


class TestRateLimiter:
    """Tests for the Redis token bucket rate limiter."""

    @pytest.mark.asyncio
    async def test_acquire_succeeds_when_bucket_has_tokens(self) -> None:
        """First request should be permitted (full bucket)."""
        mock_redis = AsyncMock()
        script_mock = AsyncMock(return_value=1)
        mock_redis.register_script = MagicMock(return_value=script_mock)

        limiter = RateLimiter(mock_redis, max_tokens=5, refill_rate=1.0)
        result = await limiter.acquire(installation_id=1234)
        assert result is True
        script_mock.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_acquire_denied_when_bucket_empty(self) -> None:
        """Request should be denied when Lua script returns 0."""
        mock_redis = AsyncMock()
        script_mock = AsyncMock(return_value=0)
        mock_redis.register_script = MagicMock(return_value=script_mock)

        limiter = RateLimiter(mock_redis, max_tokens=5, refill_rate=1.0)
        result = await limiter.acquire(installation_id=5678)
        assert result is False

    @pytest.mark.asyncio
    async def test_acquire_fails_open_on_redis_error(self) -> None:
        """If Redis is down, rate limiter fails open (allows request)."""
        mock_redis = AsyncMock()
        script_mock = AsyncMock(side_effect=ConnectionError("Redis down"))
        mock_redis.register_script = MagicMock(return_value=script_mock)

        limiter = RateLimiter(mock_redis, max_tokens=5, refill_rate=1.0)
        result = await limiter.acquire(installation_id=9999)
        assert result is True

    @pytest.mark.asyncio
    async def test_wait_for_token_succeeds_immediately(self) -> None:
        """wait_for_token should return True when a token is immediately available."""
        mock_redis = AsyncMock()
        script_mock = AsyncMock(return_value=1)
        mock_redis.register_script = MagicMock(return_value=script_mock)

        limiter = RateLimiter(mock_redis, max_tokens=5, refill_rate=1.0)
        result = await limiter.wait_for_token(
            installation_id=1111, poll_interval=0.01, max_wait=0.1
        )
        assert result is True

    @pytest.mark.asyncio
    async def test_wait_for_token_times_out(self) -> None:
        """wait_for_token should return False when max_wait is exceeded."""
        mock_redis = AsyncMock()
        script_mock = AsyncMock(return_value=0)
        mock_redis.register_script = MagicMock(return_value=script_mock)

        limiter = RateLimiter(mock_redis, max_tokens=5, refill_rate=1.0)
        result = await limiter.wait_for_token(
            installation_id=2222, poll_interval=0.01, max_wait=0.05
        )
        assert result is False

    @pytest.mark.asyncio
    async def test_different_installations_get_different_keys(self) -> None:
        """Each installation should use a separate bucket key."""
        mock_redis = AsyncMock()
        script_mock = AsyncMock(return_value=1)
        mock_redis.register_script = MagicMock(return_value=script_mock)

        limiter = RateLimiter(mock_redis, key_prefix="test:rate:")
        await limiter.acquire(installation_id=111)
        await limiter.acquire(installation_id=222)

        calls = script_mock.call_args_list
        keys_used = [call.kwargs["keys"][0] for call in calls]
        assert keys_used[0] == "test:rate:111"
        assert keys_used[1] == "test:rate:222"


# ---------------------------------------------------------------------------
# Cache tests
# ---------------------------------------------------------------------------


class TestCache:
    """Tests for the LLM result cache."""

    def test_build_cache_key_deterministic(self) -> None:
        """Same inputs should produce the same cache key."""
        key1 = _build_cache_key("model-a", "v0.1", "hash123", "chunk text")
        key2 = _build_cache_key("model-a", "v0.1", "hash123", "chunk text")
        assert key1 == key2
        assert key1.startswith("prism:llm_cache:")

    def test_build_cache_key_varies_with_input(self) -> None:
        """Different inputs should produce different cache keys."""
        key1 = _build_cache_key("model-a", "v0.1", "hash123", "chunk text")
        key2 = _build_cache_key("model-b", "v0.1", "hash123", "chunk text")
        assert key1 != key2

    def test_compute_standards_hash(self) -> None:
        """Standards hash should be a valid sha256 hex digest."""
        h = compute_standards_hash("some standards text")
        assert len(h) == 64
        assert h == hashlib.sha256(b"some standards text").hexdigest()

    @pytest.mark.asyncio
    async def test_cache_miss_when_redis_none(self) -> None:
        """Should return None when no Redis client is available."""
        result = await get_cached_result(None, "m", "v", "h", "c")
        assert result is None

    @pytest.mark.asyncio
    async def test_cache_miss_when_key_not_found(self) -> None:
        """Should return None when key doesn't exist in Redis."""
        mock_redis = AsyncMock()
        mock_redis.get = AsyncMock(return_value=None)

        result = await get_cached_result(mock_redis, "m", "v", "h", "chunk")
        assert result is None

    @pytest.mark.asyncio
    async def test_cache_hit_returns_deserialized_data(self) -> None:
        """Should return parsed JSON dict on cache hit."""
        cached_data = {"findings": [], "usage": {"tokens": 100}}
        mock_redis = AsyncMock()
        mock_redis.get = AsyncMock(return_value=json.dumps(cached_data).encode("utf-8"))

        result = await get_cached_result(mock_redis, "m", "v", "h", "chunk")
        assert result == cached_data

    @pytest.mark.asyncio
    async def test_cache_set_stores_with_ttl(self) -> None:
        """set_cached_result should call redis.set with TTL."""
        mock_redis = AsyncMock()
        mock_redis.set = AsyncMock()

        data: dict[str, Any] = {"findings": []}
        await set_cached_result(mock_redis, "model", "v0.1", "hash", "chunk", data, ttl=3600)

        mock_redis.set.assert_awaited_once()
        call_args = mock_redis.set.call_args
        assert call_args.kwargs["ex"] == 3600

    @pytest.mark.asyncio
    async def test_cache_set_noop_when_redis_none(self) -> None:
        """set_cached_result should silently no-op without Redis."""
        # Should not raise
        await set_cached_result(None, "m", "v", "h", "c", {})

    @pytest.mark.asyncio
    async def test_cache_hit_miss_counters(self) -> None:
        """Cache stats should track hits and misses."""
        # Reset module-level counters by importing and modifying
        import prism.queue.cache as cache_mod

        cache_mod._cache_hits = 0
        cache_mod._cache_misses = 0

        mock_redis = AsyncMock()
        mock_redis.get = AsyncMock(return_value=None)
        await get_cached_result(mock_redis, "m", "v", "h", "miss1")
        await get_cached_result(mock_redis, "m", "v", "h", "miss2")

        mock_redis.get = AsyncMock(return_value=json.dumps({"x": 1}).encode())
        await get_cached_result(mock_redis, "m", "v", "h", "hit1")

        stats = get_cache_stats()
        assert stats["cache_misses"] == 2
        assert stats["cache_hits"] == 1
        assert stats["total_lookups"] == 3

    @pytest.mark.asyncio
    async def test_cache_graceful_on_redis_error(self) -> None:
        """Cache read errors should return None, not raise."""
        mock_redis = AsyncMock()
        mock_redis.get = AsyncMock(side_effect=ConnectionError("down"))

        result = await get_cached_result(mock_redis, "m", "v", "h", "chunk")
        assert result is None


# ---------------------------------------------------------------------------
# Trivial diff detection tests
# ---------------------------------------------------------------------------


class TestTrivialDiffDetection:
    """Tests for _is_trivial_diff."""

    def test_empty_diffs_are_trivial(self) -> None:
        assert _is_trivial_diff([]) is True

    def test_markdown_only_is_trivial(self) -> None:
        fd = FileDiff(
            path="README.md",
            language="markdown",
            hunks=[
                Hunk(
                    old_start=1,
                    new_start=1,
                    lines=[DiffLine(kind="add", new_no=1, text="# Title")],
                )
            ],
        )
        assert _is_trivial_diff([fd]) is True

    def test_license_file_is_trivial(self) -> None:
        fd = FileDiff(
            path="LICENSE",
            language="text",
            hunks=[
                Hunk(
                    old_start=1,
                    new_start=1,
                    lines=[DiffLine(kind="add", new_no=1, text="MIT License")],
                )
            ],
        )
        assert _is_trivial_diff([fd]) is True

    def test_code_file_is_not_trivial(self) -> None:
        fd = FileDiff(
            path="src/main.py",
            language="python",
            hunks=[
                Hunk(
                    old_start=1,
                    new_start=1,
                    lines=[
                        DiffLine(kind="add", new_no=1, text="import os"),
                    ],
                )
            ],
        )
        assert _is_trivial_diff([fd]) is False

    def test_whitespace_only_changes_are_trivial(self) -> None:
        fd = FileDiff(
            path="src/main.py",
            language="python",
            hunks=[
                Hunk(
                    old_start=1,
                    new_start=1,
                    lines=[
                        DiffLine(kind="add", new_no=1, text="   "),
                        DiffLine(kind="del", old_no=1, text="\t"),
                    ],
                )
            ],
        )
        assert _is_trivial_diff([fd]) is True

    def test_mixed_trivial_and_code_is_not_trivial(self) -> None:
        md_file = FileDiff(
            path="docs/guide.md",
            language="markdown",
            hunks=[
                Hunk(
                    old_start=1,
                    new_start=1,
                    lines=[DiffLine(kind="add", new_no=1, text="## Guide")],
                )
            ],
        )
        py_file = FileDiff(
            path="src/app.py",
            language="python",
            hunks=[
                Hunk(
                    old_start=1,
                    new_start=1,
                    lines=[DiffLine(kind="add", new_no=1, text="print('hi')")],
                )
            ],
        )
        assert _is_trivial_diff([md_file, py_file]) is False

    def test_gitignore_is_trivial(self) -> None:
        fd = FileDiff(
            path=".gitignore",
            language="text",
            hunks=[
                Hunk(
                    old_start=1,
                    new_start=1,
                    lines=[DiffLine(kind="add", new_no=1, text="*.pyc")],
                )
            ],
        )
        assert _is_trivial_diff([fd]) is True


# ---------------------------------------------------------------------------
# Worker job idempotency / integration tests (no real Redis/Postgres)
# ---------------------------------------------------------------------------


class TestWorkerJobIdempotency:
    """Tests verifying that re-running the same job payload is idempotent."""

    @pytest.mark.asyncio
    async def test_same_head_sha_skips_duplicate(self) -> None:
        """Idempotency check in the webhook layer should block duplicate enqueues.

        This test verifies the Redis-based dedupe in webhook.py rather than
        re-testing it in jobs.py (jobs.py trusts the webhook layer).
        """
        from prism.github.webhook import check_and_set_idempotency

        mock_redis = AsyncMock()
        # First call: key doesn't exist → set returns True (NX succeeds)
        mock_redis.set = AsyncMock(return_value=True)

        is_new = await check_and_set_idempotency(
            redis_client=mock_redis,
            delivery_id="delivery-123",
            repo_full_name="org/repo",
            pr_number=42,
            head_sha="abc123",
        )
        assert is_new is True

        # Second call: key exists → set returns False (NX fails)
        mock_redis.set = AsyncMock(return_value=False)

        is_dup = await check_and_set_idempotency(
            redis_client=mock_redis,
            delivery_id="delivery-123",
            repo_full_name="org/repo",
            pr_number=42,
            head_sha="abc123",
        )
        assert is_dup is False


# ---------------------------------------------------------------------------
# Worker Job Lifecycle, Retry, Dead Letter & Worker Crash/Restart Tests
# ---------------------------------------------------------------------------


class TestWorkerJobLifecycle:
    """Tests for run_review retries, dead letter logging, and worker crash recovery."""

    @pytest.mark.asyncio
    async def test_job_retry_raises_and_skips_dead_letter_on_first_try(self, tmp_path: Any) -> None:
        """On initial attempt failure (try 1 < 3), run_review re-raises error for arq retry
        without dead-lettering.
        """
        db_path = tmp_path / "test_retry.db"
        db_url = f"sqlite+aiosqlite:///{db_path}"

        from prism.db import create_tables

        await create_tables(db_url)

        job_payload = {
            "installation_id": 100,
            "repo_full_name": "owner/repo",
            "pr_number": 5,
            "head_sha": "sha123",
            "delivery_id": "del-1",
        }
        ctx = {"redis": None, "job_try": 1}

        with (
            patch("prism.queue.jobs.Settings") as mock_settings_cls,
            patch(
                "prism.queue.jobs.get_installation_access_token",
                side_effect=ValueError("Token error"),
            ),
            patch("prism.queue.jobs.save_dead_letter") as mock_dead_letter,
        ):
            mock_settings = MagicMock()
            mock_settings.database_url = db_url
            mock_settings_cls.return_value = mock_settings

            with pytest.raises(ValueError, match="Token error"):
                await run_review(ctx, job_payload)

            mock_dead_letter.assert_not_called()

    @pytest.mark.asyncio
    async def test_job_records_dead_letter_on_max_tries(self, tmp_path: Any) -> None:
        """On final attempt failure (try 3 >= 3), run_review records job to dead_letter."""
        db_path = tmp_path / "test_dl.db"
        db_url = f"sqlite+aiosqlite:///{db_path}"

        from sqlalchemy import select

        from prism.db import DeadLetterJob, create_tables, get_async_session_factory

        await create_tables(db_url)

        job_payload = {
            "installation_id": 100,
            "repo_full_name": "owner/repo",
            "pr_number": 5,
            "head_sha": "sha123",
            "delivery_id": "del-1",
        }
        ctx = {"redis": None, "job_try": 3}

        with (
            patch("prism.queue.jobs.Settings") as mock_settings_cls,
            patch(
                "prism.queue.jobs.get_installation_access_token",
                side_effect=RuntimeError("API failure"),
            ),
        ):
            mock_settings = MagicMock()
            mock_settings.database_url = db_url
            mock_settings_cls.return_value = mock_settings

            with pytest.raises(RuntimeError, match="API failure"):
                await run_review(ctx, job_payload)

        # Verify dead letter entry was written to DB
        factory = get_async_session_factory(db_url)
        async with factory() as session:
            result = await session.execute(select(DeadLetterJob))
            records = result.scalars().all()
            assert len(records) == 1
            assert records[0].repo == "owner/repo"
            assert records[0].pr == 5
            assert records[0].head_sha == "sha123"
            assert records[0].attempts == 3
            assert "API failure" in records[0].error_message

    @pytest.mark.asyncio
    async def test_idempotent_re_run_same_head_sha_skips_processing(self, tmp_path: Any) -> None:
        """If review for (repo, pr, head_sha) is in DB, re-running skips execution."""
        db_path = tmp_path / "test_idempotent.db"
        db_url = f"sqlite+aiosqlite:///{db_path}"

        from sqlalchemy import select

        from prism.db import ReviewRecord, create_tables, get_async_session_factory

        await create_tables(db_url)

        # Pre-seed DB with completed review record
        factory = get_async_session_factory(db_url)
        async with factory() as session:
            rec = ReviewRecord(
                repo="owner/repo",
                pr=42,
                head_sha="sha_already_done",
                model="test-model",
                prompt_version="1.0",
                status="completed",
                findings_json="[]",
            )
            session.add(rec)
            await session.commit()

        job_payload = {
            "installation_id": 1,
            "repo_full_name": "owner/repo",
            "pr_number": 42,
            "head_sha": "sha_already_done",
        }
        ctx = {"redis": None}

        with (
            patch("prism.queue.jobs.Settings") as mock_settings_cls,
            patch("prism.queue.jobs.get_installation_access_token") as mock_auth,
        ):
            mock_settings = MagicMock()
            mock_settings.database_url = db_url
            mock_settings_cls.return_value = mock_settings

            await run_review(ctx, job_payload)

            # Auth and API calls should NOT be made because it skipped
            mock_auth.assert_not_called()

        # Check DB still has exactly 1 record
        async with factory() as session:
            result = await session.execute(select(ReviewRecord))
            assert len(result.scalars().all()) == 1


class TestPipelineCachingIntegration:
    """Integration tests for Redis LLM result caching in review_pr."""

    @pytest.mark.asyncio
    async def test_pipeline_cache_miss_and_hit(self) -> None:
        """First call stores result in Redis (miss); second call fetches from Redis (hit)."""
        from prism.models import FileDiff, Hunk, ReviewFinding, ReviewResult
        from prism.review.pipeline import PRContext, review_pr

        # In-memory Redis mock dictionary
        redis_store: dict[str, bytes] = {}

        class FakeRedis:
            async def get(self, key: str) -> bytes | None:
                return redis_store.get(key)

            async def set(self, key: str, value: str | bytes, ex: int | None = None) -> None:
                if isinstance(value, str):
                    redis_store[key] = value.encode("utf-8")
                else:
                    redis_store[key] = value

        mock_settings = Settings(
            github_app_id="123",
            github_private_key="key",
            github_webhook_secret="secret",
            database_url="sqlite://",
            redis_url="redis://localhost:6379",
        )

        mock_redis = FakeRedis()

        diff = FileDiff(
            path="src/calc.py",
            language="python",
            hunks=[
                Hunk(
                    old_start=1, new_start=1, lines=[DiffLine(kind="add", new_no=1, text="x = 100")]
                )
            ],
        )
        pr_ctx = PRContext(
            repo_full_name="org/repo",
            pr_number=1,
            head_sha="sha_cache_test",
            file_diffs=[diff],
        )

        mock_llm = AsyncMock()
        mock_finding = ReviewFinding(
            path="src/calc.py",
            line=1,
            category=IssueCategory.SECURITY,
            severity=Severity.HIGH,
            message="Hardcoded constant detected.",
            suggestion="Use config",
            confidence=0.9,
        )
        mock_response = MagicMock()
        mock_response.findings = [mock_finding]
        mock_usage = LLMUsage(
            input_tokens=50, output_tokens=20, cost_usd=0.001, latency_ms=150.0, model="gpt-4o"
        )
        mock_llm.complete_structured = AsyncMock(return_value=(mock_response, mock_usage))
        mock_llm.model = "gpt-4o"

        # 1st run: Cache miss -> calls LLM client
        res1: ReviewResult = await review_pr(
            pr_ctx, settings=mock_settings, llm_client=mock_llm, redis_client=mock_redis
        )
        assert len(res1.findings) == 1
        assert mock_llm.complete_structured.call_count == 1
        assert len(redis_store) == 1  # Key saved in cache

        # 2nd run: Cache hit -> returns cached finding WITHOUT calling LLM again
        mock_llm.complete_structured.reset_mock()
        res2: ReviewResult = await review_pr(
            pr_ctx, settings=mock_settings, llm_client=mock_llm, redis_client=mock_redis
        )
        assert len(res2.findings) == 1
        assert res2.findings[0].message == "Hardcoded constant detected."
        assert mock_llm.complete_structured.call_count == 0  # No extra LLM call!


class TestWorkerCrashAndRestartSuccessCriteria:
    """Success criteria: Kill the worker mid-job, restart it, review completes exactly once."""

    @pytest.mark.asyncio
    async def test_worker_killed_mid_job_restarts_and_completes_exactly_once(
        self, tmp_path: Any
    ) -> None:
        """Proves that killing worker mid-job and restarting completes review exactly once."""
        db_path = tmp_path / "test_worker_crash.db"
        db_url = f"sqlite+aiosqlite:///{db_path}"

        from sqlalchemy import select

        from prism.db import ReviewRecord, create_tables, get_async_session_factory

        await create_tables(db_url)

        job_payload = {
            "installation_id": 999,
            "repo_full_name": "acme/backend",
            "pr_number": 42,
            "head_sha": "crash_sha_999",
            "delivery_id": "delivery-crash-1",
        }

        # Mock GitHub client responses
        mock_gh_client = AsyncMock()
        mock_gh_client.create_check_run = AsyncMock(return_value=123)
        mock_gh_client.fetch_pull_request_diff = AsyncMock(
            return_value=(
                "diff --git a/app.py b/app.py\n"
                "--- a/app.py\n"
                "+++ b/app.py\n"
                "@@ -1,1 +1,1 @@\n"
                "-old_code()\n"
                "+new_feature_code()\n"
            )
        )
        mock_gh_client.fetch_repo_file = AsyncMock(return_value="")
        mock_gh_client.update_check_run = AsyncMock()
        mock_gh_client.close = AsyncMock()

        mock_finding = ReviewFinding(
            path="app.py",
            line=1,
            category=IssueCategory.READABILITY,
            severity=Severity.LOW,
            message="Style issue: minor formatting",
            suggestion="Fix format",
            confidence=0.85,
        )
        mock_review_res = ReviewResult(
            findings=[mock_finding],
            model="test-llm",
            prompt_version="1.0",
            input_tokens=10,
            output_tokens=10,
            latency_ms=50.0,
            cost_usd=0.0001,
        )

        # Setup in-memory Redis cache
        redis_store: dict[str, bytes] = {}

        class FakeRedis:
            async def get(self, key: str) -> bytes | None:
                return redis_store.get(key)

            async def set(self, key: str, value: str | bytes, ex: int | None = None) -> None:
                if isinstance(value, str):
                    redis_store[key] = value.encode("utf-8")
                else:
                    redis_store[key] = value

            def register_script(self, script: str) -> Any:
                return AsyncMock(return_value=1)

        mock_redis = FakeRedis()
        ctx = {"redis": mock_redis, "job_try": 1}

        # --- STEP 1: Simulate Worker Run 1 killed mid-job (e.g. SIGKILL before DB save) ---
        with (
            patch("prism.queue.jobs.Settings") as mock_settings_cls,
            patch("prism.queue.jobs.get_installation_access_token", return_value="tok"),
            patch("prism.queue.jobs.GitHubClient", return_value=mock_gh_client),
            patch(
                "prism.queue.jobs.review_pr", side_effect=asyncio.CancelledError("Worker SIGKILL")
            ),
        ):
            mock_settings = MagicMock()
            mock_settings.database_url = db_url
            mock_settings_cls.return_value = mock_settings

            with pytest.raises(asyncio.CancelledError):
                await run_review(ctx, job_payload)

        # Verify DB is clean after worker kill mid-job
        factory = get_async_session_factory(db_url)
        async with factory() as session:
            records = (await session.execute(select(ReviewRecord))).scalars().all()
            assert len(records) == 0

        # --- STEP 2: Worker Restarts and re-executes run_review ---
        ctx["job_try"] = 2
        with (
            patch("prism.queue.jobs.Settings") as mock_settings_cls,
            patch("prism.queue.jobs.get_installation_access_token", return_value="tok"),
            patch("prism.queue.jobs.GitHubClient", return_value=mock_gh_client),
            patch("prism.queue.jobs.review_pr", return_value=mock_review_res),
            patch("prism.queue.jobs.post_pull_request_review") as mock_post_review,
        ):
            mock_settings = MagicMock()
            mock_settings.database_url = db_url
            mock_settings_cls.return_value = mock_settings

            await run_review(ctx, job_payload)

            mock_post_review.assert_awaited_once()

        # --- STEP 3: Verification ---
        # 1. Exactly ONE review record persisted in Postgres
        async with factory() as session:
            records = (await session.execute(select(ReviewRecord))).scalars().all()
            assert len(records) == 1
            assert records[0].repo == "acme/backend"
            assert records[0].pr == 42
            assert records[0].head_sha == "crash_sha_999"
            assert records[0].status == "completed"

        # 2. Check run updated to completed/success
        mock_gh_client.update_check_run.assert_awaited_with(
            "acme/backend", 123, conclusion="success", summary=ANY
        )

        # --- STEP 4: If worker restarts AGAIN for same SHA, idempotency prevents dupes ---
        with (
            patch("prism.queue.jobs.Settings") as mock_settings_cls,
            patch("prism.queue.jobs.get_installation_access_token", return_value="tok"),
        ):
            mock_settings = MagicMock()
            mock_settings.database_url = db_url
            mock_settings_cls.return_value = mock_settings

            await run_review(ctx, job_payload)

        async with factory() as session:
            records = (await session.execute(select(ReviewRecord))).scalars().all()
            assert len(records) == 1  # Still exactly 1 record!

        async with factory() as session:
            records = (await session.execute(select(ReviewRecord))).scalars().all()
            assert len(records) == 1  # Still exactly 1 record!
