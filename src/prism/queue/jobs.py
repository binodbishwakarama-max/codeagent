"""Background job definitions for executing PR reviews asynchronously.

Features:
- Postgres persistence of review results
- Redis-backed LLM result caching with hit rate logging
- Per-installation token bucket rate limiting
- GitHub check run status updates (in_progress → success/neutral/failure)
- Dead-letter table for permanently failed jobs
- Trivial diff detection (skip LLM for docs-only / whitespace changes)
"""

import logging
import re
from typing import Any

from prism.config import Settings
from prism.db import get_review_by_sha, save_dead_letter, save_review_record
from prism.github.auth import get_installation_access_token
from prism.github.client import GitHubClient
from prism.github.comments import post_pull_request_review
from prism.github.diff import parse_unified_diff
from prism.models import FileDiff
from prism.observability.metrics import (
    record_cache_stats,
    record_github_api_error,
    record_review_completed,
)
from prism.observability.structured_logging import (
    bind_log_context,
    clear_log_context,
    set_correlation_id,
)
from prism.observability.tracing import (
    ReviewTrace,
    compute_config_hash,
)
from prism.queue.budget import check_daily_budget, record_daily_cost
from prism.queue.cache import get_cache_stats, log_cache_stats
from prism.queue.rate_limiter import RateLimiter
from prism.review.pipeline import PRContext, review_pr
from prism.review.redactor import redact_file_diffs
from prism.review.standards import parse_repo_standards

logger = logging.getLogger(__name__)

# File patterns considered trivial (no LLM review needed)
_TRIVIAL_PATTERNS = [
    re.compile(r"\.(md|txt|rst|adoc)$", re.IGNORECASE),
    re.compile(r"^(LICENSE|CHANGELOG|CHANGES|AUTHORS|CONTRIBUTORS)", re.IGNORECASE),
    re.compile(r"\.?(gitignore|gitattributes|editorconfig)$"),
]


def _is_trivial_diff(file_diffs: list[FileDiff]) -> bool:
    """Detect trivially small diffs that don't need LLM review.

    Returns True if all files are documentation/config-only OR
    if total changed lines (add+del) across all files is zero.
    """
    if not file_diffs:
        return True

    total_changed = 0
    all_trivial_files = True

    for fd in file_diffs:
        is_trivial_file = any(p.search(fd.path) for p in _TRIVIAL_PATTERNS)
        if not is_trivial_file:
            all_trivial_files = False

        for hunk in fd.hunks:
            for line in hunk.lines:
                if line.kind in ("add", "del"):
                    text = line.text.strip()
                    # Only count non-whitespace changes
                    if text:
                        total_changed += 1

    if all_trivial_files:
        return True

    # Pure whitespace changes
    return total_changed == 0


async def run_review(ctx: dict[str, Any], job: dict[str, Any]) -> None:
    """Async background worker job processing a pull request review.

    Full lifecycle:
    1. Check for existing completed review (idempotency check)
    2. Rate-limit check per installation
    3. Create GitHub check run (in_progress)
    4. Fetch diff and standards
    5. Skip LLM if diff is trivial
    6. Run review pipeline (with Redis caching)
    7. Post review comments to GitHub
    8. Persist to Postgres reviews table
    9. Update check run to success/neutral
    10. On permanent failure (max retries reached): write to dead_letter_jobs table

    Payload format:
    {
        "installation_id": int,
        "repo_full_name": str,
        "pr_number": int,
        "head_sha": str,
        "delivery_id": str,
    }
    """
    repo = str(job.get("repo_full_name"))
    pr_num = int(job.get("pr_number", 0))
    sha = str(job.get("head_sha"))
    installation_id = int(job.get("installation_id", 0))
    delivery_id = str(job.get("delivery_id", ""))

    set_correlation_id(delivery_id)
    bind_log_context(repo=repo, pr=pr_num, head_sha=sha, delivery_id=delivery_id)

    logger.info(
        "Starting review job for %s#%d at %s (delivery=%s)",
        repo,
        pr_num,
        sha,
        delivery_id,
    )

    settings = Settings()
    redis_client = ctx.get("redis")
    check_run_id: int | None = None
    client: GitHubClient | None = None

    # 0. Per-installation kill switch check
    blocked_inst = getattr(settings, "blocked_installations", [])
    if isinstance(blocked_inst, (list, tuple, set)) and installation_id in blocked_inst:
        logger.warning(
            "Installation %d is blocked by per-installation kill switch; skipping review for %s#%d",
            installation_id,
            repo,
            pr_num,
        )
        record_review_completed("blocked_kill_switch", 0.001, 0.0, 0)
        return

    # Idempotency check: if review for repo#pr at head_sha already completed, return cleanly
    try:
        existing = await get_review_by_sha(
            database_url=settings.database_url,
            repo=repo,
            pr=pr_num,
            head_sha=sha,
        )
        if existing is not None:
            logger.info(
                "Review for %s#%d at %s already completed (id=%d), skipping duplicate execution",
                repo,
                pr_num,
                sha,
                existing.id,
            )
            return
    except Exception as check_exc:
        logger.warning("Could not check existing review record: %s", check_exc)

    try:
        # 1. Rate-limit per installation
        if redis_client is not None:
            limiter = RateLimiter(redis_client)
            acquired = await limiter.wait_for_token(
                installation_id, poll_interval=2.0, max_wait=120.0
            )
            if not acquired:
                logger.warning(
                    "Rate limit exceeded for installation %d, deferring %s#%d",
                    installation_id,
                    repo,
                    pr_num,
                )
                return

        # 2. Authenticate and create check run
        token = await get_installation_access_token(
            installation_id=installation_id,
            settings=settings,
            redis_client=redis_client,
        )
        client = GitHubClient(token=token)

        try:
            check_run_id = await client.create_check_run(repo_full_name=repo, head_sha=sha)
            logger.info("Created check run %d for %s#%d", check_run_id, repo, pr_num)
        except Exception as exc:
            logger.warning("Could not create check run: %s", exc)
            record_github_api_error("create_check_run")

        # 3. Fetch standards to determine configuration hash
        standards_content = ""
        try:
            fetched = await client.fetch_repo_file(repo, ".prism/standards.yml", ref=sha)
            standards_content = fetched or ""
        except Exception:
            pass
        standards_file = parse_repo_standards(standards_content)
        config_hash = compute_config_hash(standards_content, settings)

        # 4. Open end-to-end ReviewTrace tagged with repo, pr, sha, config_hash
        async with ReviewTrace(
            repo=repo,
            pr=pr_num,
            head_sha=sha,
            config_hash=config_hash,
            delivery_id=delivery_id,
            settings=settings,
        ) as trace:
            # 5. Fetch and parse diff inside diff_fetch span
            async with trace.span("diff_fetch", metadata={"repo": repo, "pr": pr_num}) as diff_span:
                raw_diff = await client.fetch_pull_request_diff(repo, pr_num)
                file_diffs = parse_unified_diff(raw_diff)
                diff_span.metadata["file_count"] = len(file_diffs)

            if not file_diffs:
                logger.info("No reviewable changes in %s#%d", repo, pr_num)
                record_review_completed("neutral", 0.05, 0.0, 0)
                if check_run_id and client:
                    await client.update_check_run(
                        repo,
                        check_run_id,
                        conclusion="neutral",
                        summary="No reviewable code changes in this PR.",
                    )
                return

            # Skip trivial diffs
            if _is_trivial_diff(file_diffs):
                logger.info(
                    "Trivial diff detected for %s#%d (docs/whitespace only), skipping LLM",
                    repo,
                    pr_num,
                )
                record_review_completed("neutral", 0.05, 0.0, 0)
                if check_run_id and client:
                    await client.update_check_run(
                        repo,
                        check_run_id,
                        conclusion="neutral",
                        summary="Diff contains only documentation or whitespace changes.",
                    )
                return

            # 6. Safety check: Max cost per day budget
            budget_ok = await check_daily_budget(redis_client, settings, estimated_cost=0.03)
            if not budget_ok:
                logger.warning(
                    "Daily LLM cost budget (%.2f USD) exceeded. Disabling LLM review and posting nothing for %s#%d",
                    settings.daily_cost_budget_usd,
                    repo,
                    pr_num,
                )
                record_review_completed("budget_exceeded", 0.01, 0.0, 0)
                if check_run_id and client:
                    try:
                        await client.update_check_run(
                            repo,
                            check_run_id,
                            conclusion="neutral",
                            summary="Daily LLM review cost budget exceeded. Skipping automated review.",
                        )
                    except Exception:
                        pass
                return

            # Redact secrets from diffs before sending to LLM
            sanitized_file_diffs = redact_file_diffs(file_diffs)

            # Build context and run review pipeline (automatically traces retrieval, llm_calls, postprocess)
            pr_context = PRContext(
                repo_full_name=repo,
                pr_number=pr_num,
                head_sha=sha,
                file_diffs=sanitized_file_diffs,
                standards=standards_file.standards,
            )

            result = await review_pr(pr_context, settings=settings, redis_client=redis_client)

            # Record review cost against today's budget
            await record_daily_cost(redis_client, result.cost_usd)

            # 7. Post review to GitHub inside github_post span
            async with trace.span(
                "github_post", metadata={"repo": repo, "pr": pr_num}
            ) as post_span:
                await post_pull_request_review(
                    client=client,
                    repo_full_name=repo,
                    pr_number=pr_num,
                    head_sha=sha,
                    findings=result.findings,
                    file_diffs=file_diffs,
                )
                post_span.metadata["findings_count"] = len(result.findings)

        # 8. Persist to Postgres
        finding_dicts = [f.model_dump() for f in result.findings]
        try:
            review_id = await save_review_record(
                database_url=settings.database_url,
                repo=repo,
                pr=pr_num,
                head_sha=sha,
                model=result.model,
                prompt_version=result.prompt_version,
                status="completed",
                findings=finding_dicts,
                tokens=result.input_tokens + result.output_tokens,
                cost=result.cost_usd,
                latency_ms=result.latency_ms,
            )
            logger.info("Persisted review record id=%d for %s#%d", review_id, repo, pr_num)
        except Exception as db_exc:
            logger.error("Failed to persist review to DB: %s", db_exc)

        # 9. Update check run to completed
        num_findings = len(result.findings)
        conclusion = "success" if num_findings > 0 else "neutral"
        summary = (
            f"Found {num_findings} issue(s). "
            f"Model: {result.model}, "
            f"Cost: ${result.cost_usd:.6f}, "
            f"Latency: {result.latency_ms:.0f}ms"
        )

        if check_run_id and client:
            await client.update_check_run(
                repo,
                check_run_id,
                conclusion=conclusion,
                summary=summary,
            )

        # 10. Record Prometheus metrics
        record_review_completed(
            status="completed",
            latency_seconds=result.latency_ms / 1000.0,
            cost_usd=result.cost_usd,
            findings_count=num_findings,
        )

        # 11. Log cache stats & update Prometheus metrics
        log_cache_stats()
        c_stats = get_cache_stats()
        record_cache_stats(c_stats["cache_hits"], c_stats["cache_misses"])

        logger.info(
            "Completed review for %s#%d: %d finding(s) (cost=$%.6f, latency=%.1fms)",
            repo,
            pr_num,
            num_findings,
            result.cost_usd,
            result.latency_ms,
        )

    except Exception as exc:
        record_review_completed("failed", 0.0, 0.0, 0)
        record_github_api_error("review_pipeline")

        logger.error(
            "Review job failed for %s#%d: %s",
            repo,
            pr_num,
            exc,
            exc_info=True,
        )

        # Update check run to failure
        if check_run_id and client:
            try:
                await client.update_check_run(
                    repo,
                    check_run_id,
                    conclusion="failure",
                    summary=f"Review failed: {exc}",
                )
            except Exception:
                pass

        # Write to dead-letter table if max retries reached (or on permanent failure)
        job_try = ctx.get("job_try", 1)
        max_tries = 3
        if job_try >= max_tries:
            try:
                await save_dead_letter(
                    database_url=settings.database_url,
                    repo=repo,
                    pr=pr_num,
                    head_sha=sha,
                    delivery_id=delivery_id,
                    error_message=str(exc),
                    attempts=job_try,
                )
                logger.info("Recorded job to dead letter table after %d attempt(s)", job_try)
            except Exception as dl_exc:
                logger.error("Failed to write dead letter: %s", dl_exc)

        raise  # Re-raise so arq can track retries
    finally:
        clear_log_context()
        if client is not None:
            await client.close()


process_pull_request_review = run_review
