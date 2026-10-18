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
from prism.observability.feedback import record_thread_reply_feedback
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
from prism.review.conversation import resolve_thread_conversation
from prism.review.llm import create_llm_client
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


async def run_thread_reply(ctx: dict[str, Any], job: dict[str, Any]) -> dict[str, Any]:
    """Async background worker job processing an interactive PR review comment reply."""
    settings: Settings = ctx.get("settings") or Settings()
    redis_client = ctx.get("redis")
    llm_client = ctx.get("llm_client")

    installation_id = job.get("installation_id")
    repo = str(job.get("repo_full_name", ""))
    pr_num = int(job.get("pr_number", 0))
    comment_id = int(job.get("comment_id", 0))
    in_reply_to_id = job.get("in_reply_to_id")
    author = str(job.get("author", "developer"))
    developer_message = str(job.get("body", ""))
    parent_comment_body = str(job.get("parent_body", ""))
    file_path = str(job.get("path", ""))
    diff_hunk = job.get("diff_hunk")
    line_number = job.get("line")

    bind_log_context(repo=repo, pr=pr_num)
    logger.info(
        "Processing review thread reply for %s#%d comment %d by @%s",
        repo,
        pr_num,
        comment_id,
        author,
    )

    client: GitHubClient | None = None
    try:
        if "github_client" in ctx:
            client = ctx["github_client"]
        elif "github_token" in job:
            client = GitHubClient(token=job["github_token"])
        elif installation_id:
            token = await get_installation_access_token(
                installation_id=int(installation_id),
                settings=settings,
                redis_client=redis_client,
            )
            client = GitHubClient(token=token)

        # If parent comment body was not sent directly, retrieve it from GitHub
        target_reply_id = int(in_reply_to_id) if in_reply_to_id else comment_id
        if not parent_comment_body and client and in_reply_to_id:
            try:
                parent_comment = await client.get_pull_request_comment(repo, int(in_reply_to_id))
                if parent_comment:
                    parent_comment_body = str(parent_comment.get("body", ""))
                    if not file_path:
                        file_path = str(parent_comment.get("path", ""))
                    if not diff_hunk:
                        diff_hunk = parent_comment.get("diff_hunk")
                    if not line_number:
                        line_number = parent_comment.get("line")
            except Exception as fetch_exc:
                logger.warning("Could not fetch parent comment %s: %s", in_reply_to_id, fetch_exc)

        if llm_client is None:
            llm_client = create_llm_client(settings)

        # Fetch standards if available
        standards_context: str | None = None
        if client and repo:
            try:
                standards_context = await client.fetch_repo_file(repo, ".prism/standards.yml")
            except Exception:
                standards_context = None

        result, markdown_reply = await resolve_thread_conversation(
            llm_client=llm_client,
            developer_message=developer_message,
            developer_author=author,
            parent_comment_body=parent_comment_body,
            file_path=file_path,
            diff_hunk=diff_hunk,
            line_number=line_number,
            standards_context=standards_context,
        )

        # Post reply on GitHub
        if client and repo and pr_num and target_reply_id:
            try:
                await client.reply_to_pull_request_comment(
                    repo_full_name=repo,
                    pr_number=pr_num,
                    comment_id=target_reply_id,
                    body=markdown_reply,
                )
                logger.info(
                    "Posted inline review reply on %s#%d comment %d (decision=%s)",
                    repo,
                    pr_num,
                    target_reply_id,
                    result.decision.value,
                )
            except Exception as post_exc:
                logger.error("Failed to post comment reply on %s#%d: %s", repo, pr_num, post_exc)

        # Record feedback loop in Postgres
        if settings.database_url:
            try:
                await record_thread_reply_feedback(
                    database_url=settings.database_url,
                    repo=repo,
                    pr=pr_num,
                    comment_id=str(target_reply_id),
                    decision=result.decision.value,
                    is_false_positive=result.is_false_positive,
                    developer_message=developer_message,
                    bot_reply=markdown_reply,
                    author=author,
                    file_path=file_path,
                    line_number=line_number,
                )
            except Exception as db_exc:
                logger.warning("Failed to record thread reply feedback: %s", db_exc)

        return {
            "status": "completed",
            "decision": result.decision.value,
            "is_false_positive": result.is_false_positive,
            "explanation": result.explanation,
            "reply_body": markdown_reply,
        }
    finally:
        clear_log_context()
        if client is not None and "github_client" not in ctx:
            await client.close()


process_thread_reply = run_thread_reply
