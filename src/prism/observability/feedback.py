"""Developer feedback loop handling comment reactions, resolutions, and commit line touches.

Records feedback in database table `feedback` and computes a live acceptance rate metric.
"""

import logging
from typing import Any

from prism.db import compute_acceptance_rate, save_feedback_record
from prism.github.client import GitHubClient
from prism.models import FileDiff
from prism.observability.metrics import record_acceptance_rate

logger = logging.getLogger(__name__)


async def record_comment_reaction(
    database_url: str,
    repo: str,
    pr: int,
    comment_id: str,
    reaction_content: str,
    author: str | None = None,
    head_sha: str | None = None,
    file_path: str | None = None,
    line_number: int | None = None,
) -> int:
    """Record a thumbs-up (+1) or thumbs-down (-1) reaction to a bot comment."""
    normalized = reaction_content.lower().strip()
    if normalized in ("+1", "thumbs_up", "up", "👍"):
        feedback_type = "reaction_up"
        feedback_value = "+1"
    elif normalized in ("-1", "thumbs_down", "down", "👎"):
        feedback_type = "reaction_down"
        feedback_value = "-1"
    else:
        feedback_type = f"reaction_{normalized}"
        feedback_value = reaction_content

    record_id = await save_feedback_record(
        database_url=database_url,
        repo=repo,
        pr=pr,
        comment_id=comment_id,
        feedback_type=feedback_type,
        feedback_value=feedback_value,
        author=author,
        head_sha=head_sha,
        file_path=file_path,
        line_number=line_number,
        details=f"Developer reaction: {reaction_content}",
    )

    logger.info(
        "Recorded comment reaction %s on %s#%d comment %s (author=%s, record_id=%d)",
        feedback_type,
        repo,
        pr,
        comment_id,
        author,
        record_id,
    )

    # Refresh live acceptance rate metric in Prometheus
    await refresh_acceptance_rate_metric(database_url)
    return record_id


async def record_thread_resolved(
    database_url: str,
    repo: str,
    pr: int,
    comment_id: str,
    author: str | None = None,
    head_sha: str | None = None,
    file_path: str | None = None,
    line_number: int | None = None,
) -> int:
    """Record that a review comment thread was marked resolved by a developer."""
    record_id = await save_feedback_record(
        database_url=database_url,
        repo=repo,
        pr=pr,
        comment_id=comment_id,
        feedback_type="resolved",
        feedback_value="resolved",
        author=author,
        head_sha=head_sha,
        file_path=file_path,
        line_number=line_number,
        details="Comment thread marked resolved",
    )

    logger.info(
        "Recorded thread resolution on %s#%d comment %s (author=%s, record_id=%d)",
        repo,
        pr,
        comment_id,
        author,
        record_id,
    )

    await refresh_acceptance_rate_metric(database_url)
    return record_id


async def record_thread_reply_feedback(
    database_url: str,
    repo: str,
    pr: int,
    comment_id: str,
    decision: str,
    is_false_positive: bool,
    developer_message: str,
    bot_reply: str,
    author: str | None = None,
    head_sha: str | None = None,
    file_path: str | None = None,
    line_number: int | None = None,
) -> int:
    """Record a multi-turn developer review conversation resolution."""
    feedback_type = "false_positive" if is_false_positive else f"thread_{decision}"
    feedback_value = "conceded" if is_false_positive else decision

    record_id = await save_feedback_record(
        database_url=database_url,
        repo=repo,
        pr=pr,
        comment_id=comment_id,
        feedback_type=feedback_type,
        feedback_value=feedback_value,
        author=author,
        head_sha=head_sha,
        file_path=file_path,
        line_number=line_number,
        details=(
            f"Developer: {developer_message[:500]}\nBot Resolution ({decision}): {bot_reply[:500]}"
        ),
    )

    logger.info(
        "Recorded thread reply feedback %s (%s) on %s#%d comment %s (author=%s, record_id=%d)",
        feedback_type,
        feedback_value,
        repo,
        pr,
        comment_id,
        author,
        record_id,
    )

    await refresh_acceptance_rate_metric(database_url)
    return record_id


async def record_line_touched(
    database_url: str,
    repo: str,
    pr: int,
    comment_id: str,
    file_path: str,
    line_number: int,
    head_sha: str,
    commit_author: str | None = None,
) -> int:
    """Record that a developer pushed a commit touching a line previously commented on."""
    record_id = await save_feedback_record(
        database_url=database_url,
        repo=repo,
        pr=pr,
        comment_id=comment_id,
        feedback_type="line_touched",
        feedback_value="touched",
        author=commit_author,
        head_sha=head_sha,
        file_path=file_path,
        line_number=line_number,
        details=f"Line modified in subsequent commit {head_sha[:8]}",
    )

    logger.info(
        "Recorded line touch feedback on %s#%d %s:%d (record_id=%d)",
        repo,
        pr,
        file_path,
        line_number,
        record_id,
    )

    await refresh_acceptance_rate_metric(database_url)
    return record_id


async def detect_touched_comment_lines(
    database_url: str,
    repo: str,
    pr: int,
    new_head_sha: str,
    file_diffs: list[FileDiff],
    prior_comments: list[dict[str, Any]],
    commit_author: str | None = None,
) -> list[int]:
    """Check if new commit diffs touch lines where bot comments were placed."""
    modified_lines: set[tuple[str, int]] = set()
    for fd in file_diffs:
        for hunk in fd.hunks:
            for line in hunk.lines:
                if line.kind in ("add", "del") and line.new_no is not None:
                    modified_lines.add((fd.path, line.new_no))
                elif line.kind in ("add", "del") and line.old_no is not None:
                    modified_lines.add((fd.path, line.old_no))

    recorded_ids: list[int] = []
    for comment in prior_comments:
        c_path = comment.get("path")
        c_line = comment.get("line") or comment.get("original_line")
        c_id = str(comment.get("id", ""))
        if not c_path or not c_line or not c_id:
            continue

        if (c_path, int(c_line)) in modified_lines:
            rid = await record_line_touched(
                database_url=database_url,
                repo=repo,
                pr=pr,
                comment_id=c_id,
                file_path=c_path,
                line_number=int(c_line),
                head_sha=new_head_sha,
                commit_author=commit_author,
            )
            recorded_ids.append(rid)

    return recorded_ids


async def poll_comment_reactions(
    client: GitHubClient,
    repo: str,
    pr: int,
    database_url: str,
) -> int:
    """Poll GitHub reactions API for comments on a pull request and record new feedback."""
    count = 0
    try:
        # Fetch PR review comments
        comments = await client.list_pull_request_comments(repo, pr)
        for comment in comments:
            c_id = str(comment.get("id"))
            reactions = comment.get("reactions", {})
            plus_ones = reactions.get("+1", 0)
            minus_ones = reactions.get("-1", 0)

            if plus_ones > 0:
                await record_comment_reaction(
                    database_url=database_url,
                    repo=repo,
                    pr=pr,
                    comment_id=c_id,
                    reaction_content="+1",
                    head_sha=comment.get("commit_id"),
                    file_path=comment.get("path"),
                    line_number=comment.get("line"),
                )
                count += 1
            if minus_ones > 0:
                await record_comment_reaction(
                    database_url=database_url,
                    repo=repo,
                    pr=pr,
                    comment_id=c_id,
                    reaction_content="-1",
                    head_sha=comment.get("commit_id"),
                    file_path=comment.get("path"),
                    line_number=comment.get("line"),
                )
                count += 1
    except Exception as exc:
        logger.warning("Error during reaction polling for %s#%d: %s", repo, pr, exc)

    return count


async def refresh_acceptance_rate_metric(database_url: str) -> float:
    """Query feedback records and update the Prometheus acceptance_rate gauge."""
    rate = await compute_acceptance_rate(database_url=database_url)
    record_acceptance_rate(rate)
    return rate
