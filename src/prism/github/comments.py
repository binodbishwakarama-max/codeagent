"""Post pull request reviews with inline comments mapped to diff positions."""

import logging
from typing import Any

from prism.github.client import GitHubClient
from prism.models import FileDiff, ReviewFinding

logger = logging.getLogger(__name__)

PRISM_REVIEW_MARKER = "<!-- prism-review -->"


def build_valid_diff_positions(file_diffs: list[FileDiff]) -> set[tuple[str, int, str]]:
    """Build a lookup set of (path, line_number, side) tuples existing in the diff.

    side is 'RIGHT' for additions/context lines (new line numbers),
    and 'LEFT' for deletions (old line numbers).
    """
    valid_positions: set[tuple[str, int, str]] = set()

    for file_diff in file_diffs:
        for hunk in file_diff.hunks:
            for line in hunk.lines:
                if line.new_no is not None:
                    valid_positions.add((file_diff.path, line.new_no, "RIGHT"))
                if line.old_no is not None:
                    valid_positions.add((file_diff.path, line.old_no, "LEFT"))

    return valid_positions


def format_inline_comment(finding: ReviewFinding) -> str:
    """Format inline review finding markdown with category, message, suggestion, and rule info."""
    lines: list[str] = [
        f"**[{finding.category.value}]** {finding.message}",
    ]

    if finding.suggestion:
        lines.append("")
        lines.append("```suggestion")
        lines.append(finding.suggestion)
        lines.append("```")

    rule_meta: list[str] = []
    if finding.standard_id:
        rule_meta.append(f"Standard: `{finding.standard_id}`")
    rule_meta.append(f"Confidence: {int(finding.confidence * 100)}%")

    lines.append("")
    lines.append(f"> *{(' | '.join(rule_meta))}*")

    return "\n".join(lines)


def format_review_summary(
    findings: list[ReviewFinding],
    dropped_count: int = 0,
) -> str:
    """Generate markdown summary for the overall pull request review body."""
    lines = [
        f"{PRISM_REVIEW_MARKER}",
        "## Prism Automated Engineering Review",
        "",
    ]

    if not findings:
        lines.append(
            "No violations or suggestions identified against engineering standards. "
            "Looks good to merge!"
        )
        return "\n".join(lines)

    lines.append(
        f"Identified **{len(findings)}** actionable findings across the pull request changes:"
    )
    lines.append("")

    # Summary table by category
    categories: dict[str, int] = {}
    for f in findings:
        cat = f.category.value
        categories[cat] = categories.get(cat, 0) + 1

    lines.append("| Category | Count |")
    lines.append("| :--- | :---: |")
    for cat, count in sorted(categories.items()):
        lines.append(f"| {cat} | {count} |")
    lines.append("")

    if dropped_count > 0:
        lines.append(
            f"> *Note: {dropped_count} findings fell outside changed diff lines "
            "and were filtered out.*"
        )

    return "\n".join(lines)


async def dismiss_previous_bot_reviews(
    client: GitHubClient,
    repo_full_name: str,
    pr_number: int,
) -> None:
    """Dismiss or resolve prior Prism bot reviews so comments do not pile up on updates."""
    try:
        existing_reviews = await client.list_pull_request_reviews(repo_full_name, pr_number)
        for review in existing_reviews:
            review_id = review.get("id")
            body = review.get("body", "")
            state = review.get("state", "")
            if not review_id or state == "DISMISSED":
                continue

            # Check if this review was posted by Prism
            if PRISM_REVIEW_MARKER in body:
                logger.info(
                    "Dismissing older Prism review %s on %s#%d",
                    review_id,
                    repo_full_name,
                    pr_number,
                )
                await client.dismiss_pull_request_review(
                    repo_full_name=repo_full_name,
                    pr_number=pr_number,
                    review_id=int(review_id),
                    message="Superseded by updated commit review.",
                )
    except Exception as exc:
        logger.warning(
            "Error dismissing previous reviews on %s#%d: %s", repo_full_name, pr_number, exc
        )


async def post_pull_request_review(
    client: GitHubClient,
    repo_full_name: str,
    pr_number: int,
    head_sha: str,
    findings: list[ReviewFinding],
    file_diffs: list[FileDiff],
) -> dict[str, Any]:
    """Validate finding positions against diff and submit a single PR review."""
    # 1. Clean up old reviews on re-runs
    await dismiss_previous_bot_reviews(client, repo_full_name, pr_number)

    # 2. Map and validate inline comment positions against diff
    valid_positions = build_valid_diff_positions(file_diffs)
    postable_comments: list[dict[str, Any]] = []
    valid_findings: list[ReviewFinding] = []
    dropped_count = 0

    for finding in findings:
        pos_key = (finding.path, finding.line, finding.side)
        if pos_key not in valid_positions:
            logger.warning(
                "Dropping finding at %s:%d (side: %s): line not present in diff hunks",
                finding.path,
                finding.line,
                finding.side,
            )
            dropped_count += 1
            continue

        valid_findings.append(finding)
        postable_comments.append(
            {
                "path": finding.path,
                "line": finding.line,
                "side": finding.side,
                "body": format_inline_comment(finding),
            }
        )

    # 3. Format overall review body
    summary_body = format_review_summary(valid_findings, dropped_count=dropped_count)

    # 4. Submit review via API
    return await client.create_pull_request_review(
        repo_full_name=repo_full_name,
        pr_number=pr_number,
        commit_id=head_sha,
        body=summary_body,
        comments=postable_comments,
        event="COMMENT",
    )
