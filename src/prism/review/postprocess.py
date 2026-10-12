"""Review comment post-processing: line validation, deduplication,
confidence filtering, and ranking.
"""

import logging
import re
from collections.abc import Sequence

from prism.github.comments import build_valid_diff_positions
from prism.models import FileDiff, ReviewFinding, Severity

logger = logging.getLogger(__name__)

SEVERITY_WEIGHTS: dict[Severity, float] = {
    Severity.HIGH: 3.0,
    Severity.MEDIUM: 2.0,
    Severity.LOW: 1.0,
}


def _normalize_text(text: str) -> str:
    """Normalize text by lowercasing and stripping non-alphanumeric characters."""
    return re.sub(r"\W+", " ", text).strip().lower()


def postprocess_findings(
    findings: Sequence[ReviewFinding],
    file_diffs: list[FileDiff],
    confidence_threshold: float = 0.6,
    max_comments_per_pr: int = 10,
) -> list[ReviewFinding]:
    """Filter, deduplicate, validate, rank, and cap review findings.

    1. Validate that finding's (path, line, side) actually exists in changed diff lines.
    2. Drop findings with confidence score below confidence_threshold.
    3. Deduplicate near-identical messages on the same file/line or same semantic issue.
    4. Rank findings descending by (severity_weight * confidence).
    5. Cap at max_comments_per_pr highest-value comments.
    """
    if not findings:
        return []

    valid_positions = build_valid_diff_positions(file_diffs)
    candidates: list[ReviewFinding] = []

    # 1. Validation and confidence thresholding
    for f in findings:
        pos_key = (f.path, f.line, f.side)
        if pos_key not in valid_positions:
            logger.debug(
                "Dropped finding at %s:%d [%s]: line not in diff hunks",
                f.path,
                f.line,
                f.side,
            )
            continue

        if f.confidence < confidence_threshold:
            logger.debug(
                "Dropped low-confidence finding at %s:%d (confidence=%.2f < %.2f)",
                f.path,
                f.line,
                f.confidence,
                confidence_threshold,
            )
            continue

        candidates.append(f)

    # 2. Deduplication of identical or near-identical comments
    # Key on (path, line, category) and normalized message prefix
    deduped: list[ReviewFinding] = []
    seen_keys: set[str] = set()

    for f in candidates:
        norm_msg = _normalize_text(f.message)[:60]
        # Precise key: same location & same normalized message
        key_loc = f"{f.path}:{f.line}:{norm_msg}"
        # Category key: same file, same line, same category
        key_cat = f"{f.path}:{f.line}:{f.category.value}"

        if key_loc in seen_keys or key_cat in seen_keys:
            logger.debug("Deduplicated finding at %s:%d (%s)", f.path, f.line, f.message[:40])
            continue

        seen_keys.add(key_loc)
        seen_keys.add(key_cat)
        deduped.append(f)

    # 3. Sort by severity_weight * confidence descending
    def rank_score(item: ReviewFinding) -> tuple[float, float]:
        weight = SEVERITY_WEIGHTS.get(item.severity, 1.0)
        return (weight * item.confidence, item.confidence)

    deduped.sort(key=rank_score, reverse=True)

    # 4. Cap at max_comments_per_pr
    capped = deduped[:max_comments_per_pr]

    logger.info(
        "Post-processing completed: %d raw findings -> %d validated -> %d returned (cap=%d)",
        len(findings),
        len(candidates),
        len(capped),
        max_comments_per_pr,
    )
    return capped
