"""Labeling mined pull requests with ground-truth violations and standard categories.

This module provides utilities to:
  1. Load mined PRExamples from a JSONL file.
  2. Post-process GoldFindings: infer category/severity from comment text heuristics.
  3. Write a labeled JSONL with enriched GoldFindings.

Category inference is heuristic-only; human curators should review and override
as needed before promoting examples to the final benchmark.
"""

from __future__ import annotations

import logging
import re
from pathlib import Path

from evals.dataset.schema import FindingCategory, FindingSeverity, GoldFinding, PRExample

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Keyword patterns for category/severity inference
# ---------------------------------------------------------------------------

_CATEGORY_PATTERNS: list[tuple[FindingCategory, re.Pattern[str]]] = [
    (
        FindingCategory.SECURITY,
        re.compile(
            r"\b(sql injection|xss|csrf|injection|auth|password|secret|token|"
            r"sanitize|escape|privilege|vulnerability|exploit)\b",
            re.IGNORECASE,
        ),
    ),
    (
        FindingCategory.BUG,
        re.compile(
            r"\b(bug|crash|exception|error|fail|wrong|incorrect|broken|"
            r"off[- ]by[- ]one|race condition|deadlock|null pointer|none check)\b",
            re.IGNORECASE,
        ),
    ),
    (
        FindingCategory.PERFORMANCE,
        re.compile(
            r"\b(slow|performance|n\+1|cache|memory|leak|allocat|O\(n|O\(n²|"
            r"complexity|latency|throughput|batch)\b",
            re.IGNORECASE,
        ),
    ),
    (
        FindingCategory.ERROR_HANDLING,
        re.compile(
            r"\b(exception|error handling|catch|try[- ]except|raise|propagat|"
            r"graceful|fallback|retry|timeout)\b",
            re.IGNORECASE,
        ),
    ),
    (
        FindingCategory.TESTING,
        re.compile(
            r"\b(test|coverage|assert|mock|fixture|unit test|integration test|"
            r"missing test|should test)\b",
            re.IGNORECASE,
        ),
    ),
    (
        FindingCategory.READABILITY,
        re.compile(
            r"\b(rename|variable name|naming|readab|comment|docstring|confusing|"
            r"unclear|misleading|verbose|simplif)\b",
            re.IGNORECASE,
        ),
    ),
    (
        FindingCategory.STYLE,
        re.compile(
            r"\b(style|format|indent|whitespace|trailing|lint|pep8|pylint|"
            r"ruff|eslint|prettier|semicolon|quote)\b",
            re.IGNORECASE,
        ),
    ),
]

_SEVERITY_HIGH = re.compile(
    r"\b(security|crash|data loss|vulnerability|critical|dangerous|exploit|"
    r"sql injection|xss|must fix|blocking)\b",
    re.IGNORECASE,
)
_SEVERITY_MEDIUM = re.compile(
    r"\b(bug|incorrect|wrong|broken|fail|race|deadlock|error handling|"
    r"should fix|important)\b",
    re.IGNORECASE,
)


def _infer_category(text: str) -> FindingCategory:
    for category, pattern in _CATEGORY_PATTERNS:
        if pattern.search(text):
            return category
    return FindingCategory.OTHER


def _infer_severity(text: str) -> FindingSeverity:
    if _SEVERITY_HIGH.search(text):
        return FindingSeverity.HIGH
    if _SEVERITY_MEDIUM.search(text):
        return FindingSeverity.MEDIUM
    return FindingSeverity.LOW


def enrich_finding(finding: GoldFinding) -> GoldFinding:
    """Infer category and severity from comment text if not already set."""
    category = finding.category
    severity = finding.severity

    if category == FindingCategory.OTHER:
        category = _infer_category(finding.description)
    if severity == FindingSeverity.UNKNOWN:
        severity = _infer_severity(finding.description)

    if category == finding.category and severity == finding.severity:
        return finding

    return finding.model_copy(update={"category": category, "severity": severity})


def enrich_example(example: PRExample) -> PRExample:
    """Enrich all GoldFindings in a PRExample with inferred labels."""
    enriched_findings = [enrich_finding(f) for f in example.gold_findings]
    return example.model_copy(update={"gold_findings": enriched_findings})


# ---------------------------------------------------------------------------
# JSONL I/O helpers
# ---------------------------------------------------------------------------


def load_jsonl(path: Path) -> list[PRExample]:
    """Load PRExamples from a JSONL file."""
    examples: list[PRExample] = []
    with path.open(encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                examples.append(PRExample.model_validate_json(line))
            except Exception as exc:
                logger.warning("Skipping line %d in %s: %s", line_no, path, exc)
    return examples


def save_jsonl(examples: list[PRExample], path: Path) -> None:
    """Write PRExamples to a JSONL file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for ex in examples:
            f.write(ex.model_dump_json() + "\n")
    logger.info("Wrote %d examples to %s", len(examples), path)


def label_dataset(raw_path: Path, labeled_path: Path) -> list[PRExample]:
    """Load raw mined JSONL, enrich labels, and write labeled JSONL.

    Returns the enriched list.
    """
    examples = load_jsonl(raw_path)
    enriched = [enrich_example(ex) for ex in examples]
    save_jsonl(enriched, labeled_path)
    logger.info("Labeled %d examples → %s", len(enriched), labeled_path)
    return enriched
