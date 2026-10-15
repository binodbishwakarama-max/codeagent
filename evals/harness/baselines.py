"""Baseline strategies for comparison against the LLM review pipeline.

Implements:
1. RandomBaseline: generates random findings at random positions in changed files.
2. LinterBaseline: placeholder for running ruff/eslint on changed files (stub).

These baselines demonstrate that the LLM adds value over existing tools.
"""

from __future__ import annotations

import hashlib
import logging
import random
import re

from evals.dataset.schema import PRExample
from prism.models import IssueCategory, ReviewFinding, ReviewResult, Severity

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Random Baseline
# ---------------------------------------------------------------------------


class RandomBaseline:
    """Generates random findings at random positions in changed files.

    Serves as a lower-bound baseline: any useful review system should
    significantly outperform random placement.
    """

    def __init__(
        self,
        findings_per_pr: int = 3,
        seed: int = 42,
    ) -> None:
        self.findings_per_pr = findings_per_pr
        self.seed = seed

    def review(self, example: PRExample) -> ReviewResult:
        """Generate random findings for a PR example."""
        # Use deterministic seed per PR for reproducibility
        pr_seed = int(hashlib.sha256(f"{self.seed}:{example.id}".encode()).hexdigest()[:8], 16)
        rng = random.Random(pr_seed)

        # Extract file paths and line numbers from diff
        changed_files = _extract_changed_files(example.diff)
        if not changed_files:
            return _empty_result("random-baseline")

        findings: list[ReviewFinding] = []
        categories: list[IssueCategory] = list(IssueCategory)
        severities: list[Severity] = list(Severity)

        for _ in range(self.findings_per_pr):
            file_path, lines = rng.choice(changed_files)
            line = rng.choice(lines) if lines else 1
            category: IssueCategory = rng.choice(categories)
            severity: Severity = rng.choice(severities)

            findings.append(
                ReviewFinding(
                    path=file_path,
                    line=line,
                    category=category,
                    severity=severity,
                    message=f"Random baseline finding: potential {category.value} issue",
                    confidence=rng.uniform(0.3, 0.9),
                )
            )

        return ReviewResult(
            findings=findings,
            model="random-baseline",
            prompt_version="baseline-v1",
            input_tokens=0,
            output_tokens=0,
            latency_ms=0.0,
            cost_usd=0.0,
        )


# ---------------------------------------------------------------------------
# Linter Baseline (Stub)
# ---------------------------------------------------------------------------


class LinterBaseline:
    """Baseline using static analysis linters (ruff for Python, eslint for TS).

    This is a stub implementation that simulates linter output based on
    simple pattern matching. In production, this would shell out to actual
    linter binaries.
    """

    # Common patterns that linters detect
    _PYTHON_PATTERNS: list[tuple[str, IssueCategory, str]] = [
        (r"except\s*:", IssueCategory.ERROR_HANDLING, "Bare except clause"),
        (r"import\s+\*", IssueCategory.STANDARDS_VIOLATION, "Wildcard import"),
        (r"print\(", IssueCategory.STANDARDS_VIOLATION, "Print statement in production code"),
        (r"# ?TODO", IssueCategory.READABILITY, "Unresolved TODO comment"),
        (r"pass\s*$", IssueCategory.READABILITY, "Empty pass statement"),
    ]

    _TS_PATTERNS: list[tuple[str, IssueCategory, str]] = [
        (r"\bany\b", IssueCategory.STANDARDS_VIOLATION, "Usage of 'any' type"),
        (r"console\.log\(", IssueCategory.STANDARDS_VIOLATION, "Console.log in production"),
        (r"// ?TODO", IssueCategory.READABILITY, "Unresolved TODO comment"),
        (r"var\s+", IssueCategory.STANDARDS_VIOLATION, "Use of 'var' instead of let/const"),
    ]

    def review(self, example: PRExample) -> ReviewResult:
        """Run simulated linter analysis on the PR diff."""
        patterns = self._PYTHON_PATTERNS if example.language == "python" else self._TS_PATTERNS

        findings: list[ReviewFinding] = []
        added_lines = _extract_added_lines(example.diff)

        for file_path, line_no, line_text in added_lines:
            for pattern, category, message in patterns:
                if re.search(pattern, line_text):
                    findings.append(
                        ReviewFinding(
                            path=file_path,
                            line=line_no,
                            category=category,
                            severity=Severity.LOW,
                            message=f"Linter: {message}",
                            confidence=0.85,
                        )
                    )

        return ReviewResult(
            findings=findings,
            model=f"linter-baseline-{example.language}",
            prompt_version="baseline-v1",
            input_tokens=0,
            output_tokens=0,
            latency_ms=0.0,
            cost_usd=0.0,
        )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _extract_changed_files(diff: str) -> list[tuple[str, list[int]]]:
    """Extract file paths and changed line numbers from a unified diff."""
    files: list[tuple[str, list[int]]] = []
    current_file = ""
    current_lines: list[int] = []
    line_counter = 0

    for line in diff.split("\n"):
        if line.startswith("+++ b/"):
            if current_file and current_lines:
                files.append((current_file, current_lines))
            current_file = line[6:].strip()
            current_lines = []
            line_counter = 0
        elif line.startswith("@@ "):
            # Parse new line start from @@ -old,len +new,len @@
            match = re.search(r"\+(\d+)", line)
            if match:
                line_counter = int(match.group(1))
        elif line.startswith("+") and not line.startswith("+++"):
            current_lines.append(line_counter)
            line_counter += 1
        elif line.startswith("-") and not line.startswith("---"):
            pass  # Deleted lines don't increment new line counter
        else:
            line_counter += 1

    if current_file and current_lines:
        files.append((current_file, current_lines))

    # Fallback: at least one synthetic entry
    if not files:
        files = [("src/app.py", [1, 10, 20])]

    return files


def _extract_added_lines(diff: str) -> list[tuple[str, int, str]]:
    """Extract (file_path, line_number, line_text) for all added lines."""
    result: list[tuple[str, int, str]] = []
    current_file = ""
    line_counter = 0

    for line in diff.split("\n"):
        if line.startswith("+++ b/"):
            current_file = line[6:].strip()
            line_counter = 0
        elif line.startswith("@@ "):
            match = re.search(r"\+(\d+)", line)
            if match:
                line_counter = int(match.group(1))
        elif line.startswith("+") and not line.startswith("+++"):
            result.append((current_file, line_counter, line[1:]))
            line_counter += 1
        elif line.startswith("-") and not line.startswith("---"):
            pass
        else:
            line_counter += 1

    return result


def _empty_result(model: str) -> ReviewResult:
    """Create an empty ReviewResult for PRs with no changed files."""
    return ReviewResult(
        findings=[],
        model=model,
        prompt_version="baseline-v1",
        input_tokens=0,
        output_tokens=0,
        latency_ms=0.0,
        cost_usd=0.0,
    )
