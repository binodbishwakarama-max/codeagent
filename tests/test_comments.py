"""Tests for comments.py: position mapping and review building."""

from prism.github.comments import (
    PRISM_REVIEW_MARKER,
    build_valid_diff_positions,
    format_inline_comment,
    format_review_summary,
)
from prism.models import DiffLine, FileDiff, Hunk, IssueCategory, ReviewFinding, Severity


def make_finding(**kwargs: object) -> ReviewFinding:
    defaults = dict(
        path="src/main.py",
        line=2,
        side="RIGHT",
        category=IssueCategory.SECURITY,
        severity=Severity.HIGH,
        message="SQL injection risk.",
        confidence=0.9,
    )
    defaults.update(kwargs)
    return ReviewFinding(**defaults)  # type: ignore[arg-type]


def make_file_diff() -> FileDiff:
    """Create a simple FileDiff with one hunk covering lines 1-5 (new) and 1-4 (old)."""
    return FileDiff(
        path="src/main.py",
        language="python",
        hunks=[
            Hunk(
                old_start=1,
                new_start=1,
                lines=[
                    DiffLine(kind="ctx", old_no=1, new_no=1, text="import os"),
                    DiffLine(kind="del", old_no=2, new_no=None, text="x = 1"),
                    DiffLine(kind="add", old_no=None, new_no=2, text="x = 2"),
                    DiffLine(kind="add", old_no=None, new_no=3, text="y = 3"),
                    DiffLine(kind="ctx", old_no=3, new_no=4, text=""),
                ],
            )
        ],
    )


class TestBuildValidDiffPositions:
    def test_add_line_creates_right_position(self) -> None:
        file_diff = make_file_diff()
        positions = build_valid_diff_positions([file_diff])
        assert ("src/main.py", 2, "RIGHT") in positions

    def test_del_line_creates_left_position(self) -> None:
        file_diff = make_file_diff()
        positions = build_valid_diff_positions([file_diff])
        assert ("src/main.py", 2, "LEFT") in positions

    def test_ctx_line_creates_both_positions(self) -> None:
        file_diff = make_file_diff()
        positions = build_valid_diff_positions([file_diff])
        assert ("src/main.py", 1, "RIGHT") in positions
        assert ("src/main.py", 1, "LEFT") in positions

    def test_line_not_in_diff_is_absent(self) -> None:
        file_diff = make_file_diff()
        positions = build_valid_diff_positions([file_diff])
        assert ("src/main.py", 99, "RIGHT") not in positions

    def test_different_file_not_included(self) -> None:
        file_diff = make_file_diff()
        positions = build_valid_diff_positions([file_diff])
        assert ("src/other.py", 2, "RIGHT") not in positions

    def test_empty_file_diffs_returns_empty_set(self) -> None:
        assert build_valid_diff_positions([]) == set()


class TestFormatInlineComment:
    def test_includes_category_and_message(self) -> None:
        finding = make_finding(
            category=IssueCategory.SECURITY,
            message="SQL injection risk.",
        )
        body = format_inline_comment(finding)
        assert "SECURITY" in body
        assert "SQL injection risk." in body

    def test_includes_suggestion_block(self) -> None:
        finding = make_finding(suggestion="cursor.execute(query, (param,))")
        body = format_inline_comment(finding)
        assert "```suggestion" in body
        assert "cursor.execute(query, (param,))" in body

    def test_includes_standard_id_when_present(self) -> None:
        finding = make_finding(standard_id="SEC-002")
        body = format_inline_comment(finding)
        assert "SEC-002" in body

    def test_includes_confidence_percentage(self) -> None:
        finding = make_finding(confidence=0.85)
        body = format_inline_comment(finding)
        assert "85%" in body

    def test_no_suggestion_when_none(self) -> None:
        finding = make_finding(suggestion=None)
        body = format_inline_comment(finding)
        assert "```suggestion" not in body


class TestFormatReviewSummary:
    def test_contains_prism_marker(self) -> None:
        body = format_review_summary([])
        assert PRISM_REVIEW_MARKER in body

    def test_no_findings_shows_no_violations_message(self) -> None:
        body = format_review_summary([])
        assert "No violations" in body or "no" in body.lower()

    def test_with_findings_shows_count(self) -> None:
        findings = [
            make_finding(category=IssueCategory.SECURITY),
            make_finding(category=IssueCategory.BUG, line=3),
        ]
        body = format_review_summary(findings)
        assert "2" in body

    def test_with_dropped_count_shows_note(self) -> None:
        findings = [make_finding()]
        body = format_review_summary(findings, dropped_count=3)
        assert "3" in body

    def test_categories_listed_in_table(self) -> None:
        findings = [make_finding(category=IssueCategory.PERFORMANCE)]
        body = format_review_summary(findings)
        assert "PERFORMANCE" in body
