"""Unit tests for Pydantic v2 domain models and validation edge cases."""

from pathlib import Path

import pytest
from pydantic import ValidationError

from prism.models import (
    DiffLine,
    FileDiff,
    Hunk,
    IssueCategory,
    ReviewFinding,
    ReviewResult,
    Severity,
    Standard,
    StandardsFile,
)


def test_review_finding_valid() -> None:
    """A valid review finding instantiates cleanly."""
    finding = ReviewFinding(
        path="src/main.py",
        line=42,
        side="RIGHT",
        category=IssueCategory.SECURITY,
        severity=Severity.HIGH,
        message="SQL injection vulnerability detected.",
        suggestion="cursor.execute('SELECT * FROM users WHERE id = %s', (user_id,))",
        confidence=0.95,
        standard_id="SEC-001",
    )
    assert finding.line == 42
    assert finding.confidence == 0.95
    assert finding.category == IssueCategory.SECURITY
    assert finding.severity == Severity.HIGH


def test_review_finding_confidence_out_of_range_high() -> None:
    """Confidence > 1.0 must raise ValidationError."""
    with pytest.raises(ValidationError) as exc_info:
        ReviewFinding(
            path="src/main.py",
            line=10,
            category=IssueCategory.BUG,
            severity=Severity.LOW,
            message="Potential bug.",
            confidence=1.05,
        )
    assert "confidence" in str(exc_info.value)


def test_review_finding_confidence_out_of_range_low() -> None:
    """Confidence < 0.0 must raise ValidationError."""
    with pytest.raises(ValidationError) as exc_info:
        ReviewFinding(
            path="src/main.py",
            line=10,
            category=IssueCategory.BUG,
            severity=Severity.LOW,
            message="Potential bug.",
            confidence=-0.1,
        )
    assert "confidence" in str(exc_info.value)


def test_review_finding_line_less_than_one() -> None:
    """Line < 1 must raise ValidationError."""
    with pytest.raises(ValidationError) as exc_info:
        ReviewFinding(
            path="src/main.py",
            line=0,
            category=IssueCategory.BUG,
            severity=Severity.LOW,
            message="Line must be positive.",
            confidence=0.8,
        )
    assert "line" in str(exc_info.value)

    with pytest.raises(ValidationError) as exc_info_neg:
        ReviewFinding(
            path="src/main.py",
            line=-5,
            category=IssueCategory.BUG,
            severity=Severity.LOW,
            message="Negative line number.",
            confidence=0.8,
        )
    assert "line" in str(exc_info_neg.value)


def test_review_finding_empty_message() -> None:
    """Empty message must raise ValidationError."""
    with pytest.raises(ValidationError) as exc_info:
        ReviewFinding(
            path="src/main.py",
            line=10,
            category=IssueCategory.BUG,
            severity=Severity.LOW,
            message="",
            confidence=0.8,
        )
    assert "message" in str(exc_info.value)


def test_review_finding_whitespace_message() -> None:
    """Whitespace-only message must raise ValidationError."""
    with pytest.raises(ValidationError) as exc_info:
        ReviewFinding(
            path="src/main.py",
            line=10,
            category=IssueCategory.BUG,
            severity=Severity.LOW,
            message="    \t \n   ",
            confidence=0.8,
        )
    assert "message" in str(exc_info.value)
    assert "whitespace" in str(exc_info.value)


def test_review_finding_immutability() -> None:
    """Models must be frozen and reject mutation."""
    finding = ReviewFinding(
        path="src/main.py",
        line=10,
        category=IssueCategory.BUG,
        severity=Severity.LOW,
        message="Valid message",
        confidence=0.8,
    )
    with pytest.raises(ValidationError):
        # Trying to mutate frozen Pydantic model
        finding.line = 20


def test_diff_models_structure() -> None:
    """DiffLine, Hunk, and FileDiff construct and validate correctly."""
    line1 = DiffLine(kind="ctx", old_no=1, new_no=1, text="def foo():")
    line2 = DiffLine(kind="del", old_no=2, new_no=None, text="-    return 1")
    line3 = DiffLine(kind="add", old_no=None, new_no=2, text="+    return 2")

    hunk = Hunk(old_start=1, new_start=1, lines=[line1, line2, line3])
    file_diff = FileDiff(path="foo.py", language="python", hunks=[hunk])

    assert file_diff.path == "foo.py"
    assert len(file_diff.hunks[0].lines) == 3
    assert file_diff.hunks[0].lines[2].kind == "add"


def test_standards_file_from_yaml_dict() -> None:
    """StandardsFile parses dict format with 'standards' key."""
    yaml_content = """
    standards:
      - id: "ERR-001"
        title: "Explicit Exception Handling"
        description: "Avoid bare except statements."
        good_example: "except ValueError as e:"
        bad_example: "except:"
    """
    standards_file = StandardsFile.from_yaml(yaml_content)
    assert len(standards_file.standards) == 1
    std = standards_file.standards[0]
    assert std.id == "ERR-001"
    assert std.title == "Explicit Exception Handling"
    assert std.good_example == "except ValueError as e:"


def test_standards_file_from_yaml_list() -> None:
    """StandardsFile parses direct list format."""
    yaml_content = """
    - id: "PERF-001"
      title: "Use List Comprehensions"
      description: "Prefer list comprehensions over manual appends in tight loops."
    """
    standards_file = StandardsFile.from_yaml(yaml_content)
    assert len(standards_file.standards) == 1
    assert standards_file.standards[0].id == "PERF-001"


def test_standards_file_from_file(tmp_path: Path) -> None:
    """StandardsFile loads from file path."""
    yaml_path = tmp_path / "standards.yml"
    yaml_path.write_text(
        """
        standards:
          - id: "SEC-002"
            title: "Hash Passwords"
            description: "Always hash passwords using bcrypt/argon2."
        """,
        encoding="utf-8",
    )
    standards_file = StandardsFile.from_file(yaml_path)
    assert len(standards_file.standards) == 1
    assert standards_file.standards[0].id == "SEC-002"


def test_standard_model() -> None:
    """Standard model constructs and validates properly."""
    std = Standard(
        id="SEC-001",
        title="No Hardcoded Secrets",
        description="Never hardcode passwords or API keys.",
        good_example="key = os.getenv('API_KEY')",
        bad_example="key = 'sk-123456'",
    )
    assert std.id == "SEC-001"
    assert std.good_example is not None
    with pytest.raises(ValidationError):
        std.title = "New Title"


def test_review_result_telemetry_validation() -> None:
    """ReviewResult validates non-negative metrics."""
    result = ReviewResult(
        findings=[],
        model="claude-3-5-sonnet",
        prompt_version="v1.0",
        input_tokens=1500,
        output_tokens=300,
        latency_ms=1240.5,
        cost_usd=0.008,
    )
    assert result.input_tokens == 1500
    assert result.latency_ms == 1240.5

    with pytest.raises(ValidationError):
        ReviewResult(
            findings=[],
            model="claude-3-5-sonnet",
            prompt_version="v1.0",
            input_tokens=-1,
            output_tokens=300,
            latency_ms=100.0,
            cost_usd=0.01,
        )
