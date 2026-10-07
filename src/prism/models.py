"""Pydantic v2 domain models for Prism Review.

Includes diff models, review findings, evaluation metrics, and engineering standards.
"""

from enum import StrEnum
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator


class IssueCategory(StrEnum):
    """Classification of code issues detected during review."""

    BUG = "BUG"
    SECURITY = "SECURITY"
    PERFORMANCE = "PERFORMANCE"
    ERROR_HANDLING = "ERROR_HANDLING"
    STANDARDS_VIOLATION = "STANDARDS_VIOLATION"
    TESTING = "TESTING"
    READABILITY = "READABILITY"


class Severity(StrEnum):
    """Severity level of an identified review finding."""

    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"


class DiffLine(BaseModel):
    """A single line within a git diff hunk."""

    model_config = ConfigDict(frozen=True)

    kind: Literal["add", "del", "ctx"]
    old_no: int | None = None
    new_no: int | None = None
    text: str


class Hunk(BaseModel):
    """A unified diff hunk consisting of changes and surrounding context."""

    model_config = ConfigDict(frozen=True)

    old_start: int
    new_start: int
    lines: list[DiffLine] = Field(default_factory=list)


class FileDiff(BaseModel):
    """Structured representation of changes to a single file."""

    model_config = ConfigDict(frozen=True)

    path: str
    language: str
    hunks: list[Hunk] = Field(default_factory=list)


class ReviewFinding(BaseModel):
    """An individual actionable review finding or suggestion."""

    model_config = ConfigDict(frozen=True)

    path: str = Field(description="Relative file path in the repository")
    line: int = Field(ge=1, description="Line number in the diff (must be >= 1)")
    side: Literal["LEFT", "RIGHT"] = Field(
        default="RIGHT",
        description="Diff side for inline comment (LEFT for deleted lines, RIGHT for added)",
    )
    category: IssueCategory = Field(description="Issue category classification")
    severity: Severity = Field(description="Severity impact of the finding")
    message: str = Field(min_length=1, description="Explanatory comment text")
    suggestion: str | None = Field(default=None, description="Suggested code replacement")
    confidence: float = Field(
        ge=0.0,
        le=1.0,
        description="Model confidence score between 0.0 and 1.0 inclusive",
    )
    standard_id: str | None = Field(
        default=None,
        description="Associated engineering standard rule ID",
    )

    @field_validator("message")
    @classmethod
    def validate_message_not_whitespace(cls, v: str) -> str:
        """Reject empty or whitespace-only messages."""
        stripped = v.strip()
        if not stripped:
            raise ValueError("Review finding message cannot be empty or whitespace only")
        return stripped


class ReviewResult(BaseModel):
    """Aggregate review execution result and telemetry metadata."""

    model_config = ConfigDict(frozen=True)

    findings: list[ReviewFinding] = Field(default_factory=list)
    model: str = Field(description="Name/identifier of the model that performed the review")
    prompt_version: str = Field(description="Version of prompt template utilized")
    input_tokens: int = Field(ge=0, description="Number of prompt tokens consumed")
    output_tokens: int = Field(ge=0, description="Number of completion tokens generated")
    latency_ms: float = Field(ge=0.0, description="End-to-end execution latency in milliseconds")
    cost_usd: float = Field(ge=0.0, description="Estimated total execution cost in USD")


class Standard(BaseModel):
    """An individual engineering standard or guideline definition."""

    model_config = ConfigDict(frozen=True)

    id: str = Field(description="Unique rule identifier (e.g. SEC-001)")
    title: str = Field(description="Concise summary title of the standard")
    description: str = Field(description="Detailed explanation of rationale and requirements")
    good_example: str | None = Field(default=None, description="Exemplary code compliant with rule")
    bad_example: str | None = Field(default=None, description="Counter-example violating the rule")


class StandardsFile(BaseModel):
    """Engineering standards collection loaded from `.prism/standards.yml`."""

    model_config = ConfigDict(frozen=True)

    standards: list[Standard] = Field(default_factory=list)

    @classmethod
    def from_yaml(cls, content: str) -> "StandardsFile":
        """Parse standards from YAML string content.

        Supports either a top-level mapping with a `standards` key or a direct list of standards.
        """
        parsed = yaml.safe_load(content)
        if parsed is None:
            return cls(standards=[])
        if isinstance(parsed, list):
            return cls(standards=[Standard.model_validate(item) for item in parsed])
        if isinstance(parsed, dict):
            raw_list = parsed.get("standards", [])
            if not isinstance(raw_list, list):
                raise ValueError("Expected 'standards' in YAML file to be a list")
            return cls(standards=[Standard.model_validate(item) for item in raw_list])
        raise ValueError("Invalid YAML structure for standards file")

    @classmethod
    def from_file(cls, path: Path | str) -> "StandardsFile":
        """Load and parse standards from a filesystem path."""
        p = Path(path)
        content = p.read_text(encoding="utf-8")
        return cls.from_yaml(content)


class ReviewComment(BaseModel):
    """Inline review comment schema to be posted to a pull request."""

    model_config = ConfigDict(frozen=True)

    path: str = Field(description="File path relative to repository root")
    line: int = Field(ge=1, description="Line number in the diff to comment on")
    body: str = Field(min_length=1, description="Markdown content of the review comment")
    rule_id: str | None = Field(default=None, description="Identifier of the violated standard")


class PullRequestMetadata(BaseModel):
    """Metadata describing a GitHub pull request event."""

    model_config = ConfigDict(frozen=True)

    repo_owner: str
    repo_name: str
    pull_number: int = Field(ge=1)
    commit_sha: str
    base_sha: str
