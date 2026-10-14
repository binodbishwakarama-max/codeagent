"""Pydantic schemas for evaluation test cases, ground truth labels, and candidate outputs."""

from __future__ import annotations

from enum import StrEnum
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


class FindingSource(StrEnum):
    """Origin of a gold finding."""

    HUMAN_COMMENT = "human_comment"
    SYNTHETIC = "synthetic"


class FindingSeverity(StrEnum):
    """Severity of a gold finding — coarse-grained to ease annotation."""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    UNKNOWN = "unknown"


class FindingCategory(StrEnum):
    """Issue category of a gold finding."""

    BUG = "bug"
    SECURITY = "security"
    PERFORMANCE = "performance"
    ERROR_HANDLING = "error_handling"
    STYLE = "style"
    TESTING = "testing"
    READABILITY = "readability"
    OTHER = "other"


Split = Literal["train", "dev", "test", None]

ALLOWED_LICENSES = frozenset({"MIT", "Apache-2.0", "BSD-2-Clause", "BSD-3-Clause"})


# ---------------------------------------------------------------------------
# GoldFinding
# ---------------------------------------------------------------------------


class GoldFinding(BaseModel):
    """A single ground-truth finding derived from a human review comment."""

    model_config = ConfigDict(frozen=True)

    path: str = Field(description="Relative file path where the reviewer commented")
    line: int = Field(ge=1, description="Line number in the reviewed file (new file side)")
    category: FindingCategory = Field(
        default=FindingCategory.OTHER,
        description="Coarse issue category",
    )
    severity: FindingSeverity = Field(
        default=FindingSeverity.UNKNOWN,
        description="Estimated severity",
    )
    description: str = Field(
        min_length=1,
        description="Verbatim (or lightly cleaned) body of the human review comment",
    )
    source: FindingSource = Field(
        default=FindingSource.HUMAN_COMMENT,
        description="Whether this finding came from a human comment or is synthetic",
    )
    verified: bool = Field(
        default=False,
        description=(
            "True if the comment led to a follow-up commit touching the same file+line, "
            "confirming the reviewer identified a real issue"
        ),
    )
    reviewer: str | None = Field(
        default=None,
        description="GitHub login of the reviewer who left the comment",
    )
    thread_resolved: bool = Field(
        default=False,
        description="Whether the review thread was marked resolved on GitHub",
    )


# ---------------------------------------------------------------------------
# PRExample
# ---------------------------------------------------------------------------


class PRExample(BaseModel):
    """A single pull request example in the evaluation dataset."""

    model_config = ConfigDict(frozen=True)

    id: str = Field(description=("Stable unique ID: sha256(repo + '#' + str(pr_number))[:16] hex"))
    repo: str = Field(description="Full repository slug, e.g. 'tiangolo/fastapi'")
    language: str = Field(description="Primary programming language: 'python' or 'typescript'")
    license: str = Field(description="SPDX license identifier, e.g. 'MIT' or 'Apache-2.0'")
    pr_number: int = Field(ge=1, description="Pull request number on GitHub")
    pr_url: str = Field(description="Canonical URL of the pull request")
    base_sha: str = Field(description="SHA of the base commit (merge target)")
    head_sha: str = Field(description="SHA of the head commit reviewed")
    title: str = Field(description="Pull request title")
    body: str = Field(
        default="",
        description="Pull request description / body (may be empty)",
    )
    diff: str = Field(description="Unified diff of the PR at head_sha vs base_sha")
    changed_lines: int = Field(ge=0, description="Total added + removed lines")
    gold_findings: list[GoldFinding] = Field(
        default_factory=list,
        description="Ground-truth findings extracted from human review comments",
    )
    is_clean: bool = Field(
        default=False,
        description="True for 'clean' PRs: substantive diff, zero actionable review comments",
    )
    split: Split = Field(
        default=None,
        description="Dataset split: train/dev/test or None if unassigned",
    )


# ---------------------------------------------------------------------------
# RepoConfig — parsed from evals/configs/repos.yml
# ---------------------------------------------------------------------------


class RepoConfig(BaseModel):
    """Configuration entry for a single repository to mine."""

    model_config = ConfigDict(frozen=True)

    repo: str = Field(description="Owner/name slug")
    language: str = Field(description="Primary language: python or typescript")
    license: str = Field(description="SPDX license identifier")
    notes: str = Field(default="", description="Free-text notes about the repo")
    skip: bool = Field(default=False, description="Set to True to exclude from mining")


class ReposConfig(BaseModel):
    """Top-level config loaded from evals/configs/repos.yml."""

    model_config = ConfigDict(frozen=True)

    repos: list[RepoConfig]

    @property
    def active(self) -> list[RepoConfig]:
        """Return only non-skipped repos with allowed licenses."""
        return [r for r in self.repos if not r.skip and r.license in ALLOWED_LICENSES]
