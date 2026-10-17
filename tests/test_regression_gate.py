"""Unit tests for the CI regression evaluation gate."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from evals.harness.regression_gate import (
    DEFAULT_BASELINE_PATH,
    DEFAULT_CONFIG_PATH,
    STICKY_TAG,
    parse_pr_justification,
    run_regression_gate,
    update_baseline_file,
)


def test_baseline_file_exists_and_valid() -> None:
    """Verify evals/baselines/main.json exists with required fields."""
    assert DEFAULT_BASELINE_PATH.exists(), "evals/baselines/main.json must exist"
    data = json.loads(DEFAULT_BASELINE_PATH.read_text(encoding="utf-8"))
    assert data["name"] == "main-baseline"
    assert data["total_prs"] == 40
    assert "metrics" in data
    assert "f1" in data["metrics"]
    assert "false_positive_rate" in data["metrics"]
    assert "predictions" in data
    assert len(data["predictions"]) == 40


def test_regression_config_valid() -> None:
    """Verify evals/configs/regression.yaml is valid."""
    assert DEFAULT_CONFIG_PATH.exists(), "evals/configs/regression.yaml must exist"
    from evals.harness.config import ExperimentConfig

    config = ExperimentConfig.from_yaml(DEFAULT_CONFIG_PATH)
    assert config.name == "regression-gate"
    assert config.dataset.sample_limit == 40


def test_parse_pr_justification() -> None:
    """Verify PR body justification parsing."""
    valid_body = (
        "## Description\nFixes bug.\n\n"
        "## Eval Override Justification\n"
        "Removing stylistic comments to reduce reviewer cognitive load on clean PRs."
    )
    ok, reason = parse_pr_justification(valid_body)
    assert ok is True
    assert "Removing stylistic comments" in reason

    # Empty body
    ok_empty, _ = parse_pr_justification("")
    assert ok_empty is False

    # Template placeholder only
    template_body = "## Eval Override Justification\n[Explain why this regression is acceptable]"
    ok_tpl, _ = parse_pr_justification(template_body)
    assert ok_tpl is False


def test_branch_check_on_update_baseline(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify update_baseline aborts on non-main branch without force."""
    monkeypatch.setattr(
        "evals.harness.regression_gate.get_current_git_branch", lambda: "feature/test-branch"
    )
    ret = update_baseline_file(force=False)
    assert ret == 1


def test_regression_gate_passes_on_baseline(tmp_path: Path) -> None:
    """Verify regression gate passes on the default code state."""
    comment_file = tmp_path / "comment.md"
    ret = run_regression_gate(
        config_path=DEFAULT_CONFIG_PATH,
        baseline_path=DEFAULT_BASELINE_PATH,
        split="dev",
        output_comment=comment_file,
    )
    assert ret == 0
    assert comment_file.exists()
    content = comment_file.read_text(encoding="utf-8")
    assert STICKY_TAG in content
    assert "PASSED" in content
    assert "F1 Score" in content
    assert "False Positive Rate (clean)" in content
    assert "Cost / Review" in content
    assert "Latency (p50)" in content


def test_regression_gate_fails_on_simulated_f1_drop(tmp_path: Path) -> None:
    """Verify gate fails when candidate F1 drops by > 3 points."""
    comment_file = tmp_path / "comment.md"
    # Setting an impossibly strict threshold of 0.0000001 with candidate simulation
    # or by altering baseline expectation
    ret = run_regression_gate(
        config_path=DEFAULT_CONFIG_PATH,
        baseline_path=DEFAULT_BASELINE_PATH,
        split="dev",
        output_comment=comment_file,
        f1_drop_threshold=-0.01,  # Negative threshold forces failure if delta < 0.01
    )
    assert ret == 1
    content = comment_file.read_text(encoding="utf-8")
    assert "FAILED" in content


def test_regression_gate_override_with_justification(tmp_path: Path) -> None:
    """Verify regression gate passes when eval-override label and justification are present."""
    comment_file = tmp_path / "comment.md"
    body = (
        "## Eval Override Justification\n"
        "Accepting lower F1 score due to removal of subjective readability lint rules."
    )
    ret = run_regression_gate(
        config_path=DEFAULT_CONFIG_PATH,
        baseline_path=DEFAULT_BASELINE_PATH,
        split="dev",
        output_comment=comment_file,
        f1_drop_threshold=-0.01,  # Force regression
        pr_labels=["eval-override"],
        pr_body=body,
    )
    assert ret == 0
    content = comment_file.read_text(encoding="utf-8")
    assert "OVERRIDDEN" in content
    assert "Accepting lower F1 score" in content


def test_regression_gate_override_fails_without_justification(tmp_path: Path) -> None:
    """Verify regression gate fails if eval-override label is present but body is empty."""
    comment_file = tmp_path / "comment.md"
    ret = run_regression_gate(
        config_path=DEFAULT_CONFIG_PATH,
        baseline_path=DEFAULT_BASELINE_PATH,
        split="dev",
        output_comment=comment_file,
        f1_drop_threshold=-0.01,  # Force regression
        pr_labels=["eval-override"],
        pr_body="",  # Missing justification
    )
    assert ret == 1
    content = comment_file.read_text(encoding="utf-8")
    assert "Override Incomplete" in content
