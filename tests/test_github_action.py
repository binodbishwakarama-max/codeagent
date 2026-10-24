"""Unit tests for GitHub Actions automated pull request review runner."""

from pathlib import Path

import pytest

from prism.github.action import (
    append_github_step_summary,
    generate_action_summary_markdown,
    get_action_input,
    run_action,
    write_github_output,
)
from prism.models import IssueCategory, ReviewFinding, ReviewResult, Severity


def test_get_action_input(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("INPUT_STANDARDS_FILE", ".prism/custom.yml")
    assert get_action_input("standards-file") == ".prism/custom.yml"
    assert get_action_input("missing-var", default="default-val") == "default-val"


def test_write_github_output(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    output_file = tmp_path / "output.txt"
    monkeypatch.setenv("GITHUB_OUTPUT", str(output_file))

    write_github_output("clean_pr", "true")
    write_github_output("findings_count", "0")

    content = output_file.read_text(encoding="utf-8")
    assert "clean_pr=true\n" in content
    assert "findings_count=0\n" in content


def test_append_github_step_summary(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    summary_file = tmp_path / "step_summary.md"
    monkeypatch.setenv("GITHUB_STEP_SUMMARY", str(summary_file))

    append_github_step_summary("### Test Summary Header")
    append_github_step_summary("All checks passed.")

    content = summary_file.read_text(encoding="utf-8")
    assert "### Test Summary Header\n" in content
    assert "All checks passed.\n" in content


def test_generate_action_summary_markdown_clean() -> None:
    result = ReviewResult(
        findings=[],
        model="deterministic-standards-v1",
        prompt_version="rules-engine",
        input_tokens=100,
        output_tokens=0,
        latency_ms=25.0,
        cost_usd=0.0,
    )
    md = generate_action_summary_markdown("owner/repo", 42, result, files_count=1)
    assert "All Standards Passed (Trained Silence Guarantee)" in md
    assert "STD-SEC-01" in md
    assert "owner/repo" in md
    assert "#42" in md


def test_generate_action_summary_markdown_with_findings() -> None:
    finding = ReviewFinding(
        path="src/queries.py",
        line=38,
        category=IssueCategory.SECURITY,
        severity=Severity.HIGH,
        message="SQL injection flaw detected",
        suggestion="cursor.execute(query, (user_id,))",
        standard_id="STD-SEC-01",
        confidence=0.95,
    )
    result = ReviewResult(
        findings=[finding],
        model="qwen2.5-coder-7b",
        prompt_version="v1",
        input_tokens=500,
        output_tokens=120,
        latency_ms=45.0,
        cost_usd=0.00036,
    )
    md = generate_action_summary_markdown("owner/repo", 42, result, files_count=2)
    assert "1 Actionable Finding(s) Identified" in md
    assert "STD-SEC-01" in md
    assert "```suggestion" in md
    assert "cursor.execute(query, (user_id,))" in md


@pytest.mark.asyncio
async def test_run_action_clean_local_diff(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    output_file = tmp_path / "action_output.txt"
    summary_file = tmp_path / "action_summary.md"

    monkeypatch.setenv("GITHUB_TOKEN", "mock-token")
    monkeypatch.setenv("GITHUB_OUTPUT", str(output_file))
    monkeypatch.setenv("GITHUB_STEP_SUMMARY", str(summary_file))
    monkeypatch.setenv("INPUT_STANDARDS_FILE", ".prism/standards.yml")
    monkeypatch.setenv("INPUT_FAIL_ON_SEVERITY", "none")

    exit_code = await run_action()
    assert exit_code == 0

    assert output_file.exists()
    assert summary_file.exists()
    out_text = output_file.read_text(encoding="utf-8")
    assert "review_status=" in out_text
