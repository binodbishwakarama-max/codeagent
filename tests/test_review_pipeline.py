"""Tests for review pipeline v0: chunking, line validation, dedupe, capping, and recovery."""

import pytest

from prism.config import Settings
from prism.models import (
    DiffLine,
    FileDiff,
    Hunk,
    IssueCategory,
    ReviewFinding,
    Severity,
)
from prism.review.llm import FakeLLMClient
from prism.review.pipeline import PRContext, chunk_diffs, review_pr
from prism.review.postprocess import postprocess_findings
from prism.review.prompts import ChunkReviewResponse

# ---------------------------------------------------------------------------
# Test fixtures and helpers
# ---------------------------------------------------------------------------


def make_diff_file(path: str, lines_count: int = 5, start_line: int = 1) -> FileDiff:
    """Create a synthetic FileDiff with numbered added lines."""
    lines: list[DiffLine] = []
    for i in range(lines_count):
        line_no = start_line + i
        lines.append(
            DiffLine(
                kind="add",
                old_no=None,
                new_no=line_no,
                text=f"x_{line_no} = compute_value({line_no})",
            )
        )
    return FileDiff(
        path=path,
        language="python",
        hunks=[Hunk(old_start=1, new_start=start_line, lines=lines)],
    )


def make_finding(
    path: str = "src/main.py",
    line: int = 1,
    side: str = "RIGHT",
    category: IssueCategory = IssueCategory.BUG,
    severity: Severity = Severity.HIGH,
    message: str = "Test bug message",
    confidence: float = 0.9,
    suggestion: str | None = None,
    standard_id: str | None = None,
) -> ReviewFinding:
    return ReviewFinding(
        path=path,
        line=line,
        side=side,  # type: ignore[arg-type]
        category=category,
        severity=severity,
        message=message,
        confidence=confidence,
        suggestion=suggestion,
        standard_id=standard_id,
    )


def make_settings(**kwargs: object) -> Settings:
    defaults = dict(
        github_app_id="test-app",
        github_private_key="-----BEGIN RSA PRIVATE KEY-----\nFAKE\n-----END RSA PRIVATE KEY-----",
        github_webhook_secret="secret",
        database_url="postgresql://localhost/prism",
        redis_url="redis://localhost:6379/0",
        confidence_threshold=0.6,
        max_comments_per_pr=5,
        max_diff_tokens=100,
    )
    defaults.update(kwargs)
    return Settings(**defaults)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Chunking boundary tests
# ---------------------------------------------------------------------------


class TestChunkingBoundaries:
    def test_small_diff_fits_in_single_chunk(self) -> None:
        file1 = make_diff_file("src/a.py", lines_count=2)
        chunks = chunk_diffs([file1], max_tokens=1000)
        assert len(chunks) == 1
        assert chunks[0][0].path == "src/a.py"

    def test_multiple_files_split_when_exceeding_limit(self) -> None:
        file1 = make_diff_file("src/a.py", lines_count=20)
        file2 = make_diff_file("src/b.py", lines_count=20)
        # Small max_tokens forces them into separate chunks
        chunks = chunk_diffs([file1, file2], max_tokens=50)
        assert len(chunks) >= 2

    def test_large_single_file_splits_by_hunks(self) -> None:
        hunk1 = Hunk(
            old_start=1,
            new_start=1,
            lines=[
                DiffLine(kind="add", old_no=None, new_no=i, text=f"line_{i} " * 10)
                for i in range(1, 15)
            ],
        )
        hunk2 = Hunk(
            old_start=50,
            new_start=50,
            lines=[
                DiffLine(kind="add", old_no=None, new_no=i, text=f"line_{i} " * 10)
                for i in range(50, 65)
            ],
        )
        large_file = FileDiff(path="src/large.py", language="python", hunks=[hunk1, hunk2])

        chunks = chunk_diffs([large_file], max_tokens=50)
        assert len(chunks) >= 2

    def test_empty_diff_returns_empty_chunks(self) -> None:
        assert chunk_diffs([], max_tokens=100) == []


# ---------------------------------------------------------------------------
# Line validation tests
# ---------------------------------------------------------------------------


class TestLineValidation:
    def test_finding_on_valid_added_line_is_retained(self) -> None:
        file_diff = make_diff_file("src/main.py", lines_count=3, start_line=1)
        valid_finding = make_finding(path="src/main.py", line=2, side="RIGHT")

        result = postprocess_findings([valid_finding], [file_diff], confidence_threshold=0.5)
        assert len(result) == 1
        assert result[0].line == 2

    def test_finding_on_non_existent_line_is_dropped(self) -> None:
        file_diff = make_diff_file("src/main.py", lines_count=3, start_line=1)
        # Line 99 does not exist in the diff
        invalid_finding = make_finding(path="src/main.py", line=99, side="RIGHT")

        result = postprocess_findings([invalid_finding], [file_diff], confidence_threshold=0.5)
        assert len(result) == 0

    def test_finding_on_wrong_file_is_dropped(self) -> None:
        file_diff = make_diff_file("src/main.py", lines_count=3, start_line=1)
        invalid_finding = make_finding(path="src/other.py", line=1, side="RIGHT")

        result = postprocess_findings([invalid_finding], [file_diff], confidence_threshold=0.5)
        assert len(result) == 0


# ---------------------------------------------------------------------------
# Confidence filtering tests
# ---------------------------------------------------------------------------


class TestConfidenceFiltering:
    def test_drops_findings_below_threshold(self) -> None:
        file_diff = make_diff_file("src/main.py", lines_count=5, start_line=1)
        high = make_finding(line=1, confidence=0.85)
        medium = make_finding(line=2, confidence=0.65)
        low = make_finding(line=3, confidence=0.45)

        result = postprocess_findings(
            [high, medium, low],
            [file_diff],
            confidence_threshold=0.6,
        )
        assert len(result) == 2
        confidences = [f.confidence for f in result]
        assert 0.45 not in confidences


# ---------------------------------------------------------------------------
# Deduplication tests
# ---------------------------------------------------------------------------


class TestDeduplication:
    def test_deduplicates_near_identical_messages_on_same_line(self) -> None:
        file_diff = make_diff_file("src/main.py", lines_count=3, start_line=1)
        f1 = make_finding(line=1, message="Potential SQL injection detected here!")
        f2 = make_finding(line=1, message="Potential sql injection detected here")

        result = postprocess_findings([f1, f2], [file_diff], confidence_threshold=0.5)
        assert len(result) == 1

    def test_deduplicates_same_category_on_same_line(self) -> None:
        file_diff = make_diff_file("src/main.py", lines_count=3, start_line=1)
        f1 = make_finding(line=2, category=IssueCategory.SECURITY, message="Issue A")
        f2 = make_finding(line=2, category=IssueCategory.SECURITY, message="Issue B")

        result = postprocess_findings([f1, f2], [file_diff], confidence_threshold=0.5)
        assert len(result) == 1


# ---------------------------------------------------------------------------
# Capping and ranking tests
# ---------------------------------------------------------------------------


class TestCappingAndRanking:
    def test_ranks_by_severity_times_confidence_and_caps(self) -> None:
        file_diff = make_diff_file("src/main.py", lines_count=10, start_line=1)
        # Create 6 findings with varying severity and confidence
        f_low = make_finding(line=1, severity=Severity.LOW, confidence=0.9)  # 1 * 0.9 = 0.9
        f_med = make_finding(line=2, severity=Severity.MEDIUM, confidence=0.8)  # 2 * 0.8 = 1.6
        f_high = make_finding(line=3, severity=Severity.HIGH, confidence=0.95)  # 3 * 0.95 = 2.85
        f_med2 = make_finding(line=4, severity=Severity.MEDIUM, confidence=0.9)  # 2 * 0.9 = 1.8
        f_high2 = make_finding(line=5, severity=Severity.HIGH, confidence=0.7)  # 3 * 0.7 = 2.1

        findings = [f_low, f_med, f_high, f_med2, f_high2]
        result = postprocess_findings(
            findings,
            [file_diff],
            confidence_threshold=0.5,
            max_comments_per_pr=3,
        )

        assert len(result) == 3
        # Highest ranked first: f_high (2.85), f_high2 (2.1), f_med2 (1.8)
        assert result[0].line == 3
        assert result[1].line == 5
        assert result[2].line == 4


# ---------------------------------------------------------------------------
# Malformed output recovery in pipeline
# ---------------------------------------------------------------------------


class TestMalformedOutputRecovery:
    @pytest.mark.asyncio
    async def test_repair_retry_succeeds_after_initial_malformed_output(self) -> None:
        file_diff = make_diff_file("src/main.py", lines_count=3, start_line=1)
        pr_context = PRContext(
            repo_full_name="owner/repo",
            pr_number=1,
            title="Feature X",
            description="Testing repair",
            file_diffs=[file_diff],
        )

        valid_finding = make_finding(line=2, confidence=0.9)
        # 1st call fails with ValueError (malformed), 2nd repair call returns valid response
        fake_llm = FakeLLMClient(
            responses=[
                ValueError("Malformed JSON in LLM response"),
                ChunkReviewResponse(findings=[valid_finding]),
            ]
        )

        settings = make_settings()
        result = await review_pr(pr_context, settings=settings, llm_client=fake_llm)

        assert len(result.findings) == 1
        assert result.findings[0].line == 2
        assert fake_llm.call_count == 2

    @pytest.mark.asyncio
    async def test_repair_retry_failure_returns_empty_findings_without_raising(self) -> None:
        file_diff = make_diff_file("src/main.py", lines_count=3, start_line=1)
        pr_context = PRContext(
            repo_full_name="owner/repo",
            pr_number=1,
            title="Broken LLM",
            description="Testing full failure fallback",
            file_diffs=[file_diff],
        )

        # Both initial call and repair retry fail
        fake_llm = FakeLLMClient(
            responses=[
                RuntimeError("Initial malformed response"),
                RuntimeError("Repair prompt also failed"),
            ]
        )

        settings = make_settings()
        # Must never raise an exception!
        result = await review_pr(pr_context, settings=settings, llm_client=fake_llm)

        assert result.findings == []
        assert result.input_tokens >= 0


# ---------------------------------------------------------------------------
# End-to-end review_pr tests
# ---------------------------------------------------------------------------


class TestEndToEndReviewPR:
    @pytest.mark.asyncio
    async def test_clean_diff_returns_no_findings(self) -> None:
        file_diff = make_diff_file("src/main.py", lines_count=2, start_line=1)
        pr_context = PRContext(
            repo_full_name="owner/repo",
            pr_number=5,
            title="Clean update",
            file_diffs=[file_diff],
        )

        fake_llm = FakeLLMClient(responses=[ChunkReviewResponse(findings=[])])
        settings = make_settings()
        result = await review_pr(pr_context, settings=settings, llm_client=fake_llm)

        assert len(result.findings) == 0
        assert result.prompt_version == "v0.1"
        assert result.latency_ms >= 0.0

    @pytest.mark.asyncio
    async def test_empty_diff_skips_llm_calls(self) -> None:
        pr_context = PRContext(
            repo_full_name="owner/repo",
            pr_number=10,
            file_diffs=[],
        )
        fake_llm = FakeLLMClient()
        settings = make_settings()
        result = await review_pr(pr_context, settings=settings, llm_client=fake_llm)

        assert len(result.findings) == 0
        assert fake_llm.call_count == 0


# ---------------------------------------------------------------------------
# CLI dry-run integration test
# ---------------------------------------------------------------------------


class TestCLIDryRun:
    @pytest.mark.asyncio
    async def test_run_cli_dry_run_success(self, monkeypatch: pytest.MonkeyPatch) -> None:
        import httpx
        import respx

        from prism.review.cli import run_cli

        with respx.mock:
            diff_text = """diff --git a/src/auth.py b/src/auth.py
index abc..def 100644
--- a/src/auth.py
+++ b/src/auth.py
@@ -1,2 +1,3 @@
 def auth():
+    token = request.headers.get("Authorization")
     return token
"""
            respx.get("https://api.github.com/repos/acme/proj/pulls/101").mock(
                return_value=httpx.Response(
                    200,
                    json={
                        "title": "Refactor auth",
                        "body": "Improve token validation",
                        "head": {"sha": "sha999"},
                    },
                )
            )

            respx.post("https://api.anthropic.com/v1/messages").mock(
                return_value=httpx.Response(
                    200,
                    json={
                        "content": [
                            {
                                "type": "tool_use",
                                "name": "ChunkReviewResponse",
                                "input": {"findings": []},
                            }
                        ],
                        "usage": {"input_tokens": 120, "output_tokens": 20},
                    },
                )
            )

            monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
            monkeypatch.setenv("GITHUB_APP_ID", "test-id")
            monkeypatch.setenv("GITHUB_PRIVATE_KEY", "test-key")
            monkeypatch.setenv("GITHUB_WEBHOOK_SECRET", "test-secret")
            monkeypatch.setenv("DATABASE_URL", "postgresql://localhost/prism")
            monkeypatch.setenv("REDIS_URL", "redis://localhost:6379/0")

            async def mock_fetch(*args: object, **kwargs: object) -> tuple[dict[str, str], str]:
                return (
                    {"title": "Refactor auth", "body": "Notes", "sha": "sha999"},
                    diff_text,
                )

            monkeypatch.setattr(
                "prism.review.cli.fetch_pr_details_and_diff",
                mock_fetch,
            )

            exit_code = await run_cli(
                repo="acme/proj",
                pr_number=101,
                dry_run=True,
                token=None,
            )
            assert exit_code == 0
