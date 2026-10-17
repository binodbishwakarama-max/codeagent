"""Unit and integration tests for evals/dataset/label.py and verify_app.py."""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest

from evals.dataset.label import (
    CommentLabel,
    DeterministicMockAnnotator,
    DualAnnotatorPipeline,
    PRLabelResponse,
    compute_agreement,
    format_pr_user_prompt,
    generate_synthetic_pr_dataset,
    run_labeling_pipeline,
)
from evals.dataset.schema import (
    FindingCategory,
    FindingSeverity,
    GoldFinding,
    PRExample,
)
from evals.dataset.verify_app import is_pr_fully_verified, load_dataset_cached, save_current_dataset

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def sample_finding() -> GoldFinding:
    return GoldFinding(
        path="src/auth.py",
        line=42,
        category=FindingCategory.SECURITY,
        severity=FindingSeverity.HIGH,
        description="Potential SQL injection vulnerability with unescaped string concatenation.",
        verified=False,
    )


@pytest.fixture
def sample_example(sample_finding: GoldFinding) -> PRExample:
    return PRExample(
        id="1234567890abcdef",
        repo="testorg/testrepo",
        language="python",
        license="MIT",
        pr_number=101,
        pr_url="https://github.com/testorg/testrepo/pull/101",
        base_sha="base123",
        head_sha="head456",
        title="Add user query handling",
        body="Handles user parameters in SQL queries",
        diff=(
            "--- a/src/auth.py\n+++ b/src/auth.py\n@@ -40,3 +40,5 @@\n"
            "+query = f'SELECT * FROM users WHERE id={user_id}'\n"
        ),
        changed_lines=5,
        gold_findings=[sample_finding],
        is_clean=False,
        split=None,
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestLabelSchemas:
    def test_comment_label_validation(self) -> None:
        label = CommentLabel(
            comment_index=0,
            actionable=True,
            category=FindingCategory.BUG,
            severity=FindingSeverity.HIGH,
            description="Fix null pointer exception when user is None.",
            reasoning="Identifies concrete crash condition.",
        )
        assert label.comment_index == 0
        assert label.actionable is True
        assert label.category == FindingCategory.BUG
        assert label.severity == FindingSeverity.HIGH

    def test_pr_label_response_serialization(self) -> None:
        resp = PRLabelResponse(
            labels=[
                CommentLabel(
                    comment_index=0,
                    actionable=False,
                    reasoning="Non-actionable praise",
                )
            ]
        )
        data = resp.model_dump()
        assert len(data["labels"]) == 1
        assert data["labels"][0]["actionable"] is False


class TestPromptFormatting:
    def test_format_pr_user_prompt(self, sample_example: PRExample) -> None:
        prompt = format_pr_user_prompt(sample_example)
        assert "testorg/testrepo #101" in prompt
        assert "UNIFIED DIFF" in prompt
        assert "HUMAN REVIEW COMMENTS" in prompt
        assert "src/auth.py" in prompt

    def test_format_pr_user_prompt_truncation(self, sample_example: PRExample) -> None:
        huge_diff = "a" * 30_000
        ex = sample_example.model_copy(update={"diff": huge_diff})
        prompt = format_pr_user_prompt(ex)
        assert "... [diff truncated for context length]" in prompt


class TestAgreementLogic:
    def test_agreement_retains_agreed_category(self, sample_finding: GoldFinding) -> None:
        labels_a = [
            CommentLabel(
                comment_index=0,
                actionable=True,
                category=FindingCategory.SECURITY,
                severity=FindingSeverity.HIGH,
                description="Sanitize user_id parameter before query interpolation.",
            )
        ]
        labels_b = [
            CommentLabel(
                comment_index=0,
                actionable=True,
                category=FindingCategory.SECURITY,
                severity=FindingSeverity.MEDIUM,
                description="Prevent SQL injection in user query.",
            )
        ]

        agreed, stats = compute_agreement([sample_finding], labels_a, labels_b)
        assert len(agreed) == 1
        assert agreed[0].category == FindingCategory.SECURITY
        assert agreed[0].severity == FindingSeverity.HIGH
        assert stats.category_agreed == 1
        assert stats.agreement_rate == 1.0

    def test_agreement_discards_category_conflict(self, sample_finding: GoldFinding) -> None:
        labels_a = [
            CommentLabel(
                comment_index=0,
                actionable=True,
                category=FindingCategory.SECURITY,
            )
        ]
        labels_b = [
            CommentLabel(
                comment_index=0,
                actionable=True,
                category=FindingCategory.STYLE,
            )
        ]

        agreed, stats = compute_agreement([sample_finding], labels_a, labels_b)
        assert len(agreed) == 0
        assert stats.category_agreed == 0
        assert stats.both_actionable == 1
        assert stats.agreement_rate == 0.0

    def test_agreement_discards_non_actionable(self, sample_finding: GoldFinding) -> None:
        labels_a = [
            CommentLabel(
                comment_index=0,
                actionable=True,
                category=FindingCategory.SECURITY,
            )
        ]
        labels_b = [
            CommentLabel(
                comment_index=0,
                actionable=False,
            )
        ]

        agreed, stats = compute_agreement([sample_finding], labels_a, labels_b)
        assert len(agreed) == 0
        assert stats.both_actionable == 0
        assert stats.category_agreed == 0


@pytest.mark.asyncio
class TestPipelineExecution:
    async def test_mock_annotator(self, sample_example: PRExample) -> None:
        annotator = DeterministicMockAnnotator(annotator_id="A")
        user_prompt = format_pr_user_prompt(sample_example)
        resp, usage = await annotator.complete_structured(
            system="system prompt",
            user=user_prompt,
            schema=PRLabelResponse,
        )
        assert len(resp.labels) == 1
        assert resp.labels[0].actionable is True
        assert resp.labels[0].category == FindingCategory.SECURITY
        assert usage.latency_ms > 0

    async def test_dual_annotator_pipeline(self, sample_example: PRExample) -> None:
        pipeline = DualAnnotatorPipeline(
            client_a=DeterministicMockAnnotator(annotator_id="A"),
            client_b=DeterministicMockAnnotator(annotator_id="B"),
        )
        labeled_ex, stats = await pipeline.label_example(sample_example)
        assert len(labeled_ex.gold_findings) == 1
        assert stats.category_agreed == 1

    async def test_pipeline_clean_pr(self, sample_example: PRExample) -> None:
        clean_ex = sample_example.model_copy(update={"gold_findings": [], "is_clean": True})
        pipeline = DualAnnotatorPipeline(
            client_a=DeterministicMockAnnotator(annotator_id="A"),
            client_b=DeterministicMockAnnotator(annotator_id="B"),
        )
        labeled_ex, stats = await pipeline.label_example(clean_ex)
        assert len(labeled_ex.gold_findings) == 0
        assert labeled_ex.is_clean is True
        assert stats.total_comments == 0


class TestVerificationAppHelpers:
    def test_is_pr_fully_verified(self, sample_example: PRExample) -> None:
        # Initially unverified
        assert is_pr_fully_verified(sample_example) is False

        # After verifying finding
        v_finding = sample_example.gold_findings[0].model_copy(update={"verified": True})
        v_ex = sample_example.model_copy(update={"gold_findings": [v_finding]})
        assert is_pr_fully_verified(v_ex) is True

        # Clean PR is trivially verified
        clean_ex = sample_example.model_copy(update={"gold_findings": []})
        assert is_pr_fully_verified(clean_ex) is True

    def test_save_and_load_roundtrip(self, sample_example: PRExample) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            file_path = Path(tmp_dir) / "test.jsonl"
            save_current_dataset([sample_example], file_path)
            loaded = load_dataset_cached(str(file_path))
            assert len(loaded) == 1
            assert loaded[0].id == sample_example.id
            assert loaded[0].repo == sample_example.repo


@pytest.mark.asyncio
class TestEndToEndPipeline:
    async def test_run_labeling_pipeline_end_to_end(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            input_file = tmp_path / "raw.jsonl"
            out_dir = tmp_path / "final"
            stats_file = tmp_path / "stats.md"

            # Generate synthetic initial set
            synthetic_prs = generate_synthetic_pr_dataset(target_count=30)
            from evals.dataset.labeling import save_jsonl

            save_jsonl(synthetic_prs, input_file)

            summary = await run_labeling_pipeline(
                input_path=input_file,
                output_dir=out_dir,
                stats_out=stats_file,
                use_mock=True,
                auto_verify_test=True,
            )

            # Assert final files exist
            assert (out_dir / "train.jsonl").exists()
            assert (out_dir / "dev.jsonl").exists()
            assert (out_dir / "test.jsonl").exists()
            assert stats_file.exists()

            # Verify test split is 100% verified
            test_loaded = load_dataset_cached(str(out_dir / "test.jsonl"))
            for ex in test_loaded:
                for f in ex.gold_findings:
                    assert f.verified is True

            # Verify markdown report contains required sections
            report = stats_file.read_text(encoding="utf-8")
            assert "# Dataset Labeling & Verification Statistics" in report
            assert "Inter-Annotator Agreement" in report
            assert "Finding Category Distribution" in report
            assert "Finding Severity Distribution" in report
            assert "100% Verified" in report

            assert summary["train_count"] > 0
            assert summary["dev_count"] > 0
            assert summary["test_count"] > 0
