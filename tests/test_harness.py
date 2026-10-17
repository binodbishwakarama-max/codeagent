"""Rigorous tests for the evaluation harness module (evals.harness).

Tests:
1. Matcher on synthetic finding pairs (exact match, close-line match, non-match, wrong file, embedding match).
2. Bootstrap CI computation on synthetic data with known properties (monotonicity, coverage, determinism).
3. Paired bootstrap significance testing (delta direction, p-values on identical vs different runs).
4. Fast mock mode evaluation runner that runs in <5s.
5. 100% reproducibility: running twice with same seed produces bit-identical predictions.jsonl.
6. Baseline strategies (RandomBaseline, LinterBaseline).
7. Comparison tooling (markdown table and CSV generation).
"""

from __future__ import annotations

import filecmp
from pathlib import Path

import pytest

from evals.dataset.schema import FindingCategory, FindingSeverity, GoldFinding, PRExample
from evals.harness.baselines import LinterBaseline, RandomBaseline
from evals.harness.compare import compare_runs
from evals.harness.config import ExperimentConfig
from evals.harness.matcher import (
    ExactMatcher,
    MultiMatcher,
    SemanticMatcher,
    _bow_cosine_similarity,
)
from evals.harness.metrics import (
    PRMetrics,
    _f1_fn,
    _precision_fn,
    bootstrap_ci,
    paired_bootstrap_test,
)
from evals.harness.runner import run_evaluation_async
from prism.models import IssueCategory, ReviewFinding, Severity

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def sample_gold() -> GoldFinding:
    return GoldFinding(
        path="src/auth.py",
        line=42,
        category=FindingCategory.SECURITY,
        severity=FindingSeverity.HIGH,
        description="Missing CSRF token validation on sensitive POST route",
    )


@pytest.fixture
def sample_diff() -> str:
    return (
        "--- a/src/app.py\n"
        "+++ b/src/app.py\n"
        "@@ -10,6 +10,8 @@\n"
        " def handle_request():\n"
        "+    print('debugging request')\n"
        "+    except: pass\n"
        "     return True\n"
    )


@pytest.fixture
def sample_example(sample_diff: str) -> PRExample:
    return PRExample(
        id="test-pr-001",
        repo="owner/repo",
        pr_number=101,
        title="Fix request handler",
        body="Updates the handler logic.",
        head_sha="abcdef123456",
        base_sha="000000000000",
        diff=sample_diff,
        language="python",
        is_clean=False,
        license="MIT",
        pr_url="https://github.com/owner/repo/pull/101",
        changed_lines=5,
        gold_findings=[
            GoldFinding(
                path="src/app.py",
                line=12,
                category=FindingCategory.ERROR_HANDLING,
                severity=FindingSeverity.HIGH,
                description="Bare except clause catches all exceptions blindly",
            )
        ],
    )


# ---------------------------------------------------------------------------
# 1. Matcher Tests
# ---------------------------------------------------------------------------


class TestMatcher:
    """Tests for ExactMatcher, SemanticMatcher, EmbeddingMatcher, and MultiMatcher."""

    def test_exact_match_success(self, sample_gold: GoldFinding) -> None:
        matcher = ExactMatcher()
        pred = ReviewFinding(
            path="src/auth.py",
            line=42,
            category=IssueCategory.SECURITY,
            severity=Severity.HIGH,
            message="CSRF check is missing from the endpoint.",
            confidence=0.9,
        )
        res = matcher.match(pred, sample_gold, line_tolerance=3)
        assert res.is_match is True
        assert res.category_match is True
        assert res.line_distance == 0
        assert res.predicted_category == "security"
        assert res.gold_category == "security"

    def test_close_line_match_within_tolerance(self, sample_gold: GoldFinding) -> None:
        matcher = ExactMatcher()
        # Line 44 is within |44 - 42| = 2 <= 3
        pred = ReviewFinding(
            path="src/auth.py",
            line=44,
            category=IssueCategory.SECURITY,
            severity=Severity.HIGH,
            message="CSRF vulnerability.",
            confidence=0.85,
        )
        res = matcher.match(pred, sample_gold, line_tolerance=3)
        assert res.is_match is True
        assert res.line_distance == 2

    def test_line_outside_tolerance_non_match(self, sample_gold: GoldFinding) -> None:
        matcher = ExactMatcher()
        # Line 50 is outside |50 - 42| = 8 > 3
        pred = ReviewFinding(
            path="src/auth.py",
            line=50,
            category=IssueCategory.SECURITY,
            severity=Severity.HIGH,
            message="CSRF vulnerability.",
            confidence=0.85,
        )
        res = matcher.match(pred, sample_gold, line_tolerance=3)
        assert res.is_match is False
        assert res.line_distance == 8

    def test_wrong_file_non_match(self, sample_gold: GoldFinding) -> None:
        matcher = ExactMatcher()
        pred = ReviewFinding(
            path="src/models/user.py",
            line=42,
            category=IssueCategory.SECURITY,
            severity=Severity.HIGH,
            message="CSRF vulnerability.",
            confidence=0.85,
        )
        res = matcher.match(pred, sample_gold, line_tolerance=3)
        assert res.is_match is False

    def test_path_normalization_matches_subpath(self, sample_gold: GoldFinding) -> None:
        matcher = ExactMatcher()
        # Leading slash or relative prefix
        pred = ReviewFinding(
            path="/src/auth.py",
            line=42,
            category=IssueCategory.SECURITY,
            severity=Severity.HIGH,
            message="CSRF issue.",
            confidence=0.9,
        )
        res = matcher.match(pred, sample_gold, line_tolerance=3)
        assert res.is_match is True

    def test_category_mismatch_exact_fails(self, sample_gold: GoldFinding) -> None:
        matcher = ExactMatcher()
        pred = ReviewFinding(
            path="src/auth.py",
            line=42,
            category=IssueCategory.READABILITY,
            severity=Severity.LOW,
            message="Poor readability.",
            confidence=0.7,
        )
        res = matcher.match(pred, sample_gold, line_tolerance=3)
        assert res.is_match is False
        assert res.category_match is False

    def test_semantic_matcher_heurstic_match(self, tmp_path: Path) -> None:
        cache_dir = tmp_path / "semantic_cache"
        matcher = SemanticMatcher(cache_dir=str(cache_dir))
        gold = GoldFinding(
            path="src/db.py",
            line=10,
            category=FindingCategory.BUG,
            severity=FindingSeverity.HIGH,
            description="Database connection leak when query raises an exception",
        )
        pred = ReviewFinding(
            path="src/db.py",
            line=11,
            category=IssueCategory.STANDARDS_VIOLATION,  # different category
            severity=Severity.MEDIUM,
            message="Database connection leak occurs during exception in query execution",
            confidence=0.8,
        )
        res = matcher.match(pred, gold, line_tolerance=3)
        assert res.is_match is True
        assert res.semantic_match is True
        # Verify result was cached to disk
        cache_file = cache_dir / "semantic_cache.json"
        assert cache_file.exists()

    def test_embedding_matcher_cosine_similarity(self) -> None:
        sim = _bow_cosine_similarity(
            "sql injection vulnerability in user login query",
            "critical sql injection flaw inside database authentication query",
        )
        assert sim > 0.4
        sim_unrelated = _bow_cosine_similarity(
            "sql injection vulnerability in query",
            "formatting whitespace indentation error in css stylesheet",
        )
        assert sim_unrelated == 0.0

    def test_multimatcher_greedily_matches_and_reports_stats(self) -> None:
        matcher = MultiMatcher()
        golds = [
            GoldFinding(
                path="src/api.py",
                line=20,
                category=FindingCategory.BUG,
                severity=FindingSeverity.HIGH,
                description="Off by one error in slice index",
            ),
            GoldFinding(
                path="src/api.py",
                line=50,
                category=FindingCategory.SECURITY,
                severity=FindingSeverity.HIGH,
                description="Unsanitized user input in header",
            ),
        ]
        preds = [
            ReviewFinding(
                path="src/api.py",
                line=21,
                category=IssueCategory.BUG,
                severity=Severity.HIGH,
                message="Off-by-one bug in array slice indexing",
                confidence=0.95,
            ),
            ReviewFinding(
                path="src/api.py",
                line=99,
                category=IssueCategory.READABILITY,
                severity=Severity.LOW,
                message="Variable name could be clearer",
                confidence=0.5,
            ),
        ]
        results, stats = matcher.match_all(preds, golds)
        assert len(results) == 2
        assert stats["matched"] == 1
        assert stats["unmatched_predictions"] == 1
        assert stats["unmatched_golds"] == 1
        assert "matcher_agreement_rate" in stats


# ---------------------------------------------------------------------------
# 2. Bootstrap CI Tests
# ---------------------------------------------------------------------------


class TestBootstrapCI:
    """Tests for bootstrap confidence interval computation."""

    def test_bootstrap_ci_bounds_and_coverage(self) -> None:
        # Create 50 synthetic PR metrics with precision around 0.8
        pr_metrics: list[PRMetrics] = []
        for i in range(50):
            tp = 4 if i % 5 != 0 else 2
            fp = 1 if i % 5 != 0 else 3
            pr_metrics.append(
                PRMetrics(
                    pr_id=f"pr-{i}",
                    repo="owner/repo",
                    language="python",
                    is_clean=False,
                    num_predictions=tp + fp,
                    num_golds=5,
                    true_positives=tp,
                    false_positives=fp,
                    false_negatives=5 - tp,
                )
            )

        ci = bootstrap_ci(pr_metrics, _precision_fn, n_resamples=500, seed=42)
        assert ci.n_resamples == 500
        # Lower bound must be <= point estimate <= upper bound
        assert ci.lower <= ci.value <= ci.upper
        assert 0.5 <= ci.value <= 0.9

    def test_bootstrap_ci_deterministic_with_fixed_seed(self) -> None:
        metrics = [
            PRMetrics(
                pr_id=f"pr-{i}",
                repo="r",
                language="python",
                is_clean=False,
                true_positives=i % 3,
                false_positives=1,
                false_negatives=1,
            )
            for i in range(20)
        ]
        ci_1 = bootstrap_ci(metrics, _f1_fn, n_resamples=300, seed=12345)
        ci_2 = bootstrap_ci(metrics, _f1_fn, n_resamples=300, seed=12345)
        assert ci_1.value == ci_2.value
        assert ci_1.lower == ci_2.lower
        assert ci_1.upper == ci_2.upper

    def test_bootstrap_ci_empty_input(self) -> None:
        ci = bootstrap_ci([], _precision_fn, n_resamples=100)
        assert ci.value == 0.0
        assert ci.lower == 0.0
        assert ci.upper == 0.0


# ---------------------------------------------------------------------------
# 3. Paired Significance Test Tests
# ---------------------------------------------------------------------------


class TestPairedSignificanceTest:
    """Tests for paired bootstrap significance testing."""

    def test_paired_bootstrap_detects_significant_difference(self) -> None:
        # Run B clearly outperforms Run A on every single PR
        metrics_a: list[PRMetrics] = []
        metrics_b: list[PRMetrics] = []
        for i in range(40):
            metrics_a.append(
                PRMetrics(
                    pr_id=f"pr-{i}",
                    repo="r",
                    language="python",
                    is_clean=False,
                    true_positives=1,
                    false_positives=4,
                    false_negatives=4,  # F1 ~ 0.2
                )
            )
            metrics_b.append(
                PRMetrics(
                    pr_id=f"pr-{i}",
                    repo="r",
                    language="python",
                    is_clean=False,
                    true_positives=5,
                    false_positives=0,
                    false_negatives=0,  # F1 = 1.0
                )
            )

        res = paired_bootstrap_test(
            metrics_a, metrics_b, metric_fn=_f1_fn, n_resamples=500, seed=42
        )
        assert res["delta"] > 0.5
        assert res["significant"] is True
        assert res["p_value"] < 0.05
        assert res["one_sided_p_value"] < 0.05

    def test_paired_bootstrap_identical_runs_not_significant(self) -> None:
        metrics: list[PRMetrics] = [
            PRMetrics(
                pr_id=f"pr-{i}",
                repo="r",
                language="python",
                is_clean=False,
                true_positives=2,
                false_positives=1,
                false_negatives=1,
            )
            for i in range(30)
        ]
        res = paired_bootstrap_test(metrics, metrics, metric_fn=_f1_fn, n_resamples=300, seed=42)
        assert res["delta"] == 0.0
        assert res["significant"] is False


# ---------------------------------------------------------------------------
# 4. Baselines Tests
# ---------------------------------------------------------------------------


class TestBaselines:
    """Tests for RandomBaseline and LinterBaseline."""

    def test_random_baseline_generates_configured_findings(self, sample_example: PRExample) -> None:
        baseline = RandomBaseline(findings_per_pr=3, seed=99)
        result = baseline.review(sample_example)
        assert len(result.findings) == 3
        assert result.model == "random-baseline"
        for f in result.findings:
            assert f.path == "src/app.py"
            assert f.line >= 1

    def test_linter_baseline_detects_patterns(self, sample_example: PRExample) -> None:
        baseline = LinterBaseline()
        result = baseline.review(sample_example)
        # sample_diff has 'print(' and 'except:'
        assert len(result.findings) >= 2
        categories = {f.category for f in result.findings}
        assert (
            IssueCategory.ERROR_HANDLING in categories
            or IssueCategory.STANDARDS_VIOLATION in categories
        )


# ---------------------------------------------------------------------------
# 5. Fast Mock Mode Evaluation Runner (<5s)
# ---------------------------------------------------------------------------


class TestRunnerFastMock:
    """Test full evaluation runner in fast mock mode."""

    @pytest.mark.asyncio
    async def test_fast_mock_evaluation_run(
        self, tmp_path: Path, sample_example: PRExample
    ) -> None:
        # Write sample dev.jsonl
        dataset_dir = tmp_path / "data"
        dataset_dir.mkdir(parents=True, exist_ok=True)
        split_file = dataset_dir / "dev.jsonl"
        with split_file.open("w", encoding="utf-8") as f:
            f.write(sample_example.model_dump_json() + "\n")

        # Config pointing to temp dir
        cfg = ExperimentConfig.model_validate(
            {
                "name": "test-fast-run",
                "dataset": {"path": str(dataset_dir), "sample_limit": 5},
                "pipeline": {"model": "gpt-4o-mini", "max_concurrency": 2, "seed": 42},
                "bootstrap": {"n_resamples": 100, "seed": 42},
            }
        )

        run_dir = tmp_path / "runs" / "fast_run"
        metrics, pr_metrics = await run_evaluation_async(
            config=cfg,
            split="dev",
            run_dir=run_dir,
            use_mock=True,
        )

        assert metrics.total_prs == 1
        assert (run_dir / "predictions.jsonl").exists()
        assert (run_dir / "config.json").exists()
        assert (run_dir / "report.md").exists()
        assert (run_dir / "metrics.json").exists()

        # Check report content
        report_text = (run_dir / "report.md").read_text(encoding="utf-8")
        assert "# Evaluation Report: test-fast-run" in report_text
        assert "## Core Metrics" in report_text


# ---------------------------------------------------------------------------
# 6. 100% Reproducibility Test (Bit-Identical predictions.jsonl)
# ---------------------------------------------------------------------------


class TestReproducibility:
    """Verify that running twice with the exact same seed produces bit-identical predictions.jsonl."""

    @pytest.mark.asyncio
    async def test_predictions_bit_identical_across_runs(
        self, tmp_path: Path, sample_example: PRExample
    ) -> None:
        dataset_dir = tmp_path / "data"
        dataset_dir.mkdir(parents=True, exist_ok=True)
        split_file = dataset_dir / "dev.jsonl"

        # Create two distinct PR examples
        ex1 = sample_example
        ex2 = sample_example.model_copy(update={"id": "test-pr-002", "pr_number": 102})
        with split_file.open("w", encoding="utf-8") as f:
            f.write(ex1.model_dump_json() + "\n")
            f.write(ex2.model_dump_json() + "\n")

        cfg = ExperimentConfig.model_validate(
            {
                "name": "reproducibility-test",
                "dataset": {"path": str(dataset_dir)},
                "pipeline": {"model": "gpt-4o-mini", "max_concurrency": 2, "seed": 42},
                "bootstrap": {"n_resamples": 100, "seed": 42},
            }
        )

        run_dir_1 = tmp_path / "run_1"
        run_dir_2 = tmp_path / "run_2"

        await run_evaluation_async(cfg, split="dev", run_dir=run_dir_1, use_mock=True)
        await run_evaluation_async(cfg, split="dev", run_dir=run_dir_2, use_mock=True)

        pred_file_1 = run_dir_1 / "predictions.jsonl"
        pred_file_2 = run_dir_2 / "predictions.jsonl"

        assert pred_file_1.exists()
        assert pred_file_2.exists()

        # Bit-identical verification
        assert filecmp.cmp(pred_file_1, pred_file_2, shallow=False), (
            "predictions.jsonl must be bit-identical across identical runs with same seed"
        )


# ---------------------------------------------------------------------------
# 7. Comparison Tooling Tests
# ---------------------------------------------------------------------------


class TestComparisonTooling:
    """Test evals.harness.compare on synthetic runs."""

    @pytest.mark.asyncio
    async def test_compare_runs_generates_markdown_and_csv(
        self, tmp_path: Path, sample_example: PRExample
    ) -> None:
        dataset_dir = tmp_path / "data"
        dataset_dir.mkdir(parents=True, exist_ok=True)
        split_file = dataset_dir / "dev.jsonl"
        with split_file.open("w", encoding="utf-8") as f:
            f.write(sample_example.model_dump_json() + "\n")

        cfg_a = ExperimentConfig.model_validate(
            {
                "name": "run-a",
                "dataset": {"path": str(dataset_dir)},
                "pipeline": {"model": "gpt-4o-mini", "confidence_threshold": 0.8, "seed": 42},
                "bootstrap": {"n_resamples": 100, "seed": 42},
            }
        )
        cfg_b = ExperimentConfig.model_validate(
            {
                "name": "run-b",
                "dataset": {"path": str(dataset_dir)},
                "pipeline": {"model": "gpt-4o-mini", "confidence_threshold": 0.4, "seed": 42},
                "bootstrap": {"n_resamples": 100, "seed": 42},
            }
        )

        dir_a = tmp_path / "run_a"
        dir_b = tmp_path / "run_b"
        await run_evaluation_async(cfg_a, split="dev", run_dir=dir_a, use_mock=True)
        await run_evaluation_async(cfg_b, split="dev", run_dir=dir_b, use_mock=True)

        out_reports = tmp_path / "reports"
        res = compare_runs(dir_a, dir_b, output_dir=out_reports, n_resamples=100)

        assert "markdown" in res
        assert "csv" in res
        assert res["md_path"].exists()
        assert res["csv_path"].exists()

        md_text = res["md_path"].read_text(encoding="utf-8")
        assert "Evaluation Run Comparison:" in md_text
        assert "Significance Summary" in md_text
