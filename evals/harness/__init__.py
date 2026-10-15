"""Evaluation harness execution engine, scoring metrics, and artifact reporting."""

from evals.harness.baselines import LinterBaseline, RandomBaseline
from evals.harness.compare import compare_runs
from evals.harness.config import ExperimentConfig
from evals.harness.matcher import (
    EmbeddingMatcher,
    ExactMatcher,
    Matcher,
    MatchResult,
    MultiMatcher,
    SemanticMatcher,
    normalize_category,
)
from evals.harness.metrics import (
    CategoryMetrics,
    ConfidenceInterval,
    EvalMetrics,
    PRMetrics,
    aggregate_metrics,
    bootstrap_ci,
    cohens_kappa,
    compute_all_bootstrap_cis,
    compute_bootstrap_cis,
    compute_pr_metrics,
    paired_bootstrap_test,
)
from evals.harness.report import generate_comparison_table, generate_run_report
from evals.harness.runner import run_evaluation, run_evaluation_async

__all__ = [
    "CategoryMetrics",
    "ConfidenceInterval",
    "EmbeddingMatcher",
    "EvalMetrics",
    "ExactMatcher",
    "ExperimentConfig",
    "LinterBaseline",
    "MatchResult",
    "Matcher",
    "MultiMatcher",
    "PRMetrics",
    "RandomBaseline",
    "SemanticMatcher",
    "aggregate_metrics",
    "bootstrap_ci",
    "cohens_kappa",
    "compare_runs",
    "compute_all_bootstrap_cis",
    "compute_bootstrap_cis",
    "compute_pr_metrics",
    "generate_comparison_table",
    "generate_run_report",
    "normalize_category",
    "paired_bootstrap_test",
    "run_evaluation",
    "run_evaluation_async",
]
