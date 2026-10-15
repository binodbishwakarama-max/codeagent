"""Evaluation metrics: precision, recall, F1, false-positive rate,
actionability, severity-weighted recall, cost/latency, and bootstrap CIs.

All metrics can be computed overall, per-category, and per-language.
Bootstrap confidence intervals use PR-level resampling (1000 resamples by default).
Paired bootstrap test reports p-value on whether run B is significantly better than run A.
"""

from __future__ import annotations

import logging
import math
import random
from collections import Counter, defaultdict
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from evals.dataset.schema import FindingCategory, GoldFinding, PRExample
from evals.harness.matcher import MatchResult, MultiMatcher, normalize_category
from prism.models import ReviewFinding, ReviewResult, Severity

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Metric Results
# ---------------------------------------------------------------------------


@dataclass
class ConfidenceInterval:
    """A metric value with bootstrap confidence interval."""

    value: float
    lower: float
    upper: float
    n_resamples: int

    def __str__(self) -> str:
        return f"{self.value:.4f} [{self.lower:.4f}, {self.upper:.4f}]"


@dataclass
class CategoryMetrics:
    """Precision, recall, F1 for a single category."""

    category: str
    true_positives: int = 0
    false_positives: int = 0
    false_negatives: int = 0
    precision: float = 0.0
    recall: float = 0.0
    f1: float = 0.0


@dataclass
class PRMetrics:
    """Metrics for a single PR evaluation."""

    pr_id: str
    repo: str
    language: str
    is_clean: bool
    num_predictions: int = 0
    num_golds: int = 0
    true_positives: int = 0
    false_positives: int = 0
    false_negatives: int = 0
    match_results: list[MatchResult] = field(default_factory=list)
    gold_findings: list[GoldFinding] = field(default_factory=list)
    actionable_predictions: int = 0
    latency_ms: float = 0.0
    cost_usd: float = 0.0
    input_tokens: int = 0
    output_tokens: int = 0


@dataclass
class EvalMetrics:
    """Aggregate evaluation metrics across all PRs."""

    # Core metrics
    precision: float = 0.0
    recall: float = 0.0
    f1: float = 0.0

    # False positive rate on clean PRs
    false_positive_rate: float = 0.0
    comments_per_clean_pr: float = 0.0

    # Per-PR metrics
    comments_per_pr: float = 0.0
    actionability: float = 0.0

    # Severity-weighted recall
    severity_weighted_recall: float = 0.0

    # Cost and latency
    cost_per_review: float = 0.0
    latency_p50: float = 0.0
    latency_p95: float = 0.0
    tokens_per_review: float = 0.0

    # Counts
    total_prs: int = 0
    total_predictions: int = 0
    total_golds: int = 0
    total_true_positives: int = 0
    total_false_positives: int = 0
    total_false_negatives: int = 0
    clean_prs: int = 0

    # Bootstrap CIs
    precision_ci: ConfidenceInterval | None = None
    recall_ci: ConfidenceInterval | None = None
    f1_ci: ConfidenceInterval | None = None
    confidence_intervals: dict[str, ConfidenceInterval] = field(default_factory=dict)

    # Per-category breakdown
    per_category: list[CategoryMetrics] = field(default_factory=list)

    # Per-language breakdown
    per_language: dict[str, dict[str, float]] = field(default_factory=dict)

    # Matcher agreement
    matcher_agreement: dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Severity Weights
# ---------------------------------------------------------------------------

_SEVERITY_WEIGHT: dict[str, float] = {
    "high": 3.0,
    "medium": 2.0,
    "low": 1.0,
    "unknown": 1.0,
}


# ---------------------------------------------------------------------------
# Core Metric Computation
# ---------------------------------------------------------------------------


def compute_pr_metrics(
    pr: PRExample,
    predictions: list[ReviewFinding],
    matcher: MultiMatcher,
    review_result: ReviewResult | None = None,
) -> PRMetrics:
    """Compute metrics for a single PR by matching predictions to gold findings."""
    match_results, _ = matcher.match_all(predictions, list(pr.gold_findings))

    tp = sum(1 for r in match_results if r.is_match)
    fp = len(predictions) - tp
    fn = len(pr.gold_findings) - tp

    # Count actionable predictions: non-empty suggestion or high/medium severity
    actionable_count = sum(
        1 for f in predictions if f.suggestion or f.severity in (Severity.HIGH, Severity.MEDIUM)
    )

    latency = review_result.latency_ms if review_result else 0.0
    cost = review_result.cost_usd if review_result else 0.0
    in_tokens = review_result.input_tokens if review_result else 0
    out_tokens = review_result.output_tokens if review_result else 0

    return PRMetrics(
        pr_id=pr.id,
        repo=pr.repo,
        language=pr.language,
        is_clean=pr.is_clean,
        num_predictions=len(predictions),
        num_golds=len(pr.gold_findings),
        true_positives=tp,
        false_positives=fp,
        false_negatives=fn,
        match_results=match_results,
        gold_findings=list(pr.gold_findings),
        actionable_predictions=actionable_count,
        latency_ms=latency,
        cost_usd=cost,
        input_tokens=in_tokens,
        output_tokens=out_tokens,
    )


def aggregate_metrics(pr_metrics: list[PRMetrics]) -> EvalMetrics:
    """Aggregate per-PR metrics into overall evaluation metrics."""
    if not pr_metrics:
        return EvalMetrics()

    total_tp = sum(m.true_positives for m in pr_metrics)
    total_fp = sum(m.false_positives for m in pr_metrics)
    total_fn = sum(m.false_negatives for m in pr_metrics)

    precision = total_tp / max(1, total_tp + total_fp)
    recall = total_tp / max(1, total_tp + total_fn)
    f1 = 2 * precision * recall / max(1e-9, precision + recall)

    # Clean PR metrics
    clean_prs = [m for m in pr_metrics if m.is_clean]
    clean_predictions = sum(m.num_predictions for m in clean_prs)
    n_clean = len(clean_prs)
    fp_rate = clean_predictions / max(1, n_clean)

    # Per-PR stats
    comments_per_pr = sum(m.num_predictions for m in pr_metrics) / max(1, len(pr_metrics))

    # Actionability
    tot_preds = sum(m.num_predictions for m in pr_metrics)
    actionability = (
        sum(m.actionable_predictions for m in pr_metrics) / max(1, tot_preds)
        if tot_preds > 0
        else 1.0
    )

    # Severity-weighted recall
    tot_gold_weight = sum(
        sum(_SEVERITY_WEIGHT.get(g.severity.value, 1.0) for g in m.gold_findings)
        for m in pr_metrics
    )
    matched_gold_weight = sum(
        sum(_SEVERITY_WEIGHT.get(r.gold_severity, 1.0) for r in m.match_results if r.is_match)
        for m in pr_metrics
    )
    severity_weighted_recall = matched_gold_weight / tot_gold_weight if tot_gold_weight > 0 else 1.0

    # Latency
    latencies = sorted([m.latency_ms for m in pr_metrics if m.latency_ms > 0])
    latency_p50 = _percentile(latencies, 50) if latencies else 0.0
    latency_p95 = _percentile(latencies, 95) if latencies else 0.0

    # Cost
    total_cost = sum(m.cost_usd for m in pr_metrics)
    cost_per_review = total_cost / max(1, len(pr_metrics))

    # Tokens
    total_tokens = sum(m.input_tokens + m.output_tokens for m in pr_metrics)
    tokens_per_review = total_tokens / max(1, len(pr_metrics))

    # Per-category
    per_cat = _compute_per_category(pr_metrics)

    # Per-language
    per_lang = _compute_per_language(pr_metrics)

    return EvalMetrics(
        precision=round(precision, 4),
        recall=round(recall, 4),
        f1=round(f1, 4),
        false_positive_rate=round(fp_rate, 4),
        comments_per_clean_pr=round(fp_rate, 4),
        comments_per_pr=round(comments_per_pr, 4),
        actionability=round(actionability, 4),
        severity_weighted_recall=round(severity_weighted_recall, 4),
        cost_per_review=round(cost_per_review, 6),
        latency_p50=round(latency_p50, 2),
        latency_p95=round(latency_p95, 2),
        tokens_per_review=round(tokens_per_review, 2),
        total_prs=len(pr_metrics),
        total_predictions=tot_preds,
        total_golds=sum(m.num_golds for m in pr_metrics),
        total_true_positives=total_tp,
        total_false_positives=total_fp,
        total_false_negatives=total_fn,
        clean_prs=n_clean,
        per_category=per_cat,
        per_language=per_lang,
    )


# ---------------------------------------------------------------------------
# Bootstrap Confidence Intervals
# ---------------------------------------------------------------------------


def bootstrap_ci(
    pr_metrics: list[PRMetrics],
    metric_fn: Callable[[list[PRMetrics]], float],
    n_resamples: int = 1000,
    confidence_level: float = 0.95,
    seed: int = 42,
) -> ConfidenceInterval:
    """Compute bootstrap CI by resampling at the PR level.

    Args:
        pr_metrics: List of per-PR metric results.
        metric_fn: Callable(list[PRMetrics]) -> float returning a scalar metric.
        n_resamples: Number of bootstrap resamples.
        confidence_level: Confidence level (e.g. 0.95).
        seed: Random seed for reproducibility.

    Returns:
        ConfidenceInterval with point estimate and percentile bounds.
    """
    rng = random.Random(seed)
    n = len(pr_metrics)
    if n == 0:
        return ConfidenceInterval(value=0.0, lower=0.0, upper=0.0, n_resamples=0)

    point_estimate = metric_fn(pr_metrics)
    bootstrap_values: list[float] = []

    for _ in range(n_resamples):
        sample = [pr_metrics[rng.randint(0, n - 1)] for _ in range(n)]
        bootstrap_values.append(metric_fn(sample))

    bootstrap_values.sort()
    alpha = 1 - confidence_level
    lower_idx = max(0, int(math.floor(n_resamples * (alpha / 2))))
    upper_idx = min(n_resamples - 1, int(math.ceil(n_resamples * (1 - alpha / 2))) - 1)

    return ConfidenceInterval(
        value=round(point_estimate, 4),
        lower=round(bootstrap_values[lower_idx], 4),
        upper=round(bootstrap_values[upper_idx], 4),
        n_resamples=n_resamples,
    )


def _precision_fn(pr_metrics: list[PRMetrics]) -> float:
    """Compute precision from a list of PR metrics."""
    tp = sum(m.true_positives for m in pr_metrics)
    fp = sum(m.false_positives for m in pr_metrics)
    return tp / max(1, tp + fp)


def _recall_fn(pr_metrics: list[PRMetrics]) -> float:
    """Compute recall from a list of PR metrics."""
    tp = sum(m.true_positives for m in pr_metrics)
    fn = sum(m.false_negatives for m in pr_metrics)
    return tp / max(1, tp + fn)


def _f1_fn(pr_metrics: list[PRMetrics]) -> float:
    """Compute F1 from a list of PR metrics."""
    p = _precision_fn(pr_metrics)
    r = _recall_fn(pr_metrics)
    return 2 * p * r / max(1e-9, p + r)


def _fpr_fn(pr_metrics: list[PRMetrics]) -> float:
    """Compute false positive rate on clean PRs."""
    clean = [m for m in pr_metrics if m.is_clean]
    if not clean:
        return 0.0
    return sum(m.num_predictions for m in clean) / len(clean)


def _comments_per_pr_fn(pr_metrics: list[PRMetrics]) -> float:
    """Compute average comments per PR."""
    if not pr_metrics:
        return 0.0
    return sum(m.num_predictions for m in pr_metrics) / len(pr_metrics)


def _actionability_fn(pr_metrics: list[PRMetrics]) -> float:
    """Compute proportion of actionable predictions."""
    tot = sum(m.num_predictions for m in pr_metrics)
    if tot == 0:
        return 1.0
    return sum(m.actionable_predictions for m in pr_metrics) / tot


def _severity_weighted_recall_fn(pr_metrics: list[PRMetrics]) -> float:
    """Compute severity-weighted recall."""
    tot_weight = sum(
        sum(_SEVERITY_WEIGHT.get(g.severity.value, 1.0) for g in m.gold_findings)
        for m in pr_metrics
    )
    if tot_weight == 0:
        return 1.0
    matched_weight = sum(
        sum(_SEVERITY_WEIGHT.get(r.gold_severity, 1.0) for r in m.match_results if r.is_match)
        for m in pr_metrics
    )
    return matched_weight / tot_weight


def _cost_fn(pr_metrics: list[PRMetrics]) -> float:
    """Compute average cost per PR."""
    if not pr_metrics:
        return 0.0
    return sum(m.cost_usd for m in pr_metrics) / len(pr_metrics)


def _latency_p50_fn(pr_metrics: list[PRMetrics]) -> float:
    """Compute p50 latency."""
    latencies = sorted([m.latency_ms for m in pr_metrics if m.latency_ms > 0])
    return _percentile(latencies, 50) if latencies else 0.0


def compute_bootstrap_cis(
    pr_metrics: list[PRMetrics],
    n_resamples: int = 1000,
    confidence_level: float = 0.95,
    seed: int = 42,
) -> tuple[ConfidenceInterval, ConfidenceInterval, ConfidenceInterval]:
    """Compute bootstrap CIs for precision, recall, and F1."""
    prec_ci = bootstrap_ci(pr_metrics, _precision_fn, n_resamples, confidence_level, seed)
    rec_ci = bootstrap_ci(pr_metrics, _recall_fn, n_resamples, confidence_level, seed)
    f1_ci = bootstrap_ci(pr_metrics, _f1_fn, n_resamples, confidence_level, seed)
    return prec_ci, rec_ci, f1_ci


def compute_all_bootstrap_cis(
    pr_metrics: list[PRMetrics],
    n_resamples: int = 1000,
    confidence_level: float = 0.95,
    seed: int = 42,
) -> dict[str, ConfidenceInterval]:
    """Compute bootstrap CIs for all aggregate metrics."""
    metric_funcs: dict[str, Callable[[list[PRMetrics]], float]] = {
        "precision": _precision_fn,
        "recall": _recall_fn,
        "f1": _f1_fn,
        "false_positive_rate": _fpr_fn,
        "comments_per_pr": _comments_per_pr_fn,
        "actionability": _actionability_fn,
        "severity_weighted_recall": _severity_weighted_recall_fn,
        "cost_per_review": _cost_fn,
        "latency_p50": _latency_p50_fn,
    }
    cis: dict[str, ConfidenceInterval] = {}
    for name, fn in metric_funcs.items():
        cis[name] = bootstrap_ci(
            pr_metrics, fn, n_resamples=n_resamples, confidence_level=confidence_level, seed=seed
        )
    return cis


# ---------------------------------------------------------------------------
# Paired Bootstrap Significance Test
# ---------------------------------------------------------------------------


def paired_bootstrap_test(
    metrics_a: list[PRMetrics],
    metrics_b: list[PRMetrics],
    metric_fn: Callable[[list[PRMetrics]], float] = _f1_fn,
    n_resamples: int = 1000,
    seed: int = 42,
) -> dict[str, Any]:
    """Paired bootstrap significance test between two runs.

    Tests whether run B is significantly different/better than run A.
    Both runs must cover the exact same set of PRs (same length and order).

    Returns:
        dict containing:
        - delta: metric(B) - metric(A)
        - p_value: two-sided bootstrap p-value
        - one_sided_p_value: probability that B is not better than A (H0: B <= A)
        - significant: whether difference is statistically significant (p_value < 0.05)
    """
    if len(metrics_a) != len(metrics_b):
        raise ValueError(
            f"Paired test requires equal-length metrics: {len(metrics_a)} vs {len(metrics_b)}"
        )

    n = len(metrics_a)
    if n == 0:
        return {
            "delta": 0.0,
            "p_value": 1.0,
            "one_sided_p_value": 1.0,
            "significant": False,
        }

    val_a = metric_fn(metrics_a)
    val_b = metric_fn(metrics_b)
    observed_delta = val_b - val_a

    rng = random.Random(seed)
    diffs: list[float] = []

    for _ in range(n_resamples):
        indices = [rng.randint(0, n - 1) for _ in range(n)]
        sample_a = [metrics_a[i] for i in indices]
        sample_b = [metrics_b[i] for i in indices]
        delta_i = metric_fn(sample_b) - metric_fn(sample_a)
        diffs.append(delta_i)

    # Centered bootstrap for two-sided p-value:
    # H0: mean difference is 0. Centered differences = delta_i - observed_delta.
    two_sided_count = sum(1 for d in diffs if abs(d - observed_delta) >= abs(observed_delta))
    p_value = two_sided_count / n_resamples

    # One-sided test (H0: delta <= 0, testing if B > A):
    # Fraction of bootstrap samples where delta <= 0
    one_sided_count = (
        sum(1 for d in diffs if d <= 0) if observed_delta > 0 else sum(1 for d in diffs if d >= 0)
    )
    one_sided_p = one_sided_count / n_resamples

    return {
        "delta": round(observed_delta, 4),
        "metric_a": round(val_a, 4),
        "metric_b": round(val_b, 4),
        "p_value": round(p_value, 4),
        "one_sided_p_value": round(one_sided_p, 4),
        "significant": p_value < 0.05,
    }


# ---------------------------------------------------------------------------
# Cohen's Kappa (for judge-vs-human agreement)
# ---------------------------------------------------------------------------


def cohens_kappa(
    judgments_a: list[bool],
    judgments_b: list[bool],
) -> float:
    """Compute Cohen's kappa inter-rater agreement coefficient.

    Args:
        judgments_a: Binary judgments from rater A.
        judgments_b: Binary judgments from rater B.

    Returns:
        Kappa coefficient in [-1, 1].
    """
    n = len(judgments_a)
    if n == 0 or n != len(judgments_b):
        return 0.0

    agree = sum(1 for a, b in zip(judgments_a, judgments_b, strict=True) if a == b)
    p_o = agree / n

    a_pos = sum(1 for a in judgments_a if a) / n
    b_pos = sum(1 for b in judgments_b if b) / n
    p_e = a_pos * b_pos + (1 - a_pos) * (1 - b_pos)

    if abs(1 - p_e) < 1e-9:
        return 1.0 if p_o >= 1.0 - 1e-9 else 0.0

    return round((p_o - p_e) / (1 - p_e), 4)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _percentile(sorted_values: list[float], pct: float) -> float:
    """Compute percentile from pre-sorted values."""
    if not sorted_values:
        return 0.0
    k = (len(sorted_values) - 1) * (pct / 100.0)
    f = math.floor(k)
    c = math.ceil(k)
    if f == c:
        return sorted_values[int(k)]
    return sorted_values[int(f)] * (c - k) + sorted_values[int(c)] * (k - f)


def _compute_per_category(pr_metrics: list[PRMetrics]) -> list[CategoryMetrics]:
    """Compute precision, recall, F1 per category from match results and gold findings."""
    cat_tp: Counter[str] = Counter()
    cat_fp: Counter[str] = Counter()
    cat_gold: Counter[str] = Counter()

    for m in pr_metrics:
        # Count all gold findings per category
        for g in m.gold_findings:
            cat_gold[normalize_category(g.category.value)] += 1

        for r in m.match_results:
            pred_c = r.predicted_category
            if r.is_match:
                # Use gold category or predicted category
                c = r.gold_category or pred_c
                if c:
                    cat_tp[c] += 1
            else:
                if pred_c:
                    cat_fp[pred_c] += 1

    categories = sorted({c.value for c in FindingCategory})
    result: list[CategoryMetrics] = []
    for cat in categories:
        tp_count = cat_tp.get(cat, 0)
        fp_count = cat_fp.get(cat, 0)
        gold_count = cat_gold.get(cat, 0)
        fn_count = max(0, gold_count - tp_count)

        prec = tp_count / max(1, tp_count + fp_count) if (tp_count + fp_count) > 0 else 0.0
        rec = tp_count / max(1, tp_count + fn_count) if (tp_count + fn_count) > 0 else 0.0
        f1_val = 2 * prec * rec / max(1e-9, prec + rec) if (prec + rec) > 0 else 0.0

        result.append(
            CategoryMetrics(
                category=cat,
                true_positives=tp_count,
                false_positives=fp_count,
                false_negatives=fn_count,
                precision=round(prec, 4),
                recall=round(rec, 4),
                f1=round(f1_val, 4),
            )
        )
    return result


def _compute_per_language(pr_metrics: list[PRMetrics]) -> dict[str, dict[str, float]]:
    """Compute aggregate metrics per programming language."""
    by_lang: dict[str, list[PRMetrics]] = defaultdict(list)
    for m in pr_metrics:
        by_lang[m.language].append(m)

    result: dict[str, dict[str, float]] = {}
    for lang, metrics in sorted(by_lang.items()):
        tp = sum(m.true_positives for m in metrics)
        fp = sum(m.false_positives for m in metrics)
        fn = sum(m.false_negatives for m in metrics)
        prec = tp / max(1, tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / max(1, tp + fn) if (tp + fn) > 0 else 0.0
        f1_val = 2 * prec * rec / max(1e-9, prec + rec) if (prec + rec) > 0 else 0.0
        result[lang] = {
            "precision": round(prec, 4),
            "recall": round(rec, 4),
            "f1": round(f1_val, 4),
            "n_prs": len(metrics),
        }

    return result
