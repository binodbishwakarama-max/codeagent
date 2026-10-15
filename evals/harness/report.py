"""Markdown and CSV report generation for evaluation run summaries.

Generates:
- runs/<run_id>/report.md: Full evaluation report with metrics tables
- Comparison tables across runs (markdown + CSV)
- Paired bootstrap significance test results
"""

from __future__ import annotations

import csv
import io
import json
import logging
from pathlib import Path
from typing import Any

from evals.harness.metrics import (
    EvalMetrics,
    PRMetrics,
    paired_bootstrap_test,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Single Run Report
# ---------------------------------------------------------------------------


def generate_run_report(
    run_dir: Path,
    config: dict[str, Any],
    metrics: EvalMetrics,
    pr_metrics: list[PRMetrics],
    git_sha: str = "unknown",
    matcher_stats: dict[str, Any] | None = None,
) -> str:
    """Generate comprehensive markdown report for a single evaluation run.

    Args:
        run_dir: Directory to write report.md.
        config: Experiment configuration dict.
        metrics: Computed aggregate metrics.
        pr_metrics: Per-PR metrics for detailed breakdown.
        git_sha: Git commit SHA for reproducibility.
        matcher_stats: Optional matcher agreement statistics.

    Returns:
        Generated markdown content.
    """
    run_dir.mkdir(parents=True, exist_ok=True)
    run_name = config.get("name", "unnamed")

    # Build report
    sections: list[str] = []

    # Header
    sections.append(f"# Evaluation Report: {run_name}\n")
    sections.append(f"**Git SHA:** `{git_sha}`\n")
    sections.append(f"**Model:** `{config.get('pipeline', {}).get('model', 'unknown')}`\n")
    sections.append(f"**Split:** `{config.get('_split', 'dev')}`\n")
    sections.append(f"**Total PRs:** {metrics.total_prs}\n")
    sections.append("---\n")

    # Core Metrics
    sections.append("## Core Metrics\n")
    sections.append("| Metric | Value |")
    sections.append("|--------|-------|")
    sections.append(f"| Precision | {metrics.precision:.4f} |")
    sections.append(f"| Recall | {metrics.recall:.4f} |")
    sections.append(f"| F1 | {metrics.f1:.4f} |")
    sections.append(f"| False Positive Rate (clean PRs) | {metrics.false_positive_rate:.4f} |")
    sections.append(f"| Comments/PR | {metrics.comments_per_pr:.2f} |")
    sections.append(f"| Comments/Clean PR | {metrics.comments_per_clean_pr:.2f} |")
    sections.append(f"| Actionability | {metrics.actionability:.4f} |")
    sections.append(f"| Severity-Weighted Recall | {metrics.severity_weighted_recall:.4f} |")
    sections.append("")

    # Bootstrap CIs
    ci_dict = metrics.confidence_intervals or {}
    if not ci_dict and (metrics.precision_ci or metrics.recall_ci or metrics.f1_ci):
        if metrics.precision_ci:
            ci_dict["precision"] = metrics.precision_ci
        if metrics.recall_ci:
            ci_dict["recall"] = metrics.recall_ci
        if metrics.f1_ci:
            ci_dict["f1"] = metrics.f1_ci

    if ci_dict:
        sections.append("## Bootstrap Confidence Intervals (95%)\n")
        sections.append("| Metric | Point Estimate | 95% CI |")
        sections.append("|--------|----------------|--------|")
        for m_name, ci_val in ci_dict.items():
            sections.append(
                f"| {m_name.replace('_', ' ').title()} | {ci_val.value:.4f} | "
                f"[{ci_val.lower:.4f}, {ci_val.upper:.4f}] |"
            )
        sections.append("")

    # Cost and Latency
    sections.append("## Cost & Latency\n")
    sections.append("| Metric | Value |")
    sections.append("|--------|-------|")
    sections.append(f"| Cost/Review | ${metrics.cost_per_review:.6f} |")
    sections.append(f"| Latency p50 | {metrics.latency_p50:.1f}ms |")
    sections.append(f"| Latency p95 | {metrics.latency_p95:.1f}ms |")
    sections.append(f"| Tokens/Review | {metrics.tokens_per_review:.0f} |")
    sections.append("")

    # Per-Category
    if metrics.per_category:
        sections.append("## Per-Category Breakdown\n")
        sections.append("| Category | TP | FP | FN | Precision | Recall | F1 |")
        sections.append("|----------|----|----|----|-----------| -------|-----|")
        for cat in metrics.per_category:
            sections.append(
                f"| {cat.category} | {cat.true_positives} | {cat.false_positives} | "
                f"{cat.false_negatives} | {cat.precision:.4f} | {cat.recall:.4f} | "
                f"{cat.f1:.4f} |"
            )
        sections.append("")

    # Per-Language
    if metrics.per_language:
        sections.append("## Per-Language Breakdown\n")
        sections.append("| Language | PRs | Precision | Recall | F1 |")
        sections.append("|----------|-----|-----------|--------|-----|")
        for lang, vals in metrics.per_language.items():
            sections.append(
                f"| {lang} | {vals.get('n_prs', 0)} | {vals.get('precision', 0):.4f} | "
                f"{vals.get('recall', 0):.4f} | {vals.get('f1', 0):.4f} |"
            )
        sections.append("")

    # Matcher Agreement
    if matcher_stats:
        sections.append("## Matcher Agreement\n")
        sections.append("| Statistic | Value |")
        sections.append("|-----------|-------|")
        for k, v in matcher_stats.items():
            sections.append(f"| {k} | {v} |")
        sections.append("")

    # Counts summary
    sections.append("## Summary Counts\n")
    sections.append("| Count | Value |")
    sections.append("|-------|-------|")
    sections.append(f"| Total PRs | {metrics.total_prs} |")
    sections.append(f"| Total Predictions | {metrics.total_predictions} |")
    sections.append(f"| Total Gold Findings | {metrics.total_golds} |")
    sections.append(f"| True Positives | {metrics.total_true_positives} |")
    sections.append(f"| False Positives | {metrics.total_false_positives} |")
    sections.append(f"| False Negatives | {metrics.total_false_negatives} |")
    sections.append(f"| Clean PRs | {metrics.clean_prs} |")
    sections.append("")

    md_content = "\n".join(sections)

    # Write report
    report_path = run_dir / "report.md"
    report_path.write_text(md_content, encoding="utf-8")
    logger.info("Wrote evaluation report to %s", report_path)

    # Also save metrics as JSON for programmatic access
    metrics_json_path = run_dir / "metrics.json"
    metrics_dict = {
        "precision": metrics.precision,
        "recall": metrics.recall,
        "f1": metrics.f1,
        "false_positive_rate": metrics.false_positive_rate,
        "comments_per_pr": metrics.comments_per_pr,
        "cost_per_review": metrics.cost_per_review,
        "latency_p50": metrics.latency_p50,
        "latency_p95": metrics.latency_p95,
        "tokens_per_review": metrics.tokens_per_review,
        "total_prs": metrics.total_prs,
        "total_predictions": metrics.total_predictions,
        "total_golds": metrics.total_golds,
        "per_language": metrics.per_language,
    }
    metrics_json_path.write_text(json.dumps(metrics_dict, indent=2), encoding="utf-8")

    return md_content


# ---------------------------------------------------------------------------
# Cross-Run Comparison
# ---------------------------------------------------------------------------


def generate_comparison_table(
    run_summaries: list[dict[str, Any]],
    output_dir: Path,
) -> tuple[str, str]:
    """Generate comparison table across multiple runs.

    Args:
        run_summaries: List of dicts, each with 'name', 'metrics' (EvalMetrics-like dict).
        output_dir: Directory to write comparison files.

    Returns:
        Tuple of (markdown_content, csv_content).
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    headers = [
        "Run",
        "Model",
        "Precision",
        "Recall",
        "F1",
        "FP Rate",
        "Comments/PR",
        "Cost/Review",
        "Latency p50",
    ]

    rows: list[list[str]] = []
    for summary in run_summaries:
        name = summary.get("name", "unknown")
        m = summary.get("metrics", {})
        rows.append(
            [
                name,
                str(summary.get("model", "—")),
                f"{m.get('precision', 0):.4f}",
                f"{m.get('recall', 0):.4f}",
                f"{m.get('f1', 0):.4f}",
                f"{m.get('false_positive_rate', 0):.4f}",
                f"{m.get('comments_per_pr', 0):.2f}",
                f"${m.get('cost_per_review', 0):.6f}",
                f"{m.get('latency_p50', 0):.1f}ms",
            ]
        )

    # Markdown
    md_lines: list[str] = ["# Evaluation Comparison\n"]
    md_lines.append("| " + " | ".join(headers) + " |")
    md_lines.append("| " + " | ".join(["---"] * len(headers)) + " |")
    for row in rows:
        md_lines.append("| " + " | ".join(row) + " |")
    md_lines.append("")
    md_content = "\n".join(md_lines)

    # CSV
    csv_buf = io.StringIO()
    writer = csv.writer(csv_buf)
    writer.writerow(headers)
    writer.writerows(rows)
    csv_content = csv_buf.getvalue()

    # Write files
    (output_dir / "comparison.md").write_text(md_content, encoding="utf-8")
    (output_dir / "comparison.csv").write_text(csv_content, encoding="utf-8")

    logger.info("Wrote comparison table to %s", output_dir)
    return md_content, csv_content


def generate_significance_report(
    metrics_a: list[PRMetrics],
    metrics_b: list[PRMetrics],
    name_a: str,
    name_b: str,
    output_path: Path,
) -> str:
    """Generate significance test report comparing two runs."""
    test_result = paired_bootstrap_test(metrics_a, metrics_b)

    md = f"""# Paired Bootstrap Significance Test

**Comparing:** {name_a} vs {name_b}

| Statistic | Value |
|-----------|-------|
| F1 Delta ({name_a} - {name_b}) | {test_result["delta"]:.4f} |
| p-value | {test_result["p_value"]:.4f} |
| Significant (α=0.05) | {"✅ Yes" if test_result["significant"] else "❌ No"} |

> [!NOTE]
> Significance test uses paired bootstrap resampling at the PR level.
> Same PRs are compared across both runs to control for difficulty variation.
"""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(md, encoding="utf-8")
    logger.info("Wrote significance report to %s", output_path)
    return md
