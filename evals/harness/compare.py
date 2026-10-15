"""Cross-run comparison tooling with paired bootstrap significance testing.

Usage:
    python -m evals.harness.compare run_A run_B
    python -m evals.harness.compare runs/run_A runs/run_B --output-dir reports/

Produces a markdown table and CSV comparing two evaluation runs with
bootstrap p-values for statistical significance.
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import logging
import sys
from pathlib import Path
from typing import Any

from evals.dataset.schema import GoldFinding
from evals.harness.matcher import MatchResult
from evals.harness.metrics import (
    PRMetrics,
    _actionability_fn,
    _comments_per_pr_fn,
    _f1_fn,
    _fpr_fn,
    _precision_fn,
    _recall_fn,
    _severity_weighted_recall_fn,
    aggregate_metrics,
    paired_bootstrap_test,
)

logger = logging.getLogger(__name__)


def resolve_run_dir(run_identifier: str | Path) -> Path:
    """Resolve run directory from path or run ID."""
    p = Path(run_identifier)
    if p.exists() and p.is_dir():
        return p
    # Check runs/<identifier>
    runs_dir = Path("runs") / run_identifier
    if runs_dir.exists() and runs_dir.is_dir():
        return runs_dir
    raise FileNotFoundError(f"Run directory not found: '{run_identifier}'")


def load_run_predictions(run_dir: Path) -> list[dict[str, Any]]:
    """Load predictions.jsonl from a run directory."""
    pred_file = run_dir / "predictions.jsonl"
    if not pred_file.exists():
        raise FileNotFoundError(f"Missing predictions.jsonl in {run_dir}")
    entries: list[dict[str, Any]] = []
    with pred_file.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                entries.append(json.loads(line))
    return entries


def load_run_metrics(run_dir: Path) -> dict[str, Any]:
    """Load metrics.json from a run directory if available."""
    metrics_file = run_dir / "metrics.json"
    if metrics_file.exists():
        try:
            data = json.loads(metrics_file.read_text(encoding="utf-8"))
            if isinstance(data, dict):
                return data
        except Exception:
            pass
    return {}


def load_run_config(run_dir: Path) -> dict[str, Any]:
    """Load config.json from a run directory if available."""
    cfg_file = run_dir / "config.json"
    if cfg_file.exists():
        try:
            data = json.loads(cfg_file.read_text(encoding="utf-8"))
            if isinstance(data, dict):
                return data
        except Exception:
            pass
    return {}


def build_pr_metrics_from_predictions(entries: list[dict[str, Any]]) -> list[PRMetrics]:
    """Reconstruct PRMetrics list from predictions.jsonl records."""
    metrics_list: list[PRMetrics] = []
    for entry in entries:
        pr_id = entry.get("pr_id", "")
        repo = entry.get("repo", "")
        lang = entry.get("language", "python")
        is_clean = entry.get("is_clean", False)
        preds_raw = entry.get("predictions", [])
        golds_raw = entry.get("gold_findings", [])
        tp = entry.get("tp", 0)
        fp = entry.get("fp", len(preds_raw) - tp)
        fn = entry.get("fn", len(golds_raw) - tp)

        # Parse gold findings
        golds: list[GoldFinding] = []
        for g in golds_raw:
            try:
                golds.append(GoldFinding.model_validate(g))
            except Exception:
                pass

        # Parse match results if present, or construct synthetic approximations
        match_results_raw = entry.get("match_results", [])
        match_results: list[MatchResult] = []
        for mr in match_results_raw:
            try:
                match_results.append(MatchResult.model_validate(mr))
            except Exception:
                pass

        actionable = sum(
            1
            for p in preds_raw
            if p.get("suggestion") or p.get("severity", "").upper() in ("HIGH", "MEDIUM")
        )

        metrics_list.append(
            PRMetrics(
                pr_id=pr_id,
                repo=repo,
                language=lang,
                is_clean=is_clean or len(golds_raw) == 0,
                num_predictions=len(preds_raw),
                num_golds=len(golds_raw),
                true_positives=tp,
                false_positives=fp,
                false_negatives=fn,
                match_results=match_results,
                gold_findings=golds,
                actionable_predictions=actionable,
                latency_ms=entry.get("latency_ms", 0.0),
                cost_usd=entry.get("cost_usd", 0.0),
                input_tokens=entry.get("input_tokens", 0),
                output_tokens=entry.get("output_tokens", 0),
            )
        )
    return metrics_list


def compare_runs(
    run_a_dir: str | Path,
    run_b_dir: str | Path,
    output_dir: Path | None = None,
    n_resamples: int = 1000,
    seed: int = 42,
) -> dict[str, Any]:
    """Compare two evaluation runs and compute bootstrap significance tests.

    Args:
        run_a_dir: Path or ID of baseline run A.
        run_b_dir: Path or ID of candidate run B.
        output_dir: Optional directory to write comparison.md and comparison.csv.
        n_resamples: Number of bootstrap iterations.
        seed: Random seed.

    Returns:
        dict containing comparison rows, test results, markdown and CSV tables.
    """
    path_a = resolve_run_dir(run_a_dir)
    path_b = resolve_run_dir(run_b_dir)

    preds_a = load_run_predictions(path_a)
    preds_b = load_run_predictions(path_b)
    cfg_a = load_run_config(path_a)
    cfg_b = load_run_config(path_b)

    name_a = path_a.name
    name_b = path_b.name
    model_a = cfg_a.get("pipeline", {}).get("model", "run_A")
    model_b = cfg_b.get("pipeline", {}).get("model", "run_B")

    # Match PRs by ID
    map_a = {p["pr_id"]: p for p in preds_a}
    map_b = {p["pr_id"]: p for p in preds_b}
    common_ids = sorted(set(map_a.keys()) & set(map_b.keys()))

    if not common_ids:
        raise ValueError(
            f"No common PR IDs found between {name_a} ({len(preds_a)} PRs) and {name_b} ({len(preds_b)} PRs)"
        )

    common_preds_a = [map_a[pid] for pid in common_ids]
    common_preds_b = [map_b[pid] for pid in common_ids]

    pr_metrics_a = build_pr_metrics_from_predictions(common_preds_a)
    pr_metrics_b = build_pr_metrics_from_predictions(common_preds_b)

    eval_a = aggregate_metrics(pr_metrics_a)
    eval_b = aggregate_metrics(pr_metrics_b)

    # Perform paired bootstrap significance tests on primary metrics
    tests: dict[str, dict[str, Any]] = {
        "F1": paired_bootstrap_test(pr_metrics_a, pr_metrics_b, _f1_fn, n_resamples, seed),
        "Precision": paired_bootstrap_test(
            pr_metrics_a, pr_metrics_b, _precision_fn, n_resamples, seed
        ),
        "Recall": paired_bootstrap_test(pr_metrics_a, pr_metrics_b, _recall_fn, n_resamples, seed),
        "False Positive Rate": paired_bootstrap_test(
            pr_metrics_a, pr_metrics_b, _fpr_fn, n_resamples, seed
        ),
        "Comments/PR": paired_bootstrap_test(
            pr_metrics_a, pr_metrics_b, _comments_per_pr_fn, n_resamples, seed
        ),
        "Actionability": paired_bootstrap_test(
            pr_metrics_a, pr_metrics_b, _actionability_fn, n_resamples, seed
        ),
        "Severity-Weighted Recall": paired_bootstrap_test(
            pr_metrics_a, pr_metrics_b, _severity_weighted_recall_fn, n_resamples, seed
        ),
    }

    # Format comparison table rows
    # Columns: Metric | Run A | Run B | Delta (B - A) | p-value | Significant (α=0.05)
    headers = [
        "Metric",
        f"Run A ({name_a})",
        f"Run B ({name_b})",
        "Delta (B - A)",
        "p-value",
        "Significant (α=0.05)",
    ]

    metric_rows: list[tuple[str, float, float, str]] = [
        ("Precision", eval_a.precision, eval_b.precision, "Precision"),
        ("Recall", eval_a.recall, eval_b.recall, "Recall"),
        ("F1", eval_a.f1, eval_b.f1, "F1"),
        (
            "False Positive Rate",
            eval_a.false_positive_rate,
            eval_b.false_positive_rate,
            "False Positive Rate",
        ),
        ("Comments/PR", eval_a.comments_per_pr, eval_b.comments_per_pr, "Comments/PR"),
        ("Actionability", eval_a.actionability, eval_b.actionability, "Actionability"),
        (
            "Severity-Weighted Recall",
            eval_a.severity_weighted_recall,
            eval_b.severity_weighted_recall,
            "Severity-Weighted Recall",
        ),
        ("Cost/Review ($)", eval_a.cost_per_review, eval_b.cost_per_review, ""),
        ("Latency p50 (ms)", eval_a.latency_p50, eval_b.latency_p50, ""),
    ]

    table_data: list[list[str]] = []
    for label, val_a, val_b, test_key in metric_rows:
        delta = val_b - val_a
        delta_str = f"{delta:+.4f}"
        if test_key and test_key in tests:
            t = tests[test_key]
            pval_str = f"{t['p_value']:.4f}"
            sig_str = "✅ Yes" if t["significant"] else "❌ No"
        else:
            pval_str = "—"
            sig_str = "—"

        table_data.append(
            [
                label,
                f"{val_a:.4f}" if label != "Cost/Review ($)" else f"${val_a:.6f}",
                f"{val_b:.4f}" if label != "Cost/Review ($)" else f"${val_b:.6f}",
                delta_str,
                pval_str,
                sig_str,
            ]
        )

    # Markdown generation
    md_lines: list[str] = [
        f"# Evaluation Run Comparison: {name_a} vs {name_b}\n",
        f"- **Run A:** `{name_a}` (Model: `{model_a}`, PRs: {len(pr_metrics_a)})\n",
        f"- **Run B:** `{name_b}` (Model: `{model_b}`, PRs: {len(pr_metrics_b)})\n",
        f"- **Common PRs Analyzed:** {len(common_ids)}\n",
        f"- **Bootstrap Resamples:** {n_resamples} (seed: {seed})\n",
        "\n## Comparison Table\n",
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for row in table_data:
        md_lines.append("| " + " | ".join(row) + " |")

    f1_test = tests["F1"]
    md_lines.extend(
        [
            "",
            "## Significance Summary",
            f"- **F1 Difference:** {f1_test['delta']:+.4f}",
            f"- **F1 Two-Sided p-value:** {f1_test['p_value']:.4f}",
            f"- **One-Sided p-value (B > A):** {f1_test['one_sided_p_value']:.4f}",
            f"- **Statistically Significant at α=0.05:** {'✅ Yes' if f1_test['significant'] else '❌ No'}",
            "",
        ]
    )
    md_content = "\n".join(md_lines)

    # CSV generation
    csv_buf = io.StringIO()
    writer = csv.writer(csv_buf)
    writer.writerow(headers)
    writer.writerows(table_data)
    csv_content = csv_buf.getvalue()

    # Save outputs if output_dir specified, or default to runs/
    target_dir = output_dir or Path("runs")
    target_dir.mkdir(parents=True, exist_ok=True)
    comp_file_base = f"comparison_{name_a}_vs_{name_b}"
    md_path = target_dir / f"{comp_file_base}.md"
    csv_path = target_dir / f"{comp_file_base}.csv"

    md_path.write_text(md_content, encoding="utf-8")
    csv_path.write_text(csv_content, encoding="utf-8")

    return {
        "headers": headers,
        "rows": table_data,
        "tests": tests,
        "markdown": md_content,
        "csv": csv_content,
        "md_path": md_path,
        "csv_path": csv_path,
    }


def print_comparison_table(result: dict[str, Any]) -> None:
    """Print the formatted comparison table to stdout."""
    headers = result["headers"]
    rows = result["rows"]

    col_widths = [len(h) for h in headers]
    for row in rows:
        for idx, cell in enumerate(row):
            col_widths[idx] = max(col_widths[idx], len(cell))

    total_width = sum(col_widths) + 3 * (len(headers) - 1) + 4
    border = "=" * total_width
    header_str = " | ".join(f"{h:<{col_widths[i]}}" for i, h in enumerate(headers))
    sep_str = "-+-".join("-" * col_widths[i] for i in range(len(headers)))

    print("\n" + border)
    print("  " + header_str)
    print("  " + sep_str)
    for row in rows:
        line_str = " | ".join(f"{row[i]:<{col_widths[i]}}" for i in range(len(row)))
        print("  " + line_str)
    print(border + "\n")


def main() -> None:
    """CLI entry point for comparing two evaluation runs."""
    parser = argparse.ArgumentParser(
        description="Compare two Prism Review evaluation runs with bootstrap p-values"
    )
    parser.add_argument("run_a", type=str, help="Baseline run directory or run ID")
    parser.add_argument("run_b", type=str, help="Candidate run directory or run ID")
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Directory to save comparison.md and comparison.csv",
    )
    parser.add_argument(
        "--resamples",
        type=int,
        default=1000,
        help="Number of bootstrap resamples (default: 1000)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for bootstrap test",
    )
    args = parser.parse_args()

    out_dir = Path(args.output_dir) if args.output_dir else None

    try:
        res = compare_runs(
            run_a_dir=args.run_a,
            run_b_dir=args.run_b,
            output_dir=out_dir,
            n_resamples=args.resamples,
            seed=args.seed,
        )
        print_comparison_table(res)
        print("Saved comparison files:")
        print(f"  Markdown: {res['md_path']}")
        print(f"  CSV:      {res['csv_path']}")
    except Exception as exc:
        logger.error("Comparison failed: %s", exc)
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
