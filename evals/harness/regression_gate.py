"""Regression evaluation gate for CI pull requests.

Runs the fixed 40-PR regression evaluation set from the dev split, compares
results against evals/baselines/main.json using paired bootstrap significance testing,
and posts a sticky PR comment with metric, cost, and latency deltas.

Gate Criteria:
1. F1 score must not drop by more than 3 points (delta >= -0.03).
2. False-positive rate on clean PRs must not rise by more than 2 points beyond
   the CI noise (delta <= +0.02 or within bootstrap CI noise).
3. Total evaluation cost must not exceed the cost cap ($1.00).

Can be overridden if the PR has the 'eval-override' label AND contains a valid
justification in the PR body.
"""

from __future__ import annotations

import argparse
import asyncio
import datetime
import json
import logging
import os
import random
import re
import subprocess
import sys
from pathlib import Path
from typing import Any, cast

# Ensure stdout and stderr support UTF-8 on Windows
if sys.platform == "win32":
    try:
        if hasattr(sys.stdout, "reconfigure"):
            cast(Any, sys.stdout).reconfigure(encoding="utf-8")
        if hasattr(sys.stderr, "reconfigure"):
            cast(Any, sys.stderr).reconfigure(encoding="utf-8")
    except Exception:
        pass

from evals.harness.compare import build_pr_metrics_from_predictions
from evals.harness.config import ExperimentConfig
from evals.harness.metrics import (
    PRMetrics,
    _f1_fn,
    _fpr_fn,
    _precision_fn,
    _recall_fn,
    aggregate_metrics,
    paired_bootstrap_test,
)
from evals.harness.runner import run_evaluation_async

logger = logging.getLogger("evals.regression_gate")

STICKY_TAG = "<!-- prism-eval-regression-gate -->"
DEFAULT_BASELINE_PATH = Path("evals/baselines/main.json")
DEFAULT_CONFIG_PATH = Path("evals/configs/regression.yaml")


def get_current_git_branch() -> str:
    """Get the current active git branch name."""
    try:
        res = subprocess.run(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
            timeout=5,
        )
        return res.stdout.strip()
    except Exception:
        return "unknown"


def get_current_git_sha() -> str:
    """Get current git commit SHA."""
    try:
        res = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
            timeout=5,
        )
        return res.stdout.strip()[:12]
    except Exception:
        return "unknown"


def compute_paired_delta_ci(
    metrics_a: list[PRMetrics],
    metrics_b: list[PRMetrics],
    metric_fn: Any,
    n_resamples: int = 1000,
    confidence_level: float = 0.95,
    seed: int = 42,
) -> tuple[float, float]:
    """Compute bootstrap confidence interval for paired delta (B - A)."""
    n = len(metrics_a)
    if n == 0 or len(metrics_b) != n:
        return 0.0, 0.0

    rng = random.Random(seed)
    diffs: list[float] = []
    for _ in range(n_resamples):
        indices = [rng.randint(0, n - 1) for _ in range(n)]
        sample_a = [metrics_a[i] for i in indices]
        sample_b = [metrics_b[i] for i in indices]
        diffs.append(metric_fn(sample_b) - metric_fn(sample_a))

    diffs.sort()
    alpha = (1.0 - confidence_level) / 2.0
    lower_idx = int(alpha * n_resamples)
    upper_idx = int((1.0 - alpha) * n_resamples)
    return round(diffs[lower_idx], 4), round(diffs[upper_idx], 4)


def update_baseline_file(
    config_path: Path = DEFAULT_CONFIG_PATH,
    output_path: Path = DEFAULT_BASELINE_PATH,
    split: str = "dev",
    force: bool = False,
) -> int:
    """Refresh evals/baselines/main.json from the 40-PR regression set.

    Only runnable on main/master unless force is True.
    """
    branch = get_current_git_branch()
    if not force and branch not in ("main", "master", "HEAD"):
        logger.error(
            "Cannot update baseline: must be run on 'main' branch (current branch: '%s'). "
            "Use --force to override for testing.",
            branch,
        )
        print(f"Error: make update-baseline can only be run on main branch (current: {branch})")
        return 1

    print(f"--> Updating baseline from 40-PR regression set (split={split})...")
    config = ExperimentConfig.from_yaml(config_path)

    run_dir = Path("runs/baseline_update_tmp")
    eval_metrics, pr_metrics = asyncio.run(
        run_evaluation_async(config, split=split, run_dir=run_dir, use_mock=True)
    )

    pred_file = run_dir / "predictions.jsonl"
    predictions: list[dict[str, Any]] = []
    if pred_file.exists():
        with pred_file.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    predictions.append(json.loads(line))

    baseline_data: dict[str, Any] = {
        "name": "main-baseline",
        "git_sha": get_current_git_sha(),
        "timestamp": datetime.datetime.now(datetime.UTC).isoformat(),
        "config_name": config.name,
        "split": split,
        "sample_limit": config.dataset.sample_limit,
        "total_prs": eval_metrics.total_prs,
        "metrics": {
            "f1": round(eval_metrics.f1, 4),
            "precision": round(eval_metrics.precision, 4),
            "recall": round(eval_metrics.recall, 4),
            "false_positive_rate": round(eval_metrics.false_positive_rate, 4),
            "comments_per_pr": round(eval_metrics.comments_per_pr, 4),
            "actionability": round(eval_metrics.actionability, 4),
            "severity_weighted_recall": round(eval_metrics.severity_weighted_recall, 4),
            "cost_per_review": round(eval_metrics.cost_per_review, 6),
            "latency_p50": round(eval_metrics.latency_p50, 2),
            "latency_p95": round(eval_metrics.latency_p95, 2),
        },
        "confidence_intervals": {
            k: {
                "value": round(ci.value, 4),
                "lower": round(ci.lower, 4),
                "upper": round(ci.upper, 4),
                "n_resamples": ci.n_resamples,
            }
            for k, ci in (eval_metrics.confidence_intervals or {}).items()
        },
        "predictions": predictions,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(baseline_data, indent=2), encoding="utf-8")
    print(
        f"--> Baseline saved to {output_path} (F1: {eval_metrics.f1:.4f}, FPR: {eval_metrics.false_positive_rate:.4f})"
    )
    return 0


def parse_pr_justification(body: str) -> tuple[bool, str]:
    """Check if PR body contains a valid justification for eval-override."""
    if not body or not body.strip():
        return False, "PR description body is empty."

    # Look for a specific justification section or general explanation
    patterns = [
        r"##\s*Eval Override Justification[:\s]*(.+)",
        r"###\s*Eval Override Justification[:\s]*(.+)",
        r"(?:Justification|Override Reason)[:\s]*(.+)",
    ]
    for pat in patterns:
        m = re.search(pat, body, flags=re.IGNORECASE | re.DOTALL)
        if m:
            text = m.group(1).strip()
            # Ignore template placeholders like [Explain why...]
            cleaned = re.sub(r"\[.*?\]", "", text).strip()
            if len(cleaned) >= 20:
                return True, cleaned

    # Fallback: check if total non-template text in body is substantial
    cleaned_body = re.sub(r"<!--.*?-->", "", body, flags=re.DOTALL)
    cleaned_body = re.sub(r"\[.*?\]", "", cleaned_body).strip()
    if len(cleaned_body) >= 40:
        return True, cleaned_body[:200]

    return False, "No substantive justification found in PR description body."


def run_regression_gate(
    config_path: Path = DEFAULT_CONFIG_PATH,
    baseline_path: Path = DEFAULT_BASELINE_PATH,
    split: str = "dev",
    output_comment: Path | None = None,
    f1_drop_threshold: float = 0.03,
    fpr_rise_threshold: float = 0.02,
    cost_cap: float = 1.0,
    pr_labels: list[str] | None = None,
    pr_body: str = "",
) -> int:
    """Run regression evaluation and compare candidate metrics against baseline."""
    if not baseline_path.exists():
        logger.error(
            "Baseline file %s not found. Run 'make update-baseline' on main first.", baseline_path
        )
        print(
            f"Error: Baseline file '{baseline_path}' not found. Run 'make update-baseline' on main first."
        )
        return 1

    baseline_data: dict[str, Any] = json.loads(baseline_path.read_text(encoding="utf-8"))
    baseline_preds: list[dict[str, Any]] = baseline_data.get("predictions", [])
    if not baseline_preds:
        logger.error("Baseline file %s contains no prediction records.", baseline_path)
        return 1

    pr_metrics_baseline = build_pr_metrics_from_predictions(baseline_preds)
    base_metrics = aggregate_metrics(pr_metrics_baseline)

    # 1. Run Candidate Evaluation
    config = ExperimentConfig.from_yaml(config_path)
    run_dir = Path("runs/regression_gate_candidate")
    cand_metrics, cand_pr_metrics = asyncio.run(
        run_evaluation_async(config, split=split, run_dir=run_dir, use_mock=True)
    )

    # 2. Check Cost Cap
    total_cost = sum(m.cost_usd for m in cand_pr_metrics)
    cost_cap_exceeded = total_cost > cost_cap

    # 3. Match PRs for Paired Bootstrap
    base_map = {m.pr_id: m for m in pr_metrics_baseline}
    cand_map = {m.pr_id: m for m in cand_pr_metrics}
    common_ids = sorted(set(base_map.keys()) & set(cand_map.keys()))

    if len(common_ids) < 10:
        logger.error("Too few common PRs between baseline and candidate (%d)", len(common_ids))
        return 1

    paired_base = [base_map[pid] for pid in common_ids]
    paired_cand = [cand_map[pid] for pid in common_ids]

    # 4. Significance Tests & Paired Differences
    f1_test = paired_bootstrap_test(paired_base, paired_cand, _f1_fn, n_resamples=1000, seed=42)
    fpr_test = paired_bootstrap_test(paired_base, paired_cand, _fpr_fn, n_resamples=1000, seed=42)
    prec_test = paired_bootstrap_test(
        paired_base, paired_cand, _precision_fn, n_resamples=1000, seed=42
    )
    rec_test = paired_bootstrap_test(
        paired_base, paired_cand, _recall_fn, n_resamples=1000, seed=42
    )

    f1_delta_ci = compute_paired_delta_ci(paired_base, paired_cand, _f1_fn)
    fpr_delta_ci = compute_paired_delta_ci(paired_base, paired_cand, _fpr_fn)
    prec_delta_ci = compute_paired_delta_ci(paired_base, paired_cand, _precision_fn)
    rec_delta_ci = compute_paired_delta_ci(paired_base, paired_cand, _recall_fn)

    # Deltas
    delta_f1 = cand_metrics.f1 - base_metrics.f1
    delta_fpr = cand_metrics.false_positive_rate - base_metrics.false_positive_rate
    delta_prec = cand_metrics.precision - base_metrics.precision
    delta_rec = cand_metrics.recall - base_metrics.recall
    delta_cost = cand_metrics.cost_per_review - base_metrics.cost_per_review
    delta_latency = cand_metrics.latency_p50 - base_metrics.latency_p50

    # 5. Evaluate Thresholds
    # F1 drop: fail if F1 drops by > 3 points (delta < -0.03)
    f1_failed = delta_f1 < -f1_drop_threshold

    # FPR rise: fail if false-positive rate rises by > 2 points beyond CI noise
    # CI noise is computed from the paired bootstrap distribution
    fpr_noise_margin = max(0.0, (fpr_delta_ci[1] - fpr_delta_ci[0]) / 2.0)
    # The rise is beyond noise if delta > 0.02 and statistically significant / lower CI bound > 0
    fpr_failed = (delta_fpr > fpr_rise_threshold) and (
        fpr_test.get("significant", False)
        or fpr_delta_ci[0] > 0
        or delta_fpr > (fpr_rise_threshold + fpr_noise_margin)
    )

    is_regression = f1_failed or fpr_failed or cost_cap_exceeded

    # 6. Evaluate Override
    labels = [lbl.strip().lower() for lbl in (pr_labels or [])]
    has_override_label = "eval-override" in labels
    valid_justification, justification_reason = parse_pr_justification(pr_body)

    gate_passed: bool
    status_badge: str
    override_note: str = ""

    if not is_regression:
        gate_passed = True
        status_badge = "PASSED ✅"
    elif has_override_label and valid_justification:
        gate_passed = True
        status_badge = "OVERRIDDEN ⚠️"
        override_note = (
            "> [!NOTE]\n"
            "> **Regression Override Active**: The regression has been deliberately accepted via the `eval-override` label.\n"
            f"> **Justification**: _{justification_reason}_\n"
        )
    elif has_override_label and not valid_justification:
        gate_passed = False
        status_badge = "FAILED ❌ (Override Incomplete)"
        override_note = (
            "> [!WARNING]\n"
            "> **Override Rejected**: The PR is labeled with `eval-override`, but **lacks a valid justification** in the PR body.\n"
            "> Please add a `## Eval Override Justification` section explaining why this regression is acceptable.\n"
        )
    else:
        gate_passed = False
        status_badge = "FAILED ❌"

    # 7. Format Comment Markdown
    comment_lines: list[str] = [
        STICKY_TAG,
        f"## 🚦 Prism Review — Regression Eval Gate: {status_badge}\n",
    ]

    if override_note:
        comment_lines.append(override_note)

    comment_lines.append(
        f"Evaluated **{len(common_ids)} pull requests** from the `{split}` regression set against baseline "
        f"(`{baseline_data.get('git_sha', 'main')}`).\n"
    )

    # Markdown Table
    comment_lines.append(
        "| Metric | Baseline (`main`) | PR Candidate | Delta | 95% CI (Delta) | p-value | Gate Status |"
    )
    comment_lines.append("| :--- | :---: | :---: | :---: | :---: | :---: | :---: |")

    def _fmt_row(
        name: str,
        base_val: float,
        cand_val: float,
        delta_val: float,
        ci: tuple[float, float],
        pval: float | None,
        failed: bool,
        threshold_rule: str,
    ) -> str:
        delta_sign = "+" if delta_val > 0 else ""
        delta_str = f"{delta_sign}{delta_val:.4f}"
        ci_str = f"[{ci[0]:+.4f}, {ci[1]:+.4f}]" if ci != (0.0, 0.0) else "-"
        p_str = f"{pval:.4f}" if pval is not None else "-"
        icon = "❌ Fail" if failed else "✅ Pass"
        return f"| **{name}** | {base_val:.4f} | {cand_val:.4f} | `{delta_str}` | `{ci_str}` | {p_str} | {icon} ({threshold_rule}) |"

    comment_lines.append(
        _fmt_row(
            "F1 Score",
            base_metrics.f1,
            cand_metrics.f1,
            delta_f1,
            f1_delta_ci,
            f1_test.get("p_value"),
            f1_failed,
            f"max drop {f1_drop_threshold:.2f}",
        )
    )
    comment_lines.append(
        _fmt_row(
            "False Positive Rate (clean)",
            base_metrics.false_positive_rate,
            cand_metrics.false_positive_rate,
            delta_fpr,
            fpr_delta_ci,
            fpr_test.get("p_value"),
            fpr_failed,
            f"max rise {fpr_rise_threshold:.2f} + noise",
        )
    )
    comment_lines.append(
        _fmt_row(
            "Precision",
            base_metrics.precision,
            cand_metrics.precision,
            delta_prec,
            prec_delta_ci,
            prec_test.get("p_value"),
            False,
            "tracked",
        )
    )
    comment_lines.append(
        _fmt_row(
            "Recall",
            base_metrics.recall,
            cand_metrics.recall,
            delta_rec,
            rec_delta_ci,
            rec_test.get("p_value"),
            False,
            "tracked",
        )
    )

    # Cost and Latency rows
    cost_sign = "+" if delta_cost > 0 else ""
    lat_sign = "+" if delta_latency > 0 else ""
    comment_lines.append(
        f"| **Cost / Review** | ${base_metrics.cost_per_review:.6f} | ${cand_metrics.cost_per_review:.6f} | "
        f"`{cost_sign}${delta_cost:.6f}` | - | - | ℹ️ tracked |"
    )
    comment_lines.append(
        f"| **Latency (p50)** | {base_metrics.latency_p50:.2f} ms | {cand_metrics.latency_p50:.2f} ms | "
        f"`{lat_sign}{delta_latency:.2f} ms` | - | - | ℹ️ tracked |"
    )
    cost_status = "❌ Exceeded" if cost_cap_exceeded else "✅ OK"
    comment_lines.append(
        f"| **Total Run Cost** | - | ${total_cost:.4f} | - | - | - | {cost_status} (Cap: <${cost_cap:.2f}) |"
    )

    comment_lines.append("\n### Summary Breakdown\n")
    if is_regression and not (has_override_label and valid_justification):
        comment_lines.append("#### ❌ Failure Causes:")
        if f1_failed:
            comment_lines.append(
                f"- **F1 Score Dropped**: `{delta_f1:+.4f}` exceeds tolerance of `{-f1_drop_threshold:.4f}`."
            )
        if fpr_failed:
            comment_lines.append(
                f"- **False-Positive Rate Risen**: `{delta_fpr:+.4f}` exceeds noise-adjusted tolerance "
                f"(threshold `+{fpr_rise_threshold:.4f}` + CI noise margin `{fpr_noise_margin:.4f}`)."
            )
        if cost_cap_exceeded:
            comment_lines.append(
                f"- **Cost Cap Exceeded**: Total evaluation cost ${total_cost:.4f} exceeds ${cost_cap:.2f}."
            )

        comment_lines.append(
            "\n> **How to Proceed**:\n"
            "> 1. Fix the prompt, retrieval, or model regression to recover performance.\n"
            "> 2. If this regression is intentional (e.g. trading F1 for safety), add the `eval-override` label "
            "and detail your rationale in a `## Eval Override Justification` section in the PR description.\n"
        )
    else:
        comment_lines.append(
            "All regression gate criteria satisfied. Review quality meets baseline standards.\n"
        )

    comment_markdown = "\n".join(comment_lines)

    # Print to stdout
    print("\n" + "=" * 74)
    print(f"  REGRESSION GATE: {status_badge}")
    print("=" * 74)
    print(f"  Delta F1         : {delta_f1:+.4f} (p={f1_test.get('p_value', 1.0):.4f})")
    print(f"  Delta FPR (clean): {delta_fpr:+.4f} (p={fpr_test.get('p_value', 1.0):.4f})")
    print(f"  Delta Cost/PR    : ${delta_cost:+.6f}")
    print(f"  Delta Latency p50: {delta_latency:+.2f} ms")
    print(f"  Total Cost       : ${total_cost:.4f} (Cap: <${cost_cap:.2f})")
    print("=" * 74 + "\n")

    if output_comment:
        output_comment.parent.mkdir(parents=True, exist_ok=True)
        output_comment.write_text(comment_markdown, encoding="utf-8")
        print(f"--> Sticky comment markdown saved to {output_comment}")

    return 0 if gate_passed else 1


def main() -> None:
    """CLI entrypoint for regression gate."""
    parser = argparse.ArgumentParser(description="Prism Review Regression Gate")
    parser.add_argument(
        "--config", type=Path, default=DEFAULT_CONFIG_PATH, help="Path to regression config YAML"
    )
    parser.add_argument(
        "--baseline", type=Path, default=DEFAULT_BASELINE_PATH, help="Path to baseline main.json"
    )
    parser.add_argument("--split", type=str, default="dev", help="Dataset split to evaluate")
    parser.add_argument(
        "--update-baseline", action="store_true", help="Refresh baseline metrics and exit"
    )
    parser.add_argument(
        "--force", action="store_true", help="Force baseline update even if not on main branch"
    )
    parser.add_argument(
        "--output-comment",
        type=Path,
        default=None,
        help="File path to write sticky comment markdown",
    )
    parser.add_argument(
        "--f1-drop-threshold", type=float, default=0.03, help="Max allowable drop in F1"
    )
    parser.add_argument(
        "--fpr-rise-threshold", type=float, default=0.02, help="Max allowable rise in FPR"
    )
    parser.add_argument(
        "--cost-cap", type=float, default=1.0, help="Max allowable total eval cost in USD"
    )
    parser.add_argument(
        "--pr-labels", type=str, default=None, help="Comma-separated list of PR labels"
    )
    parser.add_argument("--pr-body", type=str, default=None, help="PR description body text")

    args = parser.parse_args()

    if args.update_baseline:
        ret = update_baseline_file(
            config_path=args.config,
            output_path=args.baseline,
            split=args.split,
            force=args.force,
        )
        sys.exit(ret)

    # Read PR context from args or environment variables (e.g. in GitHub Actions)
    pr_labels_str = args.pr_labels or os.environ.get("PR_LABELS", "")
    pr_labels_list = [lbl.strip() for lbl in pr_labels_str.split(",") if lbl.strip()]
    pr_body_text = args.pr_body or os.environ.get("PR_BODY", "")

    ret = run_regression_gate(
        config_path=args.config,
        baseline_path=args.baseline,
        split=args.split,
        output_comment=args.output_comment,
        f1_drop_threshold=args.f1_drop_threshold,
        fpr_rise_threshold=args.fpr_rise_threshold,
        cost_cap=args.cost_cap,
        pr_labels=pr_labels_list,
        pr_body=pr_body_text,
    )
    sys.exit(ret)


if __name__ == "__main__":
    main()
