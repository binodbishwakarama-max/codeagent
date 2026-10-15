"""Evaluation harness runner for executing benchmark suites against review pipelines.

Usage:
    python -m evals.harness.runner --config evals/configs/baseline.yaml --split dev
    python -m evals.harness.run --config evals/configs/baseline.yaml --split dev

Runs the review pipeline (importing prism.review directly, no GitHub calls)
over a dataset split with concurrency, caching, and a fixed seed. Saves
predictions to runs/<run_id>/predictions.jsonl plus a full copy of the config
and git SHA.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import logging
import random
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from evals.dataset.labeling import load_jsonl
from evals.dataset.schema import PRExample
from evals.harness.baselines import LinterBaseline, RandomBaseline
from evals.harness.config import ExperimentConfig, PipelineConfig
from evals.harness.matcher import EmbeddingMatcher, MultiMatcher, SemanticMatcher
from evals.harness.metrics import (
    EvalMetrics,
    PRMetrics,
    aggregate_metrics,
    compute_all_bootstrap_cis,
    compute_pr_metrics,
)
from evals.harness.report import generate_run_report
from prism.models import DiffLine, FileDiff, Hunk, ReviewFinding, ReviewResult

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Diff Parsing (unified diff string → FileDiff objects)
# ---------------------------------------------------------------------------


def parse_unified_diff(diff_text: str, language: str = "python") -> list[FileDiff]:
    """Parse a unified diff string into structured FileDiff objects.

    Allows the evaluation harness to feed diffs directly through the review
    pipeline without making any external GitHub API calls.
    """
    file_diffs: list[FileDiff] = []
    current_path = ""
    current_hunks: list[Hunk] = []
    current_lines: list[DiffLine] = []
    old_start = 0
    new_start = 0
    old_line = 0
    new_line = 0

    for raw_line in diff_text.split("\n"):
        if raw_line.startswith("+++ b/"):
            # Flush previous file
            if current_path:
                if current_lines:
                    current_hunks.append(
                        Hunk(old_start=old_start, new_start=new_start, lines=current_lines)
                    )
                file_diffs.append(
                    FileDiff(path=current_path, language=language, hunks=current_hunks)
                )
            current_path = raw_line[6:].strip()
            current_hunks = []
            current_lines = []
        elif raw_line.startswith("--- a/"):
            continue
        elif raw_line.startswith("@@ "):
            # Flush previous hunk
            if current_lines:
                current_hunks.append(
                    Hunk(old_start=old_start, new_start=new_start, lines=current_lines)
                )
                current_lines = []

            # Parse @@ -old_start,old_len +new_start,new_len @@
            import re

            match = re.search(r"-(\d+)(?:,\d+)?\s+\+(\d+)(?:,\d+)?", raw_line)
            if match:
                old_start = int(match.group(1))
                new_start = int(match.group(2))
                old_line = old_start
                new_line = new_start
        elif raw_line.startswith("+"):
            current_lines.append(
                DiffLine(kind="add", old_no=None, new_no=new_line, text=raw_line[1:])
            )
            new_line += 1
        elif raw_line.startswith("-"):
            current_lines.append(
                DiffLine(kind="del", old_no=old_line, new_no=None, text=raw_line[1:])
            )
            old_line += 1
        else:
            # Context line
            text = raw_line[1:] if raw_line.startswith(" ") else raw_line
            current_lines.append(DiffLine(kind="ctx", old_no=old_line, new_no=new_line, text=text))
            old_line += 1
            new_line += 1

    # Flush final file
    if current_path:
        if current_lines:
            current_hunks.append(
                Hunk(old_start=old_start, new_start=new_start, lines=current_lines)
            )
        file_diffs.append(FileDiff(path=current_path, language=language, hunks=current_hunks))

    return file_diffs


# ---------------------------------------------------------------------------
# Git SHA Helper
# ---------------------------------------------------------------------------


def get_git_sha() -> str:
    """Get current git HEAD SHA, or 'unknown' if not in a git repo."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode == 0:
            return result.stdout.strip()[:12]
    except (subprocess.SubprocessError, FileNotFoundError):
        pass
    return "unknown"


# ---------------------------------------------------------------------------
# Run ID Generation
# ---------------------------------------------------------------------------


def generate_run_id(config_name: str) -> str:
    """Generate a unique run ID from config name and timestamp."""
    ts = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
    return f"{config_name}_{ts}"


# ---------------------------------------------------------------------------
# Core Runner
# ---------------------------------------------------------------------------


async def run_pipeline_on_example(
    example: PRExample,
    pipeline_config: PipelineConfig,
    use_mock: bool = True,
) -> tuple[list[ReviewFinding], ReviewResult]:
    """Run the review pipeline on a single PR example without GitHub calls.

    Imports prism.review.pipeline directly and constructs a PRContext from
    the dataset example.
    """
    from prism.review.pipeline import PRContext, review_pr

    file_diffs = parse_unified_diff(example.diff, example.language)

    pr_context = PRContext(
        repo_full_name=example.repo,
        pr_number=example.pr_number,
        title=example.title,
        description=example.body,
        head_sha=example.head_sha,
        file_diffs=file_diffs,
    )

    from prism.config import Settings
    from prism.review.llm import create_llm_client

    settings = Settings(
        github_app_id="eval-app",
        github_private_key="fake-key",
        github_webhook_secret="fake-secret",
        database_url="sqlite+aiosqlite:///eval.db",
        redis_url="redis://localhost:6379/0",
        llm_model=pipeline_config.model,
        llm_timeout_seconds=pipeline_config.timeout_seconds,
        confidence_threshold=pipeline_config.confidence_threshold,
        max_comments_per_pr=pipeline_config.max_comments_per_pr,
        retrieval_enabled=pipeline_config.retrieval_enabled,
        retrieval_mode=pipeline_config.retrieval_mode,
        reranker_enabled=pipeline_config.reranker_enabled,
        retrieval_top_k=pipeline_config.retrieval_top_k,
        self_verification_enabled=pipeline_config.self_verification_enabled,
        prompt_variant=pipeline_config.prompt_variant,
    )

    if use_mock:
        from evals.dataset.label import DeterministicMockAnnotator

        mock_client = DeterministicMockAnnotator(
            annotator_id="eval",
            model=pipeline_config.model,
        )
        result = await review_pr(
            pr_context=pr_context,
            settings=settings,
            llm_client=mock_client,
        )
    else:
        client = create_llm_client(settings)
        result = await review_pr(
            pr_context=pr_context,
            settings=settings,
            llm_client=client,
        )

    # Filter findings by confidence threshold and limit max comments
    filtered = [f for f in result.findings if f.confidence >= pipeline_config.confidence_threshold][
        : pipeline_config.max_comments_per_pr
    ]

    return filtered, result


async def run_evaluation_async(
    config: ExperimentConfig,
    split: str = "dev",
    run_dir: Path | None = None,
    use_mock: bool = True,
) -> tuple[EvalMetrics, list[PRMetrics]]:
    """Run full evaluation pipeline asynchronously with concurrency and fixed seed.

    1. Load dataset split (sorted deterministically)
    2. Run pipeline and baselines concurrently (bounded by max_concurrency)
    3. Match predictions to gold findings
    4. Compute metrics and all bootstrap CIs
    5. Save predictions.jsonl, config.json, report.md, and metrics.json
    """
    # Fix random seed for reproducibility
    random.seed(config.pipeline.seed)

    run_id = generate_run_id(config.name)
    if run_dir is None:
        run_dir = Path("runs") / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    git_sha = get_git_sha()

    # Save full config and metadata
    config_copy = run_dir / "config.json"
    config_dict = config.model_dump()
    config_dict["_split"] = split
    config_dict["_git_sha"] = git_sha
    config_dict["_run_id"] = run_id
    config_copy.write_text(json.dumps(config_dict, indent=2), encoding="utf-8")

    # Load dataset split
    dataset_dir = Path(config.dataset.path)
    split_file = dataset_dir / f"{split}.jsonl"

    if not split_file.exists():
        logger.warning("Split file %s not found. Generating synthetic dataset.", split_file)
        from evals.dataset.label import run_labeling_pipeline

        await run_labeling_pipeline(
            input_path=Path("data/raw/prs.jsonl"),
            output_dir=dataset_dir,
            stats_out=Path("docs/dataset-stats.md"),
            use_mock=True,
        )

    examples = load_jsonl(split_file) if split_file.exists() else []

    # Sort examples deterministically by ID for 100% reproducible ordering
    examples = sorted(examples, key=lambda e: e.id)

    # Apply sample limit if configured
    if config.dataset.sample_limit and len(examples) > config.dataset.sample_limit:
        examples = examples[: config.dataset.sample_limit]

    logger.info(
        "Loaded %d %s examples for evaluation run '%s' (seed: %d, concurrency: %d)",
        len(examples),
        split,
        run_id,
        config.pipeline.seed,
        config.pipeline.max_concurrency,
    )

    # Initialize matchers
    semantic = SemanticMatcher(
        cache_dir=config.matcher.cache_dir,
        model=config.matcher.semantic_judge_model,
    )
    embedding = EmbeddingMatcher(threshold=config.matcher.embedding_threshold)
    multi_matcher = MultiMatcher(
        semantic=semantic,
        embedding=embedding,
        line_tolerance=config.matcher.line_tolerance,
    )

    # Initialize baselines with fixed seeds
    random_baseline = RandomBaseline(
        findings_per_pr=config.baseline.random_findings_per_pr,
        seed=config.baseline.random_seed,
    )
    linter_baseline = LinterBaseline()

    # Concurrency control
    sem = asyncio.Semaphore(config.pipeline.max_concurrency)

    # Cache directory for pipeline predictions
    cache_dir = Path(config.matcher.cache_dir) / "pipeline_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)

    async def evaluate_single_example(
        example: PRExample,
    ) -> tuple[PRMetrics, PRMetrics, PRMetrics, dict[str, Any]]:
        async with sem:
            # Deterministic cache key for pipeline result — must include all
            # variant-specific parameters to avoid cross-ablation cache collisions.
            cache_key = hashlib.sha256(
                f"{config.name}:{config.pipeline.model}:{config.pipeline.prompt_variant}:"
                f"{config.pipeline.retrieval_enabled}:{config.pipeline.retrieval_mode}:"
                f"{config.pipeline.reranker_enabled}:{config.pipeline.self_verification_enabled}:"
                f"{config.pipeline.confidence_threshold}:{config.pipeline.seed}:{example.id}".encode()
            ).hexdigest()[:24]
            cache_file = cache_dir / f"{cache_key}.json"

            findings: list[ReviewFinding]
            result: ReviewResult

            if cache_file.exists():
                try:
                    cached_data = json.loads(cache_file.read_text(encoding="utf-8"))
                    findings = [ReviewFinding.model_validate(f) for f in cached_data["findings"]]
                    result = ReviewResult.model_validate(cached_data["result"])
                except Exception:
                    findings, result = await run_pipeline_on_example(
                        example, config.pipeline, use_mock=use_mock
                    )
            else:
                try:
                    findings, result = await run_pipeline_on_example(
                        example, config.pipeline, use_mock=use_mock
                    )
                    # Cache result
                    cache_payload = {
                        "findings": [f.model_dump() for f in findings],
                        "result": result.model_dump(),
                    }
                    cache_file.write_text(json.dumps(cache_payload), encoding="utf-8")
                except Exception as exc:
                    logger.error("Pipeline failed on %s: %s", example.id, exc)
                    findings = []
                    result = ReviewResult(
                        findings=[],
                        model="error",
                        prompt_version="error",
                        input_tokens=0,
                        output_tokens=0,
                        latency_ms=0.0,
                        cost_usd=0.0,
                    )

            # Match and compute PR metrics
            pr_metric = compute_pr_metrics(example, findings, multi_matcher, result)

            # Baselines
            random_res = random_baseline.review(example)
            random_metric = compute_pr_metrics(
                example, random_res.findings, multi_matcher, random_res
            )

            linter_res = linter_baseline.review(example)
            linter_metric = compute_pr_metrics(
                example, linter_res.findings, multi_matcher, linter_res
            )

            # Prediction record
            pred_record = {
                "pr_id": example.id,
                "repo": example.repo,
                "pr_number": example.pr_number,
                "language": example.language,
                "is_clean": example.is_clean,
                "predictions": [f.model_dump() for f in findings],
                "gold_findings": [g.model_dump() for g in example.gold_findings],
                "match_results": [r.model_dump() for r in pr_metric.match_results],
                "tp": pr_metric.true_positives,
                "fp": pr_metric.false_positives,
                "fn": pr_metric.false_negatives,
                "latency_ms": result.latency_ms,
                "cost_usd": result.cost_usd,
                "input_tokens": result.input_tokens,
                "output_tokens": result.output_tokens,
            }

            return pr_metric, random_metric, linter_metric, pred_record

    # Execute all PR evaluations concurrently
    tasks = [evaluate_single_example(ex) for ex in examples]
    evaluated_results = await asyncio.gather(*tasks)

    # Unpack results in deterministic order matching examples
    all_pr_metrics = [r[0] for r in evaluated_results]
    baseline_random_metrics = [r[1] for r in evaluated_results]
    baseline_linter_metrics = [r[2] for r in evaluated_results]
    predictions_log = [r[3] for r in evaluated_results]

    # Save predictions.jsonl deterministically
    pred_path = run_dir / "predictions.jsonl"
    with pred_path.open("w", encoding="utf-8") as f:
        for entry in predictions_log:
            f.write(json.dumps(entry, sort_keys=True) + "\n")

    # Compute aggregate metrics
    metrics = aggregate_metrics(all_pr_metrics)
    random_metrics = aggregate_metrics(baseline_random_metrics)
    linter_metrics = aggregate_metrics(baseline_linter_metrics)

    # Compute bootstrap CIs for all aggregate metrics
    if len(all_pr_metrics) >= 2:
        cis = compute_all_bootstrap_cis(
            all_pr_metrics,
            n_resamples=config.bootstrap.n_resamples,
            confidence_level=config.bootstrap.confidence_level,
            seed=config.bootstrap.seed,
        )
        metrics.confidence_intervals = cis
        metrics.precision_ci = cis.get("precision")
        metrics.recall_ci = cis.get("recall")
        metrics.f1_ci = cis.get("f1")

    # Generate markdown report and metrics.json
    generate_run_report(
        run_dir=run_dir,
        config=config_dict,
        metrics=metrics,
        pr_metrics=all_pr_metrics,
        git_sha=git_sha,
    )

    # Print summary table to stdout
    _print_metrics_table(metrics, random_metrics, linter_metrics)

    logger.info("Evaluation run '%s' complete. Results in %s", run_id, run_dir)
    return metrics, all_pr_metrics


def _print_metrics_table(
    pipeline: EvalMetrics,
    random_bl: EvalMetrics,
    linter_bl: EvalMetrics,
) -> None:
    """Print a formatted metrics comparison table to stdout."""
    print("\n" + "=" * 74)
    print("  EVALUATION RESULTS")
    print("=" * 74)
    print(f"  {'Metric':<30} {'Pipeline':>12} {'Random BL':>12} {'Linter BL':>12}")
    print("-" * 74)

    rows = [
        ("Precision", pipeline.precision, random_bl.precision, linter_bl.precision),
        ("Recall", pipeline.recall, random_bl.recall, linter_bl.recall),
        ("F1", pipeline.f1, random_bl.f1, linter_bl.f1),
        (
            "FP Rate (clean)",
            pipeline.false_positive_rate,
            random_bl.false_positive_rate,
            linter_bl.false_positive_rate,
        ),
        (
            "Comments/PR",
            pipeline.comments_per_pr,
            random_bl.comments_per_pr,
            linter_bl.comments_per_pr,
        ),
        (
            "Actionability",
            pipeline.actionability,
            random_bl.actionability,
            linter_bl.actionability,
        ),
        (
            "Severity-Weighted Recall",
            pipeline.severity_weighted_recall,
            random_bl.severity_weighted_recall,
            linter_bl.severity_weighted_recall,
        ),
        (
            "Cost/Review ($)",
            pipeline.cost_per_review,
            random_bl.cost_per_review,
            linter_bl.cost_per_review,
        ),
    ]

    for name, pipe_val, rand_val, lint_val in rows:
        print(f"  {name:<30} {pipe_val:>12.4f} {rand_val:>12.4f} {lint_val:>12.4f}")

    if pipeline.confidence_intervals:
        print("\n  Bootstrap 95% CIs:")
        for m_name, ci_val in pipeline.confidence_intervals.items():
            print(f"    {m_name:<26}: {ci_val}")

    print("=" * 74 + "\n")


# ---------------------------------------------------------------------------
# Synchronous Wrapper
# ---------------------------------------------------------------------------


def run_evaluation(
    config_path: str,
    split: str = "dev",
    run_dir: Path | None = None,
    use_mock: bool = True,
) -> int:
    """Run evaluation benchmark using the provided experiment configuration file.

    Synchronous entry point called by make eval-smoke, eval-full, and tests.
    Returns 0 on success, 1 on failure.
    """
    try:
        config = ExperimentConfig.from_yaml(config_path)
    except FileNotFoundError:
        logger.error("Config file not found: %s", config_path)
        return 1
    except Exception as exc:
        logger.error("Failed to parse config %s: %s", config_path, exc)
        return 1

    try:
        asyncio.run(
            run_evaluation_async(
                config,
                split=split,
                run_dir=run_dir,
                use_mock=use_mock,
            )
        )
        return 0
    except Exception as exc:
        logger.error("Evaluation run failed: %s", exc, exc_info=True)
        return 1


# ---------------------------------------------------------------------------
# CLI Entry Point
# ---------------------------------------------------------------------------


def main() -> None:
    """CLI entrypoint for evaluation harness runner."""
    parser = argparse.ArgumentParser(description="Prism Review Evaluation Harness Runner")
    parser.add_argument(
        "--config",
        type=str,
        default="evals/configs/baseline.yaml",
        help="Path to experiment config YAML/YML",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="dev",
        choices=["train", "dev", "test"],
        help="Dataset split to evaluate on",
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        default=None,
        help="Override max concurrent evaluations",
    )
    parser.add_argument(
        "--mock",
        action="store_true",
        default=True,
        help="Use deterministic mock annotator (default: True)",
    )
    parser.add_argument(
        "--no-mock",
        dest="mock",
        action="store_false",
        help="Use live LLM client API calls",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    sys.exit(
        run_evaluation(
            config_path=args.config,
            split=args.split,
            use_mock=args.mock,
        )
    )


if __name__ == "__main__":
    main()
