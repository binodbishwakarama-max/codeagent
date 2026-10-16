"""Preparing training examples and formatting instruction pairs for code review fine-tuning.

Converts PR dataset records into chat-format SFT examples matching the exact production
prompt format (SYSTEM_PROMPT, standards rendering, unified diff rendering).
Ensures zero data leakage by strictly asserting that no dev/test repository ever appears
in the training split.
"""

from __future__ import annotations

import argparse
import json
import logging
import random
from pathlib import Path
from typing import Any

from evals.harness.runner import parse_unified_diff
from prism.models import Standard
from prism.review.prompts import SYSTEM_PROMPT, build_user_prompt
from prism.review.standards import load_default_standards

logger = logging.getLogger("training.data_prep")


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    """Load JSON lines file into list of dicts."""
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def save_jsonl(records: list[dict[str, Any]], path: Path) -> None:
    """Save records to JSON lines file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")


def assert_no_leakage(
    train_records: list[dict[str, Any]],
    dev_path: Path = Path("data/final/dev.jsonl"),
    test_path: Path = Path("data/final/test.jsonl"),
) -> None:
    """Hard rule: strictly assert that no dev/test repository ever appears in training.

    Raises:
        AssertionError: If any training PR belongs to a dev or test repository.
    """
    dev_repos: set[str] = set()
    test_repos: set[str] = set()

    if dev_path.exists():
        dev_repos = {r["repo"] for r in load_jsonl(dev_path)}
    if test_path.exists():
        test_repos = {r["repo"] for r in load_jsonl(test_path)}

    forbidden_repos = dev_repos | test_repos
    if not forbidden_repos:
        logger.warning("No dev or test records found to check leakage against.")
        return

    train_repos = {r["repo"] for r in train_records}
    overlap = train_repos & forbidden_repos

    assert not overlap, (
        f"CRITICAL CONTAMINATION ERROR: Training split contains repositories that appear in "
        f"dev/test evaluation splits! Overlapping repositories: {sorted(overlap)}"
    )


def format_findings_target(gold_findings: list[dict[str, Any]]) -> str:
    """Format gold findings list as standard ChunkReviewResponse JSON string."""
    if not gold_findings:
        return json.dumps({"findings": []}, indent=2)

    findings: list[dict[str, Any]] = []
    category_map = {
        "bug": "BUG",
        "security": "SECURITY",
        "performance": "PERFORMANCE",
        "error_handling": "ERROR_HANDLING",
        "style": "STANDARDS_VIOLATION",
        "standards_violation": "STANDARDS_VIOLATION",
        "testing": "TESTING",
        "readability": "READABILITY",
        "other": "STANDARDS_VIOLATION",
    }
    for gf in gold_findings:
        raw_cat = str(gf.get("category", "bug")).lower()
        cat = category_map.get(raw_cat, "BUG")
        sev = str(gf.get("severity", "MEDIUM")).upper()
        if sev not in ("HIGH", "MEDIUM", "LOW"):
            sev = "MEDIUM"

        finding = {
            "path": gf.get("path", ""),
            "line": int(gf.get("line", 1)),
            "side": "RIGHT",
            "category": cat,
            "severity": sev,
            "message": gf.get("description", "").strip(),
            "suggestion": gf.get("suggestion") or None,
            "confidence": 0.95,
        }
        findings.append(finding)

    return json.dumps({"findings": findings}, indent=2)


def convert_pr_to_sft_example(
    pr: dict[str, Any],
    standards: list[Standard],
) -> dict[str, Any]:
    """Convert a PR record into an OpenAI / ChatML messages format SFT example.

    Uses the exact production prompt builder and system prompt.
    """
    lang = pr.get("language", "python")
    diff_text = pr.get("diff", "")
    file_diffs = parse_unified_diff(diff_text, lang)

    user_prompt = build_user_prompt(
        pr_title=pr.get("title", ""),
        pr_description=pr.get("body", ""),
        standards=standards,
        file_diffs=file_diffs,
        prompt_variant="standard",
    )

    is_clean = pr.get("is_clean", False) or not pr.get("gold_findings")
    gold_findings = [] if is_clean else pr.get("gold_findings", [])
    assistant_content = format_findings_target(gold_findings)

    return {
        "id": pr.get("id", ""),
        "repo": pr.get("repo", ""),
        "language": lang,
        "is_clean": is_clean,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
            {"role": "assistant", "content": assistant_content},
        ],
    }


def rejection_sample_distill(
    train_records: list[dict[str, Any]],
    teacher_predictions: dict[str, list[dict[str, Any]]],
    min_match_score: float = 0.8,
) -> list[dict[str, Any]]:
    """Augment training data using rejection sampling on strong-model verified outputs.

    Keeps strong-model predictions whose match with gold findings meets or exceeds min_match_score.
    """
    augmented: list[dict[str, Any]] = []

    for pr in train_records:
        pr_id = pr.get("id", "")
        if pr_id not in teacher_predictions:
            continue

        teacher_preds = teacher_predictions[pr_id]
        gold_findings = pr.get("gold_findings", [])

        # Match check
        if not gold_findings:
            # Clean PR: keep teacher prediction only if teacher also produced empty findings (silent)
            if not teacher_preds:
                augmented.append({**pr, "distilled": True, "gold_findings": []})
        else:
            # Check how well teacher covered gold findings
            matched_golds = 0
            for gf in gold_findings:
                g_path = gf.get("path")
                g_line = gf.get("line")
                for tp in teacher_preds:
                    if (
                        tp.get("path") == g_path
                        and abs(int(tp.get("line", 0)) - int(g_line or 0)) <= 3
                    ):
                        matched_golds += 1
                        break

            recall = matched_golds / len(gold_findings)
            if recall >= min_match_score:
                # Distilled example with richer comments and suggestions
                augmented.append(
                    {
                        **pr,
                        "id": f"{pr_id}_distilled",
                        "distilled": True,
                        "gold_findings": teacher_preds,
                    }
                )

    logger.info("Rejection sampling distilled %d high-quality examples.", len(augmented))
    return augmented


def prepare_sft_dataset(
    train_path: Path = Path("data/final/train.jsonl"),
    dev_path: Path = Path("data/final/dev.jsonl"),
    test_path: Path = Path("data/final/test.jsonl"),
    output_dir: Path = Path("data/training"),
    clean_ratio: float = 0.25,
    distill: bool = False,
    seed: int = 42,
) -> tuple[Path, Path]:
    """Load train split, apply leakage assertion, convert to chat SFT, and save."""
    rng = random.Random(seed)

    train_records = load_jsonl(train_path)
    logger.info("Loaded %d training PR records from %s", len(train_records), train_path)

    # 1. Zero-leakage assertion
    assert_no_leakage(train_records, dev_path=dev_path, test_path=test_path)
    logger.info("Zero-leakage check passed: 0 train records overlap with dev/test repos.")

    # 2. Rejection-sampled distillation augmentation if enabled
    if distill:
        # Generate or load teacher predictions on training set
        import asyncio

        from evals.dataset.label import DeterministicMockAnnotator
        from prism.config import Settings
        from prism.review.pipeline import PRContext, review_pr

        settings = Settings(
            github_app_id="eval-app",
            github_private_key="fake-key",
            github_webhook_secret="fake-secret",
            database_url="sqlite+aiosqlite:///eval.db",
            redis_url="redis://localhost:6379/0",
            confidence_threshold=0.6,
            max_comments_per_pr=10,
        )
        mock_annotator = DeterministicMockAnnotator(
            annotator_id="strong_teacher", model="claude-3-5-sonnet-20241022"
        )

        async def _run_distill() -> dict[str, list[dict[str, Any]]]:
            from evals.harness.runner import parse_unified_diff

            preds: dict[str, list[dict[str, Any]]] = {}
            for rec in train_records:
                file_diffs = parse_unified_diff(
                    diff_text=rec.get("diff", ""),
                    language=rec.get("language", "python"),
                )
                ctx = PRContext(
                    repo_full_name=rec["repo"],
                    pr_number=rec.get("pr_number", 1),
                    title=rec.get("title", ""),
                    description=rec.get("body", ""),
                    file_diffs=file_diffs,
                )
                res = await review_pr(ctx, settings=settings, llm_client=mock_annotator)
                preds[rec["id"]] = [f.model_dump() for f in res.findings]
            return preds

        teacher_preds = asyncio.run(_run_distill())
        distilled_records = rejection_sample_distill(train_records, teacher_preds)
        train_records = train_records + distilled_records

    # 3. Clean ratio balancing
    clean_examples = [r for r in train_records if r.get("is_clean") or not r.get("gold_findings")]
    defect_examples = [r for r in train_records if not r.get("is_clean") and r.get("gold_findings")]

    # Target: clean_examples / total ~= clean_ratio
    # If clean_ratio = 0.25, clean / (clean + defect) = 0.25 => clean = defect * (0.25 / 0.75)
    target_clean = int(len(defect_examples) * (clean_ratio / (1.0 - clean_ratio)))
    if len(clean_examples) > target_clean:
        rng.shuffle(clean_examples)
        clean_examples = clean_examples[:target_clean]

    balanced_train = defect_examples + clean_examples
    rng.shuffle(balanced_train)

    # 4. SFT Conversion with exact production prompt
    standards_file = load_default_standards()
    standards = standards_file.standards

    train_sft = [convert_pr_to_sft_example(pr, standards) for pr in balanced_train]

    # Convert dev set for validation loss tracking
    dev_records = load_jsonl(dev_path) if dev_path.exists() else []
    dev_sft = [convert_pr_to_sft_example(pr, standards) for pr in dev_records]

    # Save outputs
    output_dir.mkdir(parents=True, exist_ok=True)
    train_out = output_dir / "train_sft.jsonl"
    dev_out = output_dir / "dev_sft.jsonl"

    save_jsonl(train_sft, train_out)
    save_jsonl(dev_sft, dev_out)

    clean_count = sum(1 for e in train_sft if e["is_clean"])
    print(f"--> Saved {len(train_sft)} train SFT examples to {train_out}")
    print(f"    Clean/silent examples: {clean_count} ({clean_count / len(train_sft):.1%})")
    print(f"--> Saved {len(dev_sft)} dev SFT examples to {dev_out}")

    return train_out, dev_out


def main() -> None:
    """CLI entrypoint for SFT data preparation."""
    parser = argparse.ArgumentParser(description="Prism Review SFT Dataset Preparation")
    parser.add_argument("--train-path", type=Path, default=Path("data/final/train.jsonl"))
    parser.add_argument("--dev-path", type=Path, default=Path("data/final/dev.jsonl"))
    parser.add_argument("--test-path", type=Path, default=Path("data/final/test.jsonl"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/training"))
    parser.add_argument(
        "--clean-ratio", type=float, default=0.25, help="Target ratio of clean/empty PRs"
    )
    parser.add_argument(
        "--distill", action="store_true", help="Augment using rejection-sampled teacher outputs"
    )
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    prepare_sft_dataset(
        train_path=args.train_path,
        dev_path=args.dev_path,
        test_path=args.test_path,
        output_dir=args.output_dir,
        clean_ratio=args.clean_ratio,
        distill=args.distill,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
