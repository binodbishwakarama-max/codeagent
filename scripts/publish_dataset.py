#!/usr/bin/env python3
"""Export and publish the Prism PR Review Evaluation Benchmark to Hugging Face Datasets.

Respects source licenses by generating:
1. `data/`: Full dataset for permissible open-source PRs (MIT, Apache-2.0, BSD).
2. `references/`: Reference-only manifests (URLs + SHAs) with fetch instructions.
3. `README.md`: Official Dataset Card formatted for Hugging Face Hub metadata.
"""

from __future__ import annotations

import argparse
import json
import logging
import shutil
import sys
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("publish_dataset")


def prepare_hf_export(
    source_dir: Path,
    output_dir: Path,
    dataset_card_path: Path,
) -> None:
    """Package dataset files and metadata for Hugging Face Hub."""
    logger.info("Preparing Hugging Face dataset export at %s...", output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    data_dir = output_dir / "data"
    refs_dir = output_dir / "references"
    data_dir.mkdir(exist_ok=True)
    refs_dir.mkdir(exist_ok=True)

    splits = ["train", "dev", "test"]
    for split in splits:
        src_file = source_dir / f"{split}.jsonl"
        if not src_file.exists():
            logger.warning("Split file %s does not exist, skipping.", src_file)
            continue

        # 1. Full permissive dataset copy
        dst_data_file = data_dir / f"{split}.jsonl"
        shutil.copy2(src_file, dst_data_file)
        logger.info("Copied %s -> %s", src_file, dst_data_file)

        # 2. Reference-only metadata export (excluding diff text)
        dst_ref_file = refs_dir / f"{split}_references.jsonl"
        ref_count = 0
        with (
            open(src_file, encoding="utf-8") as in_f,
            open(dst_ref_file, "w", encoding="utf-8") as out_f,
        ):
            for line in in_f:
                if not line.strip():
                    continue
                record = json.loads(line)
                # Strip raw diff text, keep URL and commit SHAs
                ref_record = {
                    "id": record.get("id"),
                    "repo": record.get("repo"),
                    "language": record.get("language"),
                    "license": record.get("license"),
                    "pr_number": record.get("pr_number"),
                    "pr_url": record.get("pr_url"),
                    "base_sha": record.get("base_sha"),
                    "head_sha": record.get("head_sha"),
                    "title": record.get("title"),
                    "body": record.get("body"),
                    "changed_lines": record.get("changed_lines"),
                    "gold_findings": record.get("gold_findings", []),
                    "is_clean": record.get("is_clean", False),
                    "split": record.get("split", split),
                }
                out_f.write(json.dumps(ref_record) + "\n")
                ref_count += 1
        logger.info("Generated %d reference records in %s", ref_count, dst_ref_file)

    # 3. Copy fetch script into dataset repository
    fetch_src = Path("scripts/fetch_pr_diffs.py")
    if fetch_src.exists():
        shutil.copy2(fetch_src, output_dir / "fetch_pr_diffs.py")

    # 4. Copy and augment Dataset Card as README.md
    if dataset_card_path.exists():
        card_content = dataset_card_path.read_text(encoding="utf-8")
        hf_header = """---
language:
- en
license: apache-2.0
task_categories:
- text-generation
tags:
- code-review
- software-engineering
- github
- pull-requests
- llm-benchmarking
- qwen2.5-coder
pretty_name: Prism PR Review Benchmark
size_categories:
- n<1K
---

"""
        if not card_content.startswith("---"):
            final_card = hf_header + card_content
        else:
            final_card = card_content

        (output_dir / "README.md").write_text(final_card, encoding="utf-8")
        logger.info("Wrote Hugging Face dataset README.md with YAML metadata.")

    logger.info("Dataset export prepared successfully at %s", output_dir)
    print("\n" + "=" * 80)
    print("  HUGGING FACE DATASETS EXPORT COMPLETE")
    print("=" * 80)
    print(f"Directory: {output_dir}")
    print("\nTo publish to the Hugging Face Hub, run:")
    print("  huggingface-cli login")
    print(
        "  uv run python scripts/publish_dataset.py --upload --repo-id <username>/prism-review-benchmark"
    )
    print("=" * 80 + "\n")


def upload_to_hub(output_dir: Path, repo_id: str, token: str | None = None) -> None:
    """Upload packaged dataset export directly to Hugging Face Hub."""
    import os

    from huggingface_hub import HfApi

    resolved_token = token or os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    if not resolved_token:
        token_path = Path.home() / ".cache" / "huggingface" / "token"
        if token_path.exists():
            resolved_token = token_path.read_text(encoding="utf-8").strip()

    if not resolved_token:
        raise ValueError(
            "No Hugging Face token found. Please set the HF_TOKEN environment variable, "
            "run `huggingface-cli login`, or pass `--token <hf_token>`."
        )

    logger.info("Connecting to Hugging Face Hub (repo: %s)...", repo_id)
    api = HfApi(token=resolved_token)
    api.create_repo(repo_id=repo_id, repo_type="dataset", exist_ok=True)
    logger.info("Uploading dataset files from %s to %s...", output_dir, repo_id)
    api.upload_folder(
        folder_path=str(output_dir),
        repo_id=repo_id,
        repo_type="dataset",
        commit_message="Initial benchmark dataset release v1.0.0",
    )
    print("\n" + "=" * 80)
    print(f"SUCCESS: Published dataset to https://huggingface.co/datasets/{repo_id}")
    print("=" * 80 + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Export Prism benchmark dataset for Hugging Face Hub"
    )
    parser.add_argument(
        "--source-dir", type=Path, default=Path("data/final"), help="Source data directory"
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("dist/hf_dataset"), help="Output directory"
    )
    parser.add_argument(
        "--dataset-card", type=Path, default=Path("DATASET_CARD.md"), help="Dataset Card path"
    )
    parser.add_argument("--upload", action="store_true", help="Upload directly to Hugging Face Hub")
    parser.add_argument(
        "--repo-id",
        type=str,
        default="binodbishwakarama/prism-review-benchmark",
        help="Hugging Face repo ID",
    )
    parser.add_argument("--token", type=str, default=None, help="Hugging Face access token")
    args = parser.parse_args()

    prepare_hf_export(args.source_dir, args.output_dir, args.dataset_card)

    if args.upload:
        upload_to_hub(args.output_dir, args.repo_id, args.token)


if __name__ == "__main__":
    main()
