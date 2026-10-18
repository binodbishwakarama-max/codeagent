#!/usr/bin/env python3
"""Fetch PR diffs directly from GitHub API for reference-only dataset mode.

For licensing environments where redistributing diff text is restricted,
this script downloads the original unified diffs directly from GitHub's
public API using the recorded commit SHAs and PR numbers.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path

import httpx

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("fetch_pr_diffs")


def fetch_diff(repo: str, pr_number: int, token: str | None = None) -> str:
    """Fetch raw unified diff text for a given PR from GitHub API."""
    url = f"https://api.github.com/repos/{repo}/pulls/{pr_number}"
    headers = {
        "Accept": "application/vnd.github.v3.diff",
        "User-Agent": "Prism-Dataset-Fetcher/1.0",
    }
    if token:
        headers["Authorization"] = f"Bearer {token}"

    with httpx.Client(timeout=30.0) as client:
        response = client.get(url, headers=headers, follow_redirects=True)
        if response.status_code == 403:
            logger.error("GitHub API rate limit exceeded. Set GITHUB_TOKEN environment variable.")
            sys.exit(1)
        response.raise_for_status()
        return response.text


def hydrate_dataset(input_path: Path, output_path: Path, token: str | None = None) -> None:
    """Read reference records and hydrate missing diff fields."""
    logger.info("Hydrating dataset from %s to %s", input_path, output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    records: list[dict] = []
    with open(input_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))

    hydrated_count = 0
    with open(output_path, "w", encoding="utf-8") as out_f:
        for idx, rec in enumerate(records, 1):
            repo = rec["repo"]
            pr_num = rec["pr_number"]
            if not rec.get("diff"):
                logger.info("[%d/%d] Fetching diff for %s#%d...", idx, len(records), repo, pr_num)
                try:
                    diff_text = fetch_diff(repo, pr_num, token)
                    rec["diff"] = diff_text
                    hydrated_count += 1
                    time.sleep(0.5)  # Respect API pacing
                except Exception as exc:
                    logger.warning("Failed to fetch diff for %s#%d: %s", repo, pr_num, exc)
            out_f.write(json.dumps(rec) + "\n")

    logger.info("Complete! Hydrated %d records. Saved to %s", hydrated_count, output_path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Hydrate PR diffs directly from GitHub API")
    parser.add_argument(
        "--input", type=Path, default=Path("data/final/test.jsonl"), help="Input JSONL file"
    )
    parser.add_argument(
        "--output", type=Path, default=Path("data/hydrated/test.jsonl"), help="Output JSONL file"
    )
    parser.add_argument(
        "--token",
        type=str,
        default=os.environ.get("GITHUB_TOKEN"),
        help="GitHub Personal Access Token",
    )
    args = parser.parse_args()

    hydrate_dataset(args.input, args.output, args.token)


if __name__ == "__main__":
    main()
