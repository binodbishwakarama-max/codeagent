"""Command-line interface to execute PR reviews against public or private GitHub PRs."""

import argparse
import asyncio
import os
import sys
from typing import Literal

import httpx

from prism.config import Settings
from prism.github.diff import parse_unified_diff
from prism.models import ReviewResult
from prism.review.llm import create_llm_client
from prism.review.pipeline import PRContext, review_pr
from prism.review.standards import load_default_standards


async def fetch_pr_details_and_diff(
    repo: str,
    pr_number: int,
    github_token: str | None = None,
) -> tuple[dict[str, str], str]:
    """Fetch PR metadata and unified diff from GitHub API."""
    base_url = f"https://api.github.com/repos/{repo}/pulls/{pr_number}"
    headers = {"User-Agent": "prism-review-cli"}
    if github_token:
        headers["Authorization"] = f"token {github_token}"

    async with httpx.AsyncClient(timeout=30.0) as client:
        # 1. Fetch metadata
        meta_res = await client.get(base_url, headers=headers)
        meta_res.raise_for_status()
        meta = meta_res.json()

        # 2. Fetch unified diff
        diff_headers = {**headers, "Accept": "application/vnd.github.v3.diff"}
        diff_res = await client.get(base_url, headers=diff_headers)
        diff_res.raise_for_status()
        raw_diff = diff_res.text

    title = str(meta.get("title", ""))
    body = str(meta.get("body", "") or "")
    head_sha = str(meta.get("head", {}).get("sha", ""))

    return {"title": title, "body": body, "sha": head_sha}, raw_diff


def print_review_report(result: ReviewResult, repo: str, pr_number: int, dry_run: bool) -> None:
    """Pretty-print the review output and telemetry summary to stdout."""
    print("=" * 80)
    print(f"Prism Review Report: {repo}#{pr_number}")
    print(f"Model: {result.model} | Prompt: {result.prompt_version}")
    print(
        f"Tokens: {result.input_tokens} in / {result.output_tokens} out | "
        f"Cost: ${result.cost_usd:.6f} | Latency: {result.latency_ms:.1f}ms"
    )
    print("=" * 80)

    if not result.findings:
        print("\nNo violations or issues found. Clean diff!\n")
    else:
        print(f"\nIdentified {len(result.findings)} actionable finding(s):\n")
        for idx, finding in enumerate(result.findings, 1):
            print(
                f"[{idx}] {finding.severity.value} | {finding.category.value} "
                f"at {finding.path}:{finding.line} (confidence: {int(finding.confidence * 100)}%)"
            )
            if finding.standard_id:
                print(f"    Standard: {finding.standard_id}")
            print(f"    Message: {finding.message}")
            if finding.suggestion:
                print("    Suggestion:")
                for line in finding.suggestion.splitlines():
                    print(f"      {line}")
            print("-" * 60)

    if dry_run:
        print("\n[DRY RUN] No comments posted to GitHub.\n")
    print("=" * 80)


async def run_cli(repo: str, pr_number: int, dry_run: bool, token: str | None) -> int:
    """Async CLI executor."""
    print(f"Fetching PR {repo}#{pr_number}...")
    try:
        meta, raw_diff = await fetch_pr_details_and_diff(repo, pr_number, token)
    except Exception as exc:
        print(f"Error fetching PR: {exc}", file=sys.stderr)
        return 1

    file_diffs = parse_unified_diff(raw_diff)
    print(f"Parsed {len(file_diffs)} changed code file(s) from diff.")

    standards_file = load_default_standards()
    pr_context = PRContext(
        repo_full_name=repo,
        pr_number=pr_number,
        title=meta["title"],
        description=meta["body"],
        head_sha=meta["sha"],
        file_diffs=file_diffs,
        standards=standards_file.standards,
    )

    try:
        settings = Settings()
    except Exception:
        raw_provider = os.getenv("LLM_PROVIDER", "anthropic")
        llm_provider: Literal["anthropic", "openai_compat"] = (
            "openai_compat" if raw_provider == "openai_compat" else "anthropic"
        )
        settings = Settings(
            github_app_id="cli-app",
            github_private_key="cli-key",
            github_webhook_secret="cli-secret",
            database_url="postgresql://localhost/prism",
            redis_url="redis://localhost:6379/0",
            anthropic_api_key=os.getenv("ANTHROPIC_API_KEY"),
            llm_model=os.getenv("LLM_MODEL", "claude-3-5-sonnet-20241022"),
            llm_provider=llm_provider,
            llm_base_url=os.getenv("LLM_BASE_URL"),
        )

    try:
        llm_client = create_llm_client(settings)
    except Exception as exc:
        print(f"Error initializing LLM client: {exc}", file=sys.stderr)
        return 1

    print("Executing automated engineering review...")
    result = await review_pr(pr_context, settings=settings, llm_client=llm_client)
    print_review_report(result, repo, pr_number, dry_run)
    return 0


def main() -> None:
    """CLI entrypoint."""
    parser = argparse.ArgumentParser(description="Prism Automated PR Review CLI")
    parser.add_argument("--repo", required=True, help="GitHub repository in owner/name format")
    parser.add_argument("--pr", required=True, type=int, help="Pull request number")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        default=True,
        help="Run review without posting comments (default: True)",
    )
    parser.add_argument(
        "--token",
        default=os.getenv("GITHUB_TOKEN"),
        help="Optional GitHub token for private repos or higher API rate limits",
    )

    args = parser.parse_args()
    exit_code = asyncio.run(run_cli(args.repo, args.pr, args.dry_run, args.token))
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
