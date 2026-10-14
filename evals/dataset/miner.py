"""Mining historical pull requests and review comments from GitHub repositories.

Usage (via Makefile):
    make dataset-mine

Or directly:
    GITHUB_TOKEN=ghp_... uv run python -m evals.dataset.miner \\
        --config evals/configs/repos.yml \\
        --out data/raw/prs.jsonl \\
        --target 300

Rate limits:
  - REST API  : 5,000 req/h authenticated → ~1.4 req/s.  We stay well below.
  - GraphQL   : 5,000 points/h.  We use REST to keep accounting simple.
  - Secondary : GitHub throttles burst requests; we add jitter between calls.

Caching:
  - HTTP responses are cached in .cache/github_api/ as JSON files keyed by URL hash.
  - A 24 h TTL avoids re-fetching unchanged API responses on re-runs.

Filtering heuristics:
  - Changed lines: 20–600 (inclusive).
  - At least 1 substantive inline comment (≥12 words, not bot, not "LGTM"/"nit"-only).
  - Exclude PRs from dependabot / renovate / release automation.
  - 'Clean' PRs: substantive diff + zero actionable inline comments (fp-rate baseline).
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import logging
import os
import random
import re
import time
from pathlib import Path
from typing import Any

import httpx
import yaml

from evals.dataset.schema import (
    ALLOWED_LICENSES,
    FindingSource,
    GoldFinding,
    PRExample,
    RepoConfig,
    ReposConfig,
)
from evals.dataset.splitter import assign_splits, split_summary

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

GITHUB_API = "https://api.github.com"
CACHE_DIR = Path(".cache/github_api")
CACHE_TTL_SECONDS = 86_400  # 24 h

# Filtering thresholds
MIN_CHANGED_LINES = 20
MAX_CHANGED_LINES = 600
MIN_COMMENT_WORDS = 12
TARGET_CLEAN = 60  # clean PRs to collect

# Patterns that identify non-substantive / bot comments
_BOT_PATTERN = re.compile(
    r"(dependabot|renovate|github-actions|codecov|stale|mergify|"
    r"snyk|sonarqube|deepsource|coderabbit|copilot|sweep-ai)",
    re.IGNORECASE,
)
_LGTM_PATTERN = re.compile(
    r"^\s*(lgtm\.?|looks good(?: to me)?\.?|👍+|✅+|nit:?\s*\S*)\s*$",
    re.IGNORECASE,
)
_PRAISE_PATTERN = re.compile(
    r"^\s*(nice|great|awesome|thanks?|thank you|good (work|job)|well done|perfect)[!.]?\s*$",
    re.IGNORECASE,
)
_RELEASE_TITLE = re.compile(
    r"\b(release|changelog|bump version|v\d+\.\d+\.\d+|cut \d+\.\d+)\b",
    re.IGNORECASE,
)
_BOT_AUTHOR = re.compile(
    r"(dependabot|renovate|github-actions|release-please|semantic-release)",
    re.IGNORECASE,
)

# ---------------------------------------------------------------------------
# HTTP client with caching
# ---------------------------------------------------------------------------

CACHE_DIR.mkdir(parents=True, exist_ok=True)


def _cache_path(url: str, params: dict[str, Any] | None = None) -> Path:
    key = url + json.dumps(params or {}, sort_keys=True)
    digest = hashlib.sha256(key.encode()).hexdigest()
    return CACHE_DIR / f"{digest}.json"


def _load_cache(path: Path) -> Any | None:
    if not path.exists():
        return None
    try:
        stat = path.stat()
        if time.time() - stat.st_mtime > CACHE_TTL_SECONDS:
            return None
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def _save_cache(path: Path, data: Any) -> None:
    try:
        path.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
    except Exception as exc:
        logger.debug("Cache write failed: %s", exc)


class GitHubClient:
    """Thin async GitHub REST client with per-request caching and rate-limit awareness."""

    def __init__(self, token: str) -> None:
        self._headers = {
            "Authorization": f"Bearer {token}",
            "Accept": "application/vnd.github+json",
            "X-GitHub-Api-Version": "2022-11-28",
            "User-Agent": "prism-dataset-miner/1.0",
        }
        self._client = httpx.AsyncClient(
            base_url=GITHUB_API,
            headers=self._headers,
            timeout=30.0,
            follow_redirects=True,
        )

    async def aclose(self) -> None:
        await self._client.aclose()

    async def get(
        self,
        path: str,
        params: dict[str, Any] | None = None,
        *,
        use_cache: bool = True,
    ) -> Any:
        """GET request with caching and exponential backoff on rate-limit errors."""
        url = f"{GITHUB_API}{path}"
        cache_path = _cache_path(url, params)

        if use_cache:
            cached = _load_cache(cache_path)
            if cached is not None:
                logger.debug("CACHE HIT %s", path)
                return cached

        for attempt in range(5):
            try:
                resp = await self._client.get(path, params=params)
            except httpx.HTTPError as exc:
                logger.warning("HTTP error on attempt %d: %s", attempt, exc)
                await asyncio.sleep(2**attempt + random.random())
                continue

            if resp.status_code == 200:
                data = resp.json()
                _save_cache(cache_path, data)
                # Polite delay
                await asyncio.sleep(0.25 + random.random() * 0.25)
                return data

            if resp.status_code == 403 or resp.status_code == 429:
                reset = int(resp.headers.get("x-ratelimit-reset", time.time() + 60))
                wait = max(reset - time.time(), 1)
                logger.warning("Rate-limited (HTTP %d). Sleeping %.0f s.", resp.status_code, wait)
                await asyncio.sleep(min(wait, 120))
                continue

            if resp.status_code == 404:
                logger.debug("404 for %s", path)
                return None

            logger.error("Unexpected HTTP %d for %s", resp.status_code, path)
            await asyncio.sleep(2**attempt)

        logger.error("All retries exhausted for %s", path)
        return None

    async def get_paginated(
        self,
        path: str,
        params: dict[str, Any] | None = None,
        max_pages: int = 10,
    ) -> list[Any]:
        """Collect all items across pages (up to max_pages)."""
        results: list[Any] = []
        p = dict(params or {})
        p.setdefault("per_page", 100)

        for page in range(1, max_pages + 1):
            p["page"] = page
            data = await self.get(path, params=p)
            if not data:
                break
            if isinstance(data, list):
                results.extend(data)
                if len(data) < p["per_page"]:
                    break
            else:
                # Some endpoints return an object with 'items'
                items = data.get("items", [])
                results.extend(items)
                if len(items) < p["per_page"]:
                    break
        return results


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _count_changed_lines(diff: str) -> int:
    """Count added + removed lines in a unified diff string."""
    count = 0
    for line in diff.splitlines():
        if line.startswith(("+", "-")) and not line.startswith(("+++", "---")):
            count += 1
    return count


def _is_bot_author(login: str | None) -> bool:
    if not login:
        return True
    return bool(_BOT_AUTHOR.search(login))


def _is_substantive_comment(body: str) -> bool:
    """Return True if the comment body is substantive (not LGTM/praise/nit-only)."""
    stripped = body.strip()
    if not stripped:
        return False
    words = stripped.split()
    if len(words) < MIN_COMMENT_WORDS:
        return False
    if _LGTM_PATTERN.match(stripped):
        return False
    if _PRAISE_PATTERN.match(stripped):
        return False
    return True


def _is_release_pr(title: str, author_login: str | None) -> bool:
    if _is_bot_author(author_login):
        return True
    return bool(_RELEASE_TITLE.search(title))


def _pr_id(repo: str, pr_number: int) -> str:
    raw = f"{repo}#{pr_number}"
    return hashlib.sha256(raw.encode()).hexdigest()[:16]


# ---------------------------------------------------------------------------
# Repo license verification
# ---------------------------------------------------------------------------


async def _verify_license(client: GitHubClient, repo: str, declared: str) -> bool:
    """Verify the repo's actual license matches the declared SPDX identifier."""
    data = await client.get(f"/repos/{repo}")
    if not data:
        return False
    actual = (data.get("license") or {}).get("spdx_id", "")
    if actual not in ALLOWED_LICENSES:
        logger.warning(
            "%s: actual license '%s' not in allow-list (declared: %s)", repo, actual, declared
        )
        return False
    if actual != declared:
        logger.info("%s: declared '%s' but GitHub reports '%s'", repo, declared, actual)
    return True


# ---------------------------------------------------------------------------
# Comment extraction helpers
# ---------------------------------------------------------------------------


def _comment_led_to_followup(
    comment_path: str,
    comment_line: int | None,
    commits_after_review: list[dict[str, Any]],
    files_changed_after: set[str],
) -> bool:
    """Rough heuristic: was the commented file touched in commits after this review?"""
    if comment_path in files_changed_after:
        return True
    return False


def _extract_gold_findings(
    review_comments: list[dict[str, Any]],
    commits_after_review: list[dict[str, Any]],
    files_changed_after: set[str],
) -> list[GoldFinding]:
    """Convert review comments to GoldFinding objects, filtering non-substantive ones."""
    findings: list[GoldFinding] = []
    for comment in review_comments:
        user = (comment.get("user") or {}).get("login", "")
        if _is_bot_author(user):
            continue
        body: str = comment.get("body") or ""
        if not _is_substantive_comment(body):
            continue

        path: str = comment.get("path") or ""
        # GitHub uses 'original_line' for the line in the diff, 'line' for current
        line: int | None = comment.get("line") or comment.get("original_line") or 1
        thread_resolved = bool(comment.get("subject_type") == "line")  # proxy

        verified = _comment_led_to_followup(path, line, commits_after_review, files_changed_after)

        findings.append(
            GoldFinding(
                path=path,
                line=max(1, int(line or 1)),
                description=body.strip(),
                source=FindingSource.HUMAN_COMMENT,
                verified=verified,
                reviewer=user or None,
                thread_resolved=thread_resolved,
            )
        )
    return findings


# ---------------------------------------------------------------------------
# Core mining function for a single repo
# ---------------------------------------------------------------------------


async def mine_repo(
    client: GitHubClient,
    repo_cfg: RepoConfig,
    *,
    target_per_repo: int = 20,
) -> tuple[list[PRExample], list[PRExample]]:
    """Mine PRs from one repo.

    Returns:
        (substantive_prs, clean_prs)
    """
    repo = repo_cfg.repo
    logger.info("Mining %s …", repo)

    substantive: list[PRExample] = []
    clean: list[PRExample] = []

    # Fetch merged PRs (newest first, up to 10 pages = 1000 candidates)
    prs = await client.get_paginated(
        f"/repos/{repo}/pulls",
        params={"state": "closed", "sort": "updated", "direction": "desc", "per_page": 100},
        max_pages=10,
    )

    for pr in prs:
        if len(substantive) >= target_per_repo and len(clean) >= 5:
            break

        if not pr.get("merged_at"):
            continue

        pr_number: int = pr["number"]
        title: str = pr.get("title") or ""
        body: str = pr.get("body") or ""
        author_login: str | None = (pr.get("user") or {}).get("login")

        if _is_release_pr(title, author_login):
            logger.debug("%s#%d skipped (release/bot PR)", repo, pr_number)
            continue

        head_sha: str = (pr.get("head") or {}).get("sha") or ""
        base_sha: str = (pr.get("base") or {}).get("sha") or ""
        if not head_sha or not base_sha:
            continue

        # Fetch actual diff text via GitHub diff media type
        diff_text = await _fetch_diff(client, repo, pr_number)
        if diff_text is None:
            continue

        changed_lines = _count_changed_lines(diff_text)
        if not (MIN_CHANGED_LINES <= changed_lines <= MAX_CHANGED_LINES):
            logger.debug(
                "%s#%d skipped: %d changed lines (out of [%d, %d])",
                repo,
                pr_number,
                changed_lines,
                MIN_CHANGED_LINES,
                MAX_CHANGED_LINES,
            )
            continue

        # Fetch review comments
        review_comments = await client.get_paginated(
            f"/repos/{repo}/pulls/{pr_number}/comments",
            params={"per_page": 100},
            max_pages=3,
        )

        # Fetch commits after first review comment (to detect follow-up commits)
        commits = await client.get_paginated(
            f"/repos/{repo}/pulls/{pr_number}/commits",
            params={"per_page": 100},
            max_pages=2,
        )

        # Rough 'files changed after earliest review comment' heuristic
        files_changed_after: set[str] = set()
        if review_comments and commits:
            first_comment_ts = review_comments[0].get("created_at") or ""
            for commit in commits:
                if (commit.get("commit") or {}).get("author", {}).get(
                    "date", ""
                ) > first_comment_ts:
                    commit_files = await client.get(
                        f"/repos/{repo}/commits/{commit['sha']}",
                        use_cache=True,
                    )
                    if commit_files:
                        for f in commit_files.get("files") or []:
                            files_changed_after.add(f.get("filename", ""))

        gold_findings = _extract_gold_findings(review_comments, commits, files_changed_after)

        pr_url = pr.get("html_url") or f"https://github.com/{repo}/pull/{pr_number}"
        example = PRExample(
            id=_pr_id(repo, pr_number),
            repo=repo,
            language=repo_cfg.language,
            license=repo_cfg.license,
            pr_number=pr_number,
            pr_url=pr_url,
            base_sha=base_sha,
            head_sha=head_sha,
            title=title,
            body=body,
            diff=diff_text,
            changed_lines=changed_lines,
            gold_findings=gold_findings,
            is_clean=len(gold_findings) == 0,
            split=None,
        )

        if gold_findings and len(substantive) < target_per_repo:
            substantive.append(example)
            logger.debug(
                "%s#%d → substantive (%d findings)",
                repo,
                pr_number,
                len(gold_findings),
            )
        elif not gold_findings and len(clean) < 5:
            clean.append(example)
            logger.debug("%s#%d → clean", repo, pr_number)

    logger.info("%s done: %d substantive, %d clean", repo, len(substantive), len(clean))
    return substantive, clean


async def _fetch_diff(client: GitHubClient, repo: str, pr_number: int) -> str | None:
    """Fetch unified diff for a PR via the GitHub diff media type."""
    path = f"/repos/{repo}/pulls/{pr_number}"
    cache_path = _cache_path(f"diff:{GITHUB_API}{path}")
    cached = _load_cache(cache_path)
    if cached is not None:
        return str(cached)

    url = f"{GITHUB_API}{path}"
    headers = dict(client._headers)
    headers["Accept"] = "application/vnd.github.v3.diff"

    for attempt in range(4):
        try:
            async with httpx.AsyncClient(timeout=30.0, follow_redirects=True) as c:
                resp = await c.get(url, headers=headers)
        except httpx.HTTPError as exc:
            logger.warning("Diff fetch error attempt %d: %s", attempt, exc)
            await asyncio.sleep(2**attempt)
            continue

        if resp.status_code == 200:
            text = resp.text
            _save_cache(cache_path, text)
            await asyncio.sleep(0.3 + random.random() * 0.2)
            return text

        if resp.status_code in (403, 429):
            reset = int(resp.headers.get("x-ratelimit-reset", time.time() + 60))
            wait = max(reset - time.time(), 1)
            logger.warning("Rate-limited fetching diff. Sleeping %.0f s.", wait)
            await asyncio.sleep(min(wait, 120))
            continue

        logger.debug("Diff HTTP %d for %s#%d", resp.status_code, repo, pr_number)
        return None

    return None


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------


async def mine_all(
    repos: list[RepoConfig],
    *,
    target_total: int = 300,
    token: str,
    out_path: Path,
) -> None:
    """Mine all configured repos and write JSONL output."""
    client = GitHubClient(token=token)
    try:
        all_substantive: list[PRExample] = []
        all_clean: list[PRExample] = []

        # Verify licenses first (best-effort)
        active_repos: list[RepoConfig] = []
        for r in repos:
            ok = await _verify_license(client, r.repo, r.license)
            if ok:
                active_repos.append(r)
            else:
                logger.warning("Skipping %s: license verification failed", r.repo)

        if not active_repos:
            logger.error("No repos passed license verification. Aborting.")
            return

        target_per_repo = max(15, (target_total // len(active_repos)) + 5)

        for repo_cfg in active_repos:
            try:
                substantive, clean = await mine_repo(
                    client,
                    repo_cfg,
                    target_per_repo=target_per_repo,
                )
                all_substantive.extend(substantive)
                all_clean.extend(clean)
            except Exception as exc:
                logger.error("Error mining %s: %s", repo_cfg.repo, exc, exc_info=True)
            # Polite gap between repos
            await asyncio.sleep(1.0 + random.random())

        # Dedup by id
        seen: set[str] = set()
        unique_substantive: list[PRExample] = []
        for ex in all_substantive:
            if ex.id not in seen:
                seen.add(ex.id)
                unique_substantive.append(ex)

        seen_clean: set[str] = set()
        unique_clean: list[PRExample] = []
        for ex in all_clean:
            if ex.id not in seen_clean:
                seen_clean.add(ex.id)
                unique_clean.append(ex)

        # Cap clean at TARGET_CLEAN
        unique_clean = unique_clean[:TARGET_CLEAN]

        # Assign splits (substantive only; clean has no split)
        split_examples = assign_splits(unique_substantive)

        all_examples = split_examples + unique_clean

        # Write JSONL
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("w", encoding="utf-8") as f:
            for ex in all_examples:
                f.write(ex.model_dump_json() + "\n")

        _print_summary(split_examples, unique_clean)
        logger.info(
            "Wrote %d total examples (%d substantive, %d clean) to %s",
            len(all_examples),
            len(split_examples),
            len(unique_clean),
            out_path,
        )

    finally:
        await client.aclose()


def _print_summary(substantive: list[PRExample], clean: list[PRExample]) -> None:
    """Print a summary table to stdout."""
    from collections import defaultdict

    print("\n" + "=" * 72)
    print("DATASET MINING SUMMARY")
    print("=" * 72)
    print(f"{'Repo':<40} {'Lang':<12} {'#PRs':<8} {'Split'}")
    print("-" * 72)

    by_repo: dict[str, list[PRExample]] = defaultdict(list)
    for ex in substantive:
        by_repo[ex.repo].append(ex)

    repo_splits: dict[str, str] = {}
    for ex in substantive:
        repo_splits[ex.repo] = ex.split or "?"

    for repo, prs in sorted(by_repo.items()):
        lang = prs[0].language
        split_label = repo_splits.get(repo, "?")
        print(f"  {repo:<38} {lang:<12} {len(prs):<8} {split_label}")

    print("-" * 72)
    by_lang: dict[str, int] = defaultdict(int)
    for ex in substantive:
        by_lang[ex.language] += 1

    total = len(substantive)
    for lang, count in sorted(by_lang.items()):
        print(f"  {'Total ' + lang:<38} {'':12} {count}")
    print(f"  {'TOTAL substantive':<38} {'':12} {total}")
    print(f"  {'Clean (FP baseline)':<38} {'':12} {len(clean)}")

    # Split breakdown
    by_split: dict[str | None, int] = defaultdict(int)
    for ex in substantive:
        by_split[ex.split] += 1

    print("\nSplit breakdown:")
    for s in ("train", "dev", "test"):
        print(f"  {s:<8} {by_split.get(s, 0):>5} PRs")

    print("=" * 72 + "\n")

    # Assert no cross-split repo overlap
    split_summary(substantive)


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


def _load_repos_config(config_path: Path) -> ReposConfig:
    raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    return ReposConfig.model_validate(raw)


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    parser = argparse.ArgumentParser(description="Prism dataset miner")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("evals/configs/repos.yml"),
        help="Path to repos.yml",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("data/raw/prs.jsonl"),
        help="Output JSONL file",
    )
    parser.add_argument(
        "--target",
        type=int,
        default=300,
        help="Target number of substantive PRs",
    )
    args = parser.parse_args()

    token = os.environ.get("GITHUB_TOKEN") or os.environ.get("GITHUB_PAT", "")
    if not token:
        raise SystemExit(
            "ERROR: GITHUB_TOKEN environment variable is required.\n  export GITHUB_TOKEN=ghp_..."
        )

    config = _load_repos_config(args.config)
    active = config.active

    if not active:
        raise SystemExit("No active repos found in config (all skipped or license mismatch).")

    logger.info("Mining %d repos (target %d PRs)", len(active), args.target)
    asyncio.run(mine_all(active, target_total=args.target, token=token, out_path=args.out))


if __name__ == "__main__":
    main()
