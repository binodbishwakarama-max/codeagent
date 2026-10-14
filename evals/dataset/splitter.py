"""Deterministic repo-level train/dev/test splitting.

Splitting is BY REPO (not by PR) so a repo's PRs never appear in two different splits.
This prevents data leakage where the model sees training PRs from a repo and then is
evaluated on test PRs from the same repo (which share codebase idioms).

Algorithm
---------
1. Collect the set of unique repos from the dataset.
2. Sort repos deterministically (alphabetical).
3. Assign repos to splits in 60/20/20 proportions (train/dev/test).
4. Tag each PRExample with its repo's split.

The split is stable: the same set of repos always maps to the same split,
regardless of run order or Python version, because we use sorted() and
integer arithmetic with no randomness.
"""

from __future__ import annotations

import logging
from collections import defaultdict
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from evals.dataset.schema import PRExample

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Split ratios
# ---------------------------------------------------------------------------

TRAIN_RATIO = 0.60
DEV_RATIO = 0.20
# TEST_RATIO = 1 - TRAIN_RATIO - DEV_RATIO = 0.20


def _repo_to_split(repos_sorted: list[str], repo: str) -> str:
    """Map a repo to a split based on its position in the sorted list."""
    n = len(repos_sorted)
    n_train = round(n * TRAIN_RATIO)
    n_dev = round(n * DEV_RATIO)
    idx = repos_sorted.index(repo)
    if idx < n_train:
        return "train"
    if idx < n_train + n_dev:
        return "dev"
    return "test"


def assign_splits(examples: list[PRExample]) -> list[PRExample]:
    """Return a new list of PRExamples with their 'split' field assigned.

    Assignment is deterministic: same input → same split every run.
    """
    repos = sorted({ex.repo for ex in examples})
    repo_split = {repo: _repo_to_split(repos, repo) for repo in repos}

    result: list[PRExample] = []
    for ex in examples:
        split_label = repo_split[ex.repo]
        # PRExample is frozen — create a new copy with the split field set
        result.append(ex.model_copy(update={"split": split_label}))

    return result


def split_summary(examples: list[PRExample]) -> None:
    """Print per-split repo stats and assert zero cross-split repo overlap."""
    repos_by_split: dict[str, set[str]] = defaultdict(set)
    for ex in examples:
        if ex.split:
            repos_by_split[ex.split].add(ex.repo)

    # Assert no overlap
    train_repos = repos_by_split.get("train", set())
    dev_repos = repos_by_split.get("dev", set())
    test_repos = repos_by_split.get("test", set())

    overlap_train_dev = train_repos & dev_repos
    overlap_train_test = train_repos & test_repos
    overlap_dev_test = dev_repos & test_repos

    if overlap_train_dev:
        raise AssertionError(f"Train/dev repo overlap: {overlap_train_dev}")
    if overlap_train_test:
        raise AssertionError(f"Train/test repo overlap: {overlap_train_test}")
    if overlap_dev_test:
        raise AssertionError(f"Dev/test repo overlap: {overlap_dev_test}")

    logger.info(
        "Split repo counts: train=%d, dev=%d, test=%d",
        len(train_repos),
        len(dev_repos),
        len(test_repos),
    )

    prs_by_split: dict[str, int] = defaultdict(int)
    for ex in examples:
        prs_by_split[ex.split or "unassigned"] += 1

    logger.info("Split PR counts: %s", dict(prs_by_split))
