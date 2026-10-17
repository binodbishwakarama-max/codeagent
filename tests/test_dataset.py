"""Tests for evals/dataset: schema, splitter, labeling, miner filtering heuristics.

These tests are designed to run without a GitHub token (all GitHub API calls
are mocked). The cross-split repo overlap assertion is included here so it
runs as part of `make test` as well as `make dataset-mine`.
"""

from __future__ import annotations

import hashlib
import tempfile
from pathlib import Path
from typing import Any

import pytest
from pydantic import ValidationError

from evals.dataset.labeling import (
    enrich_example,
    enrich_finding,
    label_dataset,
    load_jsonl,
    save_jsonl,
)
from evals.dataset.miner import (
    _count_changed_lines,
    _extract_gold_findings,
    _is_bot_author,
    _is_release_pr,
    _is_substantive_comment,
    _pr_id,
)
from evals.dataset.schema import (
    ALLOWED_LICENSES,
    FindingCategory,
    FindingSeverity,
    GoldFinding,
    PRExample,
    RepoConfig,
    ReposConfig,
)
from evals.dataset.splitter import assign_splits, split_summary

# ---------------------------------------------------------------------------
# Fixtures and helpers
# ---------------------------------------------------------------------------


def _make_finding(
    description: str = "This is a substantive human review comment about a bug.",
    path: str = "src/main.py",
    line: int = 42,
    verified: bool = False,
) -> GoldFinding:
    return GoldFinding(
        path=path,
        line=line,
        description=description,
        verified=verified,
    )


def _make_example(
    repo: str = "org/repo",
    pr_number: int = 1,
    split: Any = None,
    gold_findings: list[GoldFinding] | None = None,
    is_clean: bool = False,
    language: str = "python",
    license: str = "MIT",
    changed_lines: int = 50,
) -> PRExample:
    pr_id = hashlib.sha256(f"{repo}#{pr_number}".encode()).hexdigest()[:16]
    return PRExample(
        id=pr_id,
        repo=repo,
        language=language,
        license=license,
        pr_number=pr_number,
        pr_url=f"https://github.com/{repo}/pull/{pr_number}",
        base_sha="abc123",
        head_sha="def456",
        title=f"Fix issue #{pr_number}",
        body="",
        diff="--- a/src/main.py\n+++ b/src/main.py\n@@ -1,5 +1,6 @@\n+x = 1\n",
        changed_lines=changed_lines,
        gold_findings=gold_findings or [],
        is_clean=is_clean,
        split=split,
    )


# ---------------------------------------------------------------------------
# Schema tests
# ---------------------------------------------------------------------------


class TestSchema:
    def test_allowed_licenses_non_empty(self) -> None:
        assert "MIT" in ALLOWED_LICENSES
        assert "Apache-2.0" in ALLOWED_LICENSES
        assert "BSD-3-Clause" in ALLOWED_LICENSES
        assert "GPL-3.0" not in ALLOWED_LICENSES

    def test_pr_example_frozen(self) -> None:
        ex = _make_example()
        # Pydantic v2 frozen models raise ValidationError on setattr
        with pytest.raises((TypeError, ValidationError)):
            ex.__setattr__("repo", "new/repo")

    def test_gold_finding_line_ge_1(self) -> None:
        with pytest.raises(ValidationError):
            GoldFinding(path="a.py", line=0, description="x " * 13)

    def test_pr_example_id_format(self) -> None:
        ex = _make_example(repo="tiangolo/fastapi", pr_number=42)
        expected = hashlib.sha256(b"tiangolo/fastapi#42").hexdigest()[:16]
        assert ex.id == expected

    def test_repos_config_active_filters_skipped(self) -> None:
        config = ReposConfig(
            repos=[
                RepoConfig(repo="a/b", language="python", license="MIT", skip=False),
                RepoConfig(repo="c/d", language="python", license="MIT", skip=True),
                RepoConfig(repo="e/f", language="python", license="GPL-3.0", skip=False),
            ]
        )
        active = config.active
        assert len(active) == 1
        assert active[0].repo == "a/b"

    def test_repos_config_active_allows_all_permissive_licenses(self) -> None:
        repos = [
            RepoConfig(repo=f"org/repo{i}", language="python", license=lic, skip=False)
            for i, lic in enumerate(ALLOWED_LICENSES)
        ]
        config = ReposConfig(repos=repos)
        assert len(config.active) == len(ALLOWED_LICENSES)


# ---------------------------------------------------------------------------
# Miner helper tests
# ---------------------------------------------------------------------------


class TestMinerHelpers:
    def test_count_changed_lines_adds_removes(self) -> None:
        diff = (
            "--- a/f.py\n"
            "+++ b/f.py\n"
            "@@ -1,3 +1,4 @@\n"
            " context line\n"
            "-removed line\n"
            "+added line one\n"
            "+added line two\n"
        )
        assert _count_changed_lines(diff) == 3

    def test_count_changed_lines_ignores_headers(self) -> None:
        diff = "--- a/f.py\n+++ b/f.py\n"
        assert _count_changed_lines(diff) == 0

    def test_is_bot_author_dependabot(self) -> None:
        assert _is_bot_author("dependabot[bot]")
        assert _is_bot_author("renovate-bot")
        assert _is_bot_author("github-actions[bot]")

    def test_is_bot_author_human(self) -> None:
        assert not _is_bot_author("alice")
        assert not _is_bot_author("john-doe")

    def test_is_bot_author_none(self) -> None:
        assert _is_bot_author(None)

    def test_is_substantive_comment_too_short(self) -> None:
        assert not _is_substantive_comment("Looks good!")

    def test_is_substantive_comment_lgtm(self) -> None:
        assert not _is_substantive_comment("LGTM")
        assert not _is_substantive_comment("lgtm.")
        assert not _is_substantive_comment("👍👍")

    def test_is_substantive_comment_praise(self) -> None:
        assert not _is_substantive_comment("Nice!")
        assert not _is_substantive_comment("Great work")

    def test_is_substantive_comment_valid(self) -> None:
        comment = (
            "This will cause a SQL injection vulnerability if the user "
            "input is not properly sanitized before interpolation."
        )
        assert _is_substantive_comment(comment)

    def test_is_release_pr_dependabot(self) -> None:
        assert _is_release_pr("fix: bump lodash", "dependabot[bot]")

    def test_is_release_pr_release_title(self) -> None:
        assert _is_release_pr("Release v2.3.1", "alice")

    def test_is_release_pr_normal(self) -> None:
        assert not _is_release_pr("Fix null pointer in auth middleware", "alice")

    def test_pr_id_deterministic(self) -> None:
        id1 = _pr_id("tiangolo/fastapi", 99)
        id2 = _pr_id("tiangolo/fastapi", 99)
        assert id1 == id2
        assert len(id1) == 16

    def test_pr_id_unique_per_pr(self) -> None:
        assert _pr_id("org/repo", 1) != _pr_id("org/repo", 2)
        assert _pr_id("org/repo1", 1) != _pr_id("org/repo2", 1)

    def test_extract_gold_findings_filters_bots(self) -> None:
        comments = [
            {
                "user": {"login": "dependabot[bot]"},
                "body": (
                    "This is a very long and substantive review comment about something important."
                ),
                "path": "src/main.py",
                "line": 10,
            }
        ]
        findings = _extract_gold_findings(comments, [], set())
        assert findings == []

    def test_extract_gold_findings_filters_short_comments(self) -> None:
        comments = [
            {
                "user": {"login": "alice"},
                "body": "nit",
                "path": "src/main.py",
                "line": 10,
            }
        ]
        findings = _extract_gold_findings(comments, [], set())
        assert findings == []

    def test_extract_gold_findings_keeps_substantive(self) -> None:
        body = (
            "This function does not handle the case where the input list is empty, "
            "which will cause an IndexError at runtime."
        )
        comments = [
            {
                "user": {"login": "reviewer"},
                "body": body,
                "path": "src/utils.py",
                "line": 25,
            }
        ]
        findings = _extract_gold_findings(comments, [], set())
        assert len(findings) == 1
        assert findings[0].description == body.strip()
        assert findings[0].reviewer == "reviewer"
        assert findings[0].path == "src/utils.py"
        assert findings[0].line == 25

    def test_extract_gold_findings_verified_when_file_touched_after(self) -> None:
        body = "x " * 15
        comments = [
            {
                "user": {"login": "alice"},
                "body": body,
                "path": "src/utils.py",
                "line": 5,
            }
        ]
        files_changed_after = {"src/utils.py", "other.py"}
        findings = _extract_gold_findings(comments, [], files_changed_after)
        assert len(findings) == 1
        assert findings[0].verified is True

    def test_extract_gold_findings_not_verified_when_file_not_touched(self) -> None:
        body = "y " * 15
        comments = [
            {
                "user": {"login": "alice"},
                "body": body,
                "path": "src/utils.py",
                "line": 5,
            }
        ]
        findings = _extract_gold_findings(comments, [], {"other.py"})
        assert findings[0].verified is False


# ---------------------------------------------------------------------------
# Splitter tests
# ---------------------------------------------------------------------------


class TestSplitter:
    def _build_examples(self, repos: list[str], prs_per_repo: int = 5) -> list[PRExample]:
        examples = []
        for repo in repos:
            for i in range(prs_per_repo):
                examples.append(
                    _make_example(repo=repo, pr_number=i + 1, gold_findings=[_make_finding()])
                )
        return examples

    def test_assign_splits_is_deterministic(self) -> None:
        repos = [f"org/repo{i}" for i in range(10)]
        examples = self._build_examples(repos)
        result1 = assign_splits(examples)
        result2 = assign_splits(examples)
        for e1, e2 in zip(result1, result2, strict=True):
            assert e1.split == e2.split

    def test_assign_splits_covers_all_splits(self) -> None:
        repos = [f"org/repo{i}" for i in range(10)]
        examples = self._build_examples(repos)
        result = assign_splits(examples)
        splits_used = {ex.split for ex in result}
        assert "train" in splits_used
        assert "dev" in splits_used
        assert "test" in splits_used

    def test_no_cross_split_repo_overlap(self) -> None:
        """Core correctness guarantee: repos never appear in two different splits."""
        repos = [f"org/repoX{i}" for i in range(15)]
        examples = self._build_examples(repos)
        result = assign_splits(examples)
        # split_summary raises AssertionError on any overlap
        split_summary(result)  # must not raise

    def test_cross_split_overlap_is_detected(self) -> None:
        """Manually craft cross-split examples to confirm split_summary raises."""
        examples = [
            _make_example(
                repo="org/evil", pr_number=1, split="train", gold_findings=[_make_finding()]
            ),
            _make_example(
                repo="org/evil", pr_number=2, split="test", gold_findings=[_make_finding()]
            ),
        ]
        with pytest.raises(AssertionError, match="repo overlap"):
            split_summary(examples)

    def test_train_dev_test_ratios_approximate(self) -> None:
        repos = [f"org/repo{i}" for i in range(20)]
        examples = self._build_examples(repos, prs_per_repo=1)
        result = assign_splits(examples)
        splits = [ex.split for ex in result]
        n = len(splits)
        train_frac = splits.count("train") / n
        dev_frac = splits.count("dev") / n
        test_frac = splits.count("test") / n
        # Allow ±10% slack
        assert 0.50 <= train_frac <= 0.70, f"train fraction {train_frac} out of range"
        assert 0.10 <= dev_frac <= 0.30, f"dev fraction {dev_frac} out of range"
        assert 0.10 <= test_frac <= 0.30, f"test fraction {test_frac} out of range"

    def test_original_examples_unmodified(self) -> None:
        """assign_splits must not mutate the input list."""
        repos = ["org/a", "org/b", "org/c"]
        examples = self._build_examples(repos)
        for ex in examples:
            assert ex.split is None
        assign_splits(examples)
        for ex in examples:
            assert ex.split is None  # frozen model, copy was returned

    def test_split_only_substantive_examples(self) -> None:
        """Clean (is_clean=True) examples are not passed to assign_splits."""
        repos = [f"org/repo{i}" for i in range(6)]
        substantive = self._build_examples(repos)
        clean = [_make_example(repo="clean/repo", pr_number=i + 1, is_clean=True) for i in range(3)]

        split_examples = assign_splits(substantive)
        # Clean examples passed through without a split
        all_examples = split_examples + clean

        # Only substantive examples have a non-None split
        for ex in all_examples:
            if ex.is_clean:
                assert ex.split is None
            else:
                assert ex.split in ("train", "dev", "test")


# ---------------------------------------------------------------------------
# Labeling tests
# ---------------------------------------------------------------------------


class TestLabeling:
    def test_enrich_finding_infers_security_category(self) -> None:
        f = _make_finding(
            description="This could lead to SQL injection if user input is not escaped properly."
        )
        enriched = enrich_finding(f)
        assert enriched.category == FindingCategory.SECURITY

    def test_enrich_finding_infers_bug_category(self) -> None:
        f = _make_finding(description="This is incorrect and will crash when the list is empty.")
        enriched = enrich_finding(f)
        assert enriched.category == FindingCategory.BUG

    def test_enrich_finding_infers_high_severity_for_security(self) -> None:
        f = _make_finding(
            description="This is a critical security vulnerability that must be fixed immediately."
        )
        enriched = enrich_finding(f)
        assert enriched.severity == FindingSeverity.HIGH

    def test_enrich_finding_infers_medium_severity_for_bug(self) -> None:
        f = _make_finding(description="This is incorrect and will fail in edge cases.")
        enriched = enrich_finding(f)
        assert enriched.severity == FindingSeverity.MEDIUM

    def test_enrich_finding_does_not_overwrite_existing_category(self) -> None:
        f = GoldFinding(
            path="src/a.py",
            line=1,
            description="This is a security vulnerability.",
            category=FindingCategory.PERFORMANCE,  # explicitly set
        )
        enriched = enrich_finding(f)
        # Should NOT override explicitly-set category
        assert enriched.category == FindingCategory.PERFORMANCE

    def test_enrich_example_propagates_to_all_findings(self) -> None:
        ex = _make_example(
            gold_findings=[
                _make_finding("SQL injection risk if user input is unsanitized and not escaped."),
                _make_finding("This crashes when the list is empty, it is incorrect behavior."),
            ]
        )
        enriched = enrich_example(ex)
        assert enriched.gold_findings[0].category == FindingCategory.SECURITY
        assert enriched.gold_findings[1].category == FindingCategory.BUG

    def test_enrich_example_returns_new_instance(self) -> None:
        ex = _make_example(gold_findings=[_make_finding("security injection vulnerability here!")])
        enriched = enrich_example(ex)
        assert enriched is not ex

    def test_load_save_jsonl_roundtrip(self) -> None:
        examples = [
            _make_example(pr_number=i, gold_findings=[_make_finding()]) for i in range(1, 4)
        ]
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "test.jsonl"
            save_jsonl(examples, path)
            loaded = load_jsonl(path)
        assert len(loaded) == len(examples)
        for orig, loaded_ex in zip(examples, loaded, strict=True):
            assert orig.id == loaded_ex.id
            assert orig.repo == loaded_ex.repo
            assert orig.pr_number == loaded_ex.pr_number

    def test_load_jsonl_skips_malformed_lines(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "bad.jsonl"
            path.write_text('{"invalid": true}\nnot-json\n', encoding="utf-8")
            loaded = load_jsonl(path)
        assert loaded == []

    def test_label_dataset_writes_enriched_output(self) -> None:
        examples = [
            _make_example(
                pr_number=1,
                gold_findings=[
                    _make_finding(
                        "SQL injection vulnerability if not escaped properly here "
                        "because user input is not sanitized before database query."
                    )
                ],
            )
        ]
        with tempfile.TemporaryDirectory() as tmpdir:
            raw_path = Path(tmpdir) / "raw.jsonl"
            labeled_path = Path(tmpdir) / "labeled.jsonl"
            save_jsonl(examples, raw_path)
            enriched = label_dataset(raw_path, labeled_path)
            assert labeled_path.exists()
            assert len(enriched) == 1
            assert enriched[0].gold_findings[0].category == FindingCategory.SECURITY


# ---------------------------------------------------------------------------
# repos.yml config tests
# ---------------------------------------------------------------------------


class TestReposConfig:
    def test_repos_yml_is_valid(self) -> None:
        """The committed repos.yml must parse and contain ≥20 active repos."""
        import yaml

        path = Path("evals/configs/repos.yml")
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
        config = ReposConfig.model_validate(raw)
        active = config.active
        assert len(active) >= 20, f"Only {len(active)} active repos — expected ≥20"

    def test_repos_yml_all_active_have_allowed_licenses(self) -> None:
        import yaml

        path = Path("evals/configs/repos.yml")
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
        config = ReposConfig.model_validate(raw)
        for repo in config.active:
            assert repo.license in ALLOWED_LICENSES, (
                f"{repo.repo} has disallowed license {repo.license!r}"
            )

    def test_repos_yml_has_both_languages(self) -> None:
        import yaml

        path = Path("evals/configs/repos.yml")
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
        config = ReposConfig.model_validate(raw)
        langs = {r.language for r in config.active}
        assert "python" in langs
        assert "typescript" in langs


# ---------------------------------------------------------------------------
# Idempotency: same PR produces same id
# ---------------------------------------------------------------------------


class TestIdempotency:
    def test_pr_id_collision_free_across_repos(self) -> None:
        """Different repo+pr_number pairs must never produce the same id."""
        ids = set()
        for repo in ["org/a", "org/b", "user/repo"]:
            for pr_num in range(1, 50):
                pr_id = _pr_id(repo, pr_num)
                assert pr_id not in ids, f"Collision for {repo}#{pr_num}"
                ids.add(pr_id)
