"""Tests for unified diff parser: parsing, line numbers, filtering, and edge cases."""

from pathlib import Path

import pytest

from prism.github.diff import (
    VENDORED_DIR_NAMES,
    detect_language,
    parse_unified_diff,
    should_skip_file,
)

FIXTURE_DIR = Path(__file__).parent / "fixtures"


def load_fixture(name: str) -> str:
    return (FIXTURE_DIR / name).read_text(encoding="utf-8")


# ---------------------------------------------------------------------------
# should_skip_file
# ---------------------------------------------------------------------------


class TestShouldSkipFile:
    def test_lockfile_skipped(self) -> None:
        assert should_skip_file("package-lock.json") is True
        assert should_skip_file("poetry.lock") is True
        assert should_skip_file("uv.lock") is True

    def test_python_source_not_skipped(self) -> None:
        assert should_skip_file("src/main.py") is False

    def test_binary_extension_skipped(self) -> None:
        assert should_skip_file("assets/logo.png") is True
        assert should_skip_file("dist/bundle.wasm") is True

    def test_minified_js_skipped(self) -> None:
        assert should_skip_file("static/app.min.js") is True

    def test_source_map_skipped(self) -> None:
        assert should_skip_file("dist/bundle.js.map") is True

    def test_generated_protobuf_skipped(self) -> None:
        assert should_skip_file("proto/user_pb2.py") is True

    @pytest.mark.parametrize("vendor_dir", sorted(VENDORED_DIR_NAMES))
    def test_vendored_directories_skipped(self, vendor_dir: str) -> None:
        assert should_skip_file(f"{vendor_dir}/somelib/util.py") is True

    def test_nested_vendor_dir_skipped(self) -> None:
        assert should_skip_file("src/vendor/requests/api.py") is True

    def test_non_vendored_path_not_skipped(self) -> None:
        assert should_skip_file("src/services/auth.py") is False


# ---------------------------------------------------------------------------
# detect_language
# ---------------------------------------------------------------------------


class TestDetectLanguage:
    def test_python(self) -> None:
        assert detect_language("src/main.py") == "python"

    def test_typescript(self) -> None:
        assert detect_language("app/hooks/useAuth.ts") == "typescript"

    def test_yaml(self) -> None:
        assert detect_language(".prism/standards.yml") == "yaml"

    def test_unknown_defaults_to_text(self) -> None:
        assert detect_language("Makefile") == "text"


# ---------------------------------------------------------------------------
# parse_unified_diff - simple diff
# ---------------------------------------------------------------------------


class TestSimpleDiff:
    def setup_method(self) -> None:
        raw = load_fixture("simple_diff.diff")
        self.diffs = parse_unified_diff(raw)

    def test_one_file_parsed(self) -> None:
        assert len(self.diffs) == 1

    def test_correct_path(self) -> None:
        assert self.diffs[0].path == "src/main.py"

    def test_correct_language(self) -> None:
        assert self.diffs[0].language == "python"

    def test_one_hunk(self) -> None:
        assert len(self.diffs[0].hunks) == 1

    def test_hunk_old_start(self) -> None:
        assert self.diffs[0].hunks[0].old_start == 1

    def test_hunk_new_start(self) -> None:
        assert self.diffs[0].hunks[0].new_start == 1

    def test_add_line_has_correct_new_no(self) -> None:
        hunk = self.diffs[0].hunks[0]
        add_lines = [line for line in hunk.lines if line.kind == "add"]
        assert len(add_lines) >= 1
        # Check new_no is set for additions
        for line in add_lines:
            assert line.new_no is not None
            assert line.old_no is None

    def test_del_line_has_correct_old_no(self) -> None:
        hunk = self.diffs[0].hunks[0]
        del_lines = [line for line in hunk.lines if line.kind == "del"]
        assert len(del_lines) >= 1
        for line in del_lines:
            assert line.old_no is not None
            assert line.new_no is None

    def test_ctx_line_has_both_line_numbers(self) -> None:
        hunk = self.diffs[0].hunks[0]
        ctx_lines = [line for line in hunk.lines if line.kind == "ctx"]
        for line in ctx_lines:
            assert line.old_no is not None
            assert line.new_no is not None


# ---------------------------------------------------------------------------
# parse_unified_diff - rename diff
# ---------------------------------------------------------------------------


class TestRenameDiff:
    def setup_method(self) -> None:
        raw = load_fixture("rename_diff.diff")
        self.diffs = parse_unified_diff(raw)

    def test_one_file_parsed(self) -> None:
        assert len(self.diffs) == 1

    def test_new_path_used_after_rename(self) -> None:
        """After a rename, the target path should be the new (renamed-to) path."""
        assert self.diffs[0].path == "src/helpers.py"


# ---------------------------------------------------------------------------
# parse_unified_diff - deletion diff
# ---------------------------------------------------------------------------


class TestDeletionDiff:
    def setup_method(self) -> None:
        raw = load_fixture("deletion_diff.diff")
        self.diffs = parse_unified_diff(raw)

    def test_deleted_file_is_parsed(self) -> None:
        # Deleted files still produce a FileDiff for the old path
        assert len(self.diffs) == 1

    def test_all_lines_are_deletions(self) -> None:
        hunk = self.diffs[0].hunks[0]
        non_del = [line for line in hunk.lines if line.kind != "del"]
        assert len(non_del) == 0

    def test_deleted_lines_have_old_no(self) -> None:
        hunk = self.diffs[0].hunks[0]
        for line in hunk.lines:
            assert line.old_no is not None


# ---------------------------------------------------------------------------
# parse_unified_diff - multi-hunk diff
# ---------------------------------------------------------------------------


class TestMultiHunkDiff:
    def setup_method(self) -> None:
        raw = load_fixture("multi_hunk_diff.diff")
        self.diffs = parse_unified_diff(raw)

    def test_one_file_parsed(self) -> None:
        assert len(self.diffs) == 1

    def test_two_hunks_parsed(self) -> None:
        assert len(self.diffs[0].hunks) == 2

    def test_second_hunk_starts_at_correct_line(self) -> None:
        second_hunk = self.diffs[0].hunks[1]
        assert second_hunk.new_start == 16

    def test_second_hunk_has_additions(self) -> None:
        second_hunk = self.diffs[0].hunks[1]
        add_lines = [line for line in second_hunk.lines if line.kind == "add"]
        assert len(add_lines) >= 2

    def test_line_numbers_are_contiguous_within_hunk(self) -> None:
        """Context and add lines within a hunk must have ascending new_no values."""
        hunk = self.diffs[0].hunks[0]
        new_nos = [line.new_no for line in hunk.lines if line.new_no is not None]
        assert new_nos == sorted(new_nos)
        assert new_nos == list(range(new_nos[0], new_nos[0] + len(new_nos)))


# ---------------------------------------------------------------------------
# parse_unified_diff - no newline at end of file
# ---------------------------------------------------------------------------


class TestNoNewlineDiff:
    def setup_method(self) -> None:
        raw = load_fixture("no_newline_diff.diff")
        self.diffs = parse_unified_diff(raw)

    def test_file_parsed_despite_no_newline(self) -> None:
        assert len(self.diffs) == 1

    def test_no_backslash_lines_in_result(self) -> None:
        """'\\' sentinel lines must not appear in parsed DiffLines."""
        for hunk in self.diffs[0].hunks:
            for line in hunk.lines:
                assert not line.text.startswith("\\")

    def test_del_and_add_both_present(self) -> None:
        hunk = self.diffs[0].hunks[0]
        kinds = {line.kind for line in hunk.lines}
        assert "del" in kinds
        assert "add" in kinds


# ---------------------------------------------------------------------------
# parse_unified_diff - lockfile and vendored paths skipped
# ---------------------------------------------------------------------------


class TestSkippedFiles:
    def test_lockfile_diff_returns_empty(self) -> None:
        raw = """diff --git a/package-lock.json b/package-lock.json
index abc..def 100644
--- a/package-lock.json
+++ b/package-lock.json
@@ -1,3 +1,3 @@
 {
-  "version": "1.0.0"
+  "version": "1.0.1"
 }
"""
        results = parse_unified_diff(raw)
        assert len(results) == 0

    def test_vendor_dir_diff_returns_empty(self) -> None:
        raw = """diff --git a/vendor/lib/util.py b/vendor/lib/util.py
index abc..def 100644
--- a/vendor/lib/util.py
+++ b/vendor/lib/util.py
@@ -1,2 +1,2 @@
-x = 1
+x = 2
"""
        results = parse_unified_diff(raw)
        assert len(results) == 0

    def test_binary_file_diff_returns_empty(self) -> None:
        raw = """diff --git a/images/logo.png b/images/logo.png
index abc..def 100644
Binary files a/images/logo.png and b/images/logo.png differ
"""
        results = parse_unified_diff(raw)
        assert len(results) == 0

    def test_empty_diff_string_returns_empty(self) -> None:
        assert parse_unified_diff("") == []
        assert parse_unified_diff("   \n\n") == []
