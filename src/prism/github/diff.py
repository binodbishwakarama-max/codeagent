"""Unified diff parser constructing FileDiff, Hunk, and DiffLine models."""

import re
from pathlib import PurePosixPath

from prism.models import DiffLine, FileDiff, Hunk

# File patterns to skip during PR review
LOCKFILES = {
    "package-lock.json",
    "yarn.lock",
    "pnpm-lock.yaml",
    "uv.lock",
    "poetry.lock",
    "cargo.lock",
    "gemfile.lock",
    "composer.lock",
    "go.sum",
}

VENDORED_DIR_NAMES = {
    "vendor",
    "node_modules",
    "dist",
    "build",
    ".venv",
    "venv",
    "target",
}

BINARY_EXTENSIONS = {
    ".png",
    ".jpg",
    ".jpeg",
    ".gif",
    ".ico",
    ".pdf",
    ".zip",
    ".tar",
    ".gz",
    ".tgz",
    ".wasm",
    ".pyc",
    ".exe",
    ".dll",
    ".so",
    ".dylib",
    ".woff",
    ".woff2",
    ".ttf",
}

LANGUAGE_MAP = {
    ".py": "python",
    ".js": "javascript",
    ".jsx": "javascript",
    ".ts": "typescript",
    ".tsx": "typescript",
    ".go": "go",
    ".rs": "rust",
    ".java": "java",
    ".c": "c",
    ".cpp": "cpp",
    ".h": "c",
    ".hpp": "cpp",
    ".cs": "csharp",
    ".rb": "ruby",
    ".php": "php",
    ".html": "html",
    ".css": "css",
    ".scss": "scss",
    ".sql": "sql",
    ".sh": "bash",
    ".yaml": "yaml",
    ".yml": "yaml",
    ".json": "json",
    ".md": "markdown",
    ".toml": "toml",
}

MAX_FILE_LINES_CAP = 2000

HUNK_HEADER_REGEX = re.compile(r"^@@\s+-(\d+)(?:,(\d+))?\s+\+(\d+)(?:,(\d+))?\s+@@")


def detect_language(path: str) -> str:
    """Detect programming or markup language from file extension."""
    suffix = PurePosixPath(path).suffix.lower()
    return LANGUAGE_MAP.get(suffix, "text")


def should_skip_file(path: str) -> bool:
    """Determine if a file should be skipped from engineering review."""
    posix_path = PurePosixPath(path)
    filename = posix_path.name.lower()

    # Skip lockfiles
    if filename in LOCKFILES:
        return True

    # Skip binary extensions
    if posix_path.suffix.lower() in BINARY_EXTENSIONS:
        return True

    # Skip minified or source maps or generated protobufs
    if filename.endswith(".min.js") or filename.endswith(".min.css") or filename.endswith(".map"):
        return True
    if filename.endswith("_pb2.py") or filename.endswith(".pb.go"):
        return True

    # Skip vendored directories
    parts = [part.lower() for part in posix_path.parts]
    if any(part in VENDORED_DIR_NAMES for part in parts):
        return True

    return False


def parse_unified_diff(
    raw_diff: str, max_lines_per_file: int = MAX_FILE_LINES_CAP
) -> list[FileDiff]:
    """Parse unified diff text into a list of FileDiff domain models.

    Skips lockfiles, generated assets, binary files, vendored folders,
    and files exceeding the specified lines cap.
    """
    if not raw_diff or not raw_diff.strip():
        return []

    file_diffs: list[FileDiff] = []
    # Split diff by file chunks starting with "diff --git"
    raw_files = re.split(r"(?=^diff --git )", raw_diff, flags=re.MULTILINE)

    for chunk in raw_files:
        chunk = chunk.strip()
        if not chunk or not chunk.startswith("diff --git"):
            continue

        file_diff = _parse_single_file_diff(chunk, max_lines_per_file)
        if file_diff is not None:
            file_diffs.append(file_diff)

    return file_diffs


def _parse_single_file_diff(file_chunk: str, max_lines_per_file: int) -> FileDiff | None:
    lines = file_chunk.splitlines()
    header_line = lines[0]  # e.g. 'diff --git a/foo.py b/bar.py'

    # Extract default paths from diff --git a/... b/...
    git_paths = re.match(r"^diff --git a/(.+?) b/(.+?)$", header_line)
    old_path = git_paths.group(1) if git_paths else ""
    new_path = git_paths.group(2) if git_paths else ""

    # Check for binary file indication
    if any("Binary files " in line and "differ" in line for line in lines[:10]):
        return None

    # Track metadata lines to extract rename or deletion targets
    is_deleted = False
    for line in lines[1:]:
        if line.startswith("rename to "):
            new_path = line[len("rename to ") :].strip()
        elif line.startswith("deleted file mode"):
            is_deleted = True
        elif line.startswith("+++ b/"):
            new_path = line[len("+++ b/") :].strip()
        elif line.startswith("+++ /dev/null"):
            is_deleted = True
        elif line.startswith("--- a/"):
            old_path = line[len("--- a/") :].strip()
        elif line.startswith("@@"):
            break

    target_path = old_path if is_deleted and not new_path else (new_path or old_path)
    if not target_path or should_skip_file(target_path):
        return None

    # Parse hunks
    hunks: list[Hunk] = []
    current_hunk_lines: list[DiffLine] = []
    current_old_start = 0
    current_new_start = 0
    current_old_no = 0
    current_new_no = 0
    in_hunk = False
    total_file_lines = 0

    for line in lines:
        if line.startswith("@@"):
            # Close previous hunk if one was open
            if in_hunk:
                hunks.append(
                    Hunk(
                        old_start=current_old_start,
                        new_start=current_new_start,
                        lines=current_hunk_lines,
                    )
                )
                current_hunk_lines = []

            match = HUNK_HEADER_REGEX.match(line)
            if not match:
                in_hunk = False
                continue

            current_old_start = int(match.group(1))
            current_new_start = int(match.group(3))
            current_old_no = current_old_start
            current_new_no = current_new_start
            in_hunk = True
            continue

        if not in_hunk:
            continue

        # Handle lines inside hunk
        total_file_lines += 1
        if total_file_lines > max_lines_per_file:
            # File exceeds max line cap, skip parsing further
            return None

        if line.startswith(" "):
            # Context line
            current_hunk_lines.append(
                DiffLine(
                    kind="ctx",
                    old_no=current_old_no,
                    new_no=current_new_no,
                    text=line[1:],
                )
            )
            current_old_no += 1
            current_new_no += 1
        elif line.startswith("+"):
            # Added line
            current_hunk_lines.append(
                DiffLine(
                    kind="add",
                    old_no=None,
                    new_no=current_new_no,
                    text=line[1:],
                )
            )
            current_new_no += 1
        elif line.startswith("-"):
            # Deleted line
            current_hunk_lines.append(
                DiffLine(
                    kind="del",
                    old_no=current_old_no,
                    new_no=None,
                    text=line[1:],
                )
            )
            current_old_no += 1
        elif line.startswith("\\"):
            # e.g., '\ No newline at end of file'
            continue
        elif not line:
            # Empty context line in some git outputs
            current_hunk_lines.append(
                DiffLine(
                    kind="ctx",
                    old_no=current_old_no,
                    new_no=current_new_no,
                    text="",
                )
            )
            current_old_no += 1
            current_new_no += 1

    if in_hunk:
        hunks.append(
            Hunk(
                old_start=current_old_start,
                new_start=current_new_start,
                lines=current_hunk_lines,
            )
        )

    return FileDiff(
        path=target_path,
        language=detect_language(target_path),
        hunks=hunks,
    )
