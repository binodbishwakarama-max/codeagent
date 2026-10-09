"""Engineering standards loader and default standards fallback."""

from pathlib import Path

from prism.models import StandardsFile

DEFAULT_STANDARDS_PATH = Path(__file__).parent / "default_standards.yml"


def load_default_standards() -> StandardsFile:
    """Load the built-in default standards file."""
    if not DEFAULT_STANDARDS_PATH.exists():
        raise FileNotFoundError(f"Default standards file not found at {DEFAULT_STANDARDS_PATH}")
    return StandardsFile.from_file(DEFAULT_STANDARDS_PATH)


def parse_repo_standards(raw_content: str | None) -> StandardsFile:
    """Parse repository standards from YAML, falling back to built-in standards if missing."""
    if not raw_content or not raw_content.strip():
        return load_default_standards()
    try:
        return StandardsFile.from_yaml(raw_content)
    except Exception:
        # If repository standards YAML is malformed, fall back to safe default standards
        return load_default_standards()
