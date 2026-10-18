"""Secret redaction and sanitization utilities for LLM prompt hardening.

Ensures that accidental secrets, private keys, API credentials, and tokens
present in diffs or PR metadata are redacted before transmitting to LLMs.
"""

import re

from prism.models import DiffLine, FileDiff, Hunk

# Comprehensive secret regex patterns
_SECRET_PATTERNS = [
    # Private Keys (RSA, EC, OpenSSH, PGP, Generic)
    (
        re.compile(
            r"-----BEGIN\s+[A-Z\s]+PRIVATE\s+KEY-----[\s\S]*?-----END\s+[A-Z\s]+PRIVATE\s+KEY-----",
            re.MULTILINE,
        ),
        "[REDACTED_PRIVATE_KEY]",
    ),
    # OpenAI & Anthropic API keys
    (
        re.compile(r"\b(sk-[a-zA-Z0-9_-]{20,})\b"),
        "[REDACTED_API_KEY]",
    ),
    (
        re.compile(r"\b(sk-ant-[a-zA-Z0-9_-]{20,})\b"),
        "[REDACTED_API_KEY]",
    ),
    # GitHub Personal Access Tokens and Fine-grained PATs
    (
        re.compile(r"\b(gh[pousr]_[A-Za-z0-9_]{36,})\b"),
        "[REDACTED_GITHUB_TOKEN]",
    ),
    (
        re.compile(r"\b(github_pat_[A-Za-z0-9_]{50,})\b"),
        "[REDACTED_GITHUB_TOKEN]",
    ),
    # AWS Access Key IDs
    (
        re.compile(r"\b((?:AKIA|ABIA|ACCA|ASIA)[0-9A-Z]{16})\b"),
        "[REDACTED_AWS_KEY]",
    ),
    # AWS Secret Access Keys in variable assignments
    (
        re.compile(
            r"(?i)(aws_secret_access_key|aws_secret_key)\s*[:=]\s*['\"][0-9a-zA-Z/+=]{40}['\"]"
        ),
        r"\1 = '[REDACTED_AWS_SECRET]'",
    ),
    # Generic password / secret / token assignments in code
    (
        re.compile(
            r"(?i)\b(password|secret|api_key|token|auth_token|client_secret)\s*([:=])\s*['\"][^'\"]{8,}['\"]"
        ),
        r"\1 \2 '[REDACTED_SECRET]'",
    ),
    # Bearer tokens in headers or strings
    (
        re.compile(r"(?i)\bBearer\s+[a-zA-Z0-9_\-\.]{25,}\b"),
        "Bearer [REDACTED_BEARER_TOKEN]",
    ),
    # Slack tokens
    (
        re.compile(r"\b(xox[baprs]-[0-9a-zA-Z-]{20,})\b"),
        "[REDACTED_SLACK_TOKEN]",
    ),
    # Google API Key
    (
        re.compile(r"\b(AIza[0-9A-Za-z-_]{35})\b"),
        "[REDACTED_GOOGLE_API_KEY]",
    ),
    # Database URIs with embedded passwords (e.g. postgres://user:pass@host/db)
    (
        re.compile(r"(?i)\b([a-z]+://[^:]+):([^@\s/]+)@"),
        r"\1:[REDACTED_PASSWORD]@",
    ),
]


def redact_secrets(text: str) -> str:
    """Scrub known API keys, private keys, tokens, and passwords from arbitrary text."""
    if not text:
        return text

    redacted = text
    for pattern, replacement in _SECRET_PATTERNS:
        redacted = pattern.sub(replacement, redacted)

    return redacted


def redact_file_diffs(file_diffs: list[FileDiff]) -> list[FileDiff]:
    """Return a deep copy of FileDiffs with all changed/context line text sanitized."""
    sanitized_diffs: list[FileDiff] = []

    for fd in file_diffs:
        sanitized_hunks: list[Hunk] = []
        for hunk in fd.hunks:
            sanitized_lines: list[DiffLine] = []
            for line in hunk.lines:
                scrubbed_text = redact_secrets(line.text)
                sanitized_lines.append(
                    DiffLine(
                        kind=line.kind,
                        text=scrubbed_text,
                        old_no=line.old_no,
                        new_no=line.new_no,
                    )
                )
            sanitized_hunks.append(
                Hunk(
                    old_start=hunk.old_start,
                    new_start=hunk.new_start,
                    lines=sanitized_lines,
                )
            )

        sanitized_diffs.append(
            FileDiff(
                path=fd.path,
                language=fd.language,
                hunks=sanitized_hunks,
            )
        )

    return sanitized_diffs
