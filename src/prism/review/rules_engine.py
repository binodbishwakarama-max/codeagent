"""Deterministic static analysis rules engine for fast-pass PR review.

Provides high-precision, sub-millisecond pattern checks for:
1. SQL Injection (CWE-89 / STD-SEC-01 / SEC-002)
2. Hardcoded Credentials & Secrets (CWE-798 / STD-SEC-02 / SEC-001)
3. Missing Async / Network Timeouts (CWE-703 / STD-ERR-02)
4. Bare Exception Blocks (CWE-391 / STD-ERR-01 / ERR-001)

Used both as an offline fallback when zero LLM credentials are configured,
and as a pre-LLM fast pass for deterministic detection of critical
antipatterns with zero token cost.
"""

from __future__ import annotations

import re
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

from prism.models import FileDiff, IssueCategory, ReviewFinding, Severity


@dataclass(frozen=True)
class DeterministicRule:
    """Specification for a single deterministic code review rule."""

    rule_id: str
    name: str
    category: IssueCategory
    severity: Severity
    cwe: str
    confidence: float
    check: Callable[[str, str, int], ReviewFinding | None]


# ---------------------------------------------------------------------------
# Rule Checkers
# ---------------------------------------------------------------------------


def check_sql_injection(line_text: str, path: str, line_no: int) -> ReviewFinding | None:
    """Detect unparameterized SQL query assembly (CWE-89)."""
    text = line_text
    is_vulnerable = (
        'f"SELECT' in text
        or "f'SELECT" in text
        or ("SELECT %s" not in text and "SELECT" in text and "{" in text)
        or ("SELECT" in text and "+" in text and ("FROM" in text or "WHERE" in text))
        or ("execute_query(" in text and ("+" in text or 'f"' in text or "f'" in text))
    )
    if is_vulnerable:
        return ReviewFinding(
            path=path,
            line=line_no,
            category=IssueCategory.SECURITY,
            severity=Severity.HIGH,
            message=(
                "SQL Injection vulnerability (CWE-89): dynamic input is directly "
                "interpolated into query string. Use parameterized queries."
            ),
            suggestion=(
                '    query = "SELECT * FROM users WHERE id = %s"\n'
                "    cursor.execute(query, (user_id,))"
            ),
            confidence=0.96,
            standard_id="STD-SEC-01",
        )
    return None


SECRET_PATTERNS: list[tuple[re.Pattern[str], str]] = [
    (re.compile(r"sk-[a-zA-Z0-9]{20,}"), "OpenAI API token"),
    (re.compile(r"ghp_[a-zA-Z0-9]{30,}"), "GitHub Personal Access Token"),
    (re.compile(r"AKIA[0-9A-Z]{16}"), "AWS Access Key ID"),
    (re.compile(r"sk_live_[a-zA-Z0-9]{24,}"), "Stripe Live Secret Key"),
    (re.compile(r"-----BEGIN (?:RSA|EC|OPENSSH) PRIVATE KEY-----"), "Private cryptographic key"),
]


def check_hardcoded_secrets(line_text: str, path: str, line_no: int) -> ReviewFinding | None:
    """Detect raw API tokens, keys, or pre-redacted secret markers (CWE-798)."""
    for pattern, label in SECRET_PATTERNS:
        if pattern.search(line_text):
            return ReviewFinding(
                path=path,
                line=line_no,
                category=IssueCategory.SECURITY,
                severity=Severity.HIGH,
                message=(
                    f"Hardcoded credential detected ({label}, CWE-798). "
                    "Store credentials in environment variables or a secret vault."
                ),
                suggestion='    api_key = os.environ.get("SERVICE_API_KEY")',
                confidence=0.99,
                standard_id="STD-SEC-02",
            )

    if "[REDACTED_SECRET]" in line_text:
        return ReviewFinding(
            path=path,
            line=line_no,
            category=IssueCategory.SECURITY,
            severity=Severity.HIGH,
            message=(
                "Hardcoded API credential detected and automatically scrubbed (CWE-798). "
                "Store credentials in environment variables or secret vaults."
            ),
            suggestion='    api_key = os.environ.get("SERVICE_API_KEY")',
            confidence=0.99,
            standard_id="STD-SEC-02",
        )
    return None


HTTP_METHODS = (".get(", ".post(", ".put(", ".delete(")


def check_missing_timeout(line_text: str, path: str, line_no: int) -> ReviewFinding | None:
    """Detect unconstrained HTTP client requests lacking timeout deadline (CWE-703)."""
    if (
        ("httpx." in line_text or "requests." in line_text or "aiohttp." in line_text)
        and any(m in line_text for m in HTTP_METHODS)
        and "timeout=" not in line_text
    ):
        return ReviewFinding(
            path=path,
            line=line_no,
            category=IssueCategory.ERROR_HANDLING,
            severity=Severity.MEDIUM,
            message=(
                "Asynchronous HTTP invocation lacks explicit timeout (CWE-703). "
                "Network partitions will cause unconstrained connection starvation."
            ),
            suggestion='    response = await client.post("https://api.partner.com/events", timeout=10.0)',
            confidence=0.91,
            standard_id="STD-ERR-02",
        )
    return None


def check_bare_except(line_text: str, path: str, line_no: int) -> ReviewFinding | None:
    """Detect blanket bare except: blocks suppressing system signals (CWE-391)."""
    if line_text.strip().startswith("except:"):
        return ReviewFinding(
            path=path,
            line=line_no,
            category=IssueCategory.ERROR_HANDLING,
            severity=Severity.HIGH,
            message=(
                "Bare `except:` block silently suppresses system signals (CWE-391). "
                "Catch specific exceptions or log with context."
            ),
            suggestion=(
                "    except Exception as exc:\n"
                '        logger.error("Operation failed: %s", exc)\n'
                "        raise"
            ),
            confidence=0.93,
            standard_id="STD-ERR-01",
        )
    return None


# ---------------------------------------------------------------------------
# Default Registry
# ---------------------------------------------------------------------------

DEFAULT_RULES: list[DeterministicRule] = [
    DeterministicRule(
        rule_id="STD-SEC-01",
        name="Parameterized Database Queries",
        category=IssueCategory.SECURITY,
        severity=Severity.HIGH,
        cwe="CWE-89",
        confidence=0.96,
        check=check_sql_injection,
    ),
    DeterministicRule(
        rule_id="STD-SEC-02",
        name="Zero Plaintext Secrets or Credentials",
        category=IssueCategory.SECURITY,
        severity=Severity.HIGH,
        cwe="CWE-798",
        confidence=0.99,
        check=check_hardcoded_secrets,
    ),
    DeterministicRule(
        rule_id="STD-ERR-02",
        name="Explicit Network/HTTP Timeout Boundaries",
        category=IssueCategory.ERROR_HANDLING,
        severity=Severity.MEDIUM,
        cwe="CWE-703",
        confidence=0.91,
        check=check_missing_timeout,
    ),
    DeterministicRule(
        rule_id="STD-ERR-01",
        name="Explicit Exception Catching",
        category=IssueCategory.ERROR_HANDLING,
        severity=Severity.HIGH,
        cwe="CWE-391",
        confidence=0.93,
        check=check_bare_except,
    ),
]


# ---------------------------------------------------------------------------
# Execution Functions
# ---------------------------------------------------------------------------


def evaluate_rules(
    file_diffs: list[FileDiff],
    rules: Sequence[DeterministicRule] | None = None,
) -> list[ReviewFinding]:
    """Execute deterministic rules against added lines in file diffs.

    Returns deduplicated review findings matching active rules.
    """
    active_rules = rules if rules is not None else DEFAULT_RULES
    findings: list[ReviewFinding] = []
    seen: set[tuple[str, int, str]] = set()

    for fd in file_diffs:
        for hunk in fd.hunks:
            for line in hunk.lines:
                if line.kind == "add" and line.new_no is not None:
                    text = line.text
                    for rule in active_rules:
                        finding = rule.check(text, fd.path, line.new_no)
                        if finding is not None:
                            dedup_key = (finding.path, finding.line, finding.category.value)
                            if dedup_key not in seen:
                                seen.add(dedup_key)
                                findings.append(finding)
    return findings


def evaluate_rules_as_dicts(
    file_diffs: list[FileDiff],
    rules: Sequence[DeterministicRule] | None = None,
) -> list[dict[str, Any]]:
    """Execute deterministic rules and serialize findings as dictionary records."""
    findings = evaluate_rules(file_diffs, rules=rules)
    return [
        {
            "path": f.path,
            "line": f.line,
            "category": f.category.value,
            "severity": f.severity.value,
            "message": f.message,
            "suggestion": f.suggestion,
            "confidence": f.confidence,
            "standard_id": f.standard_id,
        }
        for f in findings
    ]
