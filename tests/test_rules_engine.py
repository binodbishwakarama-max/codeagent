"""Unit and integration tests for the deterministic static analysis rules engine."""

from __future__ import annotations

import pytest

from prism.config import ReviewPipelineSettings
from prism.models import DiffLine, FileDiff, Hunk, IssueCategory, Severity
from prism.review.llm import FakeLLMClient
from prism.review.pipeline import PRContext, review_pr
from prism.review.rules_engine import (
    DEFAULT_RULES,
    DeterministicRule,
    check_bare_except,
    check_hardcoded_secrets,
    check_missing_timeout,
    check_sql_injection,
    evaluate_rules,
    evaluate_rules_as_dicts,
)


class TestSQLInjectionRule:
    """Verify SQL injection pattern detection."""

    def test_detects_f_string_query(self) -> None:
        finding = check_sql_injection('query = f"SELECT * FROM users WHERE id = {user_id}"', "src/db.py", 10)
        assert finding is not None
        assert finding.path == "src/db.py"
        assert finding.line == 10
        assert finding.category == IssueCategory.SECURITY
        assert finding.severity == Severity.HIGH
        assert finding.standard_id == "STD-SEC-01"
        assert "SQL Injection" in finding.message

    def test_detects_string_concatenation_query(self) -> None:
        finding = check_sql_injection(
            'sql = "SELECT id, name FROM accounts WHERE org = " + org_id', "src/models.py", 42
        )
        assert finding is not None
        assert finding.category == IssueCategory.SECURITY

    def test_detects_execute_query_with_interpolation(self) -> None:
        finding = check_sql_injection('return execute_query("SELECT * FROM t WHERE k=" + key)', "src/repo.py", 15)
        assert finding is not None

    def test_ignores_parameterized_query(self) -> None:
        safe_line = 'cursor.execute("SELECT id, name FROM accounts WHERE org = %s", (org_id,))'
        finding = check_sql_injection(safe_line, "src/models.py", 42)
        assert finding is None


class TestHardcodedSecretsRule:
    """Verify credential and API token pattern detection."""

    def test_detects_openai_api_token(self) -> None:
        dummy_openai = f"{'sk'}-{'mocktoken1234567890123456'}"
        finding = check_hardcoded_secrets(f'api_key = "{dummy_openai}"', "src/config.py", 5)
        assert finding is not None
        assert finding.category == IssueCategory.SECURITY
        assert finding.standard_id == "STD-SEC-02"
        assert "OpenAI API token" in finding.message

    def test_detects_github_pat(self) -> None:
        dummy_ghp = f"{'ghp'}_{'mockgithubpat12345678901234567890'}"
        finding = check_hardcoded_secrets(f'token = "{dummy_ghp}"', "deploy.py", 22)
        assert finding is not None
        assert "GitHub Personal Access Token" in finding.message

    def test_detects_aws_key_id(self) -> None:
        dummy_aws = f"{'AKIA'}{'IOSFODNN7EXAMPLE'}"
        finding = check_hardcoded_secrets(f'aws_access = "{dummy_aws}"', "src/s3.py", 8)
        assert finding is not None
        assert "AWS Access Key ID" in finding.message

    def test_detects_stripe_live_key(self) -> None:
        dummy_stripe = f"{'sk_live'}_{'mockstripekey123456789012'}"
        finding = check_hardcoded_secrets(f'stripe_key = "{dummy_stripe}"', "src/billing.py", 12)
        assert finding is not None
        assert finding.severity == Severity.HIGH

    def test_detects_private_crypto_key(self) -> None:
        finding = check_hardcoded_secrets("-----BEGIN RSA PRIVATE KEY-----", "certs/key.pem", 1)
        assert finding is not None
        assert finding.severity == Severity.HIGH
        assert "Private cryptographic key" in finding.message

    def test_detects_redacted_secret_placeholder(self) -> None:
        finding = check_hardcoded_secrets("client_secret = '[REDACTED_SECRET]'", "src/oauth.py", 33)
        assert finding is not None
        assert finding.severity == Severity.HIGH

    def test_ignores_env_var_lookup(self) -> None:
        safe_line = 'api_key = os.environ.get("OPENAI_API_KEY")'
        finding = check_hardcoded_secrets(safe_line, "src/config.py", 5)
        assert finding is None


class TestMissingTimeoutRule:
    """Verify unconstrained network invocation detection."""

    def test_detects_httpx_get_without_timeout(self) -> None:
        finding = check_missing_timeout('resp = await httpx.get("https://api.example.com/data")', "src/client.py", 18)
        assert finding is not None
        assert finding.category == IssueCategory.ERROR_HANDLING
        assert finding.standard_id == "STD-ERR-02"
        assert "timeout" in finding.message.lower()

    def test_detects_requests_post_without_timeout(self) -> None:
        finding = check_missing_timeout('response = requests.post(webhook_url, json=payload)', "src/webhook.py", 45)
        assert finding is not None

    def test_ignores_calls_with_explicit_timeout(self) -> None:
        safe_line = 'resp = await httpx.get("https://api.example.com/data", timeout=5.0)'
        finding = check_missing_timeout(safe_line, "src/client.py", 18)
        assert finding is None


class TestBareExceptRule:
    """Verify bare except block detection."""

    def test_detects_bare_except_block(self) -> None:
        finding = check_bare_except("    except:", "src/worker.py", 55)
        assert finding is not None
        assert finding.category == IssueCategory.ERROR_HANDLING
        assert finding.severity == Severity.HIGH
        assert finding.standard_id == "STD-ERR-01"

    def test_ignores_specific_exception(self) -> None:
        safe_line = "    except Exception as exc:"
        finding = check_bare_except(safe_line, "src/worker.py", 55)
        assert finding is None

    def test_ignores_tuple_exceptions(self) -> None:
        safe_line = "    except (KeyError, ValueError):"
        finding = check_bare_except(safe_line, "src/worker.py", 55)
        assert finding is None


class TestEvaluateRules:
    """Verify full rules engine execution on multi-file diffs."""

    def test_evaluates_added_lines_only(self) -> None:
        file_diff = FileDiff(
            path="src/service.py",
            language="python",
            hunks=[
                Hunk(
                    old_start=1,
                    new_start=1,
                    lines=[
                        # Context lines (should be ignored)
                        DiffLine(kind="ctx", old_no=1, new_no=1, text='f"SELECT * FROM old WHERE id={id}"'),
                        # Deleted lines (should be ignored)
                        DiffLine(kind="del", old_no=2, new_no=None, text="except:"),
                        # Added line with defect (must be flagged)
                        DiffLine(kind="add", old_no=None, new_no=2, text="    except:"),
                        # Added line with SQL injection (must be flagged)
                        DiffLine(kind="add", old_no=None, new_no=3, text='query = f"SELECT * FROM items WHERE k={key}"'),
                        # Added clean line
                        DiffLine(kind="add", old_no=None, new_no=4, text="return query"),
                    ],
                )
            ],
        )

        findings = evaluate_rules([file_diff])
        assert len(findings) == 2
        categories = {f.category for f in findings}
        assert IssueCategory.ERROR_HANDLING in categories
        assert IssueCategory.SECURITY in categories
        assert {f.line for f in findings} == {2, 3}

    def test_evaluate_rules_as_dicts(self) -> None:
        file_diff = FileDiff(
            path="src/auth.py",
            language="python",
            hunks=[
                Hunk(
                    old_start=10,
                    new_start=10,
                    lines=[
                        DiffLine(kind="add", old_no=None, new_no=11, text='api_key = "[REDACTED_SECRET]"'),
                    ],
                )
            ],
        )

        records = evaluate_rules_as_dicts([file_diff])
        assert len(records) == 1
        r = records[0]
        assert r["path"] == "src/auth.py"
        assert r["line"] == 11
        assert r["category"] == "SECURITY"
        assert r["severity"] == "HIGH"
        assert r["standard_id"] == "STD-SEC-02"


@pytest.mark.asyncio
class TestPipelineFastPassIntegration:
    """Verify pre-LLM fast pass integration in review_pr pipeline."""

    async def test_fast_pass_identifies_defects_and_deduplicates(self) -> None:
        file_diff = FileDiff(
            path="src/auth.py",
            language="python",
            hunks=[
                Hunk(
                    old_start=1,
                    new_start=1,
                    lines=[
                        DiffLine(kind="ctx", old_no=1, new_no=1, text="def login():"),
                        DiffLine(
                            kind="add",
                            old_no=None,
                            new_no=2,
                            text='query = f"SELECT * FROM users WHERE user={name}"',
                        ),
                    ],
                )
            ],
        )

        ctx = PRContext(
            repo_full_name="org/repo",
            pr_number=42,
            title="Update login query",
            file_diffs=[file_diff],
        )

        settings = ReviewPipelineSettings(
            deterministic_rules_enabled=True,
            confidence_threshold=0.5,
        )

        # Mock LLM returns empty findings; deterministic fast pass still flags SQL injection
        client = FakeLLMClient(responses=[])
        result = await review_pr(ctx, settings=settings, llm_client=client)

        assert len(result.findings) == 1
        finding = result.findings[0]
        assert finding.path == "src/auth.py"
        assert finding.line == 2
        assert finding.category == IssueCategory.SECURITY
        assert finding.standard_id == "STD-SEC-01"
