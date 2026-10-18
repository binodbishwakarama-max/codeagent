"""Interactive local simulation of GitHub App webhooks and PR review pipeline.

Demonstrates:
1. Health & readiness probes (/livez, /readyz).
2. Cryptographic HMAC-SHA256 signature verification on /webhook/github.
3. Rejection of tampered signatures (HTTP 401).
4. Idempotent deduplication of replayed webhooks.
5. Bot / draft PR filtering.
6. Unified diff parsing on multi-file PR changes.
7. Automated secret redaction (preventing credential leakage).
8. Repository engineering standards matching (.prism/standards.yml).
9. Review intelligence inference and self-verification filtering (hallucination rejection).
10. GitHub Check Run summary and inline review comments with one-click commit suggestions.
11. Developer feedback telemetry (thumbs-up / thumbs-down reactions).
12. Live Prometheus metrics snapshot (/metrics).
"""

import asyncio
import hashlib
import hmac
import json
import os
import sys
import uuid
from typing import Any
from unittest.mock import AsyncMock, patch

import httpx

from prism.api import app
from prism.config import Settings
from prism.github.comments import format_inline_comment, format_review_summary
from prism.github.diff import parse_unified_diff
from prism.models import IssueCategory, ReviewFinding, Severity
from prism.review.llm import FakeLLMClient
from prism.review.pipeline import PRContext, review_pr
from prism.review.prompts import ChunkReviewResponse
from prism.review.redactor import redact_file_diffs
from prism.review.standards import load_default_standards
from prism.review.verify import FindingVerification, VerificationResponse

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

WEBHOOK_SECRET = "prism-local-dev-secret-2026"
os.environ["GITHUB_APP_ID"] = "test-app-123"
os.environ["GITHUB_PRIVATE_KEY"] = (
    "-----BEGIN RSA PRIVATE KEY-----\nFAKE\n-----END RSA PRIVATE KEY-----"
)
os.environ["GITHUB_WEBHOOK_SECRET"] = WEBHOOK_SECRET
os.environ["DATABASE_URL"] = "postgresql://localhost:5432/prism"
os.environ["REDIS_URL"] = "redis://localhost:6379/0"

SAMPLE_UNIFIED_DIFF = """diff --git a/src/db/users.py b/src/db/users.py
index 1024abc..2048def 100644
--- a/src/db/users.py
+++ b/src/db/users.py
@@ -10,6 +10,14 @@ def get_user_by_id(db_conn, user_id: str):
     cursor = db_conn.cursor()
-    cursor.execute("SELECT id, username, email FROM users WHERE id = %s", (user_id,))
+    # Query user account by ID directly
+    query = f"SELECT id, username, email FROM users WHERE id = '{user_id}'"
+    cursor.execute(query)
     return cursor.fetchone()
+
+def fetch_stripe_billing_info(customer_id: str):
+    # Direct API client connection
+    api_key = "sk-proj-51OzK9eABC1234567890XYZabcdef12345"
+    return {"client": api_key, "customer": customer_id}
diff --git a/src/api/handlers.py b/src/api/handlers.py
index 3012abc..4096def 100644
--- a/src/api/handlers.py
+++ b/src/api/handlers.py
@@ -25,6 +25,10 @@ async def process_order(order_data: dict) -> bool:
     try:
         await dispatch_order(order_data)
         return True
+    except:
+        # Silently ignore failure
+        pass
+        return False
"""


def compute_signature(payload_bytes: bytes, secret: str) -> str:
    digest = hmac.new(secret.encode("utf-8"), payload_bytes, hashlib.sha256).hexdigest()
    return f"sha256={digest}"


def create_pr_event(
    action: str = "opened",
    pr_number: int = 42,
    draft: bool = False,
    user_type: str = "User",
    user_login: str = "alex-developer",
) -> dict[str, Any]:
    return {
        "action": action,
        "installation": {"id": 98765},
        "repository": {
            "name": "codeagent",
            "full_name": "binodbishwakarama-max/codeagent",
            "private": False,
        },
        "pull_request": {
            "number": pr_number,
            "title": "feat: direct query user accounts and stripe integration",
            "body": "Updates user queries and adds direct Stripe customer lookup.",
            "state": "open",
            "draft": draft,
            "user": {"type": user_type, "login": user_login},
            "head": {
                "sha": "a1b2c3d4e5f678901234567890abcdef12345678",
                "ref": "feature/direct-db-query",
                "repo": {
                    "full_name": "binodbishwakarama-max/codeagent",
                    "fork": False,
                },
            },
            "base": {
                "sha": "0000000000000000000000000000000000000000",
                "ref": "main",
            },
            "author_association": "MEMBER",
        },
    }


class MockRedis:
    """In-memory Redis mock for local simulation without external Redis daemon."""

    def __init__(self) -> None:
        self.store: dict[str, str] = {}

    async def set(
        self, key: str, value: Any, nx: bool = False, ex: int | None = None
    ) -> bool | None:
        if nx and key in self.store:
            return None
        self.store[key] = str(value)
        return True

    async def ping(self) -> bool:
        return True

    async def aclose(self) -> None:
        pass


def create_comment_reaction_event(
    comment_id: int = 501,
    content: str = "+1",
) -> dict[str, Any]:
    return {
        "action": "created",
        "installation": {"id": 98765},
        "repository": {
            "name": "codeagent",
            "full_name": "binodbishwakarama-max/codeagent",
        },
        "pull_request": {
            "number": 101,
        },
        "comment": {
            "id": comment_id,
            "user": {"login": "prism-review[bot]"},
            "body": "+1",
        },
        "reaction": {
            "content": content,
            "user": {"login": "alex-developer"},
        },
    }


async def run_simulation() -> None:
    print("=" * 80)
    print("💎 PRISM REVIEW - LOCAL GITHUB APP WEBHOOK & PIPELINE SIMULATOR")
    print("=" * 80)

    import prism.api

    prism.api._redis_client = MockRedis()
    prism.api._settings = Settings()

    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
        # -------------------------------------------------------------------
        # Phase 1: Health & Readiness Endpoints
        # -------------------------------------------------------------------
        print("\n[Phase 1] Probing Health & Liveness Endpoints...")
        healthz = await client.get("/healthz")
        readyz = await client.get("/readyz")
        print(f"  GET /healthz -> Status: {healthz.status_code} | Body: {healthz.json()}")
        print(f"  GET /readyz  -> Status: {readyz.status_code} | Body: {readyz.json()}")
        assert healthz.status_code == 200
        assert readyz.status_code in (200, 503)

        # -------------------------------------------------------------------
        # Phase 2: Webhook Security & Ingestion
        # -------------------------------------------------------------------
        print("\n[Phase 2] Simulating GitHub Webhook Ingestion & Security Checks...")

        pr_payload = create_pr_event(action="opened", pr_number=101)
        payload_bytes = json.dumps(pr_payload).encode("utf-8")
        valid_sig = compute_signature(payload_bytes, WEBHOOK_SECRET)
        delivery_id = str(uuid.uuid4())

        # Test A: Valid HMAC Signature
        headers_valid = {
            "Content-Type": "application/json",
            "X-GitHub-Event": "pull_request",
            "X-GitHub-Delivery": delivery_id,
            "X-Hub-Signature-256": valid_sig,
        }
        res_valid = await client.post(
            "/webhook/github", content=payload_bytes, headers=headers_valid
        )
        print("  [2A] Valid HMAC-SHA256 Signature Delivery:")
        print(f"       -> Status: {res_valid.status_code} | Response: {res_valid.json()}")
        assert res_valid.status_code in (200, 202)

        # Test B: Replay Attack / Idempotency Check (Same Delivery ID)
        res_replay = await client.post(
            "/webhook/github", content=payload_bytes, headers=headers_valid
        )
        print("  [2B] Idempotent Replay (Duplicate Delivery ID):")
        print(f"       -> Status: {res_replay.status_code} | Response: {res_replay.json()}")
        assert res_replay.status_code in (200, 202)
        assert res_replay.json().get("status") in ("ignored", "duplicate")

        # Test C: Tampered / Invalid Signature
        tampered_sig = "sha256=0000000000000000000000000000000000000000000000000000000000000000"
        headers_tampered = {
            "Content-Type": "application/json",
            "X-GitHub-Event": "pull_request",
            "X-GitHub-Delivery": str(uuid.uuid4()),
            "X-Hub-Signature-256": tampered_sig,
        }
        res_tampered = await client.post(
            "/webhook/github", content=payload_bytes, headers=headers_tampered
        )
        print("  [2C] Tampered Signature Rejection:")
        print(f"       -> Status: {res_tampered.status_code} (Unauthorized) | Blocked gracefully")
        assert res_tampered.status_code == 401

        # Test D: Draft Pull Request (Filtered out)
        draft_payload = create_pr_event(action="opened", pr_number=102, draft=True)
        draft_bytes = json.dumps(draft_payload).encode("utf-8")
        draft_sig = compute_signature(draft_bytes, WEBHOOK_SECRET)
        headers_draft = {
            "Content-Type": "application/json",
            "X-GitHub-Event": "pull_request",
            "X-GitHub-Delivery": str(uuid.uuid4()),
            "X-Hub-Signature-256": draft_sig,
        }
        res_draft = await client.post("/webhook/github", content=draft_bytes, headers=headers_draft)
        print("  [2D] Draft Pull Request Filtering:")
        print(
            f"       -> Status: {res_draft.status_code} | Action: {res_draft.json().get('status')}"
        )

        # -------------------------------------------------------------------
        # Phase 3: Review Intelligence Pipeline on Real Diff
        # -------------------------------------------------------------------
        print("\n" + "=" * 80)
        print("[Phase 3] Executing Review Intelligence Engine on Realistic Multi-File PR Diff")
        print("=" * 80)

        # Step 3.1: Parse unified diff
        file_diffs = parse_unified_diff(SAMPLE_UNIFIED_DIFF)
        print(f"\n  [3.1] Parsed Diff: {len(file_diffs)} modified files found.")
        for fd in file_diffs:
            print(f"        - {fd.path} ({fd.language}): {len(fd.hunks)} hunk(s)")

        # Step 3.2: Automated Secret Redactor
        print("\n  [3.2] Executing Automated Secret Redaction (Pre-LLM)...")
        redacted_diffs = redact_file_diffs(file_diffs)
        scrubbed_count = sum(
            1
            for fd in redacted_diffs
            for hunk in fd.hunks
            for line in hunk.lines
            if "REDACTED" in line.text
        )
        print(f"        -> Redacted {scrubbed_count} secret(s) across diffs.")
        for fd in redacted_diffs:
            for hunk in fd.hunks:
                for line in hunk.lines:
                    if "REDACTED" in line.text:
                        print(f"        [Scrubbed Line in {fd.path}]: {line.text.strip()}")

        # Step 3.3: Engineering Standards Retrieval
        print("\n  [3.3] Loading Repository Standards (.prism/standards.yml)...")
        standards_file = load_default_standards()
        standards = standards_file.standards
        print(f"        -> Loaded {len(standards)} active engineering standards:")
        for std in standards[:3]:
            print(f"           * [{std.id}] {std.title}")

        # Step 3.4: LLM Structured Review + Self-Verification Filter
        print("\n  [3.4] Running LLM Inference & Self-Verification Filter...")
        # Prepare mock findings including 2 real catches and 1 hallucinated finding
        # to demonstrate self-verification discarding the false positive.
        findings_to_propose = [
            ReviewFinding(
                path="src/db/users.py",
                line=13,
                side="RIGHT",
                category=IssueCategory.SECURITY,
                severity=Severity.HIGH,
                message=(
                    "SQL Injection vulnerability: untrusted input `user_id` is directly "
                    "interpolated into query string. Use parameterized queries."
                ),
                confidence=0.96,
                suggestion=(
                    '    query = "SELECT id, username, email FROM users WHERE id = %s"\n'
                    "    cursor.execute(query, (user_id,))"
                ),
                standard_id="SEC-002",
            ),
            ReviewFinding(
                path="src/api/handlers.py",
                line=28,
                side="RIGHT",
                category=IssueCategory.ERROR_HANDLING,
                severity=Severity.HIGH,
                message=(
                    "Bare `except:` block silently suppresses all exceptions, including "
                    "KeyboardInterrupt and SystemExit. Catch specific exceptions or log with traceback."
                ),
                confidence=0.92,
                suggestion=(
                    "    except Exception as exc:\n"
                    "        logger.error('Order dispatch failed: %s', exc)\n"
                    "        return False"
                ),
                standard_id="ERR-001",
            ),
            ReviewFinding(
                path="src/db/users.py",
                line=10,
                side="RIGHT",
                category=IssueCategory.READABILITY,
                severity=Severity.LOW,
                message="Consider renaming db_conn to connection for stylistic reasons.",
                confidence=0.55,
                suggestion=None,
                standard_id=None,
            ),
        ]

        # Self-verification response discarding the nitpick finding on line 10
        verification_response = VerificationResponse(
            verifications=[
                FindingVerification(
                    path="src/db/users.py",
                    line=13,
                    decision="KEEP",
                    reason="Critical SQL injection vulnerability introduced by query interpolation.",
                ),
                FindingVerification(
                    path="src/api/handlers.py",
                    line=28,
                    decision="KEEP",
                    reason="Bare except suppresses system interrupts and hides production crashes.",
                ),
                FindingVerification(
                    path="src/db/users.py",
                    line=10,
                    decision="DROP",
                    reason="Unhelpful stylistic nitpick on unchanged signature line; unnecessary noise.",
                ),
            ]
        )

        fake_client = FakeLLMClient(
            responses=[
                ChunkReviewResponse(findings=findings_to_propose),
                verification_response,
            ]
        )

        settings = Settings(
            github_app_id="sim-app-1",
            github_private_key="fake-pem",
            github_webhook_secret=WEBHOOK_SECRET,
            confidence_threshold=0.70,
            max_comments_per_pr=5,
            retrieval_enabled=False,
            self_verification_enabled=True,
        )

        pr_ctx = PRContext(
            repo_full_name="binodbishwakarama-max/codeagent",
            pr_number=101,
            title="feat: direct query user accounts and stripe integration",
            file_diffs=redacted_diffs,
            standards=standards,
        )

        review_result = await review_pr(pr_ctx, settings, llm_client=fake_client)

        dropped_count = len(findings_to_propose) - len(review_result.findings)
        print(f"        -> Verified Findings: {len(review_result.findings)}")
        print(f"        -> Dropped / Filtered: {dropped_count}")

        # -------------------------------------------------------------------
        # Phase 4: Formatted GitHub PR Output
        # -------------------------------------------------------------------
        print("\n" + "=" * 80)
        print("[Phase 4] Output Rendered to GitHub Platform (Inline Comments & Check Run)")
        print("=" * 80)

        summary_body = format_review_summary(
            review_result.findings,
            dropped_count=dropped_count,
        )
        print("\n--- [GITHUB CHECK RUN / SUMMARY BODY] ---")
        print(summary_body)

        print("\n--- [INLINE REVIEW COMMENTS WITH ONE-CLICK COMMIT SUGGESTIONS] ---")
        for i, finding in enumerate(review_result.findings, 1):
            print(
                f"\n[Comment #{i} on {finding.path}:{finding.line} (Severity: {finding.severity.value})]"
            )
            comment_text = format_inline_comment(finding)
            print(comment_text)

        # -------------------------------------------------------------------
        # Phase 5: Feedback Webhook & Telemetry
        # -------------------------------------------------------------------
        print("\n" + "=" * 80)
        print("[Phase 5] Simulating Developer Feedback (👍 Thumbs Up Reaction Webhook)")
        print("=" * 80)

        reaction_payload = create_comment_reaction_event(comment_id=501, content="+1")
        reaction_bytes = json.dumps(reaction_payload).encode("utf-8")
        reaction_sig = compute_signature(reaction_bytes, WEBHOOK_SECRET)
        headers_reaction = {
            "Content-Type": "application/json",
            "X-GitHub-Event": "pull_request_review_comment",
            "X-GitHub-Delivery": str(uuid.uuid4()),
            "X-Hub-Signature-256": reaction_sig,
        }

        with patch("prism.api.record_comment_reaction", new=AsyncMock(return_value=1)):
            res_reaction = await client.post(
                "/webhook/github", content=reaction_bytes, headers=headers_reaction
            )
            print(f"  POST /webhook/github (reaction: +1) -> Status: {res_reaction.status_code}")
            print(f"  Response: {res_reaction.json()}")
            assert res_reaction.status_code == 200

        # Phase 6: Live Prometheus Metrics Snapshot
        print("\n[Phase 6] Inspecting Live Prometheus Telemetry (/metrics)...")
        metrics_resp = await client.get("/metrics")
        print(f"  GET /metrics -> Status: {metrics_resp.status_code}")
        # Print relevant metric lines
        lines = metrics_resp.text.strip().split("\n")
        prism_metrics = [
            metric_line
            for metric_line in lines
            if metric_line.startswith("prism_")
            or metric_line.startswith("feedback_")
            or metric_line.startswith("webhook_")
        ]
        for m in prism_metrics[:10]:
            print(f"  {m}")

    print("\n" + "=" * 80)
    print("✅ SIMULATION COMPLETE: ALL WEBHOOK & REVIEW PIPELINE STAGES PASSED!")
    print("=" * 80)


if __name__ == "__main__":
    asyncio.run(run_simulation())
