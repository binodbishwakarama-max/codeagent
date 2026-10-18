"""Unit and integration tests for multi-turn PR review thread resolution.

Tests:
1. Comment intent matching (mentions, thread replies, bot filtering, reaction exclusion).
2. Prompt construction and Markdown formatting.
3. LLM resolution logic with FakeLLMClient.
4. Background run_thread_reply job execution with mock GitHubClient and database persistence.
5. End-to-end webhook ingestion for pull_request_review_comment.
"""

import hashlib
import hmac
import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from prism.api import app
from prism.config import Settings
from prism.db import create_tables, get_feedback_records
from prism.github.client import GitHubClient
from prism.queue.jobs import run_thread_reply
from prism.review.conversation import (
    PRISM_REPLY_MARKER,
    ThreadResolutionDecision,
    ThreadResolutionResult,
    build_thread_resolution_prompt,
    format_thread_reply_markdown,
    is_prism_mention,
    resolve_thread_conversation,
    should_respond_to_comment,
)
from prism.review.llm import FakeLLMClient

TEST_DB = "sqlite+aiosqlite:///:memory:"


def make_sig(body: bytes, secret: str) -> str:
    digest = hmac.new(secret.encode("utf-8"), body, hashlib.sha256).hexdigest()
    return f"sha256={digest}"


# ---------------------------------------------------------------------------
# 1. Intent Matching Tests
# ---------------------------------------------------------------------------


class TestIntentMatching:
    def test_prism_mention_regex(self) -> None:
        assert is_prism_mention("@prism-review what is the fix?")
        assert is_prism_mention("Hey @prism can you check this?")
        assert is_prism_mention("@PRISM-REVIEW false positive")
        assert not is_prism_mention("prism is a cool geometric shape")
        assert not is_prism_mention("@prismatic-coder hello")

    def test_should_respond_to_comment_bot_filtering(self) -> None:
        should_rep, reason = should_respond_to_comment(
            comment_body="@prism-review hello",
            sender_type="Bot",
        )
        assert not should_rep
        assert "bot" in reason

        should_rep, reason = should_respond_to_comment(
            comment_body="@prism-review hello",
            sender_login="dependabot[bot]",
        )
        assert not should_rep
        assert "bot" in reason

    def test_should_respond_to_comment_reaction_exclusion(self) -> None:
        for emoji in ("+1", "-1", "👍", "👎", "thumbs_up"):
            should_rep, reason = should_respond_to_comment(comment_body=emoji)
            assert not should_rep
            assert "reaction" in reason

    def test_should_respond_to_comment_explicit_mention(self) -> None:
        should_rep, reason = should_respond_to_comment(
            comment_body="@prism-review this input is already sanitized",
            sender_login="developer1",
        )
        assert should_rep
        assert "explicit mention" in reason

    def test_should_respond_to_comment_in_thread_reply(self) -> None:
        parent_prism_body = "<!-- prism-review -->\n**[Security]** SQL injection vulnerability"
        should_rep, reason = should_respond_to_comment(
            comment_body="False alarm, this is validated upstream.",
            parent_body=parent_prism_body,
            in_reply_to_id=9876,
            sender_login="developer1",
        )
        assert should_rep
        assert "reply to prism" in reason

    def test_should_ignore_unrelated_comment(self) -> None:
        should_rep, reason = should_respond_to_comment(
            comment_body="Hey @alice can you review this PR?",
            parent_body="Can you check this method?",
            in_reply_to_id=123,
            sender_login="bob",
        )
        assert not should_rep
        assert "no mention" in reason


# ---------------------------------------------------------------------------
# 2. Prompt Construction and Formatting Tests
# ---------------------------------------------------------------------------


class TestPromptAndFormatting:
    def test_build_thread_resolution_prompt(self) -> None:
        prompt = build_thread_resolution_prompt(
            developer_message="We use parameterized queries via the ORM here.",
            developer_author="dev_sarah",
            parent_comment_body="Possible SQL injection in raw query",
            file_path="src/db/users.py",
            diff_hunk="@@ -10,3 +10,3 @@\n- select()\n+ execute()",
            line_number=24,
            standards_context="Rule SEC-01: No raw string concatenation",
        )
        assert "src/db/users.py" in prompt
        assert "Line 24" in prompt
        assert "Possible SQL injection" in prompt
        assert "dev_sarah" in prompt
        assert "SEC-01" in prompt
        assert "execute()" in prompt

    def test_format_thread_reply_markdown_resolved(self) -> None:
        result = ThreadResolutionResult(
            decision=ThreadResolutionDecision.RESOLVED,
            is_false_positive=True,
            explanation="You are correct. The input is sanitized in the auth middleware.",
            suggested_patch=None,
        )
        md = format_thread_reply_markdown(result, developer_author="alex")
        assert PRISM_REPLY_MARKER in md
        assert "Conceded (False Positive Accepted)" in md
        assert "sanitized in the auth middleware" in md
        assert "@alex" in md
        assert "```suggestion" not in md

    def test_format_thread_reply_markdown_maintained(self) -> None:
        result = ThreadResolutionResult(
            decision=ThreadResolutionDecision.MAINTAINED,
            is_false_positive=False,
            explanation="The middleware sanitizes query params, but this header remains unsanitized.",
            suggested_patch="safe_val = sanitize_header(req.header)",
        )
        md = format_thread_reply_markdown(result, developer_author="alex")
        assert PRISM_REPLY_MARKER in md
        assert "Finding Maintained" in md
        assert "safe_val = sanitize_header" in md
        assert "```suggestion" in md


# ---------------------------------------------------------------------------
# 3. LLM Resolution Logic
# ---------------------------------------------------------------------------


class TestResolutionExecution:
    @pytest.mark.asyncio
    async def test_resolve_thread_conversation_with_fake_llm(self) -> None:
        expected = ThreadResolutionResult(
            decision=ThreadResolutionDecision.RESOLVED,
            is_false_positive=True,
            explanation="Verified against schema validation: safe to proceed.",
            suggested_patch=None,
        )
        fake_llm = FakeLLMClient(responses=[expected.model_dump()])

        result, reply_md = await resolve_thread_conversation(
            llm_client=fake_llm,
            developer_message="@prism-review safe because Pydantic validates it",
            developer_author="marcus",
            parent_comment_body="Missing validation on payload dict",
            file_path="src/api.py",
        )

        assert result.decision == ThreadResolutionDecision.RESOLVED
        assert result.is_false_positive is True
        assert "Verified against schema validation" in reply_md
        assert "@marcus" in reply_md


# ---------------------------------------------------------------------------
# 4. Background Job & Database Persistence
# ---------------------------------------------------------------------------


class TestRunThreadReplyJob:
    @pytest.mark.asyncio
    async def test_run_thread_reply_full_flow(self) -> None:
        await create_tables(TEST_DB)

        mock_github = MagicMock(spec=GitHubClient)
        mock_github.get_pull_request_comment = AsyncMock(
            return_value={
                "id": 1001,
                "body": "<!-- prism-review -->\nSQL injection hazard",
                "path": "src/query.py",
                "line": 42,
                "diff_hunk": "@@ -40,3 +40,5 @@",
            }
        )
        mock_github.fetch_repo_file = AsyncMock(return_value="rules: []")
        mock_github.reply_to_pull_request_comment = AsyncMock(return_value={"id": 2002})

        resolution_result = ThreadResolutionResult(
            decision=ThreadResolutionDecision.RESOLVED,
            is_false_positive=True,
            explanation="Conceded: sanitized in QueryBuilder wrapper.",
            suggested_patch=None,
        )
        fake_llm = FakeLLMClient(responses=[resolution_result.model_dump()])

        settings = Settings(
            github_app_id="123",
            github_private_key="FAKE",
            github_webhook_secret="test-sec",
            database_url=TEST_DB,
            redis_url="redis://localhost:6379/0",
        )

        job_payload = {
            "installation_id": 999,
            "repo_full_name": "acme/backend",
            "pr_number": 42,
            "comment_id": 1002,
            "in_reply_to_id": 1001,
            "body": "@prism-review false positive, wrapped by QueryBuilder",
            "author": "sarah-dev",
            "path": "src/query.py",
            "diff_hunk": "@@ -40,3 +40,5 @@",
            "line": 42,
        }

        ctx = {
            "settings": settings,
            "github_client": mock_github,
            "llm_client": fake_llm,
        }

        outcome = await run_thread_reply(ctx, job_payload)

        assert outcome["status"] == "completed"
        assert outcome["decision"] == "resolved"
        assert outcome["is_false_positive"] is True

        # Verify GitHub reply was called
        mock_github.reply_to_pull_request_comment.assert_called_once()
        call_kwargs = mock_github.reply_to_pull_request_comment.call_args.kwargs
        assert call_kwargs["repo_full_name"] == "acme/backend"
        assert call_kwargs["pr_number"] == 42
        assert call_kwargs["comment_id"] == 1001
        assert "Conceded: sanitized in QueryBuilder wrapper." in call_kwargs["body"]

        # Verify feedback was stored in database
        records = await get_feedback_records(TEST_DB, repo="acme/backend", pr=42)
        assert len(records) == 1
        assert records[0].feedback_type == "false_positive"
        assert records[0].feedback_value == "conceded"
        assert records[0].author == "sarah-dev"


# ---------------------------------------------------------------------------
# 5. Webhook Ingestion Integration Tests
# ---------------------------------------------------------------------------


class TestWebhookReviewCommentReplies:
    def test_webhook_triggers_thread_reply_on_prism_mention(self) -> None:
        secret = "test-webhook-sec"
        settings = Settings(
            github_app_id="123",
            github_private_key="FAKE",
            github_webhook_secret=secret,
            database_url=TEST_DB,
            redis_url="redis://localhost:6379/0",
        )

        import prism.api

        prism.api._settings = settings

        client = TestClient(app)

        comment_payload = {
            "action": "created",
            "repository": {"full_name": "acme/backend"},
            "pull_request": {"number": 19},
            "comment": {
                "id": 888777,
                "body": "@prism-review can you explain why this needs a mutex?",
                "path": "src/sync.py",
                "line": 15,
                "diff_hunk": "@@ -10,6 +10,10 @@",
                "in_reply_to_id": 888100,
                "in_reply_to": {
                    "body": "<!-- prism-review -->\nRace condition detected",
                },
                "user": {"login": "chen", "type": "User"},
            },
            "sender": {"login": "chen", "type": "User"},
            "installation": {"id": 555},
        }

        raw = json.dumps(comment_payload).encode("utf-8")
        sig = make_sig(raw, secret)

        with patch("prism.queue.jobs.run_thread_reply", new_callable=AsyncMock) as mock_job:
            resp = client.post(
                "/webhook/github",
                content=raw,
                headers={
                    "X-GitHub-Event": "pull_request_review_comment",
                    "X-Hub-Signature-256": sig,
                    "X-GitHub-Delivery": "deliv-thread-reply-1",
                },
            )

            assert resp.status_code == 202
            body = resp.json()
            assert body["status"] == "accepted"
            assert body["event"] == "thread_reply"
            assert body["repo"] == "acme/backend"
            assert body["pr"] == 19
            assert body["comment_id"] == "888777"
            mock_job.assert_called_once()

    def test_webhook_ignores_unaddressed_comment(self) -> None:
        secret = "test-webhook-sec"
        settings = Settings(
            github_app_id="123",
            github_private_key="FAKE",
            github_webhook_secret=secret,
            database_url=TEST_DB,
            redis_url="redis://localhost:6379/0",
        )

        import prism.api

        prism.api._settings = settings

        client = TestClient(app)

        comment_payload = {
            "action": "created",
            "repository": {"full_name": "acme/backend"},
            "pull_request": {"number": 19},
            "comment": {
                "id": 888778,
                "body": "LGTM, ready to merge!",
                "path": "src/sync.py",
                "line": 15,
                "user": {"login": "chen", "type": "User"},
            },
            "sender": {"login": "chen", "type": "User"},
            "installation": {"id": 555},
        }

        raw = json.dumps(comment_payload).encode("utf-8")
        sig = make_sig(raw, secret)

        resp = client.post(
            "/webhook/github",
            content=raw,
            headers={
                "X-GitHub-Event": "pull_request_review_comment",
                "X-Hub-Signature-256": sig,
                "X-GitHub-Delivery": "deliv-unrelated-1",
            },
        )

        assert resp.status_code == 200
        assert resp.json()["status"] == "ignored"
