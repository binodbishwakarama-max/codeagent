"""Tests for GitHub webhook signature verification and pull_request event filtering."""

import hashlib
import hmac
from typing import Any

import pytest

from prism.github.webhook import (
    SUPPORTED_ACTIONS,
    check_and_set_idempotency,
    should_process_pull_request,
    verify_signature,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def make_signature(body: bytes, secret: str) -> str:
    """Generate a valid sha256 signature for a payload."""
    digest = hmac.new(secret.encode("utf-8"), body, hashlib.sha256).hexdigest()
    return f"sha256={digest}"


def make_pr_payload(
    action: str = "opened",
    draft: bool = False,
    user_type: str = "User",
    user_login: str = "developer",
    is_fork: bool = False,
    association: str = "CONTRIBUTOR",
) -> dict[str, Any]:
    """Build a minimal pull_request payload."""
    return {
        "action": action,
        "pull_request": {
            "number": 42,
            "draft": draft,
            "user": {"type": user_type, "login": user_login},
            "head": {
                "sha": "abc123",
                "repo": {"fork": is_fork, "full_name": "owner/fork-repo"},
            },
            "author_association": association,
        },
        "repository": {"full_name": "owner/repo"},
        "installation": {"id": 99},
    }


# ---------------------------------------------------------------------------
# verify_signature
# ---------------------------------------------------------------------------


class TestVerifySignature:
    def test_valid_signature_returns_true(self) -> None:
        secret = "my-webhook-secret"
        body = b'{"action": "opened"}'
        sig = make_signature(body, secret)
        assert verify_signature(body, secret, sig) is True

    def test_wrong_signature_returns_false(self) -> None:
        body = b'{"action": "opened"}'
        assert verify_signature(body, "correct-secret", "sha256=badhash") is False

    def test_missing_signature_returns_false(self) -> None:
        assert verify_signature(b"payload", "secret", None) is False

    def test_empty_signature_returns_false(self) -> None:
        assert verify_signature(b"payload", "secret", "") is False

    def test_signature_without_prefix_returns_false(self) -> None:
        """Signature string missing 'sha256=' prefix must be rejected."""
        body = b'{"action": "opened"}'
        raw_hash = hmac.new(b"secret", body, hashlib.sha256).hexdigest()
        assert verify_signature(body, "secret", raw_hash) is False

    def test_wrong_secret_returns_false(self) -> None:
        body = b'{"action": "opened"}'
        real_sig = make_signature(body, "correct-secret")
        assert verify_signature(body, "wrong-secret", real_sig) is False

    def test_payload_mutation_invalidates_signature(self) -> None:
        original = b'{"action": "opened"}'
        sig = make_signature(original, "secret")
        tampered = b'{"action": "closed"}'
        assert verify_signature(tampered, "secret", sig) is False


# ---------------------------------------------------------------------------
# should_process_pull_request
# ---------------------------------------------------------------------------


class TestShouldProcessPullRequest:
    def test_opened_action_is_accepted(self) -> None:
        payload = make_pr_payload(action="opened")
        ok, reason = should_process_pull_request(payload)
        assert ok is True

    @pytest.mark.parametrize("action", sorted(SUPPORTED_ACTIONS))
    def test_all_supported_actions_are_accepted(self, action: str) -> None:
        payload = make_pr_payload(action=action)
        ok, _ = should_process_pull_request(payload)
        assert ok is True

    def test_closed_action_is_rejected(self) -> None:
        payload = make_pr_payload(action="closed")
        ok, reason = should_process_pull_request(payload)
        assert ok is False
        assert "closed" in reason

    def test_draft_is_rejected(self) -> None:
        payload = make_pr_payload(draft=True)
        ok, reason = should_process_pull_request(payload)
        assert ok is False
        assert "draft" in reason

    def test_bot_type_is_rejected(self) -> None:
        payload = make_pr_payload(user_type="Bot", user_login="dependabot[bot]")
        ok, reason = should_process_pull_request(payload)
        assert ok is False
        assert "bot" in reason.lower()

    def test_bot_login_suffix_is_rejected(self) -> None:
        """Users with login ending in [bot] are treated as bots."""
        payload = make_pr_payload(user_type="User", user_login="renovate[bot]")
        ok, reason = should_process_pull_request(payload)
        assert ok is False

    def test_fork_pr_from_none_association_is_rejected(self) -> None:
        payload = make_pr_payload(is_fork=True, association="NONE")
        ok, reason = should_process_pull_request(payload)
        assert ok is False
        assert "fork" in reason.lower() or "untrusted" in reason.lower()

    def test_fork_pr_from_first_timer_is_rejected(self) -> None:
        payload = make_pr_payload(is_fork=True, association="FIRST_TIMER")
        ok, _ = should_process_pull_request(payload)
        assert ok is False

    def test_fork_pr_from_contributor_is_accepted(self) -> None:
        """Forks from existing contributors are allowed."""
        payload = make_pr_payload(is_fork=True, association="CONTRIBUTOR")
        ok, _ = should_process_pull_request(payload)
        assert ok is True

    def test_null_fork_repo_is_rejected(self) -> None:
        """If head.repo is None (inaccessible fork), reject the PR."""
        payload = make_pr_payload()
        payload["pull_request"]["head"]["repo"] = None
        ok, reason = should_process_pull_request(payload)
        assert ok is False
        assert "inaccessible" in reason.lower() or "fork" in reason.lower()

    def test_missing_pull_request_object_is_rejected(self) -> None:
        payload: dict[str, Any] = {"action": "opened"}
        ok, reason = should_process_pull_request(payload)
        assert ok is False
        assert "missing" in reason.lower()


# ---------------------------------------------------------------------------
# check_and_set_idempotency (no Redis)
# ---------------------------------------------------------------------------


class TestCheckIdempotencyWithoutRedis:
    @pytest.mark.asyncio
    async def test_without_redis_always_returns_true(self) -> None:
        """Without Redis, all events are considered new (graceful degradation)."""
        result = await check_and_set_idempotency(
            redis_client=None,
            delivery_id="abc-123",
            repo_full_name="owner/repo",
            pr_number=1,
            head_sha="sha111",
        )
        assert result is True
