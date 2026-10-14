"""GitHub webhook signature verification, event filtering, and job enqueuing."""

import hashlib
import hmac
import logging
from typing import Any

logger = logging.getLogger(__name__)

SUPPORTED_ACTIONS = {"opened", "synchronize", "reopened", "ready_for_review"}
DEDUPE_TTL_SECONDS = 86400  # 24 hours


def verify_signature(payload_bytes: bytes, secret: str, signature_header: str | None) -> bool:
    """Verify GitHub X-Hub-Signature-256 header using constant-time comparison."""
    if not signature_header or not signature_header.startswith("sha256="):
        return False

    expected_hash = hmac.new(
        secret.encode("utf-8"),
        payload_bytes,
        hashlib.sha256,
    ).hexdigest()
    expected_header = f"sha256={expected_hash}"

    return hmac.compare_digest(expected_header, signature_header)


async def check_and_set_idempotency(
    redis_client: Any | None,
    delivery_id: str | None,
    repo_full_name: str,
    pr_number: int,
    head_sha: str,
) -> bool:
    """Check and record delivery_id and (repo, pr, head_sha) deduplication keys in Redis.

    Returns True if this event is new and should be processed; False if it was already seen.
    """
    if redis_client is None:
        return True

    # 1. Deduplicate by delivery_id
    if delivery_id:
        delivery_key = f"github:delivery:{delivery_id}"
        delivery_set = await redis_client.set(delivery_key, "1", nx=True, ex=DEDUPE_TTL_SECONDS)
        if not delivery_set:
            logger.info("Ignoring duplicate delivery: %s", delivery_id)
            return False

    # 2. Deduplicate by commit head for this PR
    pr_head_key = f"github:pr_head:{repo_full_name}:{pr_number}:{head_sha}"
    pr_head_set = await redis_client.set(pr_head_key, "1", nx=True, ex=DEDUPE_TTL_SECONDS)
    if not pr_head_set:
        logger.info(
            "Ignoring duplicate review trigger for %s#%d at %s",
            repo_full_name,
            pr_number,
            head_sha,
        )
        return False

    return True


def should_process_pull_request(payload: dict[str, Any]) -> tuple[bool, str]:
    """Evaluate whether pull_request webhook payload meets review criteria.

    Filters out drafts, bot-authored PRs, untrusted forks, and unsupported actions.
    """
    action = payload.get("action")
    if action not in SUPPORTED_ACTIONS:
        return False, f"unsupported action '{action}'"

    pr = payload.get("pull_request")
    if not pr or not isinstance(pr, dict):
        return False, "missing pull_request object"

    # 1. Skip drafts
    if pr.get("draft") is True:
        return False, "pull request is in draft status"

    # 2. Skip bot authors
    author = pr.get("user", {})
    if author.get("type") == "Bot" or author.get("login", "").endswith("[bot]"):
        return False, f"bot-authored pull request by {author.get('login')}"

    # 3. Skip inaccessible forks or untrusted fork external submissions
    head = pr.get("head", {})
    head_repo = head.get("repo")
    if head_repo is None:
        return False, "inaccessible or deleted fork"

    if head_repo.get("fork") is True:
        # Check author association: external contributors without repository access
        association = pr.get("author_association", "NONE")
        if association in {"NONE", "FIRST_TIMER", "FIRST_TIME_CONTRIBUTOR"}:
            return False, f"fork PR from untrusted contributor ({association})"

    return True, "valid"
