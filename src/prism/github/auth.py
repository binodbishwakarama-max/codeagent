"""GitHub App authentication handling RS256 JWT tokens and Redis-cached installation tokens."""

import logging
import time
from datetime import UTC, datetime
from typing import Any

import httpx
import jwt

from prism.config import Settings

logger = logging.getLogger(__name__)


def generate_app_jwt(settings: Settings) -> str:
    """Generate a signed RS256 JWT for GitHub App authentication.

    The token is valid for 10 minutes with a 60-second backdate to tolerate clock skew.
    """
    now = int(time.time())
    payload = {
        "iat": now - 60,
        "exp": now + (10 * 60),
        "iss": settings.github_app_id,
    }
    private_key_pem = settings.get_private_key_pem()
    token = jwt.encode(payload, private_key_pem, algorithm="RS256")
    return token


async def get_installation_access_token(
    installation_id: int,
    settings: Settings,
    redis_client: Any | None = None,
    http_client: httpx.AsyncClient | None = None,
) -> str:
    """Obtain an installation access token, caching in Redis until 5 minutes before expiry.

    Parameters:
        installation_id: Target GitHub App installation ID.
        settings: Application settings with credentials and configuration.
        redis_client: Optional async Redis client for caching.
        http_client: Optional async HTTP client for API requests.
    """
    cache_key = f"github:installation_token:{installation_id}"

    # 1. Attempt cache lookup in Redis
    if redis_client is not None:
        try:
            cached_token = await redis_client.get(cache_key)
            if cached_token:
                logger.debug("Using cached installation token for %d", installation_id)
                if isinstance(cached_token, bytes):
                    return cached_token.decode("utf-8")
                return str(cached_token)
        except Exception as exc:
            logger.warning("Redis cache error reading token for %d: %s", installation_id, exc)

    # 2. Exchange JWT for new installation access token
    app_jwt = generate_app_jwt(settings)
    url = f"https://api.github.com/app/installations/{installation_id}/access_tokens"
    headers = {
        "Accept": "application/vnd.github+json",
        "Authorization": f"Bearer {app_jwt}",
        "X-GitHub-Api-Version": "2022-11-28",
    }

    should_close_client = False
    if http_client is None:
        http_client = httpx.AsyncClient(timeout=15.0)
        should_close_client = True

    try:
        response = await http_client.post(url, headers=headers)
        response.raise_for_status()
        data = response.json()
    finally:
        if should_close_client:
            await http_client.aclose()

    token = str(data["token"])
    expires_at_str = str(data["expires_at"])

    # 3. Calculate TTL (until 5 minutes before expiry)
    try:
        # e.g., '2026-10-07T12:00:00Z'
        expires_at = datetime.fromisoformat(expires_at_str.replace("Z", "+00:00"))
        remaining_seconds = int((expires_at - datetime.now(UTC)).total_seconds())
        # Subtract 300s (5 minutes buffer)
        ttl_seconds = remaining_seconds - 300
    except Exception as exc:
        logger.warning("Failed parsing token expiration %s: %s", expires_at_str, exc)
        ttl_seconds = 3000  # Default ~50 minutes safety fallback

    # 4. Cache in Redis
    if redis_client is not None and ttl_seconds > 0:
        try:
            await redis_client.set(cache_key, token, ex=ttl_seconds)
            logger.debug(
                "Cached installation token for %d with TTL %d seconds",
                installation_id,
                ttl_seconds,
            )
        except Exception as exc:
            logger.warning("Failed to cache installation token in Redis: %s", exc)

    return token
