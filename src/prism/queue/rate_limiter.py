"""Redis-based token bucket rate limiter for per-installation concurrency control."""

import logging
import time
from typing import Any

logger = logging.getLogger(__name__)

# Lua script implementing atomic token bucket with refill
# Keys: [bucket_key]
# Args: [max_tokens, refill_rate_per_sec, now, tokens_requested]
_TOKEN_BUCKET_LUA = """
local key = KEYS[1]
local max_tokens = tonumber(ARGV[1])
local refill_rate = tonumber(ARGV[2])
local now = tonumber(ARGV[3])
local requested = tonumber(ARGV[4])

local bucket = redis.call('HMGET', key, 'tokens', 'last_refill')
local tokens = tonumber(bucket[1])
local last_refill = tonumber(bucket[2])

if tokens == nil then
    tokens = max_tokens
    last_refill = now
end

-- Refill tokens based on elapsed time
local elapsed = math.max(0, now - last_refill)
local refill = elapsed * refill_rate
tokens = math.min(max_tokens, tokens + refill)

-- Check if enough tokens
if tokens >= requested then
    tokens = tokens - requested
    redis.call('HMSET', key, 'tokens', tokens, 'last_refill', now)
    redis.call('EXPIRE', key, 3600)
    return 1
else
    redis.call('HMSET', key, 'tokens', tokens, 'last_refill', now)
    redis.call('EXPIRE', key, 3600)
    return 0
end
"""


class RateLimiter:
    """Per-installation token bucket rate limiter backed by Redis.

    Prevents any single installation from starving others by bounding
    concurrent job throughput per installation.
    """

    def __init__(
        self,
        redis_client: Any,
        max_tokens: int = 5,
        refill_rate: float = 1.0,
        key_prefix: str = "prism:rate:",
    ) -> None:
        self.redis = redis_client
        self.max_tokens = max_tokens
        self.refill_rate = refill_rate
        self.key_prefix = key_prefix
        self._script: Any = None

    async def _get_script(self) -> Any:
        """Lazily register the Lua script with Redis."""
        if self._script is None:
            self._script = self.redis.register_script(_TOKEN_BUCKET_LUA)
        return self._script

    async def acquire(self, installation_id: int, tokens: int = 1) -> bool:
        """Attempt to acquire tokens for an installation.

        Returns True if the request was permitted, False if rate-limited.
        """
        key = f"{self.key_prefix}{installation_id}"
        now = time.time()

        try:
            script = await self._get_script()
            result = await script(
                keys=[key],
                args=[self.max_tokens, self.refill_rate, now, tokens],
            )
            allowed = int(result) == 1
            if not allowed:
                logger.info(
                    "Rate limited installation %d (bucket: %s)",
                    installation_id,
                    key,
                )
            return allowed
        except Exception as exc:
            # If Redis is down, fail open (allow the request) to avoid blocking
            logger.warning(
                "Rate limiter Redis error for installation %d: %s (failing open)",
                installation_id,
                exc,
            )
            return True

    async def wait_for_token(
        self,
        installation_id: int,
        poll_interval: float = 1.0,
        max_wait: float = 60.0,
    ) -> bool:
        """Block until a token becomes available or max_wait is exceeded.

        Returns True if a token was acquired, False if timed out.
        """
        import asyncio

        elapsed = 0.0
        while elapsed < max_wait:
            if await self.acquire(installation_id):
                return True
            await asyncio.sleep(poll_interval)
            elapsed += poll_interval

        logger.warning(
            "Rate limiter timed out for installation %d after %.1fs",
            installation_id,
            max_wait,
        )
        return False
