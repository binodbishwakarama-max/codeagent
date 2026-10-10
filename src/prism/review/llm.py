"""Provider-agnostic LLM client interfaces, implementations, and retry handlers."""

import asyncio
import json
import logging
import random
import time
from dataclasses import dataclass
from typing import Any, Protocol, TypeVar

import httpx
from pydantic import BaseModel

from prism.config import Settings, calculate_token_cost
from prism.observability.tracing import trace_llm_call

logger = logging.getLogger(__name__)

T = TypeVar("T", bound=BaseModel)


@dataclass(frozen=True)
class LLMUsage:
    """Usage telemetry for an LLM completion."""

    input_tokens: int
    output_tokens: int
    cost_usd: float
    latency_ms: float
    model: str


class LLMClient(Protocol):
    """Protocol defining the standard structured LLM completion contract."""

    async def complete_structured(
        self,
        system: str,
        user: str,
        schema: type[T],
        timeout: float | None = None,
    ) -> tuple[T, LLMUsage]:
        """Request structured completion matching the provided Pydantic schema."""
        ...


# ---------------------------------------------------------------------------
# Retry helper with exponential backoff and jitter
# ---------------------------------------------------------------------------


async def _execute_with_retry(
    request_fn: Any,
    max_retries: int = 3,
    initial_delay: float = 1.0,
    max_delay: float = 10.0,
) -> httpx.Response:
    """Execute async HTTP request with exponential backoff and jitter for 429/5xx."""
    last_exc: Exception | None = None

    for attempt in range(max_retries):
        try:
            response: httpx.Response = await request_fn()
            # Retry on rate limits or server errors
            if response.status_code == 429 or response.status_code >= 500:
                delay = min(initial_delay * (2**attempt) + random.uniform(0.0, 0.5), max_delay)
                logger.warning(
                    "LLM HTTP %d on attempt %d/%d; retrying in %.2fs",
                    response.status_code,
                    attempt + 1,
                    max_retries,
                    delay,
                )
                await asyncio.sleep(delay)
                continue
            return response
        except (httpx.TimeoutException, httpx.NetworkError) as exc:
            last_exc = exc
            delay = min(initial_delay * (2**attempt) + random.uniform(0.0, 0.5), max_delay)
            logger.warning(
                "LLM network exception '%s' on attempt %d/%d; retrying in %.2fs",
                exc,
                attempt + 1,
                max_retries,
                delay,
            )
            await asyncio.sleep(delay)

    if last_exc is not None:
        raise last_exc
    raise httpx.HTTPStatusError("Exceeded max retries", request=None, response=response)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Anthropic Client
# ---------------------------------------------------------------------------


class AnthropicClient:
    """Client for Anthropic Claude models using tool use for guaranteed structured output."""

    def __init__(
        self,
        api_key: str | None = None,
        model: str = "claude-3-5-sonnet-20241022",
        timeout: float = 30.0,
    ) -> None:
        self.api_key = api_key or ""
        self.model = model
        self.default_timeout = timeout
        self.base_url = "https://api.anthropic.com/v1/messages"

    async def complete_structured(
        self,
        system: str,
        user: str,
        schema: type[T],
        timeout: float | None = None,
    ) -> tuple[T, LLMUsage]:
        """Invoke Anthropic Messages API with tool definition matching schema."""
        deadline = timeout or self.default_timeout
        start_time = time.time()

        tool_name = schema.__name__
        tool_definition = {
            "name": tool_name,
            "description": f"Output structured data matching {tool_name}",
            "input_schema": schema.model_json_schema(),
        }

        payload: dict[str, Any] = {
            "model": self.model,
            "max_tokens": 4096,
            "system": system,
            "messages": [{"role": "user", "content": user}],
            "tools": [tool_definition],
            "tool_choice": {"type": "tool", "name": tool_name},
        }

        headers = {
            "x-api-key": self.api_key,
            "anthropic-version": "2023-06-01",
            "content-type": "application/json",
        }

        async with trace_llm_call(f"anthropic:{self.model}", model=self.model) as span:
            async with httpx.AsyncClient(timeout=deadline) as client:
                response = await _execute_with_retry(
                    lambda: client.post(self.base_url, headers=headers, json=payload)
                )
                response.raise_for_status()
                data = response.json()

            usage_info = data.get("usage", {})
            in_tokens = usage_info.get("input_tokens", 0)
            out_tokens = usage_info.get("output_tokens", 0)
            cost = calculate_token_cost(self.model, in_tokens, out_tokens)
            latency_ms = round((time.time() - start_time) * 1000.0, 2)

            span.record_usage(input_tokens=in_tokens, output_tokens=out_tokens, cost_usd=cost)

            # Find tool_use block
            tool_input: dict[str, Any] | None = None
            for block in data.get("content", []):
                if block.get("type") == "tool_use" and block.get("name") == tool_name:
                    tool_input = block.get("input")
                    break

            if tool_input is None:
                raise ValueError(f"Anthropic response missing tool_use block for '{tool_name}'")

            parsed = schema.model_validate(tool_input)
            usage = LLMUsage(
                input_tokens=in_tokens,
                output_tokens=out_tokens,
                cost_usd=cost,
                latency_ms=latency_ms,
                model=self.model,
            )
            return parsed, usage


# ---------------------------------------------------------------------------
# OpenAI-Compatible Client (vLLM / self-hosted / fine-tuned)
# ---------------------------------------------------------------------------


class OpenAICompatClient:
    """Client for OpenAI-compatible endpoints (vLLM, Ollama, self-hosted fine-tuned models)."""

    def __init__(
        self,
        base_url: str | None = None,
        api_key: str | None = None,
        model: str = "prism-ft",
        timeout: float = 30.0,
    ) -> None:
        self.base_url = (base_url or "http://localhost:8000/v1").rstrip("/")
        self.api_key = api_key or "none"
        self.model = model
        self.default_timeout = timeout

    async def complete_structured(
        self,
        system: str,
        user: str,
        schema: type[T],
        timeout: float | None = None,
    ) -> tuple[T, LLMUsage]:
        """Invoke OpenAI-compatible chat completion with JSON/tools structured output."""
        deadline = timeout or self.default_timeout
        start_time = time.time()
        url = f"{self.base_url}/chat/completions"

        tool_name = schema.__name__
        tool_definition = {
            "type": "function",
            "function": {
                "name": tool_name,
                "description": f"Output structured data matching {tool_name}",
                "parameters": schema.model_json_schema(),
            },
        }

        payload: dict[str, Any] = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "tools": [tool_definition],
            "tool_choice": {"type": "function", "function": {"name": tool_name}},
        }

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        async with trace_llm_call(f"openai_compat:{self.model}", model=self.model) as span:
            async with httpx.AsyncClient(timeout=deadline) as client:
                response = await _execute_with_retry(
                    lambda: client.post(url, headers=headers, json=payload)
                )
                response.raise_for_status()
                data = response.json()

            usage_info = data.get("usage", {})
            in_tokens = usage_info.get("prompt_tokens", 0)
            out_tokens = usage_info.get("completion_tokens", 0)
            cost = calculate_token_cost(self.model, in_tokens, out_tokens)
            latency_ms = round((time.time() - start_time) * 1000.0, 2)

            span.record_usage(input_tokens=in_tokens, output_tokens=out_tokens, cost_usd=cost)

            choices = data.get("choices", [])
            if not choices:
                raise ValueError("OpenAI-compatible endpoint returned empty choices")

            choice = choices[0]
            message = choice.get("message", {})

            # 1. Attempt extracting from tool_calls
            tool_calls = message.get("tool_calls", [])
            raw_args: str | None = None
            if tool_calls:
                fn = tool_calls[0].get("function", {})
                raw_args = fn.get("arguments")

            # 2. Fallback to message content
            if not raw_args:
                raw_args = message.get("content")

            if not raw_args:
                raise ValueError("No arguments or content found in completion response")

            if isinstance(raw_args, str):
                json_data = json.loads(raw_args)
            else:
                json_data = raw_args

            parsed = schema.model_validate(json_data)
            usage = LLMUsage(
                input_tokens=in_tokens,
                output_tokens=out_tokens,
                cost_usd=cost,
                latency_ms=latency_ms,
                model=self.model,
            )
            return parsed, usage


# ---------------------------------------------------------------------------
# Fake / Mock LLM Client for Testing
# ---------------------------------------------------------------------------


class FakeLLMClient:
    """Mock LLM client for deterministic unit testing without network or token spend."""

    def __init__(
        self,
        responses: list[Any] | None = None,
        model: str = "fake-model",
    ) -> None:
        self.responses: list[Any] = responses or []
        self.call_count = 0
        self.calls: list[dict[str, Any]] = []
        self.model = model

    async def complete_structured(
        self,
        system: str,
        user: str,
        schema: type[T],
        timeout: float | None = None,
    ) -> tuple[T, LLMUsage]:
        self.calls.append({"system": system, "user": user, "schema": schema})
        self.call_count += 1

        if not self.responses:
            # Return empty schema instance by default
            return schema(), LLMUsage(
                input_tokens=100,
                output_tokens=50,
                cost_usd=0.0,
                latency_ms=10.0,
                model=self.model,
            )

        resp = self.responses.pop(0)
        if isinstance(resp, Exception):
            raise resp

        if isinstance(resp, dict):
            parsed = schema.model_validate(resp)
        elif isinstance(resp, schema):
            parsed = resp
        else:
            parsed = schema.model_validate(resp)

        return parsed, LLMUsage(
            input_tokens=150,
            output_tokens=75,
            cost_usd=0.0,
            latency_ms=15.0,
            model=self.model,
        )


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


def create_llm_client(settings: Settings) -> LLMClient:
    """Instantiate appropriate LLM client configured by application settings."""
    if settings.llm_base_url or settings.llm_provider == "openai_compat":
        return OpenAICompatClient(
            base_url=settings.llm_base_url,
            api_key=None,
            model=settings.llm_model,
            timeout=settings.llm_timeout_seconds,
        )
    return AnthropicClient(
        api_key=settings.anthropic_api_key,
        model=settings.llm_model,
        timeout=settings.llm_timeout_seconds,
    )
