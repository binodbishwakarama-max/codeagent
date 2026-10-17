"""Tests for AnthropicClient, OpenAICompatClient, and exponential backoff retries using respx."""

import httpx
import pytest
import respx
from pydantic import BaseModel

from prism.review.llm import AnthropicClient, OpenAICompatClient


class DummySchema(BaseModel):
    message: str


# ---------------------------------------------------------------------------
# AnthropicClient tests
# ---------------------------------------------------------------------------


class TestAnthropicClient:
    @pytest.mark.asyncio
    @respx.mock
    async def test_successful_structured_completion(self) -> None:
        client = AnthropicClient(api_key="test-key", model="claude-3-5-sonnet-20241022")

        respx.post("https://api.anthropic.com/v1/messages").mock(
            return_value=httpx.Response(
                200,
                json={
                    "content": [
                        {
                            "type": "tool_use",
                            "name": "DummySchema",
                            "input": {"message": "hello world"},
                        }
                    ],
                    "usage": {"input_tokens": 100, "output_tokens": 25},
                },
            )
        )

        parsed, usage = await client.complete_structured(
            system="system prompt",
            user="user prompt",
            schema=DummySchema,
        )

        assert parsed.message == "hello world"
        assert usage.input_tokens == 100
        assert usage.output_tokens == 25
        assert usage.cost_usd > 0.0

    @pytest.mark.asyncio
    @respx.mock
    async def test_retry_on_429_then_succeed(self) -> None:
        client = AnthropicClient(api_key="test-key", model="claude-3-5-sonnet-20241022")

        route = respx.post("https://api.anthropic.com/v1/messages")
        route.side_effect = [
            httpx.Response(429, json={"error": "rate limit"}),
            httpx.Response(
                200,
                json={
                    "content": [
                        {
                            "type": "tool_use",
                            "name": "DummySchema",
                            "input": {"message": "retry success"},
                        }
                    ],
                    "usage": {"input_tokens": 50, "output_tokens": 10},
                },
            ),
        ]

        parsed, usage = await client.complete_structured(
            system="system",
            user="user",
            schema=DummySchema,
        )

        assert parsed.message == "retry success"
        assert route.call_count == 2


# ---------------------------------------------------------------------------
# OpenAICompatClient tests
# ---------------------------------------------------------------------------


class TestOpenAICompatClient:
    @pytest.mark.asyncio
    @respx.mock
    async def test_successful_tool_call_completion(self) -> None:
        client = OpenAICompatClient(base_url="http://localhost:8000/v1", model="prism-ft")

        respx.post("http://localhost:8000/v1/chat/completions").mock(
            return_value=httpx.Response(
                200,
                json={
                    "choices": [
                        {
                            "message": {
                                "tool_calls": [
                                    {
                                        "function": {
                                            "name": "DummySchema",
                                            "arguments": '{"message": "openai success"}',
                                        }
                                    }
                                ]
                            }
                        }
                    ],
                    "usage": {"prompt_tokens": 80, "completion_tokens": 30},
                },
            )
        )

        parsed, usage = await client.complete_structured(
            system="system",
            user="user",
            schema=DummySchema,
        )

        assert parsed.message == "openai success"
        assert usage.input_tokens == 80
        assert usage.output_tokens == 30
