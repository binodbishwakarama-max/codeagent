"""Integration tests verifying vLLM OpenAI-compatible serving with the evaluation harness."""

from __future__ import annotations

import httpx
import pytest
from fastapi.testclient import TestClient

from evals.dataset.schema import GoldFinding, PRExample
from evals.harness.config import PipelineConfig
from evals.harness.runner import run_pipeline_on_example
from prism.config import Settings
from prism.models import DiffLine, FileDiff, Hunk
from prism.review.llm import OpenAICompatClient
from prism.review.pipeline import PRContext, review_pr
from prism.review.prompts import ChunkReviewResponse
from training.serve import create_mock_app


@pytest.fixture
def vllm_test_client() -> TestClient:
    """Create in-process test client for OpenAI-compatible mock server."""
    app = create_mock_app(model_name="Qwen/Qwen2.5-Coder-7B-Instruct-Prism-Review")
    return TestClient(app)


def test_vllm_mock_endpoints(vllm_test_client: TestClient) -> None:
    """Verify vLLM mock server exposes standard OpenAI endpoints."""
    # Health check
    resp_health = vllm_test_client.get("/health")
    assert resp_health.status_code == 200
    assert resp_health.json()["status"] == "healthy"

    # List models
    resp_models = vllm_test_client.get("/v1/models")
    assert resp_models.status_code == 200
    data = resp_models.json()
    assert len(data["data"]) >= 1
    assert "Qwen" in data["data"][0]["id"]


@pytest.mark.asyncio
async def test_openai_compat_client_direct(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify OpenAICompatClient completes structured requests against mock vLLM."""
    app = create_mock_app(model_name="Qwen/Qwen2.5-Coder-7B-Instruct-Prism-Review")
    orig_client = httpx.AsyncClient

    def _mock_async_client(*args: object, **kwargs: object) -> httpx.AsyncClient:
        return orig_client(transport=httpx.ASGITransport(app=app))

    monkeypatch.setattr(httpx, "AsyncClient", _mock_async_client)

    client = OpenAICompatClient(
        base_url="http://testserver/v1",
        api_key=None,
        model="Qwen/Qwen2.5-Coder-7B-Instruct-Prism-Review",
    )
    resp, usage = await client.complete_structured(
        system="You are an expert reviewer.",
        user="Check this diff for vulnerabilities.",
        schema=ChunkReviewResponse,
    )
    assert isinstance(resp, ChunkReviewResponse)
    assert usage.input_tokens > 0
    assert usage.output_tokens > 0


@pytest.mark.asyncio
async def test_review_pr_with_vllm_openai_compat(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify review_pr executes seamlessly with OpenAICompatClient."""
    app = create_mock_app(model_name="Qwen/Qwen2.5-Coder-7B-Instruct-Prism-Review")
    orig_client = httpx.AsyncClient

    def _mock_async_client(*args: object, **kwargs: object) -> httpx.AsyncClient:
        return orig_client(transport=httpx.ASGITransport(app=app))

    monkeypatch.setattr(httpx, "AsyncClient", _mock_async_client)

    client = OpenAICompatClient(
        base_url="http://testserver/v1",
        model="Qwen/Qwen2.5-Coder-7B-Instruct-Prism-Review",
    )

    file_diff = FileDiff(
        path="src/auth.py",
        language="python",
        hunks=[
            Hunk(
                old_start=1,
                new_start=1,
                lines=[
                    DiffLine(kind="ctx", old_no=1, new_no=1, text="import os"),
                    DiffLine(kind="add", old_no=None, new_no=2, text="password = 'hardcoded'"),
                ],
            )
        ],
    )

    ctx = PRContext(
        repo_full_name="tiangolo/fastapi",
        pr_number=101,
        title="Update auth",
        file_diffs=[file_diff],
    )

    settings = Settings(
        github_app_id="app",
        github_private_key="key",
        github_webhook_secret="sec",
        database_url="sqlite+aiosqlite:///db.sqlite",
        redis_url="redis://localhost:6379/0",
        llm_provider="openai_compat",
        llm_base_url="http://testserver/v1",
        llm_model="Qwen/Qwen2.5-Coder-7B-Instruct-Prism-Review",
    )

    result = await review_pr(ctx, settings=settings, llm_client=client)
    assert result.model == "Qwen/Qwen2.5-Coder-7B-Instruct-Prism-Review"
    assert result.input_tokens > 0


@pytest.mark.asyncio
async def test_eval_harness_against_openai_compat_endpoint(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify eval harness calls OpenAICompatClient via LLM_BASE_URL with zero code changes."""
    app = create_mock_app(model_name="Qwen/Qwen2.5-Coder-7B-Instruct-Prism-Review")
    orig_client = httpx.AsyncClient

    def _mock_async_client(*args: object, **kwargs: object) -> httpx.AsyncClient:
        return orig_client(transport=httpx.ASGITransport(app=app))

    monkeypatch.setattr(httpx, "AsyncClient", _mock_async_client)
    monkeypatch.setenv("LLM_BASE_URL", "http://testserver/v1")
    monkeypatch.setenv("LLM_MODEL", "Qwen/Qwen2.5-Coder-7B-Instruct-Prism-Review")

    example = PRExample(
        id="tiangolo/fastapi#202",
        repo="tiangolo/fastapi",
        language="python",
        license="MIT",
        pr_number=202,
        pr_url="https://github.com/tiangolo/fastapi/pull/202",
        base_sha="base1234",
        head_sha="head1234",
        title="Fix path traversal",
        body="Sanitize file paths before read",
        changed_lines=2,
        diff="""diff --git a/fastapi/utils.py b/fastapi/utils.py
--- a/fastapi/utils.py
+++ b/fastapi/utils.py
@@ -10,2 +10,3 @@
 def read_file(path: str) -> str:
+    safe_path = os.path.basename(path)
     return open(safe_path).read()
""",
        split="dev",
        gold_findings=[
            GoldFinding(
                path="fastapi/utils.py",
                line=11,
                description="Potential security issue",
            )
        ],
    )

    pipeline_cfg = PipelineConfig(
        model="Qwen/Qwen2.5-Coder-7B-Instruct-Prism-Review",
        confidence_threshold=0.5,
        max_comments_per_pr=10,
    )

    # Execute harness runner with use_mock=False to test actual create_llm_client path
    findings, review_res = await run_pipeline_on_example(example, pipeline_cfg, use_mock=False)
    assert review_res is not None
    assert review_res.model == "Qwen/Qwen2.5-Coder-7B-Instruct-Prism-Review"
    assert review_res.input_tokens > 0
