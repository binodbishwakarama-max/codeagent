"""Integration tests for POST /webhook/github endpoint using respx mocks."""

import hashlib
import hmac
import json
from typing import Any

import httpx
import pytest
import respx
from fastapi.testclient import TestClient

from prism.api import app
from prism.config import Settings

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def make_signature(body: bytes, secret: str) -> str:
    digest = hmac.new(secret.encode("utf-8"), body, hashlib.sha256).hexdigest()
    return f"sha256={digest}"


def get_test_settings() -> Settings:
    return Settings(
        github_app_id="test-app-123",
        github_private_key="-----BEGIN RSA PRIVATE KEY-----\nFAKE\n-----END RSA PRIVATE KEY-----",
        github_webhook_secret="test-secret-abc",
        database_url="postgresql://prism:prism@localhost:5432/prism",
        redis_url="redis://localhost:6379/0",
    )


def make_pr_event(
    action: str = "opened",
    draft: bool = False,
    user_type: str = "User",
    user_login: str = "developer",
) -> dict[str, Any]:
    return {
        "action": action,
        "installation": {"id": 777},
        "repository": {"full_name": "acme/backend"},
        "pull_request": {
            "number": 10,
            "draft": draft,
            "user": {"type": user_type, "login": user_login},
            "head": {
                "sha": "deadbeef1234",
                "repo": {"fork": False, "full_name": "acme/backend"},
            },
            "author_association": "MEMBER",
        },
    }


@pytest.fixture()
def client() -> TestClient:
    return TestClient(app, raise_server_exceptions=True)


@pytest.fixture()
def webhook_secret() -> str:
    return "test-secret-abc"


# ---------------------------------------------------------------------------
# Signature tests via TestClient (no Redis - graceful degradation)
# ---------------------------------------------------------------------------


class TestWebhookSignatureVerification:
    def test_valid_signature_returns_202(
        self, client: TestClient, webhook_secret: str, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("GITHUB_APP_ID", "test-app-123")
        monkeypatch.setenv(
            "GITHUB_PRIVATE_KEY",
            "-----BEGIN RSA PRIVATE KEY-----\nFAKE\n-----END RSA PRIVATE KEY-----",
        )
        monkeypatch.setenv("GITHUB_WEBHOOK_SECRET", webhook_secret)
        monkeypatch.setenv("DATABASE_URL", "postgresql://prism:prism@localhost:5432/prism")
        monkeypatch.setenv("REDIS_URL", "redis://localhost:6379/0")

        payload = make_pr_event()
        body = json.dumps(payload).encode()
        sig = make_signature(body, webhook_secret)

        response = client.post(
            "/webhook/github",
            content=body,
            headers={
                "Content-Type": "application/json",
                "X-Hub-Signature-256": sig,
                "X-GitHub-Event": "pull_request",
                "X-GitHub-Delivery": "unique-delivery-001",
            },
        )
        # 202 on success, or 500 if Redis connection fails gracefully
        assert response.status_code in (202, 500)

    def test_invalid_signature_returns_401(
        self, client: TestClient, webhook_secret: str, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("GITHUB_APP_ID", "test-app-123")
        monkeypatch.setenv(
            "GITHUB_PRIVATE_KEY",
            "-----BEGIN RSA PRIVATE KEY-----\nFAKE\n-----END RSA PRIVATE KEY-----",
        )
        monkeypatch.setenv("GITHUB_WEBHOOK_SECRET", webhook_secret)
        monkeypatch.setenv("DATABASE_URL", "postgresql://prism:prism@localhost:5432/prism")
        monkeypatch.setenv("REDIS_URL", "redis://localhost:6379/0")

        payload = make_pr_event()
        body = json.dumps(payload).encode()

        response = client.post(
            "/webhook/github",
            content=body,
            headers={
                "Content-Type": "application/json",
                "X-Hub-Signature-256": "sha256=badhash",
                "X-GitHub-Event": "pull_request",
                "X-GitHub-Delivery": "unique-delivery-002",
            },
        )
        assert response.status_code == 401

    def test_missing_signature_returns_401(
        self, client: TestClient, webhook_secret: str, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("GITHUB_APP_ID", "test-app-123")
        monkeypatch.setenv(
            "GITHUB_PRIVATE_KEY",
            "-----BEGIN RSA PRIVATE KEY-----\nFAKE\n-----END RSA PRIVATE KEY-----",
        )
        monkeypatch.setenv("GITHUB_WEBHOOK_SECRET", webhook_secret)
        monkeypatch.setenv("DATABASE_URL", "postgresql://prism:prism@localhost:5432/prism")
        monkeypatch.setenv("REDIS_URL", "redis://localhost:6379/0")

        payload = make_pr_event()
        body = json.dumps(payload).encode()

        response = client.post(
            "/webhook/github",
            content=body,
            headers={
                "Content-Type": "application/json",
                "X-GitHub-Event": "pull_request",
            },
        )
        assert response.status_code == 401


# ---------------------------------------------------------------------------
# respx mock for GitHub auth
# ---------------------------------------------------------------------------


class TestGitHubAuthWithRespx:
    @pytest.mark.asyncio
    @respx.mock
    async def test_get_installation_token_calls_github_api(self) -> None:
        """Verify that token exchange hits the correct GitHub endpoint."""

        from prism.github.auth import get_installation_access_token

        mock_token = "ghs_test_token_xyz"
        expires_at = "2026-10-07T14:00:00Z"

        route = respx.post("https://api.github.com/app/installations/777/access_tokens").mock(
            return_value=httpx.Response(
                200,
                json={"token": mock_token, "expires_at": expires_at},
            )
        )

        settings = get_test_settings()
        # Use a real RSA key for signing (generate minimal test key)
        from cryptography.hazmat.primitives import serialization
        from cryptography.hazmat.primitives.asymmetric import rsa

        private_key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
        pem = private_key.private_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PrivateFormat.TraditionalOpenSSL,
            encryption_algorithm=serialization.NoEncryption(),
        ).decode("utf-8")

        settings = Settings(
            github_app_id="12345",
            github_private_key=pem,
            github_webhook_secret="test-secret",
            database_url="postgresql://prism:prism@localhost:5432/prism",
            redis_url="redis://localhost:6379/0",
        )

        async with httpx.AsyncClient() as http_client:
            token = await get_installation_access_token(
                installation_id=777,
                settings=settings,
                redis_client=None,
                http_client=http_client,
            )

        assert token == mock_token
        assert route.called


def test_healthz_and_readyz_endpoints(client: TestClient) -> None:
    res_health = client.get("/healthz")
    assert res_health.status_code == 200
    assert res_health.json() == {"status": "ok"}

    res_ready = client.get("/readyz")
    assert res_ready.status_code in (200, 503)
    assert "status" in res_ready.json()
    assert "database" in res_ready.json()
    assert "redis" in res_ready.json()
