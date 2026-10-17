"""Unit tests for configuration validation and fail-fast behavior."""

from pathlib import Path

import pytest
from pydantic import ValidationError

from prism.config import Settings


def test_missing_github_webhook_secret_raises_readable_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Settings must raise a clear error when GITHUB_WEBHOOK_SECRET is missing."""
    monkeypatch.setenv("GITHUB_APP_ID", "123456")
    monkeypatch.setenv("GITHUB_PRIVATE_KEY", "fake-pem-key")
    monkeypatch.setenv("DATABASE_URL", "postgresql://user:pass@localhost:5432/db")
    monkeypatch.setenv("REDIS_URL", "redis://localhost:6379/0")
    monkeypatch.delenv("GITHUB_WEBHOOK_SECRET", raising=False)
    with pytest.raises(ValidationError) as exc_info:
        Settings()

    error_str = str(exc_info.value)
    assert "github_webhook_secret" in error_str
    assert "is required and must not be empty or blank" in error_str


def test_blank_github_webhook_secret_raises_readable_error() -> None:
    """Whitespace-only secret must fail validation."""
    with pytest.raises(ValidationError) as exc_info:
        Settings(
            github_app_id="123456",
            github_private_key="fake-pem-key",
            github_webhook_secret="   ",
            database_url="postgresql://user:pass@localhost:5432/db",
            redis_url="redis://localhost:6379/0",
        )

    error_str = str(exc_info.value)
    assert "github_webhook_secret is required and must not be empty or blank" in error_str


def test_valid_settings_defaults() -> None:
    """Settings instantiates with default thresholds when required fields are supplied."""
    settings = Settings(
        github_app_id="prism-app",
        github_private_key=(
            "-----BEGIN RSA PRIVATE KEY-----\nMIIE...\n-----END RSA PRIVATE KEY-----"
        ),
        github_webhook_secret="webhook-secret-123",
        database_url="postgresql://prism:prism@localhost:5432/prism",
        redis_url="redis://localhost:6379/0",
    )

    assert settings.github_app_id == "prism-app"
    assert settings.max_comments_per_pr == 10
    assert settings.confidence_threshold == 0.6
    assert settings.llm_provider == "anthropic"
    assert settings.llm_model == "claude-3-5-sonnet-20241022"
    assert settings.max_diff_tokens == 8000
    assert settings.get_private_key_pem().startswith("-----BEGIN RSA")


def test_private_key_file_resolution(tmp_path: Path) -> None:
    """get_private_key_pem resolves file content when path points to an existing file."""
    pem_file = tmp_path / "app.private-key.pem"
    pem_file.write_text("PEM_FILE_CONTENT", encoding="utf-8")

    settings = Settings(
        github_app_id="prism-app",
        github_private_key=str(pem_file),
        github_webhook_secret="webhook-secret-123",
        database_url="postgresql://prism:prism@localhost:5432/prism",
        redis_url="redis://localhost:6379/0",
    )

    assert settings.get_private_key_pem() == "PEM_FILE_CONTENT"
