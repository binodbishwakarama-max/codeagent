"""Application configuration validated at startup via pydantic-settings."""

from pathlib import Path
from typing import Any, Literal

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Runtime configuration schema for Prism Review.

    Fails fast at startup if critical credentials or connection strings are missing.
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    # GitHub App Credentials
    github_app_id: str = Field(
        default="",
        description="GitHub App ID",
    )
    github_private_key: str = Field(
        default="",
        description="GitHub App private key PEM string or filesystem path to .pem file",
    )
    github_webhook_secret: str = Field(
        default="",
        description="Secret token used to verify incoming GitHub webhook signatures",
    )

    # Database & Cache
    database_url: str = Field(
        default="",
        description="PostgreSQL connection string with pgvector support",
    )
    redis_url: str = Field(
        default="",
        description="Redis connection string for arq job queue",
    )

    # LLM Settings
    anthropic_api_key: str | None = Field(
        default=None,
        description="API key for Anthropic Claude models",
    )
    llm_provider: Literal["anthropic", "openai_compat"] = Field(
        default="anthropic",
        description="LLM provider architecture",
    )
    llm_model: str = Field(
        default="claude-3-5-sonnet-20241022",
        description="Model name/id to use for code review",
    )
    llm_base_url: str | None = Field(
        default=None,
        description="Custom base URL for self-hosted or OpenAI-compatible endpoints",
    )
    llm_timeout_seconds: float = Field(
        default=30.0,
        gt=0.0,
        description="Timeout deadline in seconds for LLM invocations",
    )

    # Langfuse Observability
    langfuse_public_key: str | None = Field(
        default=None,
        description="Langfuse public API key",
    )
    langfuse_secret_key: str | None = Field(
        default=None,
        description="Langfuse secret API key",
    )
    langfuse_host: str = Field(
        default="https://cloud.langfuse.com",
        description="Langfuse server host endpoint",
    )

    # Review Pipeline Thresholds
    max_diff_tokens: int = Field(
        default=8000,
        gt=0,
        description="Maximum tokens of diff context to send to LLM",
    )
    max_comments_per_pr: int = Field(
        default=10,
        ge=1,
        description="Maximum inline comments posted per pull request review",
    )
    confidence_threshold: float = Field(
        default=0.6,
        ge=0.0,
        le=1.0,
        description="Minimum confidence score required to post an inline finding",
    )

    # Safety & Operational Controls
    blocked_installations: Any = Field(
        default_factory=lambda: [],
        description="List of GitHub App installation IDs blocked by per-installation kill switch",
    )
    daily_cost_budget_usd: float = Field(
        default=10.0,
        ge=0.0,
        description="Maximum USD cost budget allowed per UTC day across all LLM reviews",
    )
    budget_enforced: bool = Field(
        default=True,
        description="Whether to enforce the daily cost budget cap and disable reviews when exceeded",
    )

    # Retrieval & Context Augmentation
    retrieval_enabled: bool = Field(
        default=False,
        description="Whether to retrieve repository context and augment review prompt",
    )
    retrieval_mode: Literal["vector", "hybrid"] = Field(
        default="hybrid",
        description="Retrieval strategy: vector only or hybrid BM25 + vector",
    )
    reranker_enabled: bool = Field(
        default=False,
        description="Whether to rerank top candidates using cross-encoder",
    )
    retrieval_top_k: int = Field(
        default=6,
        ge=1,
        le=30,
        description="Number of context chunks to include in review prompt",
    )
    retrieval_model: str = Field(
        default="BAAI/bge-small-en-v1.5",
        description="Embedding model name for vector retrieval",
    )
    reranker_model: str = Field(
        default="BAAI/bge-reranker-base",
        description="Cross-encoder model name for reranking",
    )

    # Self-Verification Pass
    self_verification_enabled: bool = Field(
        default=False,
        description="Whether to run a secondary self-verification pass to filter false positives",
    )
    self_verification_model: str | None = Field(
        default=None,
        description="Model to use for self-verification pass (defaults to llm_model)",
    )
    prompt_variant: Literal["standard", "no_standards", "terse", "cot", "few_shot"] = Field(
        default="standard",
        description="Prompt template variant to utilize",
    )

    @property
    def retrieval(self) -> Any:
        """Namespace proxy supporting settings.retrieval.enabled access."""

        class _RetrievalProxy:
            def __init__(self, s: "Settings") -> None:
                self.enabled = s.retrieval_enabled
                self.mode = s.retrieval_mode
                self.top_k = s.retrieval_top_k
                self.reranker_enabled = s.reranker_enabled
                self.model = s.retrieval_model
                self.reranker_model = s.reranker_model

        return _RetrievalProxy(self)

    @property
    def self_verification(self) -> Any:
        """Namespace proxy supporting settings.self_verification.enabled access."""

        class _VerifyProxy:
            def __init__(self, s: "Settings") -> None:
                self.enabled = s.self_verification_enabled
                self.model = s.self_verification_model

        return _VerifyProxy(self)

    @field_validator("blocked_installations", mode="after")
    @classmethod
    def parse_blocked_installations(cls, v: Any) -> list[int]:
        """Support comma-separated strings, ints, or collections for blocked installations."""
        if isinstance(v, str):
            if not v.strip():
                return []
            return [int(x.strip()) for x in v.split(",") if x.strip()]
        if isinstance(v, (list, tuple, set)):
            return [int(x) for x in v]
        if isinstance(v, int):
            return [v]
        return []

    @field_validator(
        "github_webhook_secret",
        "github_app_id",
        "github_private_key",
        "database_url",
        "redis_url",
    )
    @classmethod
    def validate_non_empty_strings(cls, v: str, info: object) -> str:
        """Ensure required string variables are not blank whitespace."""
        stripped = v.strip()
        if not stripped:
            field_name = getattr(info, "field_name", "variable")
            raise ValueError(f"{field_name} is required and must not be empty or blank")
        return stripped

    def get_private_key_pem(self) -> str:
        """Resolve and return private key as PEM text.

        Reads from disk if `github_private_key` points to an existing file,
        otherwise returns the raw key content.
        """
        key_path = Path(self.github_private_key)
        if key_path.is_file():
            return key_path.read_text(encoding="utf-8").strip()
        return self.github_private_key


# ---------------------------------------------------------------------------
# Pricing table (USD per 1M tokens)
# ---------------------------------------------------------------------------

MODEL_PRICING_PER_MILLION: dict[str, dict[str, float]] = {
    "claude-3-5-sonnet-20241022": {"input": 3.0, "output": 15.0},
    "claude-3-5-haiku-20241022": {"input": 0.8, "output": 4.0},
    "claude-3-opus-20240229": {"input": 15.0, "output": 75.0},
    "gpt-4o": {"input": 2.5, "output": 10.0},
    "gpt-4o-mini": {"input": 0.15, "output": 0.60},
    "qwen/qwen-2.5-coder-32b-instruct": {"input": 0.0, "output": 0.0},
    "prism-ft": {"input": 0.0, "output": 0.0},
}


def calculate_token_cost(model: str, input_tokens: int, output_tokens: int) -> float:
    """Calculate total USD cost for an LLM call based on model pricing table."""
    rates = MODEL_PRICING_PER_MILLION.get(model, {"input": 0.0, "output": 0.0})
    input_cost = (input_tokens * rates["input"]) / 1_000_000.0
    output_cost = (output_tokens * rates["output"]) / 1_000_000.0
    return round(input_cost + output_cost, 6)
