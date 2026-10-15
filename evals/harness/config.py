"""Experiment configuration schema for evaluation harness runs.

Loaded from YAML config files in evals/configs/. Captures all parameters
needed for a fully reproducible evaluation run.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field


class DatasetConfig(BaseModel):
    """Configuration for the evaluation dataset source."""

    model_config = ConfigDict(frozen=True)

    path: str = Field(
        default="data/final",
        description="Directory containing {train,dev,test}.jsonl split files",
    )
    sample_limit: int | None = Field(
        default=None,
        description="Maximum number of examples to evaluate (None = all)",
    )


class PipelineConfig(BaseModel):
    """Configuration for the review pipeline under evaluation."""

    model_config = ConfigDict(frozen=True)

    model: str = Field(
        default="gpt-4o-mini",
        description="LLM model identifier for the review pipeline",
    )
    timeout_seconds: float = Field(
        default=30.0,
        gt=0.0,
        description="Per-call timeout deadline for LLM invocations",
    )
    max_concurrency: int = Field(
        default=5,
        ge=1,
        description="Maximum concurrent LLM calls during evaluation",
    )
    confidence_threshold: float = Field(
        default=0.6,
        ge=0.0,
        le=1.0,
        description="Minimum confidence score for a finding to count as a prediction",
    )
    max_comments_per_pr: int = Field(
        default=10,
        ge=1,
        description="Maximum findings per PR from the pipeline",
    )
    seed: int = Field(
        default=42,
        description="Fixed random seed for reproducibility",
    )
    retrieval_enabled: bool = Field(
        default=False,
        description="Whether retrieval context augmentation is enabled",
    )
    retrieval_mode: Literal["vector", "hybrid"] = Field(
        default="hybrid",
        description="Retrieval strategy: vector only or hybrid BM25 + vector",
    )
    reranker_enabled: bool = Field(
        default=False,
        description="Whether cross-encoder reranking is enabled",
    )
    retrieval_top_k: int = Field(
        default=6,
        ge=1,
        description="Number of retrieved chunks in review prompt",
    )
    self_verification_enabled: bool = Field(
        default=False,
        description="Whether to run secondary self-verification pass to filter false positives",
    )
    prompt_variant: Literal["standard", "no_standards", "terse", "cot", "few_shot"] = Field(
        default="standard",
        description="Prompt template variant to utilize",
    )

    @property
    def retrieval(self) -> Any:
        """Namespace proxy supporting pipeline.retrieval.enabled access."""

        class _RetrievalProxy:
            def __init__(self, s: PipelineConfig) -> None:
                self.enabled = s.retrieval_enabled
                self.mode = s.retrieval_mode
                self.top_k = s.retrieval_top_k
                self.reranker_enabled = s.reranker_enabled

        return _RetrievalProxy(self)

    @property
    def self_verification(self) -> Any:
        """Namespace proxy supporting pipeline.self_verification.enabled access."""

        class _VerifyProxy:
            def __init__(self, s: PipelineConfig) -> None:
                self.enabled = s.self_verification_enabled

        return _VerifyProxy(self)


class MatcherConfig(BaseModel):
    """Configuration for the prediction-to-gold matching strategy."""

    model_config = ConfigDict(frozen=True)

    line_tolerance: int = Field(
        default=3,
        ge=0,
        description="Maximum line number difference for a position match",
    )
    strategy: Literal["exact", "semantic", "embedding", "both"] = Field(
        default="both",
        description=(
            "Matching strategy: 'exact' = category equality only, "
            "'semantic' = LLM judge, 'embedding' = cosine similarity, "
            "'both' = report both semantic and embedding"
        ),
    )
    semantic_judge_model: str = Field(
        default="gpt-4o-mini",
        description="Model for the semantic match LLM judge",
    )
    embedding_threshold: float = Field(
        default=0.75,
        ge=0.0,
        le=1.0,
        description="Minimum cosine similarity for embedding-based match",
    )
    cache_dir: str = Field(
        default=".cache/matcher",
        description="Directory for caching match judgments",
    )


class BootstrapConfig(BaseModel):
    """Configuration for bootstrap confidence interval estimation."""

    model_config = ConfigDict(frozen=True)

    n_resamples: int = Field(
        default=1000,
        ge=100,
        description="Number of bootstrap resamples for CI computation",
    )
    confidence_level: float = Field(
        default=0.95,
        gt=0.0,
        lt=1.0,
        description="Confidence level for bootstrap intervals (e.g. 0.95 = 95%)",
    )
    seed: int = Field(
        default=42,
        description="Random seed for bootstrap resampling",
    )


class BaselineConfig(BaseModel):
    """Configuration for baseline comparison strategies."""

    model_config = ConfigDict(frozen=True)

    random_findings_per_pr: int = Field(
        default=3,
        ge=1,
        description="Number of random findings per PR for the random baseline",
    )
    random_seed: int = Field(
        default=42,
        description="Seed for the random baseline",
    )


class ExperimentConfig(BaseModel):
    """Top-level experiment configuration loaded from YAML."""

    model_config = ConfigDict(frozen=True)

    name: str = Field(description="Human-readable experiment name")
    version: str = Field(default="1.0.0", description="Config schema version")
    dataset: DatasetConfig = Field(default_factory=DatasetConfig)
    pipeline: PipelineConfig = Field(default_factory=PipelineConfig)
    matcher: MatcherConfig = Field(default_factory=MatcherConfig)
    bootstrap: BootstrapConfig = Field(default_factory=BootstrapConfig)
    baseline: BaselineConfig = Field(default_factory=BaselineConfig)
    metrics: list[str] = Field(
        default_factory=lambda: [
            "precision",
            "recall",
            "f1",
            "false_positive_rate",
            "comments_per_pr",
            "actionability",
            "severity_weighted_recall",
            "cost_per_review",
            "latency_p50",
            "latency_p95",
            "tokens_per_review",
        ],
        description="List of metric names to compute and report",
    )

    @classmethod
    def from_yaml(cls, path: str | Path) -> ExperimentConfig:
        """Load experiment configuration from a YAML file."""
        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(f"Config file not found: {p}")
        raw: dict[str, Any] = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
        return cls.model_validate(raw)
