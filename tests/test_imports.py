"""Test that all prism, evals, and training modules can be imported cleanly."""

import importlib

import pytest

MODULES_TO_TEST = [
    "prism",
    "prism.config",
    "prism.models",
    "prism.api",
    "prism.github",
    "prism.github.auth",
    "prism.github.webhook",
    "prism.github.client",
    "prism.github.diff",
    "prism.github.comments",
    "prism.review",
    "prism.review.pipeline",
    "prism.review.prompts",
    "prism.review.standards",
    "prism.review.llm",
    "prism.review.postprocess",
    "prism.retrieval",
    "prism.retrieval.indexer",
    "prism.retrieval.search",
    "prism.retrieval.rerank",
    "prism.queue",
    "prism.queue.worker",
    "prism.queue.jobs",
    "prism.observability",
    "prism.observability.tracing",
    "prism.observability.metrics",
    "evals",
    "evals.dataset",
    "evals.dataset.mining",
    "evals.dataset.labeling",
    "evals.dataset.schema",
    "evals.harness",
    "evals.harness.runner",
    "evals.harness.metrics",
    "evals.harness.report",
    "training",
    "training.data_prep",
    "training.train",
    "training.merge",
    "training.serve",
]


@pytest.mark.parametrize("module_name", MODULES_TO_TEST)
def test_module_import(module_name: str) -> None:
    """Ensure module can be imported without errors."""
    mod = importlib.import_module(module_name)
    assert mod is not None
