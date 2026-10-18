"""Smoke test validating evaluation harness baseline configuration and runner."""

from pathlib import Path

from evals.harness.runner import run_evaluation


def test_eval_baseline_config_exists() -> None:
    """Verify that the baseline evaluation configuration file exists."""
    config_file = Path("evals/configs/baseline.yaml")
    assert config_file.exists(), "evals/configs/baseline.yaml must exist"


def test_eval_smoke_run() -> None:
    """Smoke test running the evaluation harness against baseline configuration."""
    exit_code = run_evaluation("evals/configs/baseline.yaml")
    assert exit_code == 0, "Evaluation runner should succeed on baseline config"
