"""CLI entrypoint for running evaluation benchmark.

Usage:
    python -m evals.harness.run --config evals/configs/baseline.yaml --split dev
"""

from evals.harness.runner import main

if __name__ == "__main__":
    main()
