"""Modal serverless deployment script for Prism Review vLLM serving.

Scales to zero when no PRs are active; cold-starts in ~10 seconds on A10G.

Run with:
    modal serve ops/serving/modal_serve.py   # For local interactive testing
    modal deploy ops/serving/modal_serve.py  # For permanent production deployment
"""

from __future__ import annotations

import os

try:
    import modal
except ImportError:
    modal = None

if modal is not None:
    app = modal.App("prism-vllm-serve")

    # Define container image with CUDA and vLLM
    vllm_image = (
        modal.Image.debian_slim(python_version="3.11")
        .pip_install(
            "vllm>=0.6.0",
            "torch==2.4.0",
            "transformers>=4.44.0",
            "fastapi>=0.110.0",
            "huggingface_hub",
        )
    )

    volume = modal.Volume.from_name("prism-model-cache", create_if_missing=True)

    MODEL_DIR = "/root/models/qwen2.5-coder-7b-prism/merged_16bit"

    @app.function(
        image=vllm_image,
        gpu="A10G",
        timeout=1800,
        scaledown_window=300,  # Scale down after 5 minutes of idle
        volumes={"/root/models": volume},
        secrets=[modal.Secret.from_name("huggingface-secret")] if "huggingface-secret" in os.environ else [],
    )
    @modal.web_server(port=8000, startup_timeout=120)
    def serve():
        """Launch vLLM server process inside container."""
        import subprocess

        cmd = [
            "vllm",
            "serve",
            MODEL_DIR if os.path.exists(MODEL_DIR) else "Qwen/Qwen2.5-Coder-7B-Instruct",
            "--host",
            "0.0.0.0",
            "--port",
            "8000",
            "--max-model-len",
            "8192",
            "--gpu-memory-utilization",
            "0.90",
            "--trust-remote-code",
        ]
        subprocess.Popen(cmd)
else:

    def main() -> None:
        print("Modal SDK not installed in local environment. Run 'pip install modal' to use.")

    if __name__ == "__main__":
        main()

