"""Modal serverless script for fine-tuning Qwen2.5-Coder-7B-Instruct with QLoRA.

Run with:
    modal run training/modal_train.py
    modal run training/modal_train.py --rank 32 --epochs 3
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

try:
    import modal
except ImportError:
    modal = None

if modal is not None:
    app = modal.App("prism-qwen-coder-qlora")

    # Define custom container image with Unsloth, CUDA 12.1, and PyTorch
    qlora_image = (
        modal.Image.debian_slim(python_version="3.11")
        .pip_install(
            "torch==2.3.0",
            "transformers==4.42.3",
            "datasets",
            "accelerate",
            "peft",
            "bitsandbytes",
            "trl==0.9.4",
            "pydantic>=2.7.0",
            "pyyaml",
            "httpx",
        )
        .run_commands(
            "pip install --no-deps 'unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git'"
        )
    )

    volume = modal.Volume.from_name("prism-model-cache", create_if_missing=True)

    @app.function(
        image=qlora_image,
        gpu="A10G",
        timeout=7200,
        volumes={"/root/models": volume},
    )
    def train_on_modal(
        rank: int = 32,
        alpha: int = 64,
        epochs: int = 3,
        learning_rate: float = 2e-4,
        clean_ratio: float = 0.25,
        push_to_hub: bool = False,
        hub_repo_id: str | None = None,
    ) -> dict[str, Any]:
        """Execute QLoRA fine-tuning on Modal cloud GPU."""
        import sys

        sys.path.append("/root/codeagent")

        from training.data_prep import prepare_sft_dataset
        from training.train import TrainingConfig, train

        output_dir = "/root/models/qwen2.5-coder-7b-prism"

        print(f"--> Preparing SFT dataset (clean ratio: {clean_ratio:.1%})...")
        train_path, dev_path = prepare_sft_dataset(
            output_dir=Path("/root/data/training"),
            clean_ratio=clean_ratio,
            distill=True,
        )

        config = TrainingConfig(
            base_model="Qwen/Qwen2.5-Coder-7B-Instruct",
            train_data_path=str(train_path),
            dev_data_path=str(dev_path),
            output_dir=output_dir,
            adapter_dir=f"{output_dir}/adapter",
            merged_dir=f"{output_dir}/merged_16bit",
            lora_r=rank,
            lora_alpha=alpha,
            learning_rate=learning_rate,
            num_train_epochs=epochs,
            per_device_train_batch_size=2,
            gradient_accumulation_steps=8,
            max_seq_length=8192,
            packing=False,
            gradient_checkpointing=True,
            load_in_4bit=True,
        )

        print("--> Launching QLoRA training on A10G...")
        result = train(config, force_framework="unsloth")
        volume.commit()
        return result

    @app.local_entrypoint()
    def main(
        rank: int = 32,
        epochs: int = 3,
        lr: float = 2e-4,
    ) -> None:
        print(f"Triggering serverless fine-tuning job on Modal (rank={rank}, epochs={epochs})...")
        res = train_on_modal.remote(rank=rank, alpha=2 * rank, epochs=epochs, learning_rate=lr)
        print("Modal execution completed:", res)
else:

    def main() -> None:
        print("Modal SDK not installed in local environment. Run 'pip install modal' to use.")


if __name__ == "__main__":
    main()
