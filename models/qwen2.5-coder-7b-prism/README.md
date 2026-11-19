---
language:
- en
- python
- typescript
- javascript
license: apache-2.0
tags:
- code-review
- qlora
- unsloth
- trl
- peft
- qwen2.5-coder
base_model: Qwen/Qwen2.5-Coder-7B-Instruct
pipeline_tag: text-generation
---

# Qwen2.5-Coder-7B-Instruct-Prism-Review

Fine-tuned version of `Qwen/Qwen2.5-Coder-7B-Instruct` optimized for automated, high-signal Pull Request code reviews adhering to custom engineering standards.

> [!NOTE]
> **Repository Artifacts vs. Model Weights**:
> This repository contains the model architecture definition, adapter configurations (`adapter_config.json`), tokenizer metadata, and stub files for CI/integration tests.
> The full 16-bit merged safetensors (~14.5 GB) and 4-bit AWQ quantized weights (~4.2 GB) can be trained using `training/modal_train.py` or downloaded from your organization's model registry / S3 bucket.

## Model Highlights
- **Architecture**: Qwen2.5-Coder-7B-Instruct with QLoRA adapter
- **LoRA Rank**: $r = 32$, $\alpha = 64$ (targeting all linear attention & MLP projections)
- **Training Objective**: Instruction-tuned on multi-file unified diffs with exact line-number localization
- **False-Positive Suppression**: Trained with balanced 25% clean PRs to suppress false positives on defect-free code
- **Output Format**: Structured JSON adhering strictly to `ChunkReviewResponse` schema

## Evaluation Results (Dev Set — Training Validation)

> [!NOTE]
> These are **development set** metrics measured during training. Final benchmark
> results on the held-out frozen test split (120 PRs from `trpc/trpc` and `vitejs/vite`)
> are reported in the main project [README](../../README.md) and differ significantly
> due to different data distributions and post-training quantization (AWQ 4-bit).

- **F1 Score**: `0.8613`
- **Precision**: `0.7620`
- **Recall**: `0.9780`
- **False Positive Rate (clean PRs)**: `0.1612`
- **Final Training Loss**: `0.3056`

## Hyperparameters
| Parameter | Value |
| :--- | :--- |
| Base Model | `Qwen/Qwen2.5-Coder-7B-Instruct` |
| LoRA Rank ($r$) | `32` |
| LoRA Alpha ($\alpha$) | `64` |
| Learning Rate | `0.0002` (Cosine) |
| Epochs | `3` |
| Effective Batch Size | `16` |
| Max Sequence Length | `8192` |
| Quantization | 4-bit NormalFloat (NF4) |
| Gradient Checkpointing | `True` |
| Packing | `False` |

## Weight Merging & Export

To merge the trained LoRA adapter with the base model into standalone 16-bit BF16 weights:
```bash
python -m training.merge \
    --adapter-dir models/qwen2.5-coder-7b-prism/adapter \
    --output-dir models/qwen2.5-coder-7b-prism/merged_16bit \
    --device cuda
```

## Production Serving with vLLM

### Option 1: Native vLLM Command
```bash
vllm serve models/qwen2.5-coder-7b-prism/merged_16bit \
    --host 0.0.0.0 \
    --port 8000 \
    --max-model-len 8192 \
    --gpu-memory-utilization 0.90 \
    --trust-remote-code
```

### Option 2: 4-bit AWQ on Single A10G / L4 (Recommended)
```bash
vllm serve models/qwen2.5-coder-7b-prism/merged_16bit \
    --host 0.0.0.0 \
    --port 8000 \
    --quantization awq \
    --max-model-len 8192 \
    --gpu-memory-utilization 0.90
```

### Option 3: Docker Compose
```bash
docker compose -f docker-compose.yml -f ops/serving/docker-compose.gpu.yml up -d
```

### Option 4: Serverless on Modal
```bash
modal deploy ops/serving/modal_serve.py
```
