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

## Model Highlights
- **Architecture**: Qwen2.5-Coder-7B-Instruct with QLoRA adapter
- **LoRA Rank**: $r = 64$, $\alpha = 128$ (targeting all linear attention & MLP projections)
- **Training Objective**: Instruction-tuned on multi-file unified diffs with exact line-number localization
- **False-Positive Suppression**: Trained with balanced 25% clean PRs to enforce silence on defect-free code
- **Output Format**: Structured JSON adhering strictly to `ChunkReviewResponse` schema

## Evaluation Results (Dev Set Benchmark)
- **F1 Score**: `0.8573`
- **Precision**: `0.7620`
- **Recall**: `0.9780`
- **False Positive Rate (clean PRs)**: `0.1612`
- **Final Training Loss**: `0.1813`

## Hyperparameters
| Parameter | Value |
| :--- | :--- |
| Base Model | `Qwen/Qwen2.5-Coder-7B-Instruct` |
| LoRA Rank ($r$) | `64` |
| LoRA Alpha ($\alpha$) | `128` |
| Learning Rate | `0.00015` (Cosine) |
| Epochs | `3` |
| Effective Batch Size | `16` |
| Max Sequence Length | `8192` |
| Quantization | 4-bit NormalFloat (NF4) |
| Gradient Checkpointing | `True` |
| Packing | `False` |

## Usage with vLLM
```bash
vllm serve models/experiments/config_3_r64/merged_16bit \
    --port 8000 \
    --max-model-len 8192 \
    --gpu-memory-utilization 0.90
```
