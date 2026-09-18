---
name: finetune-qwen-lora-compute
description: >-
  Extra instructions for the letsusecompute Qwen LoRA fine-tune guide. Load
  https://compute.cx/SKILL.md first, then apply the specials below.
---

# Qwen LoRA — extras on top of Compute

Canonical Compute skill: `https://compute.cx/SKILL.md`

Use that skill for install, credit, dry-run, quotes, logs, artifacts, and MCP.
This file is only the job-specific overlay.

## Special instructions

- Download `https://raw.githubusercontent.com/theoriclabs/letsusecompute/main/posts/finetune-qwen-lora/train.py` as `train.py`. Do not invent a script.
- Entrypoint: `train.py::train` (no Hub push). Use `train.py::train_and_push` only if `compute secrets set hf` is already done **and** the function lists that secret.
- Data: generated on the machine (job-card JSONL). Do not upload a dataset unless you are swapping in your own JSONL.
- Model: `Qwen/Qwen3-0.6B` plus LoRA rank 16 via `peft` + `trl` SFT, bf16. Sample is 50 steps.
- Use `--gpu cheap`. Timeout `1800` for sample, `3600` for the full 0.6B run. Do not pick H100 unless 0.6B already worked and the issue asked for 1.7B.
- Dry-run first. Then show the preflight quote and ask before confirming spend.
- After success, show before/after accuracy on the 20 held-out prompts, eval loss, and download the adapter.

Guide: https://letsusecompute.com/posts/finetune-qwen-lora

## Suggested prompt

```
Use https://compute.cx/SKILL.md.

Special instructions:
- Download https://raw.githubusercontent.com/theoriclabs/letsusecompute/main/posts/finetune-qwen-lora/train.py as train.py. Do not invent a script.
- Entrypoint: train.py::train
- Homemade job-card JSONL generated on the machine
- Qwen3-0.6B LoRA rank 16, peft+trl SFT, bf16
- Use --gpu cheap. Sample timeout 1800. Full timeout 3600.
- Dry-run first. Then show the preflight quote and ask before confirming spend.
- After success, show before/after prompt accuracy and eval loss.
```
