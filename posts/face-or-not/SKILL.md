---
name: face-or-not-compute
description: >-
  Extra instructions for training the letsusecompute Face or Not classifier.
  Load https://compute.cx/SKILL.md first, then apply this job-specific overlay.
---

# Face or Not — extras on top of Compute

Canonical Compute skill: `https://compute.cx/SKILL.md`

Use that skill for setup, credit, dry-runs, quotes, logs, receipts, artifacts,
secrets, and reporting. This file only defines the Face or Not workload.

## Special instructions

- Download `https://raw.githubusercontent.com/theoriclabs/letsusecompute/main/posts/face-or-not/train.py` as `train.py`; do not invent a replacement.
- Plain entrypoint: `train.py::train`. Publishing entrypoint: `train.py::train_and_push`.
- Dataset: `theoriclabs/face-or-not` pinned to revision `7c81a2453be2cb48bb0b48192db91a826108e3de`; the script downloads it on the GPU machine and records manifest SHA-256 `8026d6deb5fcf6f9d9945c996d73a7dc69322ff466b05269591cd95d3105e8fa` in its outputs.
- Task: binary `no_face` / `face` classification on 128x128 crops. Bounding-box detection and face recognition are out of scope.
- Model: torchvision ResNet18 with `IMAGENET1K_V1` initialization; warm the linear head for four epochs, then fine-tune only `layer4` and the head. Earlier layers remain frozen. Train for at most 30 epochs, select on validation only, and evaluate held-out test once in this run. This is the second model measured on the same 400-row benchmark after the 87% baseline.
- Required target: at least 95% held-out test accuracy.
- Use `--gpu cheap` and `--timeout 1800`. Do not choose a premium SKU manually.
- Total issue budget is $5, including failed attempts.
- Run `--dry-run` first. It creates no machine.
- For a paid run, show the preflight provider, SKU, rate, timeout-budget estimate, and balance. Ask before confirming. Never add `--yes` unless the user explicitly authorizes the quoted spend.
- After success, report the run ID, provider, GPU, wall time, final receipt cost, best validation accuracy, and held-out test metrics. Download the artifact.
- Use `train_and_push` only with a stored `hf` secret. It publishes weights only if test accuracy reaches 95%.
- If Compute is confusing or broken, prepare `compute report "<what happened>" --run <run_id>` text and let the user decide whether to send it.
- A previous random-CNN baseline reached 87% held-out accuracy for $0.12. Its aggregate test score is known. The correction was chosen from training/validation history; do not use held-out prediction identities or images to tune it.
- Corrected run `run_e1abf4b4c24a49baa0d8f62e2d94a3c3` already completed on RunPod A100 for $0.13 and reached 98.25% held-out accuracy. Do not launch another run.
- Its Compute artifact is missing. Report `rpt_b40a746d9514b1a9b543ff003368fc7a` requests recovery. Do not reconstruct or publish weights; publish only the exact recovered qualifying checkpoint after hash verification.

## Commands

```bash
compute run train.py::train --gpu cheap --dry-run
compute run train.py::train --gpu cheap --timeout 1800 --wait

compute artifacts list <run_id>
compute artifacts get <run_id> <artifact_id> <version> --out ./weights
compute runs receipt <run_id>
```

For one training run that also publishes a qualifying checkpoint:

```bash
compute secrets set hf
compute run train.py::train_and_push --gpu cheap --timeout 1800 --wait
```

Dataset: https://huggingface.co/datasets/theoriclabs/face-or-not

Weights: https://huggingface.co/theoriclabs/face-or-not-cnn

Guide: https://letsusecompute.com/posts/face-or-not

## Suggested prompt

```text
Use https://compute.cx/SKILL.md and the Face or Not overlay in this file.
Dry-run train.py::train first with --gpu cheap and --timeout 1800. Show the
preflight quote before any spend and keep all attempts within $5. The task is
128x128 face/no-face classification, not bounding-box detection. After success,
retrieve the artifact and receipt and report the held-out test accuracy.
```
