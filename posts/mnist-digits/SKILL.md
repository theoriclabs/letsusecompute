---
name: mnist-digits-compute
description: >-
  Extra instructions for the letsusecompute MNIST digits guide. Load
  https://compute.cx/SKILL.md first, then apply the specials below.
---

# MNIST Digits — extras on top of Compute

Canonical Compute skill: `https://compute.cx/SKILL.md`

Use that skill for install, credit, dry-run, quotes, logs, artifacts, and MCP.
This file is only the job-specific overlay.

## Special instructions

- Download `https://raw.githubusercontent.com/theoriclabs/letsusecompute/main/posts/mnist-digits/train.py` as `train.py`. Do not invent a script.
- Entrypoint: `train.py::train`
- Dataset: MNIST via `torchvision.datasets.MNIST` (the script downloads it on the machine)
- Randomly initialized CNN, ~106k params, CrossEntropyLoss, 10 epochs, 28×28
- Use `--gpu cheap`. Do not pick H100 or MI300X. Timeout `900`.
- Dry-run first. Then show the preflight quote and ask before confirming spend.
- After success, show the printed loss/accuracy and download artifacts with `compute artifacts list` / `get`.

Guide: https://letsusecompute.com/posts/mnist-digits
Weights: kept as the run's `mnist-cnn` compute artifact only — this guide's run had no Hugging Face token set up, so nothing was published to the Hub. Use `train_and_push` after `compute secrets set hf` if you want a Hub checkpoint.

## Suggested prompt

```
Use https://compute.cx/SKILL.md.

Special instructions:
- Download https://raw.githubusercontent.com/theoriclabs/letsusecompute/main/posts/mnist-digits/train.py as train.py. Do not invent a script.
- Entrypoint: train.py::train
- Dataset: MNIST via torchvision.datasets.MNIST (the script downloads it on the machine)
- Randomly initialized CNN, ~106k params, CrossEntropyLoss, 10 epochs, 28x28
- Use --gpu cheap. Do not pick H100 or MI300X. Timeout 900.
- Dry-run first. Then show the preflight quote and ask before confirming spend.
- After success, show the printed loss/accuracy and download artifacts with compute artifacts list / get.
```
