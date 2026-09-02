---
name: not-hotdog-compute
description: >-
  Extra instructions for the letsusecompute Not Hotdog guide. Load
  https://compute.cx/SKILL.md first, then apply the specials below.
---

# Not Hotdog — extras on top of Compute

Canonical Compute skill: `https://compute.cx/SKILL.md`

Use that skill for install, credit, dry-run, quotes, logs, artifacts, and MCP.
This file is only the job-specific overlay.

## Special instructions

- Download `https://raw.githubusercontent.com/theoriclabs/letsusecompute/main/posts/not-hotdog/train.py` as `train.py`. Do not invent a script.
- Entrypoint: `train.py::train`
- Dataset: `theoriclabs/hot-dog-not-hot-dog` (the script downloads it on the machine)
- Randomly initialized CNN, ~93k params, CrossEntropyLoss, 50 epochs, 128×128
- Use `--gpu cheap`. Do not pick H100 or MI300X.
- Dry-run first. Then show the preflight quote and ask before confirming spend.
- After success, show the printed loss/accuracy and download artifacts with `compute artifacts list` / `get`.

Guide: https://letsusecompute.com/posts/not-hotdog  
Published weights (skip training): https://huggingface.co/theoriclabs/not-hotdog-cnn

## Suggested prompt

```
Use https://compute.cx/SKILL.md.

Special instructions:
- Download https://raw.githubusercontent.com/theoriclabs/letsusecompute/main/posts/not-hotdog/train.py as train.py. Do not invent a script.
- Entrypoint: train.py::train
- Dataset: theoriclabs/hot-dog-not-hot-dog (the script downloads it on the machine)
- Randomly initialized CNN, ~93k params, CrossEntropyLoss, 50 epochs, 128x128
- Use --gpu cheap. Do not pick H100 or MI300X.
- Dry-run first. Then show the preflight quote and ask before confirming spend.
- After success, show the printed loss/accuracy and download artifacts with compute artifacts list / get.
```
