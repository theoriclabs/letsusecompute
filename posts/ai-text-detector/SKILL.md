---
name: ai-text-detector-compute
description: >-
  Extra instructions for the letsusecompute AI-text detector guide. Load
  https://compute.cx/SKILL.md first, then apply the specials below.
---

# AI-text detector — extras on top of Compute

Canonical Compute skill: `https://compute.cx/SKILL.md`

Use that skill for install, credit, dry-run, quotes, logs, artifacts, and MCP.
This file is only the job-specific overlay.

## Special instructions

- Download `https://raw.githubusercontent.com/theoriclabs/letsusecompute/main/posts/ai-text-detector/train.py` as `train.py`. Do not invent a script.
- Entrypoint: `train.py::train`
- Dataset: `Hello-SimpleAI/HC3` English (`all`), plus a streamed `liamdugan/raid` slice for a different generator. The script downloads both on the machine.
- TF-IDF logistic baseline + DistilBERT (~67M params), binary `human` vs `ai`.
- Prefer `--gpu cheap`. If cheap quotes a Vast SKU with no interruptible offers, use `--provider runpod --gpu A100-PCIe-80GB`. Timeout `2400`.
- Dry-run first. Then show the preflight quote and ask before confirming spend.
- After success, show in-distribution accuracy, human false-positive rate, medicine-domain numbers, RAID/OOD numbers, and download artifacts with `compute artifacts list` / `get`.

Guide: https://letsusecompute.com/posts/ai-text-detector

## Suggested prompt

```
Use https://compute.cx/SKILL.md.

Special instructions:
- Download https://raw.githubusercontent.com/theoriclabs/letsusecompute/main/posts/ai-text-detector/train.py as train.py. Do not invent a script.
- Entrypoint: train.py::train
- Dataset: Hello-SimpleAI/HC3 English (all), plus a streamed liamdugan/raid slice for a different generator. The script downloads both on the machine.
- TF-IDF logistic baseline + DistilBERT (~67M params), binary human vs ai.
- Prefer --gpu cheap. If cheap quotes a Vast SKU with no interruptible offers, use --provider runpod --gpu A100-PCIe-80GB. Timeout 2400.
- Dry-run first. Then show the preflight quote and ask before confirming spend.
- After success, show in-distribution accuracy, human false-positive rate, medicine-domain numbers, RAID/OOD numbers, and download artifacts with compute artifacts list / get.
```
