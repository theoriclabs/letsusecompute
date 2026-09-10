---
name: model-router-compute
description: >-
  Extra instructions for the letsusecompute prompt-router guide. Load
  https://compute.cx/SKILL.md first, then apply the specials below.
---

# Model Router — extras on top of Compute

Canonical Compute skill: `https://compute.cx/SKILL.md`

Use that skill for install, credit, dry-run, quotes, logs, artifacts, and MCP.
This file is only the job-specific overlay.

## Special instructions

- Download `https://raw.githubusercontent.com/theoriclabs/letsusecompute/main/posts/model-router/train.py` as `train.py`. Do not invent a script.
- Entrypoint: `train.py::train`
- Dataset: `routellm/gpt4_dataset` (the script downloads it on the machine)
- TF-IDF logistic baseline + DistilBERT (~67M params), binary `cheap_suffices` (Mixtral score ≥ 4)
- Use `--gpu cheap`. Do not pick H100 or MI300X unless a cheap run exceeds 40 minutes. Timeout `3600`.
- Dry-run first. Then show the preflight quote and ask before confirming spend.
- After success, show routing accuracy, the cheap-traffic fraction at the 95% quality target, the implied cost cut, and download artifacts with `compute artifacts list` / `get`.

Guide: https://letsusecompute.com/posts/model-router

## Suggested prompt

```
Use https://compute.cx/SKILL.md.

Special instructions:
- Download https://raw.githubusercontent.com/theoriclabs/letsusecompute/main/posts/model-router/train.py as train.py. Do not invent a script.
- Entrypoint: train.py::train
- Dataset: routellm/gpt4_dataset (the script downloads it on the machine)
- TF-IDF logistic baseline + DistilBERT (~67M params), binary cheap_suffices (Mixtral score >= 4)
- Use --gpu cheap. Do not pick H100 or MI300X unless a cheap run exceeds 40 minutes. Timeout 3600.
- Dry-run first. Then show the preflight quote and ask before confirming spend.
- After success, show routing accuracy, the cheap-traffic fraction at the 95% quality target, the implied cost cut, and download artifacts with compute artifacts list / get.
```
