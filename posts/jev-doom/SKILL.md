---
name: jev-doom-compute
description: >-
  Extra instructions for the letsusecompute jev-doom guide. Load
  https://compute.cx/SKILL.md first, then apply the specials below.
---

# Jev Doom — extras on top of Compute

Canonical Compute skill: `https://compute.cx/SKILL.md`

Use that skill for install, credit, dry-run, quotes, logs, artifacts, and MCP.
This file is only the job-specific overlay.

## Special instructions

- Download `https://raw.githubusercontent.com/theoriclabs/letsusecompute/main/posts/jev-doom/train.py` as `train.py`. Do not invent a script.
- Entrypoint: `train.py::train`
- Data: generated on the machine from ViZDoom `defend_the_center.cfg`. Freedoom WADs ship with `vizdoom`. Do not upload trajectories. Structured observations are object bearings and visible-label boxes, not pixels.
- Model: Qwen2.5-0.5B-Instruct plus a shared scalar candidate-scoring head. Stage A freezes the backbone. Stage B is LoRA rank 16 at 2e-5. Then one DAgger relabel round. A fitted aim-cone controller is trained on the same labels. This is supervised cloning, not RLCD.
- Use `--gpu cheap` (also accepted: `cheapest`). Timeout `5400`. Do not pick H100 or MI300X unless a cheap run cannot boot or persist artifacts.
- Dry-run first. Then show the preflight quote and ask before confirming spend.
- After success, show closed-loop mean kills vs random and teacher, offline agreement, `reload_ok`, `deployed_policy`, and download the replay MP4s.

Guide: https://letsusecompute.com/posts/jev-doom

## Suggested prompt

```
Use https://compute.cx/SKILL.md.

Special instructions:
- Download https://raw.githubusercontent.com/theoriclabs/letsusecompute/main/posts/jev-doom/train.py as train.py. Do not invent a script.
- Entrypoint: train.py::train
- Data generated on the machine from ViZDoom defend_the_center.cfg
- Qwen2.5-0.5B-Instruct + scoring head; Stage A frozen, Stage B LoRA 16, then DAgger
- Use --gpu cheap. Timeout 5400.
- Dry-run first. Then show the preflight quote and ask before confirming spend.
- After success, show mean kills vs random/teacher, reload_ok, and deployed_policy.
```
