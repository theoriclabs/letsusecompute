---
name: jev-games-compute
description: >-
  Extra instructions for the letsusecompute jev-games guide. Load
  https://compute.cx/SKILL.md first, then apply the specials below.
---

# Jev games — extras on top of Compute

Canonical Compute skill: `https://compute.cx/SKILL.md`

Use that skill for install, credit, dry-run, quotes, logs, artifacts, and MCP.
This file is only the job-specific overlay.

## Special instructions

- Download `https://raw.githubusercontent.com/theoriclabs/letsusecompute/main/posts/jev-games/train.py` as `train.py`. Do not invent a script.
- Entrypoint: `train.py::train`
- Data: generated on the machine from `ALE/Pong-v5`. Do not upload trajectories. Structured observations are paddle/ball boxes extracted from official ALE pixels.
- Model: Qwen2.5-0.5B-Instruct plus a shared scalar candidate-scoring head. Stage A freezes the backbone. Stage B is LoRA rank 16. Then one DAgger relabel round. This is supervised cloning, not RLCD.
- Use `--gpu cheap` (also accepted: `cheapest`). Timeout `5400`. Do not pick H100 or MI300X unless a cheap run cannot boot or persist artifacts.
- Dry-run first. Then show the preflight quote and ask before confirming spend.
- After success, show closed-loop Pong return and win rate vs random and teacher, offline agreement, `reload_ok`, and download the replay MP4s.

Guide: https://letsusecompute.com/posts/jev-games

## Suggested prompt

```
Use https://compute.cx/SKILL.md.

Special instructions:
- Download https://raw.githubusercontent.com/theoriclabs/letsusecompute/main/posts/jev-games/train.py as train.py. Do not invent a script.
- Entrypoint: train.py::train
- Data generated on the machine from ALE/Pong-v5
- Qwen2.5-0.5B-Instruct + scoring head; Stage A frozen, Stage B LoRA 16, then DAgger
- Use --gpu cheap. Timeout 5400.
- Dry-run first. Then show the preflight quote and ask before confirming spend.
- After success, show Pong return/win-rate vs random/teacher and reload_ok.
```
