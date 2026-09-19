---
name: chess-transformer-compute
description: >-
  Extra instructions for the letsusecompute chess-transformer guide. Load
  https://compute.cx/SKILL.md first, then apply the specials below.
---

# Chess move transformer — extras on top of Compute

Canonical Compute skill: `https://compute.cx/SKILL.md`

Use that skill for install, credit, dry-run, quotes, logs, artifacts, and MCP.
This file is only the job-specific overlay.

## Special instructions

- Download `https://raw.githubusercontent.com/theoriclabs/letsusecompute/main/posts/chess-transformer/train.py` as `train.py`. Do not invent a script.
- Entrypoint: `train.py::train`
- Dataset: `Lichess/standard-chess-games` (one recent month, streamed on the machine). Fallback: `adamkarvonen/chess_games` `lichess_100mb.zip`
- SAN-token GPT. Pipeline preset: `--args '{"sample": true}'` (4 layers, 256-wide, 4 heads, 100k positions, 20 games/opponent). Headline: 8 layers, 512-wide, 8 heads, 4M positions, 100 games/opponent
- Use `--gpu cheap` for the pipeline run. `H100-SXM` is allowed for the headline run. Do not pick MI300X.
- Dry-run first. Then show the preflight quote and ask before confirming spend.
- After success, show top-1 accuracy, sampled legal-move rate, match W/D/L, and download artifacts with `compute artifacts list` / `get`.

Guide: https://letsusecompute.com/posts/chess-transformer

## Suggested prompt

```
Use https://compute.cx/SKILL.md.

Special instructions:
- Download https://raw.githubusercontent.com/theoriclabs/letsusecompute/main/posts/chess-transformer/train.py as train.py. Do not invent a script.
- Entrypoint: train.py::train
- Dataset: Lichess/standard-chess-games (one recent month, streamed on the machine); fallback adamkarvonen/chess_games lichess_100mb.zip
- SAN-token GPT. Pipeline: --args '{"sample": true}'. Headline: 8/512/8, 4M positions.
- Use --gpu cheap. Timeout 1800 for sample, 3600 for headline.
- Dry-run first. Then show the preflight quote and ask before confirming spend.
- After success, show top-1, legal-move rate, and match W/D/L, then download artifacts with compute artifacts list / get.
```
