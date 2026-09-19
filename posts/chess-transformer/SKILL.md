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
- SAN-token GPT. Pipeline preset: `--args '{"sample": true}'` (4 layers, 256-wide, 4 heads, 100k positions, 20 games/opponent). Headline A (script default): 8 layers, 512-wide, 8 heads, 4M positions, 8 epochs, 100 games/opponent. Headline B (this guide): 16M positions, 3 epochs, batch 64, lr 4e-4
- The script downloads Stockfish 17.1 as a pinned Linux binary. Do not try apt; `compute.Image` has no apt API.
- Use `--gpu cheap` for the pipeline run. Headline: `--gpu H100-PCIe --provider runpod --timeout 4500`. Do not pick MI300X.
- Dry-run first. Then show the preflight quote and ask before confirming spend. `--dry-run` is an upload plan, not a price quote.
- After success, show top-1 accuracy, sampled and argmax legal-move rates, match W/D/L, and download artifacts with `compute artifacts list` / `get`.

Guide: https://letsusecompute.com/posts/chess-transformer

## Suggested prompt

```
Use https://compute.cx/SKILL.md.

Special instructions:
- Download https://raw.githubusercontent.com/theoriclabs/letsusecompute/main/posts/chess-transformer/train.py as train.py. Do not invent a script.
- Entrypoint: train.py::train
- Dataset: Lichess/standard-chess-games (one recent month, streamed on the machine); fallback adamkarvonen/chess_games lichess_100mb.zip
- SAN-token GPT. Pipeline: --args '{"sample": true}'. Headline B: 8/512/8, 16M positions, 3 epochs, batch 64, lr 4e-4. The script downloads Stockfish 17.1 itself.
- Use --gpu cheap for the sample (timeout 1800). Headline: --gpu H100-PCIe --provider runpod --timeout 4500.
- Dry-run first. Then show the preflight quote and ask before confirming spend.
- After success, show top-1, legal-move rate (sampled and argmax), and match W/D/L, then download artifacts with compute artifacts list / get.
```
