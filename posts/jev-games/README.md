# Pong + Doom: a Jev-inspired decision model

The combined guide covers Atari Pong and ViZDoom, Jev’s calibrated-decision training objective, and our smaller supervised-imitation experiment. This directory contains the Pong script; the Doom script remains in [`../jev-doom`](../jev-doom).

Guide: https://letsusecompute.com/posts/jev-games

## Quick start

```bash
curl -fsSL https://raw.githubusercontent.com/theoriclabs/letsusecompute/main/posts/jev-games/train.py -o train.py
compute run train.py::train --gpu cheap --dry-run
compute run train.py::train --gpu cheap --timeout 2400 --wait --args '{"sample": true}'
compute run train.py::train --gpu cheap --timeout 5400 --wait
```

## This guide’s run

- Sample: `run_78b6f5ec13f4af481ca9af88aa42c61e` Vast RTX-3090, $0.03
- Qwen headline: `run_84d48446e804cd4772e33b240d882643` RunPod A100-SXM, $0.45. Offline agree 0.983, closed-loop 0–5
- Fitted intercept (CPU, same labels): first-to-5 is 5–3, win rate 100% on 6 episodes
- Revision spend $1.57. Issue total $2.25 including the first toy-grid post
- Environment: `ALE/Pong-v5`, frameskip 4, sticky-action probability 0
- Observation: ball/paddle boxes from official RGB. Menu: FIRE, RIGHT (up), LEFT (down)

## Observation contract

| Field | Meaning |
|---|---|
| `ball_rel` | `above_far` / `above` / `aligned` / `below` / `below_far` |
| `approach` | `toward` the player, `away`, or `none` |
| `delta` | ball_y − paddle_y in playfield pixels |
| Menu | FIRE, RIGHT (up), LEFT (down) |

The Qwen model never sees the RGB tensor. Replays are the real ALE frames. The deployed winner is the intercept with lead and deadzone fit from those labels.

## Licenses

[Arcade Learning Environment](https://github.com/Farama-Foundation/Arcade-Learning-Environment) / `ale-py` and the bundled Pong ROM. We do not vendor ALE source.

## Combined replay with decisions

From the repository root, using Python 3.12:

```bash
python -m pip install -r posts/jev-games/assets/requirements-video.txt
python posts/jev-games/assets/make_video.py
```

This renders locally on CPU. It records fresh seed-77 gameplay using the published fitted parameters (Pong lead 1.6 / deadzone 6; Doom cone 8°), then shows contiguous decisions 20–99 from each game. Each pre-action frame is paired with the observation and action actually used to advance the engine. Playback holds each decision for 0.2 seconds, slowing both games. The action tiles are choices, not neural confidence scores. No Qwen checkpoint or Jev API is used.

Outputs: `assets/social/jev-games.mp4`, `.gif`, `poster.jpg`, `captions.vtt`, and `decisions.json`. The JSON records source-code hashes, fitted parameters, game seeds, original frame indices and video timestamps. These short excerpts explain the controller; the historical evaluation results remain in the article and receipts.

Both training scripts also export `replays/<policy>_decisions.json` for future runs. Its `frame` refers to the original captured frames; the raw replay MP4 samples every second frame, up to 360 frames. A final Pong terminal frame can have no next action.

Combined revision GPU spend: $1.62 ($1.57 Pong + $0.05 Doom); $2.30 including the earlier $0.68 prototype. `assets/results/summary.json` and `run-result.json` are archived results from that earlier Snake / DoorKey / Breakout prototype, not the Pong evaluation. The Pong revision’s run record is `assets/results/receipt.txt`.
