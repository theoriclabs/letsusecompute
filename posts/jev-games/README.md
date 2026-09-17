# Atari Pong decision model on compute.cx

Real `ALE/Pong-v5`. Structured paddle/ball observations extracted from official ALE pixels. A fitted intercept controller wins first-to-5. Qwen2.5-0.5B candidate scoring copies the labels offline and loses closed-loop.

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
