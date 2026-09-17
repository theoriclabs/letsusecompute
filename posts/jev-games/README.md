# Jev-inspired decision games on compute.cx

One shared Qwen2.5-0.5B-Instruct candidate scorer for Snake, fully observed DoorKey-5x5, and MinAtar-style Breakout. Supervised cloning, not RLCD.

Guide: https://letsusecompute.com/posts/jev-games

## Quick start

```bash
curl -fsSL https://raw.githubusercontent.com/theoriclabs/letsusecompute/main/posts/jev-games/train.py -o train.py
compute run train.py::train --gpu cheap --dry-run
compute run train.py::train --gpu cheap --timeout 2400 --wait --args '{"sample": true}'
compute run train.py::train --gpu cheap --timeout 2400 --wait --args '{"states_per_game": 900, "episodes": 20, "epochs_head": 5, "epochs_lora": 3}'
```

## This guide’s run

- Sample: `run_9c6ce2dd7976a1d01bd3498cf9d43429` Vast RTX-3090, $0.01
- Headline: `run_fbac11a02be7662387a54c834561393a` RunPod A100-SXM-80GB, $0.33
- Issue spend: $0.68 including lost Vast jobs and a hung collection
- DoorKey closed-loop success 90% vs random 0% vs teacher 100% (20 episodes)
- Snake food 0.15 vs 0.05 vs 11.7
- Breakout bricks 1.50 vs 1.15 vs 15.3
- Offline tie-aware agreement 64% / 86% / 63%
- Head changed, reload matched. Artifacts persisted on the A100 rerun. No Hub push — no `hf` secret.

## Observation contract

| Game | Observation | Menu | Teacher |
|---|---|---|---|
| Snake | ordered body, heading, food | left / straight / right | legal + nearest food |
| DoorKey | full 5x5 board, pose, key, door, goal | MiniGrid 0–6 | BFS, ties kept |
| Breakout | paddle, ball, last, dir, brick mask | MinAtar IDs 0, 1, 3 | short-rollout intercept |

Structured state only. Pixel Breakout is out of scope for this post.

## Licenses

Environments are original implementations matching public rules. MinAtar is GPL-3.0; we did not vendor its source. MiniGrid DoorKey is the published 5x5 fully observed variant, not the default partial-obs benchmark.
