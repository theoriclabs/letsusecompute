# Doom + Pong: a Jev-inspired decision model

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

## Two replays with real model probabilities

The guide leads with Doom, followed by Pong. Each video records two separate seed-77 episodes: Qwen controls the first segment; the fitted rule controls the second. Each uses up to the first 80 consecutive decisions, held for 0.2 seconds per decision for readable playback. A shorter episode ends at death rather than being padded with invented decisions.

The Qwen segment displays the actual softmax output from the saved scoring head and LoRA adapter. The chosen action is the argmax of that vector and is the action passed to the engine. The rule segment has no probability estimate. These are action probabilities, not calibrated probabilities of a kill or a win. The fresh recordings do not replace the historical evaluation table.

Reproduce with Python 3.12 and the saved artifacts from the account that owns the runs:

```bash
python -m pip install -r posts/jev-games/assets/requirements-video.txt
compute artifacts get run_577746f2c18b58372a8c404c2b8f2a1c artifact_9160b4ca29eb1f71514b07432a961087 1 --out /tmp/jev-checkpoints/doom
compute artifacts get run_84d48446e804cd4772e33b240d882643 artifact_838cf65caaa78db4a7ad5ea18e697ba4 1 --out /tmp/jev-checkpoints/pong
python posts/jev-games/assets/make_video.py --checkpoints /tmp/jev-checkpoints
```

The renderer loads Alibaba Qwen’s `Qwen/Qwen2.5-0.5B-Instruct`, pinned to base revision `7ae557604adf67be50417f59c2c2f167def9a775`, then restores each run’s head, adapter, and evaluation stage. Inference uses float32 on Apple MPS if available, otherwise CPU. The checkpoint folder for each game must contain `head.pt`, `summary.json`, and `adapter/adapter_config.json` plus `adapter/adapter_model.safetensors`. No new GPU rental or training is launched.

Outputs under `assets/social/`: `doom.mp4`, `pong.mp4`, matching `*-poster.jpg`, `.vtt`, and `*-decisions.json`. The logs include the run/artifact IDs, checkpoint and source hashes, original frame indices, raw RGB hashes, full-precision probabilities, and video timestamps. The earlier combined `jev-games.mp4` / `.gif` and `decisions.json` remain as archived assets; the page uses the two new videos.

Both training scripts’ `record_episode` functions accept a plain action or a `model_prediction` result and preserve the probabilities from the latter. The exported `frame` refers to the original captured frames; training-run replay MP4s sample every second frame up to 360 frames. A final Pong terminal frame can have no next action.

Combined revision GPU spend: $1.62 ($1.57 Pong + $0.05 Doom); $2.30 including the earlier $0.68 prototype. Local replay inference is outside those charges. `assets/results/summary.json` and `run-result.json` are archived results from the earlier Snake / DoorKey / Breakout prototype, not the Pong evaluation. The Pong revision’s run record is `assets/results/receipt.txt`.

## Attribution

- Base model: [Alibaba Qwen team’s Qwen2.5-0.5B-Instruct](https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct).
- Decision-interface inspiration: [TypeSafe Jev](https://typesafe.ai/blog/introducing-system-one-models-and-jev).
- Candidate-scoring and training reference: [vinnylarouge/jevlike](https://github.com/vinnylarouge/jevlike).
- Reviewed research references in [issue #18](https://github.com/theoriclabs/letsusecompute/issues/18): [harshatheg/Qwen-2.5-1B-RLCD](https://huggingface.co/harshatheg/Qwen-2.5-1B-RLCD) and [AlexWortega/OpenJev](https://huggingface.co/AlexWortega/openjev). Their checkpoints were not used in these game runs.
- Environments/assets: Farama’s [ViZDoom](https://github.com/Farama-Foundation/ViZDoom), [Arcade Learning Environment](https://github.com/Farama-Foundation/Arcade-Learning-Environment), and [Gymnasium](https://github.com/Farama-Foundation/Gymnasium); [Freedoom](https://freedoom.github.io/).
- Model tooling: Hugging Face [Transformers](https://github.com/huggingface/transformers) and [PEFT](https://github.com/huggingface/peft).
