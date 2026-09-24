---
name: qwen-pokemon-compute
description: Reproduce the Qwen3-0.6B Pokémon Showdown imitation experiment on Compute.
---

# Teach Qwen to play Pokémon on Compute

Load [the public Compute skill](https://compute.cx/SKILL.md) first. Work only from public documentation and this guide.

- Download this folder's `train.py`. It is self-contained: on the GPU machine it downloads Node `v22.22.0` and Pokémon Showdown at commit `a5df8274`, builds it, starts a local server on port 8000, and generates its own training data. No ROM, no dataset download, no API labels.
- Format is `gen9randombattle`. Every evaluation opponent is poke-env's `MaxBasePowerPlayer`. The teacher is poke-env's `SimpleHeuristicsPlayer` with terastallization turned off, so it plays the same action space as the model.
- The model sees the battle as text with numbered options (moves first, then switches) and answers one digit. Both the untuned and the fine-tuned Qwen players take the highest-probability legal digit; do not compare a constrained model against an unconstrained one.
- Dry-run first: `compute run train.py::train --gpu runpod/H100-SXM --dry-run`. Bare `H100-SXM` is ambiguous between providers at quote time; qualify it.
- Pipeline check: `train.py::train --args '{"sample": true}'` (60 expert battles, 20 eval battles per player, one epoch). It cost $0.29 on H100-SXM. It is not the headline result.
- Full run: 3,000 expert battles (~66k decisions), one epoch of LoRA r=16 on the digit only, 200 eval battles per player. Use `--timeout 3600`; it finished well inside that. Issue #10 budget: $10 before a go-ahead.
- `train_and_push` injects `Secret.from_name("hf")` and publishes the adapter to `theoriclabs/qwen3-0.6b-pokemon-showdown`. Never push a sample run.
- If `--wait` stops streaming, the run is usually still going. Check `compute runs status <run_id>` and `compute logs <run_id>` instead of cancelling.
- Report 95% intervals (the job computes Wilson intervals). 200 battles cannot separate two players a few points apart.
- After the run: `compute runs receipt`, `compute artifacts list`, `compute artifacts get ... --out`, and confirm `compute machines` is empty. Keep `results.json`, the expert decisions, and the saved replays.
