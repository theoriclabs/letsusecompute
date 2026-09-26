---
name: alphazero-small-compute
description: Train AlphaZero on Connect Four with Compute, checkpointing every iteration and resuming across runs.
---

# AlphaZero on Connect Four with Compute

Load [the public Compute skill](https://compute.cx/SKILL.md) first. Work only from public documentation and this guide.

- `train.py` is self-contained: game, batched PUCT MCTS, a 374,810-parameter residual net, self-play workers, training, and five evaluation opponents. No dataset download.
- Always qualify the GPU: `--gpu runpod/H100-SXM`. Dry-run every entrypoint first; the CLI's static check rejects an entrypoint whose `compute.App` is not a plain module-level import.
- Sample: `train.py::train --args '{"sample": true}' --timeout 300` (3 iterations, 64 games each, ~$0.35). It proves the worker pool, the GPU and the artifact; it is not a result.
- Self-play runs in worker subprocesses started as `python -c <boot> train.py --az-worker ...` and connected over a localhost socket. Do not switch to `multiprocessing` fork (the parent may hold a CUDA context) or spawn (the Compute runner imports `train.py` under a generated module name that children cannot import).
- Every iteration overwrites `$COMPUTE_ARTIFACT_DIR/alphazero-c4/checkpoint.pt` and `history.json`. A run killed by `--timeout` still uploads the artifact; verified on this guide's first run, which timed out after iteration 10 and left a complete iteration-10 checkpoint.
- There is no documented way to hand one run's artifact to another. Two resume paths:
  - `resume` / `resume_and_push <run_id>`: the job installs the Compute CLI and runs `compute artifacts get` itself. Needs a revocable Compute API key stored as the `COMPUTE_API_KEY` secret. Do not create that secret on the user's behalf; ask them to.
  - `resume_from_hub`: `train_and_push` also mirrors each checkpoint to `theoriclabs/alphazero-connect4` under `checkpoint/`, and this entrypoint continues from there with only the `hf` secret. This is what the published run used.
- Use `max_seconds` so a run stops itself cleanly; keep `--timeout` a few minutes above it. `eval_every=2` halves evaluation time, which is otherwise ~70% of an iteration.
- The one-ply heuristic is too weak to show progress: untrained network plus 64-simulation search already beats it ~80%. Read the rollout-MCTS and raw-policy curves.
- Record run ids, receipts, and whether each artifact is `completed`. Confirm `compute machines` is empty.
