# Teach Qwen to furnish a room

Trained **Qwen3.5-9B with a rank-4 LoRA adapter** on 192 original procedural
Blender programs. The full run, exact-weight reload and held-out evaluation
completed on Compute. **Completed pilot GPU spend: $7.82.**
This report covers the original seven finalized runs. The separately budgeted
inference-only repair experiment is in progress and excluded from these totals.

**Robustness limitation:** with Blender API hints, all eight adapter
programs in a validation-only diagnostic failed with a list/tuple color-helper
error. The original validation used batch 1, while this diagnostic used batch 8,
so it does not isolate instructions from batching. The adapter remains limited
to the tested setup.

| Test policy | Rendered | Geometry valid | Token limit |
|---|---:|---:|---:|
| BASE | 0/32 | 0/32 | 3/32 |
| SFT | 30/32 | 18/32 | 1/32 |

One base test program was blocked by the frozen guard for a valid
`bpy.ops.object.material_slot_add` call. Its execution/geometry is unmeasured;
lifting that one restriction could add at most one base success. Another
guard rejection imported `random`, contrary to the explicit task. See
[evidence/test-guard-limitation.json](evidence/test-guard-limitation.json).

These are 32 prompts from **four held-out layout groups**, each with eight
correlated palette variants. Shared furniture primitives and a restricted
Blender task limit the claim. This is a procedural-data control, not 192
independent Astra demonstrations, an open-ended design benchmark, or a
comparison against Merve's different multi-turn GEPA/GRPO environment.

Corrected validation: base **0/8 rendered,
0/8 geometry valid**; SFT **7/8 rendered,
5/8 geometry valid**. Required object names and
axis-aligned bounds are diagnostics, not semantic, collision or aesthetic proof.
There is no independent human/VLM preference score.

A separate **post-hoc validation-only API-hint diagnostic** appended identical
Blender 4.5 compatibility notes to both system prompts. Base rendered
**1/8** and passed geometry on **0/8**;
SFT rendered **0/8** and passed geometry on **0/8**.
This diagnostic changed no model weights or primary test outputs. Its batch
size also differed from original validation (8 versus 1), so it is not an
isolated causal test of the API hints. See
[evidence/api-hint-control-plan.json](evidence/api-hint-control-plan.json).

## Results and actual weights

- [Published article](index.html), [all base/SFT comparisons](assets/results/gallery.html),
  [loss plot](assets/results/loss.png), [machine-readable results](assets/results/summary.json).
- Actual adapter: `weights/sft/adapter/adapter_model.safetensors` (21,713,472 bytes).
  [Model card](MODEL_CARD.md). The ignored `weights/` directory is retained locally.
- Portable full archive: `weights/sft-export.tar.gz`; final evaluation archive:
  `weights/test-export.tar.gz`. Their bytes and member hashes were verified.
- [Execution and receipts](EXECUTION.md), [validation checks](VALIDATION.md),
  [research plan](PLAN.md), [weight verification](evidence/trained-weights-verification.json).

Base/tokenizer revision: `c202236235762e1c871ad0ccb60c8ee5ba337b9a`.
Adapter SHA-256: `a3ee9475724bfe2d84c264134c64f32e6b96452a8d5d594981921551e4360402`.
All 496 tensors are finite bf16 values; all 248 LoRA-B tensors are nonzero.
The selected checkpoint is step 72 of 72, after three epochs.
Validation completion losses: **0.09904 → 0.02291 → 0.01696**.
The training/validation loop took **37.16 minutes**; GPU receipts also include setup,
model downloads, generation, rendering, and teardown.

## Reproduce

Use Python 3.11 for local Blender preparation. The separate Compute SDK
interpreter runs `submit.py` and `retrieve.py`; this experiment used the
neighboring Compute checkout (SDK 0.1.17). Install/sign in through the
[Compute quickstart](https://compute.cx/quickstart), or use that checkout's
`.venv/bin/python`. `submit.py` uses its current client/quote interfaces.

```bash
cd posts/blender-rooms
uv venv --python 3.11 .venv
uv pip install --python .venv/bin/python -r requirements.txt
.venv/bin/python rooms.py --output data
.venv/bin/python render.py --output artifacts
.venv/bin/python -m pytest -q
```

All 256 trusted teacher programs were executed, rendered and geometry-checked.
The model sees text and bpy targets; teacher images are quality-control assets.
Completion/padding labels are masked. Targets are never truncated; observed
training sequence lengths were 2,414–4,240 tokens, below the 6,144 limit.

The deployed Compute agent had artifact-transfer bugs during this experiment.
The included receiver uses immutable, authenticated, size-bounded archive slots.
Start it in one terminal, then connect a temporary HTTPS tunnel in another:

```bash
.venv/bin/python receiver.py --root transfer/received --key-file transfer/worker.key --port 8891
cloudflared tunnel --url http://127.0.0.1:8891 --protocol http2
```

Save the printed public HTTPS origin in `transfer/base-url`. The private token
is generated in `transfer/worker.key` and read directly by `submit.py`; keep
that file private. The transfer directory is ignored by Git. Keep both
processes running until all archives arrive, then stop them.

Set `COMPUTE_PY` to the interpreter containing the Compute SDK. Run one stage
at a time; use the returned run ID with `compute logs RUN_ID --follow` and
`compute runs status RUN_ID`. The launcher refuses overlapping allocations
and quotes beyond the remaining ledger budget. Use a fresh ledger for a new
experiment, and verify a current GPU offer before each stage.

```bash
COMPUTE_PY=/path/to/compute/.venv/bin/python
"$COMPUTE_PY" submit.py smoke --gpu runpod/A100-PCIe-80GB --timeout 1800 --transfer-dir transfer --ledger evidence/my-budget.json --budget 25
"$COMPUTE_PY" retrieve.py SMOKE_RUN_ID --archive transfer/received/smoke.tar.gz --out weights/my-smoke
"$COMPUTE_PY" submit.py train --gpu runpod/H100-SXM --timeout 10800 --transfer-dir transfer --ledger evidence/my-budget.json
"$COMPUTE_PY" retrieve.py TRAIN_RUN_ID --archive transfer/received/sft.tar.gz --out weights/my-sft
"$COMPUTE_PY" submit.py test --gpu runpod/H100-SXM --timeout 7200 --count 32 --training-run TRAIN_RUN_ID --transfer-dir transfer --ledger evidence/my-budget.json
"$COMPUTE_PY" retrieve.py TEST_RUN_ID --archive transfer/received/test.tar.gz --out weights/my-test
```

Add `--plan-only` to a submission command to resolve source without network
or spending. Full SFT uses learning rate 2e-4 with cosine decay, rank 4/alpha 8,
batch 1 with eight accumulated steps, bf16 and gradient checkpointing. It selects
by validation completion loss. Test generation uses identical left-padded
batches of eight for both policies, greedy decoding, and 4,096 new tokens.

The evaluator was corrected using validation outputs before the test: ordinary
mesh/object operators are permitted and supplied lighting is fixed. The exact
saved validation programs were replayed for both policies. Geometry thresholds
and generated programs were unchanged. See the
[correction record](evidence/evaluator-correction.json). Original and final
source snapshots, hashes and archives are saved with each run.

```bash
uv pip install --python .venv/bin/python -r requirements-report.txt
.venv/bin/python report.py --training weights/my-sft --test weights/my-test --output assets/my-results
```

## Use the adapter

On a suitable NVIDIA GPU, load `Qwen/Qwen3.5-9B` at the pinned revision with
`AutoModelForImageTextToText` in bf16, then
`PeftModel.from_pretrained(base, "weights/sft/adapter")`. Use the pinned tokenizer,
`rooms.SYSTEM` and the same non-thinking chat template. `inference.batch_tokens`
returns code tokens without executing them; `evaluate.py` is a complete loading
example. This adapter was evaluated on the four procedural room families only.

## Review and next experiment

Share only [the anonymous review folder](assets/results/review/index.html)
with reviewers before showing labeled results. It contains shuffled A/B images,
the brief, and a local JSON export of their own choices. The identity key remains
separate. No preferences have been filled in automatically. Because the primary base
policy produced no completed renders, those pairs do not establish a
fine-grained aesthetic advantage.

The optional control can be reproduced with `submit.py control --count 8` and
the same transfer/training arguments. It uses the receiver's `probe` slot;
preserve any earlier throughput probe before that run.

The predeclared 80% validation-geometry gate was not met (5/8).
GRPO/GEPA was therefore not started. The next useful experiment is richer,
independently authored layouts and a repair/geometry curriculum, followed by
independent visual review. The completed pilot left **$17.18** before the separate follow-up.

Generated programs are untrusted. The AST filter is not an OS sandbox; execute
model code only on disposable workers without account secrets or sensitive
mounts. Local preparation/tests execute this project's trusted demonstrations.
Article: https://letsusecompute.com/posts/blender-rooms/
[Shareable GIF, MP4, and post copy](SOCIAL.md). No model was uploaded to Hub.
The follow-up code is included for inspection; this article reports only the completed pilot.
Private payloads, credentials, raw account records, and model archives are excluded from publication.
