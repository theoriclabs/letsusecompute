# GPU execution — completed September 8, 2026

This record covers the original seven finalized pilot runs, totaling **$7.82**.
The adapter and full archives are retained locally. The separately budgeted
repair experiment is outside this completed-pilot record.

| Run | Outcome | Final debit |
|---|---|---:|
| `run_d663d5abc85ccb99441fd11c3d95d027` | A100 setup failed on removed Transformers `warmup_ratio` argument, before training. | $0.11 |
| `run_2bd67d01e05b221f9047bfdd066fc674` | A100 optimizer update/reload passed; oversized result hung agent polling. Cancelled. | $0.29 |
| `run_74699ed064f07bbb83e7c77332aa7f69` | A100 smoke update, reload, generation and archive retrieval passed. | $0.11 |
| `run_0f6686e157bd8bb7ce1187bf01320e63` | H100 PCIe provisioning failed with provider HTTP 500. | $0.00 |
| `run_16141ae2479299d3c5fd0686ab08b405` | H100 SXM: full SFT, selected adapter export and validation generation. | $5.24 |
| `run_f62cfd31c65501be529a975b1eb6a139` | H100 SXM: replay corrected validation, then untouched 32-prompt test. | $1.54 |
| `run_d6f7173297b6c3aed5062e8c22fd6f3c` | H100 SXM: post-hoc eight-prompt validation API-hint diagnostic. | $0.53 |
| **Total** | **All receipts finalized** | **$7.82** |

A [public receipt summary](evidence/pilot-receipts.json) records the finalized costs.
Full account receipts, logs, and payloads are retained privately.
Launch records contain the actual quote and timeout; initial planning quotes
in the ledger are historical estimates. The successful H100 SXM training run
used a 10,800-second cap, with an $10.91 quoted hold. The account was never
charged beyond its authorized $25 experiment budget.

## Training and retained weights

Base/tokenizer: `Qwen/Qwen3.5-9B`, revision
`c202236235762e1c871ad0ccb60c8ee5ba337b9a`. The original dataset contains 192
training programs, 32 validation programs and 32 test programs, grouped by
layout before palette expansion. LoRA uses rank 4, alpha 8, dropout 0.05,
10,819,584 trainable text-projection parameters, bf16, seed 42, completion-only
loss, learning rate 2e-4 with cosine decay, batch 1 and accumulation 8.
Three epochs completed 72 optimizer steps. Validation losses at steps 24/48/72
were 0.0990414, 0.0229079 and 0.0169640. Step 72 was selected by validation loss.
The training/validation loop took 2,229.78 seconds (37.16 minutes). Receipts
also cover setup, downloads, generation, rendering and machine teardown.

The selected adapter was exported before evaluation, then exported with the
complete run. Saved bf16 weights were reloaded before generation. Test and
diagnostic workers independently loaded the same full archive after SHA checks.
The retained adapter is `weights/sft/adapter/adapter_model.safetensors`:
21,713,472 bytes, 496 finite bf16 tensors, all 248 LoRA-B tensors nonzero.
SHA-256: `a3ee9475724bfe2d84c264134c64f32e6b96452a8d5d594981921551e4360402`.
See [weight verification](evidence/trained-weights-verification.json).

| Retained archive | Bytes | SHA-256 |
|---|---:|---|
| `weights/sft-export.tar.gz` | 18,311,333 | `bd4f0d8b878eb618662c2a938430e2dfe6a5d314319c2ae2353aa951a47a0ba3` |
| `weights/test-export.tar.gz` | 7,151,964 | `783ffbb780d3062b5b9995d442bd78276f5015e503cc79834d3382c17c2667db` |
| `weights/control-export.tar.gz` | 131,710 | `5698844371d1bcaddd5e96be27236114aab2a337bdaa434919437781ca87e27b` |

Archives and individual members were verified (96 full-training files, 473 test
files, 86 diagnostic files). The original training source and frozen test
payload are preserved. The ignored `weights/` directory is durable local
storage, not a published model registry. The article and evidence are published; weights remain local.

## Evaluation and stopping decision

Corrected validation: SFT 7/8 rendered, 5/8 geometry-valid; base 0/8 for both.
The original operator guard rejected ordinary Blender operations. Before test,
we corrected it using validation outputs and replayed identical saved programs
for both policies, without changing geometry thresholds. Both evaluator
versions and the correction record are retained.

Primary test: SFT 30/32 rendered, 18/32 geometry-valid; base observed 0/32 for
both. One base case was blocked by a valid material-slot operation missing
from the frozen guard, so its actual execution and geometry are unmeasured.
Allowing that one operation could add at most one base success. The test was
not regenerated or re-scored. All failures and raw outputs appear in the
[complete gallery](assets/results/gallery.html).

The separate diagnostic appended identical Blender 4.5 API hints to the eight
validation prompts for both policies. Base produced one empty render and no
geometry-valid rooms. SFT produced no renders: all eight helpers concatenated
an RGB list and an alpha tuple. Original validation used batch 1; this
post-hoc diagnostic used batch 8. It shows failure under that combined variant,
without isolating the effect of instructions from batching. Saved weights and
primary test outputs were unchanged.

The four test layout groups have eight correlated palette variants each.
Object-name and bounds checks do not establish semantic or aesthetic quality.
The anonymous review packet is ready, but no independent preference score was
collected. The predeclared 80% validation geometry gate was missed (5/8), so
GRPO/GEPA was not started. A separate inference-only follow-up is underway.

## Compute transfer issues and cleanup

The deployed artifact upload returned HTTP 411; the checked-out agent already
contains a Content-Length fix. Separately, the disk supervisor polls Python
through `execFile` with an 8 MiB output limit, while terminal results include
base64 payloads. A roughly 14 MB result exceeded this limit and the agent
continued polling a completed process. The local
buffer reproduction records
`ERR_CHILD_PROCESS_STDIO_MAXBUFFER`. No Compute source or deployment was changed.

The user explicitly approved a temporary authenticated Cloudflare tunnel.
`receiver.py` exposed only fixed, authenticated archive slots, capped at 64 MiB,
with SHA verification and immutable completed uploads. It exposed no arbitrary
workspace paths. The fresh transfer token had no account privileges; rendering
children received no account credentials. The worker parent uploaded archives
and returned a small receipt to Compute. An 8 MiB throughput probe and successful
smoke verified this path before full training.

After all archives and final receipts were retained, the receiver, tunnel and
sleep-prevention process were stopped. Port 8891 was confirmed closed and the
scoped transfer token was removed. The cleanup record is retained privately.
