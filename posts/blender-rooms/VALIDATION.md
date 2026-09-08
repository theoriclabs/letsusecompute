# Validation — completed September 8, 2026

- All **256 trusted teacher programs** executed, rendered and passed geometry
  checks: 192 training, 32 validation and 32 test. Per-example code, PNG, blend,
  bounds and Blender logs are retained under ignored `artifacts/teacher/`.
  Mesh counts were 33–60; evaluated polygon counts were 788–1,490.
- The pinned tokenizer measured **2,414–4,240 tokens** per example, below the
  6,144-token limit. No targets were truncated. Layout groups and exact targets
  are separated between splits; palette variants within each group correlate.
- **17 tests passed in 25.89 seconds**. They include a real miniature Qwen3.5
  adapter update and exact reload logits, Trainer checkpoint selection with
  gradient checkpointing, unequal-length batched generation, geometry and fixed
  lighting behavior, untrusted-code guards, authenticated immutable transfer,
  bounded archive extraction and early checkpoint transfer on evaluation failure.
  Two PEFT warnings concern the tiny test model's missing repository config.
  [Final test output](evidence/tests.txt). This local architecture test is
  separate from the completed full 9B training run.
- The actual 9B smoke performed a finite optimizer step, saved/reloaded bf16
  weights, generated tokens and successfully transferred verified artifacts.
  Full SFT subsequently completed all 72 steps and its saved adapter was loaded
  again for test and the diagnostic. [Weight verification](evidence/trained-weights-verification.json).
- Original full-training source hashes and the frozen test source/payload are
  retained under their run IDs. Archive and per-member checks passed for the
  smoke, full training, test and diagnostic. Hashes and receipts are recorded
  in [EXECUTION.md](EXECUTION.md).

## Measurement scope

The corrected frozen evaluator was applied to both models. Identical saved
validation outputs were replayed after fixing ordinary operators and evaluator
lighting, before the primary test. Geometry thresholds were unchanged. An extra
bright program light produces identical evaluator pixels in the trusted test.

The primary comparison reports 30/32 SFT renders and 18/32 geometry passes.
One baseline case remains unmeasured because the frozen guard rejected a valid
material-slot API; lifting only that restriction could add at most one base
success. It has not been silently corrected after observing test results.

The API-hint diagnostic is post-hoc and validation-only. It changed both the
system instruction and generation batch size relative to original validation
(8 versus 1), so it does not isolate their effects. All eight SFT programs
failed at list/tuple concatenation. The one base render was empty. Primary
weights, prompts and results remained unchanged.

Geometry uses axis-aligned bounds, names and support connectivity. It does not
prove collision freedom, recognizability or aesthetics. The article shows the
predetermined first palette from all four test families; the complete gallery
retains all 48 pairs across the three comparisons. Unblinded assistant image
inspection is not independent review. The review packet contains no automatic
preferences and no human/VLM preference score is claimed.

## Retained artifacts and checks

The public `evidence/` subset holds source snapshots, checks, receipt summaries,
and measurement limitations. Raw account logs and private payloads remain local. `assets/results/` holds actual renders,
raw output text, the measured loss chart, image hashes, summary and review pages.
`data/`, `artifacts/` and `weights/` are ignored but retained locally.

Final report lint and browser/link checks are recorded separately in
`evidence/report-verification.json`. No Compute monorepo source changed, so
unrelated monorepo integration suites were not run. The original seven pilot GPUs and temporary transfer were stopped.
The separate follow-up is excluded from this completed-pilot record.
