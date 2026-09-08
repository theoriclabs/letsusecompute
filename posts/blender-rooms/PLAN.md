# Teach Qwen to furnish a room

## Hypothesis and scope

First test whether supervised examples improve **Qwen/Qwen3.5-9B** on
text-to-Blender room generation. Keep the base model fixed. Merve's experiment
already tried GRPO and GEPA; repeating sparse-reward RL before teaching the
model what successful furniture looks like is a weak first bet.

This pilot uses original, programmatically varied bpy demonstrations authored
in this project. They are **not 192 independent frontier-model generations**.
Their renders are quality-control and evaluation assets; the first training
run is text-to-code, not image-conditioned. Generalization within this small
procedural grammar is the first claim to test. It cannot establish arbitrary
interior-design ability or a win against Merve's different, multi-turn setup.

## Ordered experiment

1. Generate 192 training examples: four room families, six layout groups per
   family, eight palettes. Produce 32 validation and 32 test examples from two
   additional layout groups per family. Split before palette expansion.
   Save prompts, complete bpy programs, scene measurements, renders and hashes.
2. Execute every teacher program with Blender. Reject missing shells,
   nonfinite/degenerate meshes, excessive polygon counts, furniture outside
   the floor, missing semantic groups, and floating parts. Inspect a training
   contact sheet visually. Record the dataset's limitations explicitly.
3. On a single NVIDIA 80 GB GPU, run a one-step Qwen LoRA smoke test with a
   short generation, save/reload the adapter, and download and verify its scoped archive export.
   Do not spend on a full run until the smoke passes.
4. Run the frozen base on fixed validation prompts, then completion-only LoRA
   SFT (rank 4, 2e-4 learning rate, three epochs, effective batch 8, bf16,
   gradient checkpointing). No truncation of code targets. Select the
   checkpoint by validation completion loss, then compare base/SFT generations
   under identical decoding, token and render budgets.
5. Use blind paired human review of shuffled render pairs for visual detail,
   recognizability, instruction adherence and layout. Geometry metrics remain
   separate; object names and part counts alone cannot establish quality.
   Evaluate the untouched test prompts once after selecting the recipe.
6. Only if SFT produces plausible scenes, add 150–200 independently authored
   teacher programs and compare with this procedural-data control. Then test
   rejection-sampling SFT or a short GRPO curriculum. Keep unsafe execution a
   hard failure, but log shell, bounds, support, coverage and detail components
   separately instead of collapsing every geometric flaw into zero advantage.

## Evidence and decision rules

- Primary artifact: adapter weights + pinned base/tokenizer revision.
- Required evidence: exact dataset/source hashes, environment versions, seed,
  training/validation losses, raw baseline/SFT outputs including failures,
  render measurements, paired review manifest, Compute run IDs and receipts.
- Pilot readiness: all teacher examples pass the checks and the GPU smoke
  trains with finite loss, reloads the adapter and preserves artifacts.
- Success claim: a completed fixed comparison with all failures counted;
  report paired results and uncertainty, including flat or negative results.
  Do not equate lower teacher-forced loss with better rendered rooms.
- Next research stage: at least 80% render/geometry validity on validation
  and a positive blind preference signal. These are continuation gates, not
  a guarantee of a publishable model.
- Budget: the user authorized and funded $25. Receipts, current allocations,
  and remaining budget are recorded in `evidence/execution-budget.json`.
  The smoke adapter is downloaded and verified. The user explicitly approved
  the temporary authenticated archive transfer; no account keys are sent to
  rendering workers. Obtain a fresh quote before each allocation. Never exceed
  the remaining authorized budget.

## Sources inspected on 2026-09-08

- [Merve's experiment](https://huggingface.co/buckets/merve/blender-grpo-gepa)
- [Task generator](https://huggingface.co/buckets/merve/blender-grpo-gepa/tree/blender_rl/tasks.py)
- [Rewards](https://huggingface.co/buckets/merve/blender-grpo-gepa/tree/blender_rl/rewards.py)
- [Training code](https://huggingface.co/buckets/merve/blender-grpo-gepa/tree/blender_rl/train.py)
- [Qwen model card](https://huggingface.co/Qwen/Qwen3.5-9B)
- [Transformers Qwen3.5 API](https://huggingface.co/docs/transformers/model_doc/qwen3_5)

The source task generator chooses among four room types and three palettes;
changing seeds does not ensure distinct prompt text. Its published code also
hard-gates geometry before visual judging. Those observations motivate our
split audit and supervised first experiment. The bucket is mutable, and the
README describes an earlier reward configuration than the inspected code.

## Final outcome — September 8, 2026

The supervised pilot, exact-weight reload and untouched 32-prompt test are
complete. SFT rendered 30/32 test rooms; 18/32 passed geometry. Corrected
validation reached 5/8 geometry-valid, below the predeclared 80% continuation
gate. Independent visual review is still open; its packet is prepared.
GRPO/GEPA was not started.

A post-hoc validation-only API-hint diagnostic produced no SFT renders. Its
instructions and batch size both differed from original validation, so no
isolated causal claim is made. One primary baseline program was blocked by a
valid API absent from the frozen guard; that case remains unmeasured.

The completed pilot cost $7.82 of $25. Its seven GPU runs and temporary
transfer were stopped, and the adapter is retained locally. The article and
completed-pilot evidence are published. A separate inference-only repair
experiment is underway and excluded from this record.
