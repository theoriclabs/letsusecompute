# Qwen3.5-9B Blender-room LoRA — experimental

Base: `Qwen/Qwen3.5-9B` at `c202236235762e1c871ad0ccb60c8ee5ba337b9a`.
Adapter: `weights/sft/adapter/`; SHA-256 `a3ee9475724bfe2d84c264134c64f32e6b96452a8d5d594981921551e4360402`.
10,819,584 trainable parameters; rank 4, alpha 8, dropout 0.05; text projections
only. The vision backbone and base weights were frozen. Three epochs over 192
procedural prompt/bpy demonstrations, completion-only loss, seed 42, bf16.

Selected by validation loss at optimizer step 72. Exact saved bf16 weights
were reloaded before validation generation and independently loaded from the
verified full archive for test. Training run: `run_16141ae2479299d3c5fd0686ab08b405`.
Final test run: `run_f62cfd31c65501be529a975b1eb6a139`. Validation-only API-hint control: `run_d6f7173297b6c3aed5062e8c22fd6f3c`.

Test: 30/32 SFT renders and 18/32 geometry-valid outputs versus
0/32 base renders and 0/32 geometry-valid outputs. Same prompts,
greedy decoding, 4,096-token cap, batches of eight, fixed Blender 4.5.3 renderer.
One base test case was blocked by an ordinary material-slot API restriction;
its execution is unmeasured, giving at most one possible additional base success.
Validation: 5/8 geometry-valid SFT programs. See the
[full evidence](assets/results/summary.json) and [gallery](assets/results/gallery.html).

The test consists of four held-out layout groups with eight palette variants
per group. With identical API hints in the validation diagnostic, all eight SFT
programs failed with list/tuple concatenation errors. Both the system prompt and batch size differed from original validation,
so causality is not isolated and instruction robustness is not established. The shared procedural grammar limits diversity; palette cases are
correlated. No independent aesthetic preference or general interior-design
ability is established. This is not a reproduction of Merve's multi-turn benchmark.

Use for research with the `rooms.SYSTEM` prompt and non-thinking chat template.
The adapter can still emit incomplete programs, outdated bpy calls, or geometry
outside the room. Generated Python must run only on disposable, secret-free
workers; the AST filter is not a security sandbox. No model was uploaded to Hub.
