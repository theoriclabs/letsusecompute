# Merge two trained models and measure what survives

Issue: https://github.com/theoriclabs/letsusecompute/issues/8

Two fresh LoRA adapters on the same pinned Qwen3-0.6B base: one learns custom support queue codes, the other extracts order records. We export full parent weights and run **mergekit 0.1.4** with balanced linear, skewed linear and TIES configurations. No inference router or ensemble is involved.

Issue #7 produced one adapter, not the two complementary parents needed here. This experiment trains both parents from the same base and records the generated dataset. It does not reuse #7's adapter or claim these are independent real-world domains.

## Measured result

Full run: `run_f4a8f6a0f6a7093a78f04cd454669f86`, RunPod A100-PCIe-80GB. **$0.20**, eight billed minutes. **$0.42 total** including samples and failures. Workload time before publication/upload: 157.0 seconds. Every receipt is final; every machine is confirmed terminated.

| Model | Queue code | Strict order JSON |
|---|---:|---:|
| Base | 0/48 | 0/48 |
| Triage parent | 48/48 | 0/48 |
| Extraction parent | 0/48 | 47/48 |
| Balanced linear | 48/48 | 48/48 |
| Skewed linear | 48/48 | 0/48 |
| TIES | 48/48 | 48/48 |

The base's extraction values are correct on all 48 prompts after removing one outer Markdown fence. Its official score measures strict JSON-only adherence, not extraction knowledge. The triage parent and skewed merge remain at 0/48 after that diagnostic. The extraction parent's one failure is quantity `3` instead of `34`.

Balanced linear and TIES both score 24/24 per skill on validation. The preset tie-break selects balanced linear. No recipe was changed after observing full-run results.

Published model: [theoriclabs/qwen3-0.6b-support-order-merge](https://huggingface.co/theoriclabs/qwen3-0.6b-support-order-merge), revision `948b19edd3f61d6911adf4b96156bfeb69e32643`.

Both adapters train 10,092,544 parameters, change 392 tensors and reproduce saved-checkpoint responses. The selected 596,049,920-parameter merged model reproduces all final test responses after reload. See [results](assets/results/full/results.json), [format diagnostic](assets/results/full/format-diagnostic.json), and [run evidence](assets/results/RUNS.md).

The full command actually used was:

```bash
compute run train.py::train_and_push --gpu runpod/A100-PCIe-80GB --timeout 1800
```

The cheap selector refused estimation; an RTX-3090 sample trained successfully but had an invalid artifact manifest. A corrected RTX-3090 retry stalled during dependency installation. The A100 sample proved artifact retrieval before the full run. Details and public reports are in the evidence log.

## Reproduce

```bash
curl -fsSL https://raw.githubusercontent.com/theoriclabs/letsusecompute/main/posts/merge-models/train.py -o train.py
curl -fsSL https://compute.cx/install.sh | sh
compute setup
compute run train.py::train --gpu cheap --dry-run
compute run train.py::train --gpu cheap --timeout 1800 --args '{"sample": true}' --wait
```

Review the quote before spending. The CLI's local dry-run checks packaging but does not quote or create a machine. If the cheap selector cannot estimate this workload, use `compute gpu list` and request an explicit NVIDIA SKU, e.g. `--gpu vastai/RTX-3090`. Actual availability and rates vary.

After the sample proves training, merging, evaluation, save/reload and artifact retrieval, omit `--args` for the full run. Full mode trains 480 examples per parent for three epochs. Sample mode uses 60 examples per parent for one epoch and is only a pipeline check.

To publish the full run's validation-selected merge:

```bash
compute secrets set hf
compute run train.py::train_and_push --gpu cheap --dry-run
compute run train.py::train_and_push --gpu cheap --timeout 1800 --wait
```

`Secret.from_name("hf")` is attached to `train_and_push`. Storing a secret alone does not inject it. `train` requires no token for this public base model. Never publish sample mode over the full model.

```bash
compute runs receipt <run_id>
compute artifacts list <run_id>
compute artifacts get <run_id> <artifact_id> <version> --out ./weights
compute machines
```

## What is trained

- Base: `Qwen/Qwen3-0.6B`, revision `c1899de289a04d12100db370d81485cdf75e47ca`.
- Independent LoRA adapters, rank 16, alpha 32, no dropout; all attention projections and MLP gate/up/down projections.
- Response-only causal cross-entropy; AdamW, learning rate `2e-4`, weight decay `0.01`, batch 8, seed 42, gradient clipping 1.0. BF16 base weights, PEFT trainable adapters. Maximum checked sequence length 192; no silent truncation.
- Training examples are generated in the job, not downloaded from a private dataset. All JSONL splits are saved with SHA-256 hashes.
- Triage: RUBY = billing, JADE = login, AMBER = shipping. The prompt requests a queue code but does not provide the mapping. Four recurring issue phrases per category.
- Extraction: `{"sku":"AX-00100","qty":1}`. Four SKU prefixes, integer quantities from 1 to 40, synthetic order wording.
- Different templates and identifiers for train, validation and test. Intent phrases and SKU prefixes recur across splits. This is a deliberately narrow test of learned conventions and formatting, not broad semantic generalization.

## Merge settings and evaluation

Full parent weights are exported in FP32 after folding in LoRA. mergekit performs the merge in FP32; evaluation and the saved best model use BF16. Linear operates on full weights, not on separately averaged LoRA A and B matrices.

| Variant | Triage weight | Extraction weight | Density |
|---|---:|---:|---:|
| Linear balanced | 0.50 | 0.50 | all |
| Linear skewed | 0.95 | 0.05 | all |
| TIES | 0.50 | 0.50 | 0.20 |

TIES uses the original base to compute task deltas, trims small updates and resolves sign disagreements. All three configurations normalize weights. These are three preset recipes, not an exhaustive search.

Each model is tested on the same 24 validation prompts and 48 final test prompts **per skill** (sample: six each). Decode greedily, thinking disabled, maximum 48 new tokens. Queue scoring strips surrounding whitespace and requires the exact code. Extraction permits JSON key ordering/whitespace differences, but requires exactly `sku` and `qty`, correct values, and integer quantity. Explanations, code fences and extra fields fail.

The best merge is selected only on validation: maximize the worse skill's accuracy, then summed accuracy; ties keep the first preset. Test results do not enter selection. Every response is stored in `results.json`. The triage test contains 12 issue phrases crossed with two templates and repeated with different IDs; 48 rows do not represent 48 independent semantic intents.

The job verifies nonzero gradients, changed adapter tensors, exact response reproduction after reloading each adapter, and reproduction of all final test outputs after reloading the selected merge.

## Files and checks

- `train.py`: complete Compute job and deterministic data generator.
- `test_data.py`: local split and scoring tests, without GPU dependencies.
- `SKILL.md`: agent workflow and budget constraints.
- `assets/results/`: measured outputs, data, receipts, exact submitted sources and download checksums.
- `verify_results.py`: audit data hashes, splits, responses, scores, training evidence and selection.
- `analyze_format.py`: fence-removal diagnostic; never used for selection.
- `plot_results.py`: regenerate the SVG figures with matplotlib.

```bash
python3 test_data.py
python3 -m py_compile train.py
python3 verify_results.py
python3 analyze_format.py
python3 plot_results.py
```

Base model: Apache-2.0. Synthetic examples originate in this script. mergekit is a dependency; its license applies to that software, not a claim that its source code is ours. See [mergekit](https://github.com/arcee-ai/mergekit) and the [Qwen model card](https://huggingface.co/Qwen/Qwen3-0.6B).
