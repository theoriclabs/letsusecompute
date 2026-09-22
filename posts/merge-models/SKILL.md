---
name: merge-models-compute
description: Reproduce the Qwen3 support/order model-merging experiment on Compute.
---

# Merge models on Compute

Load [the public Compute skill](https://compute.cx/SKILL.md) first. Work only from public documentation and this guide.

- Download this folder's `train.py`; it generates its own synthetic data on the GPU machine and downloads the pinned Qwen3-0.6B revision.
- Start with `train.py::train --args '{"sample": true}'`. Sample mode tests the whole pipeline with 60 examples per parent, one epoch, and six test prompts per skill. It is not the headline result.
- Run `python3 test_data.py` locally if the repository is available. Dry-run before submitting.
- Try `--gpu cheap`; if estimation is refused, record the request ID and use an explicit NVIDIA SKU from the live catalog. Do not silently approve selector fallback.
- Confirm the preflight quote unless the user has already authorized the run budget. Issue #8 has a $10 total ceiling including failures. Timeout is 1800 seconds per run; it excludes provisioning overhead.
- Then run full mode: 480 examples per parent, three epochs, 24 validation and 48 test prompts per skill. Base, parents, and three preset merges see identical evaluation inputs. Never tune on test results.
- `train_and_push` injects `Secret.from_name("hf")` and publishes the validation-selected merge under `theoriclabs/qwen3-0.6b-support-order-merge`. Storing a secret alone does not inject it. Do not overwrite a published model with a sample-mode run.
- Verify nonzero gradients, changed adapter tensors, both adapter reload checks, and merged-model reload. Retrieve the artifact after teardown; metadata alone is not retrieval.
- Keep every output, including failed responses. Queue scoring is exact; JSON scoring checks exact keys, values and integer type. Do not replace measured results with expectations.
- Compare both skills separately. This synthetic test measures custom labels and output formatting on held-out templates, not broad domain expertise.
- Record actual run receipts, GPU/provider, elapsed time, total cost and public `compute report` IDs in evidence. Confirm machine teardown and final settlement.
