# Issue #8 run evidence

Budget authorized by the user: $10 total, including failed runs. No top-up authorized or needed.

## Friction

- `rpt_1ffb86cc0f4125045846dcd67ce26ee8`: CLI 0.1.13 `--gpu cheap` could not estimate the LoRA/mergekit job and refused before creating a run. Request `req_f9b2022d799b5319727f40b5caaa87c5`. Explicit `vastai/RTX-3090` obtained a quote.

## Sample

- Run: `run_41a3710dbab118477c86431e0910c4f5`
- Entrypoint: `train.py::train`, arguments `{"sample":true}`
- Vast RTX-3090, $0.15/hour + 7.5% fee, timeout 1,800 seconds.
- Quote: 31 billed minutes (timeout + estimated teardown), approximately $0.09, excluding boot/install/download overhead. Capacity unverified at quote time.
- Scope: 60 examples per parent, one epoch; six validation and six test prompts per skill.
- Exact submitted source: `sample/source.py`.
- Training, all merges and merged reload completed in 105.1 seconds. Artifact scan failed because the submitted marker used unsupported `kind=model`. No artifact bytes survived.
- Final receipt: **$0.03**, eight billed minutes, cleanup confirmed (`provider_absent`), settlement final. See `sample/receipt.json` and `sample/status.json`.

- `rpt_64044bc7eceb0a63896c189f878a1059`: artifact schema/error-message documentation feedback; server categorized it as a duplicate of the selector report because both referenced the same run. The manifest bug was ours and is corrected to `kind=output`. Also noted that CLI 0.1.13 MCP does not expose the skill's `compute_inspect_run`; final receipts provide settlement and cleanup evidence.

## Corrected sample

- Run: `run_4c6f2bfd64c1433347d58a61c3e56d7a`
- Same sample settings and GPU/rate; manifest fixed to `kind=output`.
- Quote: approximately $0.08 at timeout plus estimated teardown, excluding provisioning/install overhead. Approved under the user's $10 total budget.
- Submitted source: `sample-fixed/source.py`.
- Cancelled after ~11 minutes in startup (four boot, seven preparation), before the entrypoint ran. $0.03; cleanup confirmed. See final receipt.
- `rpt_a7f4edc0ae1f560fb6d60d2f5b3f473a`: no dependency-install progress/diagnostics during the wait.

## A100 sample

- Run: `run_c1f3785b58bc63a3daa1873bba764d4a`
- RunPod A100-PCIe-80GB, $1.45/hour + 7.5%, timeout 1,800 seconds.
- Quote: approximately $0.80 at timeout plus estimated teardown, excluding setup overhead. Within the authorized $10 total.
- Submitted source: `sample-a100/source.py`; sample settings unchanged.
- Succeeded: 114.6 seconds in the Python workload, six billed minutes, **$0.16**. Final settlement and cleanup confirmed.
- Artifact `artifact_cdbccd60e1807b21af78a29eca64aea3`, version 1. All 35 files downloaded and SHA-256 verified. The 1,192,135,096-byte merged checkpoint was streamed through a FIFO hash verifier because the local disk had less than 1 GB free; it was not retained locally. Small adapter and evaluation files were downloaded normally. See `sample-a100/download-verification.json`.
- Local evidence audit passed: generated data and hashes match, every score recomputes from saved outputs, 392 changed tensors per adapter, nonzero gradients, exact reload outputs, validation-only selection.
- Base extraction responses contained correct values inside Markdown fences; strict JSON-only scoring rejected them. This is format adherence, not a claim that the base cannot extract values.
- Settled total before full run: **$0.22**.

## Full run

- Run: `run_f4a8f6a0f6a7093a78f04cd454669f86`
- Entrypoint: `train.py::train_and_push`, default full settings: 480 examples per parent, three epochs, 24 validation and 48 test prompts per skill.
- RunPod A100-PCIe-80GB, $1.45/hour + 7.5%, timeout 1,800 seconds; quote approximately $0.81 at timeout plus estimated teardown, excluding setup overhead.
- Submitted source: `full/source.py`. Recipe and data unchanged from the predeclared full configuration. `hf` secret explicitly injected by the entrypoint.
- Succeeded. 157.0 seconds for download/training/merging/evaluation/reload before Hugging Face publication and Compute artifact persistence. Full receipt **$0.20**, eight billed minutes, final settlement and confirmed cleanup.
- Triage parent: 48/48 queues, 0/48 strict extraction. Extraction parent: 0/48 queues, 47/48 extraction. Balanced linear and TIES: 48/48 on both. Skewed linear: 48/48 queues, 0/48 extraction.
- Balanced linear and TIES tie at 24/24 per skill on validation. Predeclared ordering selects balanced linear. Full results were not used for recipe tuning.
- Published model: https://huggingface.co/theoriclabs/qwen3-0.6b-support-order-merge ; revision `948b19edd3f61d6911adf4b96156bfeb69e32643`.
- Published weight SHA-256: `bcd8a3cdd3c1b9f812500ceee5a90620779ebacdd0dad00418cd26041e5a224a` (1,192,135,096 bytes).
- Artifact: `artifact_7401e2566955129225db83039ee93344`, version 1.
- Final experiment spend: **$0.42**, four attempts. `machines-after.json` is empty. No top-up, continuing endpoint, or machine left running.
- All 40 full-run artifact files downloaded and SHA-256 verified. The large weight file was streamed through the same FIFO verifier rather than retained locally. Its hash matches the published Hugging Face LFS object exactly. See `full/download-verification.json` and `full/hub.json`.
- Local audit of the final Compute artifact passed. The Hub's embedded `results.json` is the pre-publication snapshot (`hub_repo: null`); the Compute artifact records the successful repository name. Scores, datasets and model bytes match.

Console logs are normalized to line breaks with terminal control sequences and trailing whitespace removed. Metrics and result JSON are unchanged.
