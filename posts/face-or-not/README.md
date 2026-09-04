# Face or Not on compute.cx

Fine-tune a compact ResNet18 that classifies a 128x128 image crop as `face` or `no_face`
on a fresh cheap GPU with [compute.cx](https://compute.cx).

This is binary image classification. It does not find faces in a larger image,
draw bounding boxes, identify anyone, or perform face recognition.

Guide: https://letsusecompute.com/posts/face-or-not

## Publication status

The validated dataset is published at
[`theoriclabs/face-or-not`](https://huggingface.co/datasets/theoriclabs/face-or-not)
at revision `7c81a2453be2cb48bb0b48192db91a826108e3de`. Model weights have not
been published because the qualifying run's Compute artifact is missing.
Recovery report `rpt_b40a746d9514b1a9b543ff003368fc7a` is open.

`train.py` pins that exact revision and records it together with manifest
SHA-256 `8026d6deb5fcf6f9d9945c996d73a7dc69322ff466b05269591cd95d3105e8fa`
in the checkpoint, metrics, model card, and Compute artifact metadata.

## Quick start

```bash
curl -fsSL https://raw.githubusercontent.com/theoriclabs/letsusecompute/main/posts/face-or-not/train.py -o train.py

curl -fsSL https://compute.cx/install.sh | sh
compute setup
compute credits add 10

compute run train.py::train --gpu cheap --dry-run
compute run train.py::train --gpu cheap --timeout 1800 --wait
```

`--dry-run` prints the upload plan and creates no machine. The second command
shows the GPU, locked rate, timeout-budget estimate, and balance before asking
you to confirm spend. Keep the issue's total spend at or below $5.

With the public Compute CLI 0.1.13, the dry-run found one source file, the
CUDA-PyTorch image, the 1,800-second decorator timeout, no secrets, and the
declared pip packages. The workload estimator could not model the training
loop, so the CLI fell back to the decorator's `vastai/RTX-3090`. The observed
preflight locked $0.15/hr and quoted approximately $0.08 total at the full
31-billed-minute timeout budget. The prompt was declined; no GPU started.

## Dataset

The published [`theoriclabs/face-or-not`](https://huggingface.co/datasets/theoriclabs/face-or-not)
dataset contains 4,000 balanced 128x128 RGB crops:

| split | `no_face` | `face` | total |
|---|---:|---:|---:|
| train | 1,600 | 1,600 | 3,200 |
| validation | 200 | 200 | 400 |
| test | 200 | 200 | 400 |

All examples derive from Open Images V7, so both labels share the same source
and image pipeline. Positive rows use verified `Human face` boxes. Negative rows
come only from images with a human-verified negative `Human face` label. Original
image IDs never cross splits, and the test split is evaluated once after model
selection within each run. The corrected ResNet18 is the second model measured
on the same 400-row test benchmark after the 87% random-CNN baseline.

Open Images lists each selected image under CC BY 2.0 and licenses its
annotations under CC BY 4.0. Each dataset row retains its author, source and
landing URLs, license URL, Open Images ID, and crop coordinates. See the dataset
card for attribution and Open Images' per-image license caveat.

## Model and evaluation

- ResNet18 initialized from torchvision `IMAGENET1K_V1` weights
- Four head-only warmup epochs; then only `layer4` and the two-class head train
- Earlier layers and their BatchNorm state remain frozen
- AdamW, cosine schedule for the fine-tuning stage, cross-entropy with label smoothing
- 30 epochs maximum with validation-based early stopping
- Final checkpoint selected by validation accuracy, with validation loss breaking ties
- Each run evaluates the held-out test once after selection; the corrected model is the second model measured on this benchmark
- Target: at least 95% held-out test accuracy

The script writes `model.pt`, `metrics.json`, `history.json`, an accuracy/loss
SVG, a confusion matrix, held-out prediction records and grid, a model card, and
`.compute-artifact.json` under `$COMPUTE_ARTIFACT_DIR`.

## Compute evidence so far

Two Vast.ai allocations failed before training and cost $0.00. The first failed
with an HTTP 410 while creating an RTX 3090 instance
(`run_fb50b85e773418b2d7ea5163d2d9becb`); the second found no interruptible
RTX 6000 Ada offer (`run_028dd10761dc44afa8123d6c3d44fb8c`). Compute reports
`rpt_7f235359c68c52af6b64e38981c832d7` and
`rpt_e97eb68f61e64b002a2c6e5d74b7a56b` record those failures.

A RunPod A100 baseline (`run_5e61db1cc0b501b3bfa6d18829669c62`)
completed for $0.12. Its randomly initialized 315,426-parameter CNN reached
90.0% validation accuracy and 87.0% held-out test accuracy, below the 95%
target, so its publication gate held. The staged ResNet18 correction above was
chosen from the training/validation history without examining held-out example
identities or images. The aggregate 87% baseline test score was already known.
Its quoted RunPod rate is $1.45/hr, with an approximately
$0.81 timeout-budget estimate for 31 billed minutes.

After explicit approval, corrected run
`run_e1abf4b4c24a49baa0d8f62e2d94a3c3` completed on a RunPod A100 80GB
PCIe for $0.13. It selected epoch 17 at 98.5% validation accuracy and reached
98.25% on the held-out test: 98.98% face precision, 97.5% recall, 98.24% F1,
and confusion matrix `[[198, 2], [5, 195]]`. The target gate passed.

Compute returned the metrics and archived logs but listed no completed artifact,
despite the result naming the machine-local artifact directory. Report
`rpt_b40a746d9514b1a9b543ff003368fc7a` requests recovery. Without the exact
checkpoint and prediction records, model publication remains pending and no
corrected prediction grid is shown. Total issue spend is $0.25.

The production reproduction led to Compute fix `8c7a1d2`: declared artifact
persistence now has bounded retries and must finish before a run can succeed.
That prevents recurrence but cannot restore this run's destroyed checkpoint.

After a successful run:

```bash
compute artifacts list <run_id>
compute artifacts get <run_id> <artifact_id> <version> --out ./weights
compute runs receipt <run_id>
```

The receipt gives the provider, GPU, wall time, and final cost to record in the
issue alongside the held-out metric.

## Article assets

`assets/build_assets.py` deterministically renders the balanced training-sample
grid, task-flow diagram, architecture diagram, and a per-image attribution
ledger from the validated local DatasetDict. It never reads validation or test
images.

`assets/render_results.py` accepts a downloaded artifact directory and renders
publication-ready accuracy/loss,
confusion-matrix, and prediction assets. It verifies the prediction rows against
the reported aggregate metrics before rendering and writes both a human-readable
Markdown credit table and a machine-readable JSONL ledger for every held-out
image shown.

The baseline artifact is rendered under `assets/results` and labeled as the 87%
random-CNN baseline. Because the corrected artifact is missing,
`assets/render_run_evidence.py` renders only its training curves and confusion
matrix from archived logs and exact result JSON under `assets/final-results`.
It records that the corrected checkpoint, prediction records, prediction grid,
and per-image credits remain unavailable.

```bash
python assets/build_assets.py
python assets/render_results.py ./downloaded-artifact --output ./assets/results --model-label "model label"
python assets/render_run_evidence.py ./evidence/run_id --output ./assets/final-results
```

## Publish a qualifying checkpoint

Store a Hugging Face write token, then use the secrets-backed entrypoint. The
script publishes only when held-out test accuracy meets the 95% target.

```bash
compute secrets set hf
compute run train.py::train_and_push --gpu cheap --timeout 1800 --wait
```

`Secret.from_name("hf")` injects the token only into `train_and_push`. The plain
`train` entrypoint needs no secret.

For the completed qualifying run in this record, do not retrain. Publish only
the exact recovered `model.pt` and its generated card after verifying the
Compute artifact manifest and hashes. No model URL or revision exists yet.

## Agent path

Give your agent [`SKILL.md`](./SKILL.md). It adds this job's constraints to the
canonical [Compute skill](https://compute.cx/SKILL.md).

## Account caps

- One active run per account
- Six run creates per hour
- A balance at or below $1 blocks new runs
- Provider capacity can refuse a create even with sufficient balance
