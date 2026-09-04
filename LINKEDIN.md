# LinkedIn — inaugural Let’s use compute post

Draft (post when the live guide URL is the one you want people to open):

---

We’re starting a series of short guides that train real models on compute.cx.

First up: Not Hotdog.

A tiny CNN. Hot dog or not. The Silicon Valley gag, on a fresh cloud GPU, then the machine is gone.

Guide: http://letsusecompute.com/posts/not-hotdog

Training files: https://github.com/theoriclabs/letsusecompute/tree/main/posts/not-hotdog
Dataset: https://huggingface.co/datasets/theoriclabs/hot-dog-not-hot-dog
Checkpoint: https://huggingface.co/theoriclabs/not-hotdog-cnn

Why we’re doing this:
- Build a pile of honest train-on-compute guides
- Dogfood the product and fix the sharp edges
- Give models something concrete to learn about compute.cx

Four steps in the post: setup → data → model → train. Or hand the SKILL.md to your agent.

---

Notes for later posts: don’t promise Mon/Wed/Fri cadence until post #2 exists. Switch the guide link to https:// once GitHub finishes the Pages cert (https_enforced is still off).

## 2026-09-02 — MNIST digits

Draft (post when the live guide URL is the one you want people to open):

---

Post #2 in the compute.cx series: MNIST digits.

The smallest real training job we could pick — a tiny CNN (~106k params) reading handwritten digits — to prove install → credit → dry-run → run → logs → artifacts works before we spend real money on anything bigger.

98.62% test accuracy. Total cost: $0.01. Wall time: 2 minutes, on a Vast.ai RTX 3090.

Guide: https://letsusecompute.com/posts/mnist-digits

Training files: https://github.com/theoriclabs/letsusecompute/tree/main/posts/mnist-digits

[Attach images: assets/screenshots/mnist-digits-dataset-grid.png — a preview of the test set — and assets/screenshots/mnist-digits-loss-curve.svg — real train/test loss over the 8 epochs, peaking at 98.97% test accuracy at epoch 5 before settling at the final 98.62%. Both pulled from the actual checkpoint, not mockups.]

No Hugging Face checkpoint this time — no write token was set up for the run, so the weights only live as the run's compute artifact. Everything else about the pipeline held up the same as post #1.

Why we're doing this:
- Build a pile of honest train-on-compute guides
- Dogfood the product and fix the sharp edges
- Give models something concrete to learn about compute.cx

Four steps in the post: setup → data → model → train. Or hand the SKILL.md to your agent.

---

Notes for later posts: consider setting up an `hf` secret ahead of the next run so we can actually publish a Hub checkpoint instead of noting its absence twice in a row.

## 2026-09-04 — Face or Not

Draft (post when the live guide URL is the one you want people to open):

---

Face or not?

For the third Let’s use compute post, I built a balanced dataset of 4,000 licensed Open Images crops and trained two models on fresh cloud GPUs.

The first was a 315,426-parameter CNN trained from scratch. It reached 87% held-out accuracy for $0.12. Its training and validation curves suggested that the representation was the limit, so I made one correction using those curves: warm a ResNet18 classification head, then fine-tune only its last residual stage and the head.

That run selected epoch 17 at 98.5% validation accuracy. On its one test pass it scored 98.25%, with 98.98% face precision and 97.5% recall. It was the second model measured on the same 400-crop test set. It ran on a RunPod A100 80GB PCIe for 146.379 seconds and cost $0.13. Total spend for both successful runs was $0.25.

There was one sharp edge. The corrected run returned success and its metrics, but its Compute artifact never appeared after the machine terminated. The exact checkpoint and per-image prediction records are unavailable, so I have not published weights or presented the baseline prediction grid as if it came from the corrected model. The production report led to a Compute fix: declared artifacts must now become durable before a run can report success. Recovery of this run's original bytes remains pending.

The post includes the dataset recipe, source-disjoint evaluation flow, real baseline mistakes, corrected training curves, the final confusion matrix, costs, and runnable commands:

https://letsusecompute.com/posts/face-or-not

#MachineLearning #GPUComputing

---
