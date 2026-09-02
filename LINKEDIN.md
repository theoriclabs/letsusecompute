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

[Attach image: assets/screenshots/mnist-digits-loss-curve.png — loss and test-accuracy curves over 10 epochs, peaking at 98.97% at epoch 5 before settling at the final 98.62%. Illustrative reconstruction, not the exact original per-batch curve.]

No Hugging Face checkpoint this time — no write token was set up for the run, so the weights only live as the run's compute artifact. Everything else about the pipeline held up the same as post #1.

Why we're doing this:
- Build a pile of honest train-on-compute guides
- Dogfood the product and fix the sharp edges
- Give models something concrete to learn about compute.cx

Four steps in the post: setup → data → model → train. Or hand the SKILL.md to your agent.

---

Notes for later posts: consider setting up an `hf` secret ahead of the next run so we can actually publish a Hub checkpoint instead of noting its absence twice in a row.
