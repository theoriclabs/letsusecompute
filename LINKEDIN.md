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

## 2026-09-10 — Model router

Draft (post when the live guide URL is the one you want people to open):

---

Post #5 in the compute.cx series: a prompt router.

Most prompts do not need the expensive model. We trained DistilBERT — plus a TF-IDF baseline — on RouteLLM’s Mixtral-vs-GPT-4 scores to decide which ones do.

On 10,000 held-out prompts the DistilBERT router keeps 95% of always-GPT-4 quality, sends 68% of traffic to Mixtral, and cuts list-price cost 72%. Always-Mixtral is cheaper still and misses the quality bar. The TF-IDF baseline lands on the same 95% line and spends a bit more.

Headline run: Vast.ai RTX 3090, 21 billed minutes, $0.07. Whole issue $0.08, including a 10% sample and a boot that never came up.

One sharp edge, again: the job returned ok=true and the metrics, then artifact PUT failed with HTTP 411. No downloadable weights. The result JSON persisted, which is what the post uses.

Guide: https://letsusecompute.com/posts/model-router

#MachineLearning #GPUComputing

---

## 2026-09-12 — AI-text detector

Draft (post when the live guide URL is the one you want people to open):

---

Post #6 in the compute.cx series: an AI-text detector.

We trained DistilBERT — plus a TF-IDF baseline — on HC3 to tell human answers from ChatGPT. On 4,502 held-out answers it scores 99.4% and flags 1.0% of the human text. Medicine, still ChatGPT, stays at 98%. Then we changed the generator.

On a 600-row RAID slice of GPT-4, Llama-chat, and human text, accuracy is 48%. DistilBERT catches 23% of the new-generator answers and almost none of the humans. That is as far as $0.38 got us from a product like Pangram.

Headline run: RunPod A100-SXM, 9 billed training minutes, $0.30. Whole issue $0.38, including a $0.08 sample. Artifacts persisted this time. No Hub checkpoint — no write token was set.

Guide: https://letsusecompute.com/posts/ai-text-detector

#MachineLearning #GPUComputing

---

## 2026-09-14 — Text compressor

Draft (post when the live guide URL is the one you want people to open):

---

Post #7 in the compute.cx series: a language-model compressor.

We trained a 3.4M-parameter character GPT on the first 90 MB of enwik8 and used its next-byte probabilities as a range coder. On the last 10 MB the bitstream is 1.82 bits per byte. gzip -9 needs 2.87. xz -9 needs 2.14. The first 32 KB of decode matched the original.

Count the 6.7 MB of weights and it loses to gzip. That is the difference between a shared codec and a file you send once.

Headline run: RunPod A100-SXM, 28 billed training minutes, $0.91. Whole issue $1.34, including a $0.21 sample and two failed creates. Artifacts persisted. No Hub checkpoint — no write token was set.

Guide: https://letsusecompute.com/posts/text-compressor

#MachineLearning #GPUComputing

---

## 2026-09-17 — Doom + Pong

Draft (post when the live guide URL is the one you want people to open):

---

Post #8: can a small model learn to play Doom and Pong?

We trained a half-billion-parameter model to choose the next button to press. In Pong, it copied 98.3% of the teacher’s offline answers—and still lost 0–5. In Doom, the sample model kept turning right. It never fired.

Two simple rules fitted to the same teaching examples did better: 5–3 in Pong and 9 kills per episode in Doom.

The inspiration was TypeSafe’s Jev: give a model a situation and get a decision your code can use. Jev’s RLCD training aims for calibrated probabilities. Our experiment uses supervised imitation, so it explores the interface without claiming to reproduce Jev.

The combined guide explains the training, the failures, and what the rules learned. Two videos show Doom first, then Pong. Each starts with the saved Qwen model’s real action probabilities and gameplay, then switches to the fitted rule. These are recorded replays; the probabilities express action preferences, not calibrated chances of winning.

GPU runs: $1.62, including failed and cancelled attempts. Doom’s full model run was interrupted; its Qwen result comes from the sample.

Guide: https://letsusecompute.com/posts/jev-games

#MachineLearning #GPUComputing

---
