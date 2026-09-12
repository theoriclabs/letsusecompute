# AI-text detector on compute.cx

Train a small classifier that tells human-written answers from ChatGPT answers — then measure how badly it fails on a different generator — on a fresh cloud GPU via [compute.cx](https://compute.cx).

Guide: https://letsusecompute.com/posts/ai-text-detector

## Quick start

```bash
curl -fsSL https://raw.githubusercontent.com/theoriclabs/letsusecompute/main/posts/ai-text-detector/train.py -o train.py

curl -fsSL https://compute.cx/install.sh | sh
compute setup
compute credits add 10

compute run train.py::train --gpu cheap --dry-run
compute run train.py::train --provider runpod --gpu A100-PCIe-80GB --timeout 2400 --wait
```

`--dry-run` only prints the upload plan. Cost and GPU show up on the real run, in the preflight quote, before you confirm spend.

`--gpu cheap` is the issue default. On this account it quoted the decorator SKU and Vast had no interruptible offers, so the guide run used RunPod A100-PCIe-80GB.

A short sample is enough to check the metric code:

```bash
compute run train.py::train --provider runpod --gpu A100-PCIe-80GB --timeout 2400 --wait --args '{"max_train": 2000, "max_eval": 400, "epochs": 1, "ood_per_model": 40, "ood_scan": 20000}'
```

To publish weights to Hugging Face, store a write token and use the secrets-backed entrypoint:

```bash
compute secrets set hf
compute run train.py::train_and_push --provider runpod --gpu A100-PCIe-80GB --timeout 2400 --wait
```

`compute secrets set hf` stores the token. `Secret.from_name("hf")` on `train_and_push` is what injects it into the job. Storing the secret alone does not put it on the machine.

After the run succeeds and the machine is gone:

```bash
compute artifacts list <run_id>
compute artifacts get <run_id> <artifact_id> <version> --out ./weights
```

## What this trains

- Label: `ai` vs `human` on [`Hello-SimpleAI/HC3`](https://huggingface.co/datasets/Hello-SimpleAI/HC3) English answers. One human answer and one ChatGPT answer per question. The script downloads the dataset on the machine.
- Held-out slice: 10% of non-medicine questions, split by question so both answers stay together.
- Domain holdout: every `medicine` question.
- Different generator: a streamed RAID slice (`liamdugan/raid`) of human / GPT-4 / Llama-chat text with no adversarial attacks.
- Baseline: TF-IDF (1–2 grams, 50k features) + logistic regression.
- Model: DistilBERT (`distilbert-base-uncased`, ~67M params) with a two-class head.
- Metric that matters: accuracy and the false-positive rate on human text.

## This guide's run

- Sample: `run_3eaf70692b424b122f8656610c28d501` on RunPod A100-PCIe-80GB, $0.08
- Headline: `run_ea7b0d5c56f928808c3096a232e3e0bd` on RunPod A100-SXM-80GB, 9 billed training minutes, $0.30
- Issue spend: $0.38 (boot failures were $0)
- DistilBERT on 4,502 HC3 holdout answers: 99.42% accuracy, 1.02% human false-positive rate (TF-IDF 97.07% / 2.44%)
- Medicine holdout: 98.11% accuracy, 3.46% human FPR
- RAID OOD (200 human / 200 GPT-4 / 200 Llama-chat): 48.33% accuracy, 0.50% human FPR, 22.75% AI recall

Artifacts persisted. No Hub checkpoint — no `hf` secret was set. File a write token and use `train_and_push` if you want `theoriclabs/ai-text-detector-small`.

## Agent path

Point your agent at [`SKILL.md`](./SKILL.md) and ask it to train the detector on compute.

## Caps to know

- One active run per account
- New accounts: first-day spend cap (~$50)
- Balance ≤ $1 blocks new runs
- Weights via artifacts are kept for a limited window
