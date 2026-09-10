# Model Router on compute.cx

Train a small classifier that looks at a prompt and decides whether Mixtral is good enough or GPT-4 is needed — then read the dollars saved at a fixed quality target — on a fresh cloud GPU via [compute.cx](https://compute.cx).

Guide: https://letsusecompute.com/posts/model-router

## Quick start

```bash
curl -fsSL https://raw.githubusercontent.com/theoriclabs/letsusecompute/main/posts/model-router/train.py -o train.py

curl -fsSL https://compute.cx/install.sh | sh
compute setup
compute credits add 10

compute run train.py::train --gpu cheap --dry-run
compute run train.py::train --gpu cheap --timeout 3600 --wait
```

`--dry-run` only prints the upload plan. Cost and GPU show up on the real run, in the preflight quote, before you confirm spend.

A 10% sample is enough to check the metric code:

```bash
compute run train.py::train --gpu cheap --timeout 3600 --wait --args '{"max_train": 10900, "max_eval": 1000, "epochs": 1}'
```

To publish weights to Hugging Face, store a write token and use the secrets-backed entrypoint:

```bash
compute secrets set hf
compute run train.py::train_and_push --gpu cheap --timeout 3600 --wait
```

`compute secrets set hf` stores the token. `Secret.from_name("hf")` on `train_and_push` is what injects it into the job. Storing the secret alone does not put it on the machine.

After the run succeeds and the machine is gone:

```bash
compute artifacts list <run_id>
compute artifacts get <run_id> <artifact_id> <version> --out ./weights
```

## What this trains

- Label: `cheap_suffices` when Mixtral's judge score is at least 4 / 5 on [`routellm/gpt4_dataset`](https://huggingface.co/datasets/routellm/gpt4_dataset)
- Baseline: TF-IDF (1–2 grams, 50k features) + logistic regression
- Model: DistilBERT (`distilbert-base-uncased`, ~67M params) with a two-class head
- Metric that matters: the cost-vs-quality curve at 95% of always-GPT-4 quality, using public list prices (Mixtral 8x7B $0.24 / 1M tokens; GPT-4 Turbo $10 / $30 per 1M)
- Dataset is downloaded on the machine, not uploaded by you

## This guide's run

- Run id: `run_2538f6a593a1fedfce8b4d4ce694c3d0`
- Provider / GPU: vastai, RTX-3090
- Billed: 21 minutes of training, $0.07 (sample + boot-fail + headline = $0.08 of the $15 issue budget)
- DistilBERT accuracy: 76.88% (TF-IDF 74.52%)
- At 95% of always-GPT-4 quality: 67.8% of traffic to Mixtral, 72.0% list-price cut vs always GPT-4

The function returned `ok: true`. Artifact PUT then failed with HTTP 411, so the run is marked failed and there is no downloadable Compute artifact or Hub checkpoint. The result JSON persisted; that is what the guide page uses. File a write token and use `train_and_push` if you want a Hub checkpoint once artifacts work.

## Agent path

Point your agent at [`SKILL.md`](./SKILL.md) and ask it to train the prompt router on compute.

## Caps to know

- One active run per account
- New accounts: first-day spend cap (~$50)
- Balance ≤ $1 blocks new runs
- Weights via artifacts are kept for a limited window
