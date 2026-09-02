# MNIST Digits on compute.cx

Train a tiny CNN that reads handwritten digits — the classic MNIST task — on a fresh cloud GPU via [compute.cx](https://compute.cx).

Guide: https://letsusecompute.com/posts/mnist-digits

## Quick start

```bash
curl -fsSL https://raw.githubusercontent.com/theoriclabs/letsusecompute/main/posts/mnist-digits/train.py -o train.py

curl -fsSL https://compute.cx/install.sh | sh
compute setup
compute credits add 10

compute run train.py::train --gpu cheap --dry-run
compute run train.py::train --gpu cheap --timeout 900 --wait
```

`--dry-run` only prints the upload plan. Cost and GPU show up on the real run, in the preflight quote, before you confirm spend.

To publish weights to Hugging Face, store a write token and use the secrets-backed entrypoint:

```bash
compute secrets set hf
compute run train.py::train_and_push --gpu cheap --timeout 900 --wait
```

`compute secrets set hf` stores the token. `Secret.from_name("hf")` on `train_and_push` is what injects it into the job. Storing the secret alone does not put it on the machine.

After the run succeeds and the machine is gone:

```bash
compute artifacts list <run_id>
compute artifacts get <run_id> <artifact_id> <version> --out ./weights
```

## What this trains

- Randomly initialized CNN (~106k params): 2 conv blocks → linear → linear
- Loss: cross-entropy over 10 digit classes
- Default: 10 epochs, 28×28 images, batch size 128
- Dataset: MNIST via `torchvision.datasets.MNIST` (downloaded on the machine, not uploaded by you)

## This guide's run

- Run id: `run_11d87f76dffbea350f2f7e75513aa25e`
- Provider / GPU: vastai, RTX-3090
- Billed: 2 minutes, $0.01 total (of the $3 issue budget)
- Test accuracy: 98.62% final (peak 98.97% at epoch 5, then mild overfit)

This run used the plain `train` entrypoint — no Hugging Face write token was set up for it, so the weights exist only as the run's `mnist-cnn` compute artifact, not on the Hub. Run `compute secrets set hf` and use `train_and_push` if you want a Hub checkpoint.

## Agent path

Point your agent at [`SKILL.md`](./SKILL.md) and ask it to train MNIST digits on compute.

## Caps to know

- One active run per account
- New accounts: first-day spend cap (~$50)
- Balance ≤ $1 blocks new runs
- Weights via artifacts are kept for a limited window; this run's weights were not additionally published to Hugging Face (see above)
