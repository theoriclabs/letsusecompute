# Text compressor on compute.cx

Train a small character language model and use it as a compressor — model plus a range coder — then compare bits per byte against gzip and xz on a fresh cloud GPU via [compute.cx](https://compute.cx).

Guide: https://letsusecompute.com/posts/text-compressor

## Quick start

```bash
curl -fsSL https://raw.githubusercontent.com/theoriclabs/letsusecompute/main/posts/text-compressor/train.py -o train.py

curl -fsSL https://compute.cx/install.sh | sh
compute setup
compute credits add 10

compute run train.py::train --gpu cheap --dry-run
compute run train.py::train --gpu cheap --timeout 2400 --wait
```

`--dry-run` only prints the upload plan. Cost and GPU show up on the real run, in the preflight quote, before you confirm spend.

A 1 MB slice is enough to check the coder:

```bash
compute run train.py::train --gpu cheap --timeout 2400 --wait --args '{"sample": true}'
```

To publish weights to Hugging Face, store a write token and use the secrets-backed entrypoint:

```bash
compute secrets set hf
compute run train.py::train_and_push --gpu cheap --timeout 2400 --wait
```

`compute secrets set hf` stores the token. `Secret.from_name("hf")` on `train_and_push` is what injects it into the job. Storing the secret alone does not put it on the machine.

After the run succeeds and the machine is gone:

```bash
compute artifacts list <run_id>
compute artifacts get <run_id> <artifact_id> <version> --out ./weights
```

## What this trains

- Data: enwik8, first 90 MB train, last 10 MB test, downloaded on the machine
- Model: 4-layer character GPT, 256-wide, 256-byte context, 3,356,160 parameters
- Compression: next-byte probabilities + 32-bit rANS; gzip -9 and xz -9 on the same bytes
- Metric that matters: bits per byte of the bitstream, plus a decode round-trip

## This guide's run

- Run id: `run_ea0ca7f6a97ca6bf787f075bb8dc1af1`
- Provider / GPU: runpod, A100-SXM-80GB
- Billed: 28 minutes of training, $0.91 (issue total $1.34 including samples and two failed creates)
- Model + rANS: 1.817 bits/byte (2,271,393 bytes)
- gzip -9: 2.871 bits/byte
- xz -9: 2.144 bits/byte
- Decode of the first 32,768 holdout bytes matched

The function returned `ok: true` and persisted a five-file artifact. There is no Hub checkpoint — no `hf` secret was set.

## Agent path

Point your agent at [`SKILL.md`](./SKILL.md) and ask it to train the text compressor on compute.

## Caps to know

- One active run per account
- Six run creates per hour
- New accounts: first-day spend cap (~$50)
- Balance ≤ $1 blocks new runs
- Weights via artifacts are kept for a limited window
