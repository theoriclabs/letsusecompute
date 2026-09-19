# Chess move transformer on compute.cx

Train a small GPT on SAN move tokens from public Lichess games — the model never sees the rules, only games — then play it against a random mover and Stockfish skill 1 on a fresh cloud GPU via [compute.cx](https://compute.cx).

Guide: https://letsusecompute.com/posts/chess-transformer

## Quick start

```bash
curl -fsSL https://raw.githubusercontent.com/theoriclabs/letsusecompute/main/posts/chess-transformer/train.py -o train.py

curl -fsSL https://compute.cx/install.sh | sh
compute setup
compute credits add 10

compute run train.py::train --gpu cheap --dry-run
compute run train.py::train --gpu cheap --timeout 1800 --wait --args '{"sample": true}'
```

`--dry-run` only prints the upload plan. Cost and GPU show up on the real run, in the preflight quote, before you confirm spend.

`--args '{"sample": true}'` is the pipeline preset: 4 layers / 256-wide / 4 heads, 100k positions, 20 games per opponent.

Headline run (8 layers / 512-wide / 8 heads, 4M positions, 100 games per opponent):

```bash
compute run train.py::train --gpu H100-SXM --timeout 3600 --wait
```

To publish weights to Hugging Face, store a write token and use the secrets-backed entrypoint:

```bash
compute secrets set hf
compute run train.py::train_and_push --gpu cheap --timeout 1800 --wait --args '{"sample": true}'
```

`compute secrets set hf` stores the token. `Secret.from_name("hf")` on `train_and_push` is what injects it into the job. Storing the secret alone does not put it on the machine. `train` works with no token.

After the run succeeds and the machine is gone:

```bash
compute artifacts list <run_id>
compute artifacts get <run_id> <artifact_id> <version> --out ./weights
```

## What this trains

- Data: one recent month of [`Lichess/standard-chess-games`](https://huggingface.co/datasets/Lichess/standard-chess-games), streamed on the machine. Filter: both Elos ≥ 1800, ≥ 20 plies, drop Abandoned / Rules infraction. Fallback: [`adamkarvonen/chess_games`](https://huggingface.co/datasets/adamkarvonen/chess_games) `lichess_100mb.zip`
- Tokens: one SAN move (`Nf3`, `O-O`, `e8=Q`). Vocab from the training games plus PAD/BOS/EOS/UNK
- Headline model: 8-layer decoder, d_model 512, 8 heads, block 256, ~25M parameters
- Sample preset: 4 layers, 256-wide, 4 heads, 100k positions
- Metrics: next-move top-1, unmasked legal-move rate (sample and argmax), W/D/L vs random and vs Stockfish skill 1 (`Limit(time=0.02)`)

## This guide's run

TODO after runs

## Agent path

Point your agent at [`SKILL.md`](./SKILL.md) and ask it to train the chess move transformer on compute.

## Caps to know

- One active run per account
- New accounts: first-day spend cap (~$50)
- Balance ≤ $1 blocks new runs
- Weights via artifacts are kept for a limited window
