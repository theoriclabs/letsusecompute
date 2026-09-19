# Chess move transformer on compute.cx

Train a small GPT on SAN move tokens from public Lichess games — the model never sees the rules, only games — then play it against a random mover and Stockfish skill 1 on a fresh cloud GPU via [compute.cx](https://compute.cx).

Guide: https://letsusecompute.com/posts/chess-transformer

Weights: https://huggingface.co/theoriclabs/chess-move-transformer (headline B, the 16M run, pushed 08:58 UTC after run A)

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

`--args '{"sample": true}'` is the pipeline preset: 4 layers / 256-wide / 4 heads, 100k positions, 20 games per opponent. The script downloads Stockfish 17.1 as a pinned Linux binary (the Compute image has no apt).

Headline B (8 layers / 512-wide / 8 heads, 16M positions, 3 epochs, batch 64, lr 4e-4, 100 games per opponent):

```bash
compute run train.py::train --gpu H100-PCIe --provider runpod --timeout 4500 --wait --args '{"max_positions": 16000000, "epochs": 3, "batch_size": 64, "lr": 0.0004}'
```

Omit the `--args` object on that line for headline A (4M positions, 8 epochs). Timeout 4500 covers download, training, eval, and 200 matches.

To publish weights to Hugging Face, store a write token and use the secrets-backed entrypoint:

```bash
compute secrets set hf
compute run train.py::train_and_push --gpu H100-PCIe --provider runpod --timeout 4500 --wait --args '{"max_positions": 16000000, "epochs": 3, "batch_size": 64, "lr": 0.0004}'
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
- Headline model: 8-layer decoder, d_model 512, 8 heads, block 256. 28,386,304 params at 4M positions (vocab 5,928); 29,502,976 params at 16M positions (vocab 8,109)
- Sample preset: 4 layers, 256-wide, 4 heads, 100k positions, 3,787,264 params
- Metrics: next-move top-1, unmasked legal-move rate (sample and argmax), W/D/L vs random and vs Stockfish skill 1 (`Limit(time=0.02)`)

## This guide's run

All five jobs, $2.14 of a $20 budget.

| run | what | GPU | billed | cost |
|---|---|---|---|---|
| `run_ef3aa0058ce5f3699e73a9d57fb259af` | first pipeline attempt, `--gpu cheap` | Vast RTX-3090 | 17 min hung in "provider reports creating", cancelled | $0.05 |
| `run_496b871f1d3790e0eef6c77447fae548` | second attempt | Vast L4 | boot_failed: no interruptible L4 offers | $0.00 |
| `run_0ffa481485213ea2e182a409b3b6fc86` | sample preset, `--args '{"sample": true}'` | RunPod A100-PCIe-80GB | 3 min | $0.07 |
| `run_b9d92718d37fbe9b36c31ad2e21815d8` | headline A: 4.08M positions, 8 epochs, 13,440 steps | RunPod H100-PCIe | 25 min | $1.08 |
| `run_dc7002880b6daab7cb3cb6f61e3e063c` | headline B: 16.3M positions, 3 epochs, 10,143 steps | RunPod H100-PCIe | 22 min | $0.94 |

Headline B (the numbers on the guide): top-1 38.55% on 328,530 holdout positions; legal sampled 84.84% (n=5,000); legal argmax 94.26%; vs random 50 W / 50 D / 0 L; vs Stockfish 2 W / 10 D / 88 L; 1,290 illegal attempts across 200 games.

Headline A: top-1 34.18% on 82,506 positions; legal sampled 79.02%; legal argmax 90.30%; vs random 40 W / 59 D / 1 L; vs Stockfish 0 W / 9 D / 91 L; 2,182 illegal attempts.

The ticket asked for legal rate ≥ 99% and ≥ 95 wins of 100 vs random. Neither is met.

Friction: `rpt_d5dd2c8585511c999fd81010a5d14301` (no non-interactive price quote), `rpt_023af546ab619039ec080dba1279136f` (`--gpu cheap` used the source SKU after `workload_estimate_unsupported`), `rpt_b9e65fd29961222fe7239f378aa010b4` (hung Vast RTX-3090 boot), `rpt_022179b60e28c3aacba224f311769068` (catalog L4 with no offers), `rpt_259d10a7adcf0d20e2dd795c43a3d72e` (`compute logs -f` crash).

## Agent path

Point your agent at [`SKILL.md`](./SKILL.md) and ask it to train the chess move transformer on compute.

## Caps to know

- Two active runs per account
- New accounts: first-day spend cap (~$50)
- Balance ≤ $1 blocks new runs
- Weights via artifacts are kept for a limited window
