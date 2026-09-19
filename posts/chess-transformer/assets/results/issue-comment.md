Draft comment for [theoriclabs/letsusecompute#9](https://github.com/theoriclabs/letsusecompute/issues/9). Do not post from the agent.

## Runs

| run | what | GPU / provider | billed | cost |
|---|---|---|---|---|
| `run_ef3aa0058ce5f3699e73a9d57fb259af` | first pipeline attempt, `--gpu cheap` | Vast RTX-3090 | 17 min hung in "provider reports creating", cancelled | $0.05 |
| `run_496b871f1d3790e0eef6c77447fae548` | second attempt | Vast L4 | `boot_failed`: no interruptible L4 offers | $0.00 |
| `run_0ffa481485213ea2e182a409b3b6fc86` | sample preset, `--args '{"sample": true}'` (4/256, 100k positions, 126 steps, 20+20 games) | RunPod A100-PCIe-80GB | 3 min | $0.07 |
| `run_b9d92718d37fbe9b36c31ad2e21815d8` | headline A: 4.08M positions, 28,386,304 params, 8 epochs, 13,440 steps in 1,108 s, 100+100 games | RunPod H100-PCIe | 25 min | $1.08 |
| `run_dc7002880b6daab7cb3cb6f61e3e063c` | headline B: 16,326,581 positions, 29,502,976 params, 3 epochs, batch 64, lr 4e-4, 10,143 steps in 837 s, 100+100 games | RunPod H100-PCIe | 22 min | $0.94 |

**Total issue spend: $2.14** of the $20 budget.

Weights: https://huggingface.co/theoriclabs/chess-move-transformer holds headline B (`model.pt`, `vocab.json`, `results.json`), pushed 08:58 UTC after run A.

Guide: https://letsusecompute.com/posts/chess-transformer

## Metrics

| | Sample | Headline A | Headline B |
|---|---|---|---|
| params | 3,787,264 | 28,386,304 | 29,502,976 |
| positions | 102,098 | 4,081,734 | 16,326,581 |
| top-1 | 2.59% (n=2,082) | 34.18% (n=82,506) | 38.55% (n=328,530) |
| legal sampled | 8.89% (n=2,082) | 79.02% (n=5,000) | 84.84% (n=5,000) |
| legal argmax | 9.70% | 90.30% | 94.26% |
| vs random W/D/L | 0 / 18 / 2 | 40 / 59 / 1 | 50 / 50 / 0 |
| vs Stockfish W/D/L | 0 / 1 / 19 | 0 / 9 / 91 | 2 / 10 / 88 |
| illegal attempts (200 games; 40 for sample) | 2,273 | 2,182 | 1,290 |

Draws vs random, headline B, from replaying the PGNs with python-chess: 50 = 39 threefold repetition, 10 stalemate, 1 ply cap. Material at the end: model ahead by ≥3 in 23, behind in 24, level in 3. Median 103 plies. vs Stockfish: 10 draws, all threefold repetition.

## Friction

- `rpt_d5dd2c8585511c999fd81010a5d14301` — CLI has no non-interactive price quote; `--dry-run` is an upload plan, not a quote
- `rpt_023af546ab619039ec080dba1279136f` — `--gpu cheap` silently used the SKU declared in the source after `workload_estimate_unsupported`
- `rpt_b9e65fd29961222fe7239f378aa010b4` — hung Vast RTX-3090 boot billed 17 minutes; one report per run blocks filing distinct issues
- `rpt_022179b60e28c3aacba224f311769068` — catalog listed a Vast L4 that had no offers
- `rpt_259d10a7adcf0d20e2dd795c43a3d72e` — `compute logs -f` crashes after ~15 min with `ValueError: invalid literal for int() with base 16: b''`

Stockfish via apt is impossible (`compute.Image` has no apt API). The pinned GitHub 17.1 binary works. The `compute report` summary is capped at 200 characters.

## Done-when

The issue said done at legal rate ≥ 99% and ≥ 95 wins of 100 vs random. Headline B is 94.26% legal by argmax (84.84% sampled) and 50 wins vs random. Neither bar is met. 16 million positions moved every number relative to the 4M run and still left a model that tracks the board imperfectly (illegal argmax, then the highest legal alternative) and, on a deterministic argmax, repeats instead of converting extra material. Chess-GPT-style results sit at hundreds of millions of positions. No Elo.
