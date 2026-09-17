# Jev-style Doom on compute.cx

Real ViZDoom Defend the Center. Freedoom frames. Structured monster bearings from the engine, not pixels. A fitted aim-cone controller averages 9 kills. Random averages 1.1.

Guide: https://letsusecompute.com/posts/jev-doom

## Quick start

```bash
curl -fsSL https://raw.githubusercontent.com/theoriclabs/letsusecompute/main/posts/jev-doom/train.py -o train.py
compute run train.py::train --gpu cheap --dry-run
compute run train.py::train --gpu L4 --timeout 2400 --wait --args '{"sample": true}'
compute run train.py::train --gpu L4 --timeout 5400 --wait
```

`--gpu cheap` currently quotes the SKU written in `@app.function`, which may be sold out. Pass an available cheap SKU from `compute gpu list` if create is refused.

## This guide’s run

- Scenario: ViZDoom `defend_the_center.cfg`, Freedoom WAD bundled with `vizdoom==1.3.0`. No ROM to supply.
- Observation: nearest-monster distance and bearing from `state.objects`; optional visible box from `state.labels`. Menu: turn left, turn right, attack.
- Sample: `run_577746f2c18b58372a8c404c2b8f2a1c` Vast L4, $0.03. Offline agree 0.662. Qwen 0 kills (always turn right). Threshold 10.3 kills. `reload_ok`
- Full: `run_6390d2547371c10b15795294dfe811e9` Vast L4, lost after head epoch 1, $0.02
- Teacher / fitted cone (8°): 9.0 mean kills, 8/8 episodes with ≥3 kills
- Random: 1.1 mean kills
- Revision spend $0.05
- Video is a recorded engine replay, not live inference

## Observation contract

| Field | Meaning |
|---|---|
| `bearing` | signed degrees from player facing to the nearest monster |
| `dist` | Euclidean world distance to that monster |
| `visible` / `offset` | on-screen label, if any |
| Menu | turn left, turn right, attack |

The Qwen model never sees the RGB tensor. Replays are the real Freedoom frames.

## Licenses

[ViZDoom](https://github.com/Farama-Foundation/ViZDoom) and the bundled [Freedoom](https://freedoom.github.io/) WADs. We do not vendor engine source.
