---
name: text-compressor-compute
description: >-
  Extra instructions for the letsusecompute text-compressor guide. Load
  https://compute.cx/SKILL.md first, then apply the specials below.
---

# Text compressor — extras on top of Compute

Canonical Compute skill: `https://compute.cx/SKILL.md`

Use that skill for install, credit, dry-run, quotes, logs, artifacts, and MCP.
This file is only the job-specific overlay.

## Special instructions

- Download `https://raw.githubusercontent.com/theoriclabs/letsusecompute/main/posts/text-compressor/train.py` as `train.py`. Do not invent a script.
- Entrypoint: `train.py::train`
- Dataset: enwik8 (first 90 MB train, last 10 MB test). The script downloads it on the machine. If the Hutter zip is blocked, it falls back to `Salesforce/wikitext` `wikitext-103-raw-v1`.
- Model: a ~3.4M-parameter character GPT plus a 32-bit rANS coder. Compare bits/byte to gzip -9 and xz -9. Decode must match a prefix of the holdout.
- Use `--gpu cheap`. Timeout `2400`. Do not pick H100 or MI300X unless a cheap run cannot boot.
- Dry-run first. Then show the preflight quote and ask before confirming spend.
- After success, show model / gzip / xz bits per byte, the decode-check result, and download artifacts with `compute artifacts list` / `get`.

Guide: https://letsusecompute.com/posts/text-compressor

## Suggested prompt

```
Use https://compute.cx/SKILL.md.

Special instructions:
- Download https://raw.githubusercontent.com/theoriclabs/letsusecompute/main/posts/text-compressor/train.py as train.py. Do not invent a script.
- Entrypoint: train.py::train
- Dataset: enwik8 (90/10 MB), downloaded on the machine
- ~3.4M-parameter character GPT + rANS; compare bits/byte to gzip -9 and xz -9
- Use --gpu cheap. Timeout 2400.
- Dry-run first. Then show the preflight quote and ask before confirming spend.
- After success, show model / gzip / xz bits per byte and the decode-check result.
```
