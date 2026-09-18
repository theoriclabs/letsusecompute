# Fine-tune Qwen with LoRA on compute.cx

Homemade job-card JSONL. Qwen3-0.6B plus LoRA rank 16. Twenty held-out prompts, scored before and after.

Guide: https://letsusecompute.com/posts/finetune-qwen-lora

## Quick start

```bash
curl -fsSL https://raw.githubusercontent.com/theoriclabs/letsusecompute/main/posts/finetune-qwen-lora/train.py -o train.py
compute run train.py::train --gpu cheap --dry-run
compute run train.py::train --gpu cheap --timeout 1800 --wait --args '{"sample": true}'
compute run train.py::train --gpu cheap --timeout 3600 --wait
```

`train.py::train` does not see the Hugging Face secret. Storing `compute secrets set hf` is not enough; `train_and_push` is the function that lists it.

## Swap in your own data

The builtin set is four-line job cards (`gpu`, `timeout_s`, `command`, `note`). Replace it by publishing a JSONL of `{"user","assistant"}` rows and pointing the script at that repo once you fork the generator.

## This guide’s run

- Sample: `run_236bc844e5cda41ee3af6464daa15558` Vast L4, $0.03. Before 0/20, after 12/20, eval loss 0.415
- Full: `run_18c4da14109e36d7cefc654c7a9d7e05` Vast L4, $0.03. Two epochs, 414/46 rows. Before 0/20, after 18/20, eval loss 0.051
- Hung 3090 boot cancelled $0 (`rpt_6768d7f550ed072120730df3121c76d6`). Earlier full L4 HTTP 400 (`rpt_cdc45ea3a1798c18fe074109a7c2dde7`). ADA no offers (`rpt_5f049580d1aad2cad58ac0cd0d357249`).
- No `hf` secret on the account, so no Hub push
- Issue spend $0.06
