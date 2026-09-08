"""Bounded, left-padded inference; never execute generated programs locally."""

from __future__ import annotations

import json
import time

from pipeline import dump
from render import render_code


def completion_tokens(tokens, eos_id):
    """Remove only padding after EOS, preserving each example's stopping status."""
    return tokens[: tokens.index(eos_id) + 1] if eos_id in tokens else tokens


def batch_tokens(model, tokenizer, prompts, max_new_tokens=4096):
    import torch

    old_padding = tokenizer.padding_side
    tokenizer.padding_side = "left"
    try:
        inputs = tokenizer(prompts, padding=True, return_tensors="pt", add_special_tokens=False).to(
            model.device
        )
    finally:
        tokenizer.padding_side = old_padding
    with torch.inference_mode():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            use_cache=True,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    width = inputs["input_ids"].shape[1]
    return [completion_tokens(row[width:].tolist(), tokenizer.eos_token_id) for row in outputs]


def generate_batched(model, tokenizer, rows, root, label, *, batch_size=8, max_new_tokens=4096):
    if batch_size < 1:
        raise ValueError("Batch size must be positive")
    model.eval()
    results = []
    for start in range(0, len(rows), batch_size):
        group = rows[start : start + batch_size]
        prompts = [
            tokenizer.apply_chat_template(
                r["messages"][:2], tokenize=False, add_generation_prompt=True, enable_thinking=False
            )
            for r in group
        ]
        begin = time.monotonic()
        completions = batch_tokens(model, tokenizer, prompts, max_new_tokens)
        generation_seconds = time.monotonic() - begin
        for row, tokens in zip(group, completions, strict=True):
            directory = root / label / row["id"]
            directory.mkdir(parents=True, exist_ok=False)
            raw = tokenizer.decode(tokens, skip_special_tokens=True)
            (directory / "raw.txt").write_text(raw)
            exhausted = len(tokens) >= max_new_tokens and tokens[-1] != tokenizer.eos_token_id
            begin = time.monotonic()
            metrics = (
                {
                    "valid": False,
                    "executed": False,
                    "coverage": 0.0,
                    "failures": ["generation_token_limit"],
                }
                if exhausted
                else render_code(raw, directory, row["required"], resolution=384, samples=12)
            )
            metrics.update(
                {
                    "id": row["id"],
                    "family": row["family"],
                    "split": row["split"],
                    "label": label,
                    "output_tokens": len(tokens),
                    "token_limit": exhausted,
                    "seconds": generation_seconds / len(group) + time.monotonic() - begin,
                    "generation_batch_seconds": generation_seconds,
                    "generation_batch_size": len(group),
                    "batch_ids": [r["id"] for r in group],
                }
            )
            dump(directory / "metrics.json", metrics)
            results.append(metrics)
            print(json.dumps({"phase": label, **metrics}), flush=True)
        dump(root / (label + "-results.json"), results)
    return results
