"""Completion-only Qwen LoRA training, paired evaluation and durable evidence."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import math
import os
import random
import shutil
import tarfile
import time
from pathlib import Path

from export import bundle_result
from render import render_code
from rooms import FAMILIES, MODEL, REVISION, write_dataset


def dump(path: Path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def encode_row(row, tokenizer, max_length):
    prefix = tokenizer.apply_chat_template(
        row["messages"][:2], tokenize=False, add_generation_prompt=True, enable_thinking=False
    )
    prompt_ids = tokenizer(prefix, add_special_tokens=False)["input_ids"]
    completion_ids = tokenizer(row["messages"][2]["content"], add_special_tokens=False)["input_ids"]
    completion_ids = completion_ids + [tokenizer.eos_token_id]
    ids = prompt_ids + completion_ids
    if len(ids) > max_length:
        raise ValueError(
            f"{row['id']}: {len(ids)} tokens exceed {max_length}; refusing to truncate bpy"
        )
    return {
        "input_ids": ids,
        "attention_mask": [1] * len(ids),
        "labels": [-100] * len(prompt_ids) + completion_ids,
    }


def collate(batch, pad_token_id):
    import torch

    length = max(len(row["input_ids"]) for row in batch)
    return {
        key: torch.tensor([row[key] + [fill] * (length - len(row[key])) for row in batch])
        for key, fill in [("input_ids", pad_token_id), ("attention_mask", 0), ("labels", -100)]
    }


def lora_targets(model):
    import torch

    # Target both DeltaNet and full-attention/MLP projections, excluding vision
    # and the enormous vocabulary output head. Resolve actual module names.
    targets = [
        name
        for name, module in model.named_modules()
        if isinstance(module, torch.nn.Linear) and ".language_model.layers." in "." + name
    ]
    if not targets:
        raise RuntimeError("No Qwen text backbone projection modules found")
    return targets


def artifact_marker(root, metadata):
    identity = {
        "model": MODEL,
        "revision": REVISION,
        "dataset": metadata["dataset"],
        "mode": metadata["mode"],
        "schema": 1,
    }
    key = (
        "blender-rooms-" + hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
    )
    dump(
        root / ".compute-artifact.json",
        {
            "version": 1,
            "kind": "output",
            "name": "blender-rooms",
            "compatibility_key": key,
            "metadata": metadata,
        },
    )


def select_balanced(rows, count):
    if count < 4 or count % 4 or count > len(rows):
        raise ValueError("Evaluation count must be a multiple of four, between 4 and split size")
    return [
        row
        for family in FAMILIES
        for row in [r for r in rows if r["family"] == family][: count // 4]
    ]


def generate(model, tokenizer, rows, root, label, max_new_tokens=4096):
    import torch

    model.eval()
    results = []
    for row in rows:
        directory = root / label / row["id"]
        directory.mkdir(parents=True, exist_ok=True)
        prompt = tokenizer.apply_chat_template(
            row["messages"][:2], tokenize=False, add_generation_prompt=True, enable_thinking=False
        )
        inputs = tokenizer(prompt, return_tensors="pt", add_special_tokens=False).to(model.device)
        start = time.monotonic()
        with torch.inference_mode():
            output = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                use_cache=True,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
        tokens = output[0, inputs["input_ids"].shape[1] :].tolist()
        raw = tokenizer.decode(tokens, skip_special_tokens=True)
        (directory / "raw.txt").write_text(raw)
        # Reject exhausted generations even if a truncated prefix happens to
        # parse: silently incomplete rooms are failures, not shorter successes.
        exhausted = len(tokens) >= max_new_tokens and tokens[-1] != tokenizer.eos_token_id
        if exhausted:
            metrics = {
                "valid": False,
                "executed": False,
                "coverage": 0.0,
                "failures": ["generation_token_limit"],
            }
        else:
            metrics = render_code(raw, directory, row["required"], resolution=384, samples=12)
        metrics.update(
            {
                "id": row["id"],
                "family": row["family"],
                "split": row["split"],
                "label": label,
                "output_tokens": len(tokens),
                "token_limit": exhausted,
                "seconds": time.monotonic() - start,
            }
        )
        dump(directory / "metrics.json", metrics)
        results.append(metrics)
        print(json.dumps({"phase": label, **metrics}), flush=True)
    dump(root / (label + "-results.json"), results)
    return results


def paired_review(root, rows, baseline, trained):
    rng = random.Random(72391)
    blinded = []
    key = []
    for i, row in enumerate(rows):
        order = ["base", "sft"]
        rng.shuffle(order)
        paths = {label: str(Path(label) / row["id"] / "render.png") for label in order}
        blinded.append(
            {
                "pair_id": f"pair-{i:03d}",
                "prompt": row["messages"][1]["content"],
                "A": paths[order[0]],
                "B": paths[order[1]],
                "preference": None,
                "recognizable_furniture": None,
                "detail": None,
                "notes": "",
            }
        )
        key.append({"pair_id": f"pair-{i:03d}", "A": order[0], "B": order[1], "id": row["id"]})
    # The review export copies images to neutral names; never show this key to
    # reviewers until they have recorded their choices.
    import shutil

    review = root / "review"
    review.mkdir(exist_ok=True)
    for pair in blinded:
        for side in ("A", "B"):
            source = root / pair[side]
            name = pair["pair_id"] + "-" + side + ".png"
            if source.exists():
                shutil.copy2(source, review / name)
                pair[side] = name
            else:
                pair[side] = None
    dump(review / "pairs.json", blinded)
    dump(root / "review-key.json", key)
    summary = {
        label: {
            "count": len(result),
            "valid": sum(r["valid"] for r in result),
            "valid_fraction": sum(r["valid"] for r in result) / len(result),
            "mean_coverage": sum(r.get("coverage", 0.0) for r in result) / len(result),
        }
        for label, result in [("base", baseline), ("sft", trained)]
    }
    summary["blind_preference"] = "pending human review; no aesthetic claim"
    dump(root / "comparison.json", summary)
    return summary


def training_arguments(root, *, smoke, train_rows, epochs=3.0, use_cpu=False):
    from transformers import TrainingArguments

    steps_per_epoch = math.ceil(train_rows / 8)
    return TrainingArguments(
        output_dir=str(root / "checkpoints"),
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=1 if smoke else 8,
        learning_rate=2e-4,
        num_train_epochs=epochs,
        max_steps=1 if smoke else -1,
        warmup_steps=0 if smoke else math.ceil(steps_per_epoch * epochs * 0.05),
        lr_scheduler_type="cosine",
        bf16=not use_cpu,
        use_cpu=use_cpu,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        logging_steps=1,
        eval_strategy="steps",
        eval_steps=1 if smoke else steps_per_epoch,
        save_strategy="steps",
        save_steps=1 if smoke else steps_per_epoch,
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        report_to="none",
        seed=42,
        data_seed=42,
        prediction_loss_only=True,
        dataloader_num_workers=0,
        optim="adamw_torch",
        remove_unused_columns=False,
    )


def train_impl(*, smoke=False, max_length=6144, epochs=3.0, eval_count=8):
    from functools import partial

    import torch
    from datasets import Dataset
    from peft import LoraConfig, PeftModel, get_peft_model
    from transformers import (
        AutoModelForImageTextToText,
        AutoTokenizer,
        Trainer,
        set_seed,
    )

    if not torch.cuda.is_available():
        raise RuntimeError("An NVIDIA GPU is required for the 9B run")
    if not torch.cuda.is_bf16_supported():
        raise RuntimeError("bf16 support is required")
    if not 0 < epochs <= 5:
        raise ValueError("epochs must be in (0,5]")
    if not 1024 <= max_length <= 8192:
        raise ValueError("max_length must be in [1024,8192]")
    if any(os.getenv(name) for name in ("HF_TOKEN", "hf", "OPENAI_API_KEY", "COMPUTE_API_KEY")):
        raise RuntimeError(
            "Use the secret-free entrypoint: generated bpy must not share a credentialed worker"
        )
    set_seed(42)
    root = Path("/tmp/blender-result-export") / ("smoke" if smoke else "sft")
    root.mkdir(parents=True, exist_ok=True)
    rows, dataset_manifest = write_dataset(root / "data")
    train_rows = [r for r in rows if r["split"] == "train"]
    val_rows = [r for r in rows if r["split"] == "validation"]
    evaluation = select_balanced(val_rows, 4 if smoke else eval_count)
    tokenizer = AutoTokenizer.from_pretrained(MODEL, revision=REVISION, trust_remote_code=False)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    encoded_train = [encode_row(r, tokenizer, max_length) for r in train_rows]
    encoded_val = [encode_row(r, tokenizer, max_length) for r in val_rows]
    # Validate all teacher geometry before loading the expensive base model.
    teacher_results = []
    for row in select_balanced(train_rows, 4) if smoke else train_rows + val_rows:
        result = render_code(
            row["messages"][-1]["content"],
            root / "teacher" / row["id"],
            row["required"],
            render=False,
        )
        teacher_results.append({"id": row["id"], **result})
    dump(root / "teacher-validation.json", teacher_results)
    # Keep the evidence but avoid hundreds of sequential artifact PUT requests.
    with tarfile.open(root / "teacher-validation.tar.gz", "w:gz") as archive:
        archive.add(root / "teacher", arcname="teacher")
    shutil.rmtree(root / "teacher")
    if not all(r["valid"] for r in teacher_results):
        raise RuntimeError("Teacher geometry check failed")
    metadata = {
        "model": MODEL,
        "revision": REVISION,
        "mode": "smoke" if smoke else "sft",
        "dataset": dataset_manifest,
        "seed": 42,
        "gpu": torch.cuda.get_device_name(0),
        "versions": {
            pkg: importlib.metadata.version(pkg)
            for pkg in ["torch", "transformers", "peft", "accelerate", "bpy", "fla-core"]
        },
        "max_length": max_length,
        "epochs": epochs,
        "lora_rank": 4,
        "lora_alpha": 8,
        "export_dtype": "bfloat16",
        "export_transport": "authenticated scoped archive transfer; inline Compute receipt",
        "eval_ids": [r["id"] for r in evaluation],
        "train_max_tokens": max(len(r["input_ids"]) for r in encoded_train),
        "source_sha256": {
            p.name: hashlib.sha256(p.read_bytes()).hexdigest()
            for p in Path(__file__).parent.glob("*.py")
        },
    }
    dump(root / "run-config.json", metadata)
    dump(
        root / "environment.json",
        {
            distribution.metadata["Name"]: distribution.version
            for distribution in importlib.metadata.distributions()
        },
    )
    model = AutoModelForImageTextToText.from_pretrained(
        MODEL,
        revision=REVISION,
        dtype=torch.bfloat16,
        attn_implementation="sdpa",
        trust_remote_code=False,
    ).to("cuda")
    baseline = []
    if not smoke:
        baseline = generate(model, tokenizer, evaluation, root, "base")
    targets = lora_targets(model)
    model = get_peft_model(
        model,
        LoraConfig(
            r=4,
            lora_alpha=8,
            lora_dropout=0.05,
            task_type="CAUSAL_LM",
            target_modules=targets,
            revision=REVISION,
        ),
    )
    model.config.use_cache = False
    model.enable_input_require_grads()
    model.print_trainable_parameters()
    if smoke:
        # One example of each family, not four copies of the first bedroom.
        smoke_ids = {r["id"] for r in select_balanced(train_rows, 4)}
        encoded_train = [
            x for r, x in zip(train_rows, encoded_train, strict=False) if r["id"] in smoke_ids
        ]
        encoded_val = encoded_val[:2]
    args = training_arguments(root, smoke=smoke, train_rows=len(encoded_train), epochs=epochs)
    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=Dataset.from_list(encoded_train),
        eval_dataset=Dataset.from_list(encoded_val),
        data_collator=partial(collate, pad_token_id=tokenizer.pad_token_id),
    )
    start = time.monotonic()
    result = trainer.train()
    if not math.isfinite(result.training_loss):
        raise RuntimeError("Nonfinite training loss")
    adapter_updated = any(
        bool(torch.count_nonzero(parameter.detach()).item())
        for name, parameter in model.named_parameters()
        if "lora_B" in name
    )
    if not adapter_updated:
        raise RuntimeError("No LoRA B weights changed; the optimizer did not update the adapter")
    adapter = root / "adapter"
    trainer.save_model(str(adapter))
    # Save the final adapter in the dtype used by the bf16 forward pass. The
    # exact exported weights are reloaded below before any quality evaluation.
    from safetensors.torch import load_file, save_file

    weight_path = adapter / "adapter_model.safetensors"
    weights = {k: v.to(torch.bfloat16) for k, v in load_file(weight_path).items()}
    save_file(weights, weight_path, metadata={"format": "pt"})
    del weights
    dump(root / "history.json", trainer.state.log_history)
    dump(
        root / "training.json",
        {
            "metrics": result.metrics,
            "seconds": time.monotonic() - start,
            "best_checkpoint": trainer.state.best_model_checkpoint,
            "best_validation_loss": trainer.state.best_metric,
            "targets": targets,
            "adapter_updated": adapter_updated,
        },
    )
    if not smoke:
        # The bootstrap parent transfers selected weights before the longer
        # render comparison. Preserve the model even if evaluation later fails.
        dump(
            root / "early-export.json",
            bundle_result(
                root,
                {
                    "mode": "sft_checkpoint",
                    "base_revision": REVISION,
                    "train_rows": len(encoded_train),
                    "evaluation_complete": False,
                },
            ),
        )
    # Reload the exact on-disk adapter, rather than only testing in-memory weights.
    del trainer, model
    import gc

    gc.collect()
    torch.cuda.empty_cache()
    base = AutoModelForImageTextToText.from_pretrained(
        MODEL,
        revision=REVISION,
        dtype=torch.bfloat16,
        attn_implementation="sdpa",
        trust_remote_code=False,
    ).to("cuda")
    model = PeftModel.from_pretrained(base, adapter)
    model.config.use_cache = True
    trained = generate(
        model,
        tokenizer,
        evaluation[:1] if smoke else evaluation,
        root,
        "sft",
        max_new_tokens=64 if smoke else 4096,
    )
    comparison = (
        {
            "smoke": True,
            "adapter_reload": True,
            "training_loss": result.training_loss,
            "note": "64-token generation is a loading check, not a complete-room evaluation",
        }
        if smoke
        else paired_review(root, evaluation, baseline, trained)
    )
    dump(root / "comparison.json", comparison)
    (adapter / "README.md").write_text(f"""---
base_model: {MODEL}
library_name: peft
tags: [blender, lora, qwen3_5]
---
# Blender rooms — {"smoke adapter; not a trained product" if smoke else "experimental SFT adapter"}

Base revision: `{REVISION}`. Original procedural text-to-bpy demonstrations.
See adjacent run-config.json, training.json and comparison.json for measured
results. Rendered quality requires blind review. Generated Python must execute
only in a disposable environment without credentials. No public benchmark claim.
""")
    return bundle_result(
        root,
        {
            "mode": metadata["mode"],
            "comparison": comparison,
            "base_revision": REVISION,
            "train_rows": len(encoded_train),
        },
    )
