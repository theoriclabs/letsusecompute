"""Fine-tune Qwen3-0.6B with LoRA on a homemade job-card dataset.

The model should answer GPU-job questions as four fixed lines (gpu,
timeout_s, command, note). Data is generated on the machine; readers can
point --args at their own JSONL later. Supervised SFT via peft + trl.

    compute run train.py::train --gpu cheap --dry-run
    compute run train.py::train --gpu cheap --timeout 1800 --wait --args '{"sample": true}'
    compute run train.py::train --gpu cheap --timeout 3600 --wait
"""

from __future__ import annotations

import json
import os
import random
import tempfile
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import compute

app = compute.App("finetune-qwen-lora")
image = compute.Image.cuda_pytorch().pip_install(
    "--extra-index-url",
    "https://download.pytorch.org/whl/cu124",
    "torch==2.5.1+cu124",
    "numpy",
    "transformers==4.51.3",
    "peft==0.15.2",
    "trl==0.16.1",
    "accelerate==1.6.0",
    "datasets==3.5.0",
    "huggingface_hub>=0.33,<1",
)
hf_secret = compute.Secret.from_name("hf")

DEFAULT_MODEL = "Qwen/Qwen3-0.6B"
DEFAULT_HUB_REPO = "theoriclabs/qwen3-0.6b-jobcards-lora"
DEFAULT_DATASET_REPO = "theoriclabs/compute-job-cards"
WORKLOAD_SUBDIR = "finetune-qwen-lora"
ARTIFACT_NAME = "qwen-jobcards-lora"
ARTIFACT_MARKER = ".compute-artifact.json"
DEFAULT_ARTIFACT_FALLBACK = Path("/tmp/compute-qwen-lora")
MAX_LEN = 384

SYSTEM = (
    "You write Compute job cards. Reply with exactly four lines, no markdown, "
    "in this order:\n"
    "gpu: <cheap|L4|RTX-3090|H100-SXM>\n"
    "timeout_s: <integer>\n"
    "command: <one compute command>\n"
    "note: <one sentence>"
)


def _bridge_hf_token() -> None:
    token = (
        os.environ.get("HF_TOKEN")
        or os.environ.get("HUGGING_FACE_HUB_TOKEN")
        or os.environ.get("hf")
    )
    if token:
        os.environ.setdefault("HF_TOKEN", token)
        os.environ.setdefault("HUGGING_FACE_HUB_TOKEN", token)


def _py(value: Any) -> Any:
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            pass
    if isinstance(value, float):
        return float(value)
    if isinstance(value, int):
        return int(value)
    return value


def write_artifact_marker(
    directory: Path | str,
    *,
    name: str,
    kind: str,
    compatibility_key: str,
    metadata: dict[str, Any],
) -> Path:
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    marker_path = directory / ARTIFACT_MARKER
    payload = {
        "compatibility_key": compatibility_key,
        "kind": kind,
        "metadata": metadata,
        "name": name,
        "version": 1,
    }
    data = json.dumps(payload, sort_keys=True, indent=2, ensure_ascii=False) + "\n"
    fd, tmp_name = tempfile.mkstemp(prefix=".compute-artifact.", suffix=".tmp", dir=str(directory))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp_name, marker_path)
        dir_fd = os.open(str(directory), os.O_RDONLY)
        try:
            os.fsync(dir_fd)
        finally:
            os.close(dir_fd)
    except Exception:
        try:
            os.unlink(tmp_name)
        except OSError:
            pass
        raise
    for leftover in directory.glob(".compute-artifact.*.tmp"):
        try:
            leftover.unlink()
        except OSError:
            pass
    return marker_path


def resolve_artifact_dirs(env: Mapping[str, str] | None = None) -> Path:
    environ = os.environ if env is None else env
    base = environ.get("COMPUTE_ARTIFACT_DIR")
    root = Path(base) / WORKLOAD_SUBDIR if base else DEFAULT_ARTIFACT_FALLBACK
    root.mkdir(parents=True, exist_ok=True)
    return root


def card(gpu: str, timeout_s: int, command: str, note: str) -> str:
    return f"gpu: {gpu}\ntimeout_s: {timeout_s}\ncommand: {command}\nnote: {note}"


def build_examples(seed: int = 0) -> list[dict[str, str]]:
    rng = random.Random(seed)
    scripts = ["train.py", "sft.py", "fit.py", "run.py"]
    entrypoints = ["train", "train_and_push"]
    rows: list[dict[str, str]] = []

    def add(user: str, assistant: str, tag: str) -> None:
        rows.append({"user": user.strip(), "assistant": assistant.strip(), "tag": tag})

    for _ in range(90):
        script = rng.choice(scripts)
        fn = rng.choice(entrypoints)
        add(
            rng.choice(
                [
                    f"First command for {script}::{fn} on a cheap GPU?",
                    f"I have {script}::{fn}. What do I run before spending?",
                    f"How do I dry-run {script}::{fn}?",
                    f"Preflight {script}::{fn} without creating a machine.",
                ]
            ),
            card(
                "cheap",
                1800,
                f"compute run {script}::{fn} --gpu cheap --dry-run",
                "Dry-run prints the upload plan and spends nothing.",
            ),
            "dry_run",
        )

    for _ in range(80):
        script = rng.choice(scripts)
        minutes = rng.choice([15, 20, 30, 40, 45])
        add(
            rng.choice(
                [
                    f"Fine-tune Qwen3-0.6B LoRA with {script}::train. About {minutes} minutes.",
                    f"0.6B LoRA SFT, {script}::train, keep it cheap.",
                    f"Small Qwen LoRA, one GPU, {script}::train, under an hour.",
                    f"Run the 0.6B sample in {script}::train.",
                ]
            ),
            card(
                "cheap",
                3600,
                f"compute run {script}::train --gpu cheap --timeout 3600 --wait",
                "0.6B LoRA fits a cheap GPU; confirm the quote before it starts.",
            ),
            "small_lora",
        )

    for _ in range(50):
        script = rng.choice(scripts)
        add(
            rng.choice(
                [
                    f"Qwen3-1.7B LoRA, {script}::train, after 0.6B already worked.",
                    f"Bump {script}::train to 1.7B. What GPU?",
                    f"1.7B SFT is allowed now. Command for {script}::train?",
                ]
            ),
            card(
                "H100-SXM",
                3600,
                f"compute run {script}::train --gpu H100-SXM --timeout 3600 --wait",
                "1.7B is the larger pin; only after 0.6B worked end to end.",
            ),
            "large_lora",
        )

    for _ in range(50):
        sku = rng.choice(["RTX-3090", "L4", "RTX-6000-ADA"])
        add(
            rng.choice(
                [
                    f"--gpu cheap quoted a sold-out SKU. Try {sku} for train.py::train.",
                    f"gpu list shows {sku} available. Use that for the sample.",
                    f"Create was refused on the decorator SKU. Rerun train.py::train on {sku}.",
                ]
            ),
            card(
                sku if sku in {"L4", "RTX-3090"} else "cheap",
                1800,
                f"compute run train.py::train --gpu {sku} --timeout 1800 --wait --args '{{\"sample\": true}}'",
                "Pass an available SKU from compute gpu list when cheap quotes a miss.",
            ),
            "sku_fallback",
        )

    for _ in range(40):
        add(
            rng.choice(
                [
                    "The last run was marked lost after epoch 1. Same script.",
                    "Spot machine vanished mid-train. Retry?",
                    "run status is lost. What next for train.py::train?",
                ]
            ),
            card(
                "cheap",
                3600,
                "compute run train.py::train --gpu cheap --timeout 3600 --wait",
                "Spot reclaim ends the machine; retry the same entrypoint on a live SKU.",
            ),
            "lost",
        )

    for _ in range(40):
        add(
            rng.choice(
                [
                    "Balance is $4. How do I add credit before the full run?",
                    "create refused: hold exceeds remaining credit.",
                    "I need the minimum top-up.",
                ]
            ),
            card(
                "cheap",
                600,
                "compute credits add 10",
                "Minimum Checkout is $10; balance updates after the webhook.",
            ),
            "credits",
        )

    for _ in range(40):
        add(
            rng.choice(
                [
                    "I set compute secrets set hf. Does train.py::train see the token?",
                    "HF secret exists. Will the model download use it automatically?",
                    "Explain Secret.from_name('hf') vs listing it on the function.",
                ]
            ),
            card(
                "cheap",
                1800,
                "compute run train.py::train_and_push --gpu cheap --timeout 1800 --wait",
                "Storing a secret does not inject it; the function must list secrets=[hf_secret].",
            ),
            "secrets",
        )

    for _ in range(40):
        add(
            rng.choice(
                [
                    "CUDA OOM on batch 8 for 0.6B LoRA. Next command?",
                    "The 0.6B run died with out of memory. What do I change?",
                    "OOM during SFT. Keep cheap GPU.",
                ]
            ),
            card(
                "cheap",
                3600,
                "compute run train.py::train --gpu cheap --timeout 3600 --wait --args '{\"batch_size\": 1}'",
                "Drop the batch size before jumping to a bigger SKU.",
            ),
            "oom",
        )

    for _ in range(30):
        add(
            rng.choice(
                [
                    "Show the quote then start train.py::train. I already approved spend.",
                    "Preflight looked fine. Confirm train.py::train on cheap.",
                ]
            ),
            card(
                "cheap",
                3600,
                "compute run train.py::train --gpu cheap --timeout 3600 --wait --yes",
                "--yes skips the price prompt and still spends prepaid credit.",
            ),
            "confirm",
        )

    rng.shuffle(rows)
    return rows


EVAL_PROMPTS: list[dict[str, Any]] = [
    {
        "user": "First command for train.py::train? Do not spend yet.",
        "expect_gpu": "cheap",
        "expect_in_command": ["dry-run"],
        "tag": "dry_run",
    },
    {
        "user": "How do I dry-run sft.py::train on a cheap GPU?",
        "expect_gpu": "cheap",
        "expect_in_command": ["dry-run"],
        "tag": "dry_run",
    },
    {
        "user": "Fine-tune Qwen3-0.6B LoRA with train.py::train. Keep it cheap.",
        "expect_gpu": "cheap",
        "expect_in_command": ["train.py::train", "--gpu"],
        "tag": "small_lora",
    },
    {
        "user": "0.6B LoRA SFT, one GPU, under an hour.",
        "expect_gpu": "cheap",
        "expect_in_command": ["--gpu"],
        "tag": "small_lora",
    },
    {
        "user": "Qwen3-1.7B LoRA after the 0.6B run already worked.",
        "expect_gpu": "H100-SXM",
        "expect_in_command": ["H100"],
        "tag": "large_lora",
    },
    {
        "user": "Bump train.py::train to 1.7B. What GPU?",
        "expect_gpu": "H100-SXM",
        "expect_in_command": ["H100"],
        "tag": "large_lora",
    },
    {
        "user": "--gpu cheap quoted a sold-out SKU. Try L4 for train.py::train.",
        "expect_gpu": "L4",
        "expect_in_command": ["L4"],
        "tag": "sku_fallback",
    },
    {
        "user": "gpu list shows RTX-3090 available. Use that for the sample.",
        "expect_gpu": "RTX-3090",
        "expect_in_command": ["RTX-3090"],
        "tag": "sku_fallback",
    },
    {
        "user": "The last run was marked lost after epoch 1. Same script train.py::train.",
        "expect_gpu": "cheap",
        "expect_in_command": ["train.py::train"],
        "tag": "lost",
    },
    {
        "user": "Spot machine vanished mid-train. Retry train.py::train?",
        "expect_gpu": "cheap",
        "expect_in_command": ["train.py::train"],
        "tag": "lost",
    },
    {
        "user": "Balance is $4. How do I add credit before the full run?",
        "expect_gpu": "cheap",
        "expect_in_command": ["credits add"],
        "tag": "credits",
    },
    {
        "user": "create refused: hold exceeds remaining credit.",
        "expect_gpu": "cheap",
        "expect_in_command": ["credits"],
        "tag": "credits",
    },
    {
        "user": "I set compute secrets set hf. Does train.py::train see the token?",
        "expect_gpu": "cheap",
        "expect_in_command": ["train_and_push"],
        "tag": "secrets",
    },
    {
        "user": "Explain Secret.from_name('hf') vs listing it on the function.",
        "expect_gpu": "cheap",
        "expect_in_command": ["train_and_push"],
        "tag": "secrets",
    },
    {
        "user": "CUDA OOM on batch 8 for 0.6B LoRA. Next command?",
        "expect_gpu": "cheap",
        "expect_in_command": ["batch_size"],
        "tag": "oom",
    },
    {
        "user": "The 0.6B run died with out of memory. Keep cheap GPU.",
        "expect_gpu": "cheap",
        "expect_in_command": ["batch_size"],
        "tag": "oom",
    },
    {
        "user": "Preflight looked fine. Confirm train.py::train on cheap. I approved spend.",
        "expect_gpu": "cheap",
        "expect_in_command": ["--yes"],
        "tag": "confirm",
    },
    {
        "user": "Show the quote then start train.py::train. I already approved spend.",
        "expect_gpu": "cheap",
        "expect_in_command": ["--yes"],
        "tag": "confirm",
    },
    {
        "user": "Preflight train.py::train_and_push without creating a machine.",
        "expect_gpu": "cheap",
        "expect_in_command": ["dry-run"],
        "tag": "dry_run",
    },
    {
        "user": "Small Qwen LoRA sample in train.py::train, cheap GPU.",
        "expect_gpu": "cheap",
        "expect_in_command": ["train.py::train"],
        "tag": "small_lora",
    },
]


def parse_card(text: str) -> dict[str, str]:
    out = {"gpu": "", "timeout_s": "", "command": "", "note": ""}
    for raw in text.splitlines():
        line = raw.strip()
        if line.startswith("gpu:"):
            out["gpu"] = line.split(":", 1)[1].strip()
        elif line.startswith("timeout_s:"):
            out["timeout_s"] = line.split(":", 1)[1].strip()
        elif line.startswith("command:"):
            out["command"] = line.split(":", 1)[1].strip()
        elif line.startswith("note:"):
            out["note"] = line.split(":", 1)[1].strip()
    return out


def score_generation(text: str, expect: dict[str, Any]) -> dict[str, Any]:
    parsed = parse_card(text)
    gpu_ok = parsed["gpu"] == expect["expect_gpu"]
    cmd = parsed["command"]
    hits = [token for token in expect["expect_in_command"] if token in cmd]
    cmd_ok = len(hits) == len(expect["expect_in_command"])
    four_lines = all(parsed.values())
    return {
        "gpu_ok": gpu_ok,
        "cmd_ok": cmd_ok,
        "format_ok": four_lines,
        "ok": gpu_ok and cmd_ok and four_lines,
        "parsed": parsed,
    }


def messages_for(user: str, assistant: str | None = None) -> list[dict[str, str]]:
    msgs = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": user}]
    if assistant is not None:
        msgs.append({"role": "assistant", "content": assistant})
    return msgs


def split_rows(rows: list[dict[str, str]], seed: int) -> dict[str, list[dict[str, str]]]:
    rng = random.Random(seed)
    rows = list(rows)
    rng.shuffle(rows)
    n = len(rows)
    n_val = max(32, int(0.1 * n))
    return {"train": rows[n_val:], "val": rows[:n_val]}


def generate_one(model, tokenizer, user: str, device, max_new: int = 96) -> str:
    import torch

    prompt = tokenizer.apply_chat_template(
        messages_for(user),
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    enc = tokenizer(prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        out = model.generate(
            **enc,
            max_new_tokens=max_new,
            do_sample=False,
            temperature=1.0,
            top_p=1.0,
            top_k=0,
            pad_token_id=tokenizer.pad_token_id,
        )
    new_tokens = out[0, enc["input_ids"].shape[1] :]
    return tokenizer.decode(new_tokens, skip_special_tokens=True).strip()


def eval_prompts(model, tokenizer, device, prompts: list[dict[str, Any]]) -> dict[str, Any]:
    pairs = []
    hits = 0
    for item in prompts:
        text = generate_one(model, tokenizer, item["user"], device)
        scored = score_generation(text, item)
        hits += int(scored["ok"])
        pairs.append(
            {
                "user": item["user"],
                "tag": item["tag"],
                "generation": text,
                **{k: scored[k] for k in ("ok", "gpu_ok", "cmd_ok", "format_ok")},
                "parsed": scored["parsed"],
            }
        )
    return {"n": len(prompts), "hits": hits, "accuracy": hits / max(len(prompts), 1), "pairs": pairs}


def _train_impl(
    *,
    sample: bool,
    model_id: str,
    epochs: int,
    batch_size: int,
    max_steps: int,
    lora_rank: int,
    seed: int,
    push_to_hub: bool,
    hub_repo: str,
    dataset_repo: str,
    push_dataset: bool,
) -> dict:
    import time

    import torch
    from datasets import Dataset
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from trl import SFTConfig, SFTTrainer

    random.seed(seed)
    torch.manual_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device_name = torch.cuda.get_device_name(0) if device.type == "cuda" else "cpu"
    use_bf16 = device.type == "cuda" and torch.cuda.is_bf16_supported()
    if sample:
        epochs = epochs or 1
        batch_size = min(batch_size, 2)
        max_steps = 50 if max_steps <= 0 else min(max_steps, 50)
    else:
        epochs = epochs or 2
        batch_size = batch_size or 2
        max_steps = -1 if max_steps == 0 else max_steps

    rows = build_examples(seed)
    splits = split_rows(rows, seed)
    if sample:
        splits["train"] = splits["train"][:120]
        splits["val"] = splits["val"][:24]
    print(
        f"device={device_name} sample={sample} train={len(splits['train'])} "
        f"val={len(splits['val'])} model={model_id} bf16={use_bf16}",
        flush=True,
    )

    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    def to_messages(row: dict[str, str]) -> dict[str, Any]:
        return {"messages": messages_for(row["user"], row["assistant"])}

    train_ds = Dataset.from_list(splits["train"]).map(to_messages)
    val_ds = Dataset.from_list(splits["val"]).map(to_messages)

    print(f"loading {model_id}", flush=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16 if use_bf16 else torch.float32,
        trust_remote_code=True,
    )
    model.to(device)
    if tokenizer.pad_token_id is None:
        model.config.pad_token_id = tokenizer.eos_token_id

    print("eval before", flush=True)
    before = eval_prompts(model, tokenizer, device, EVAL_PROMPTS)
    print(f"before_acc={before['accuracy']:.3f} hits={before['hits']}/{before['n']}", flush=True)

    targets = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    present = {n.split(".")[-1] for n, _ in model.named_modules()}
    targets = [t for t in targets if t in present]
    print(f"lora targets={targets}", flush=True)
    model = get_peft_model(
        model,
        LoraConfig(
            r=lora_rank,
            lora_alpha=lora_rank,
            lora_dropout=0.05,
            target_modules=targets,
            bias="none",
            task_type="CAUSAL_LM",
        ),
    )
    model.print_trainable_parameters()
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)

    out_dir = resolve_artifact_dirs()
    sft_dir = out_dir / "sft"
    t0 = time.time()
    args = SFTConfig(
        output_dir=str(sft_dir),
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        gradient_accumulation_steps=4,
        num_train_epochs=epochs,
        max_steps=max_steps,
        learning_rate=2e-4,
        logging_steps=5,
        eval_strategy="steps" if len(val_ds) else "no",
        eval_steps=25,
        save_strategy="no",
        bf16=use_bf16,
        fp16=False,
        report_to="none",
        max_length=MAX_LEN,
        packing=False,
        seed=seed,
    )
    trainer = SFTTrainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        processing_class=tokenizer,
    )
    train_out = trainer.train()
    metrics = {k: _py(v) for k, v in (train_out.metrics or {}).items()}
    eval_after_loss = None
    try:
        ev = trainer.evaluate()
        eval_after_loss = _py(ev.get("eval_loss"))
        metrics.update({f"eval_{k}": _py(v) for k, v in ev.items()})
    except Exception as err:  # noqa: BLE001
        print(f"trainer.evaluate failed: {err}", flush=True)
    print(f"train_metrics={metrics}", flush=True)

    print("eval after", flush=True)
    after = eval_prompts(model, tokenizer, device, EVAL_PROMPTS)
    print(f"after_acc={after['accuracy']:.3f} hits={after['hits']}/{after['n']}", flush=True)

    adapter_dir = out_dir / "adapter"
    model.save_pretrained(adapter_dir)
    tokenizer.save_pretrained(adapter_dir)
    data_dir = out_dir / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    (data_dir / "train.jsonl").write_text("".join(json.dumps(r) + "\n" for r in splits["train"]))
    (data_dir / "val.jsonl").write_text("".join(json.dumps(r) + "\n" for r in splits["val"]))
    (data_dir / "eval_prompts.json").write_text(json.dumps(EVAL_PROMPTS, indent=2) + "\n")

    summary = {
        "model_id": model_id,
        "sample": sample,
        "device_name": device_name,
        "bf16": use_bf16,
        "train_n": len(splits["train"]),
        "val_n": len(splits["val"]),
        "lora_rank": lora_rank,
        "lora_targets": targets,
        "trainable": trainable,
        "metrics": metrics,
        "eval_loss": eval_after_loss,
        "before": {"accuracy": before["accuracy"], "hits": before["hits"], "n": before["n"]},
        "after": {"accuracy": after["accuracy"], "hits": after["hits"], "n": after["n"]},
        "pairs_before": before["pairs"],
        "pairs_after": after["pairs"],
        "seconds": time.time() - t0,
        "claim": "LoRA SFT on homemade job cards, not a general chatbot",
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, default=_py) + "\n")
    write_artifact_marker(
        out_dir,
        name=ARTIFACT_NAME,
        kind="output",
        compatibility_key="qwen-jobcards-v1",
        metadata={
            "filename": "adapter",
            "model_id": model_id,
            "after_accuracy": after["accuracy"],
            "before_accuracy": before["accuracy"],
        },
    )

    hub_url = None
    dataset_url = None
    if push_to_hub or push_dataset:
        _bridge_hf_token()
        try:
            from huggingface_hub import HfApi

            api = HfApi()
            if push_dataset:
                api.create_repo(dataset_repo, exist_ok=True, repo_type="dataset", private=False)
                api.upload_folder(folder_path=str(data_dir), repo_id=dataset_repo, repo_type="dataset")
                dataset_url = f"https://huggingface.co/datasets/{dataset_repo}"
            if push_to_hub:
                api.create_repo(hub_repo, exist_ok=True, private=False)
                api.upload_folder(folder_path=str(adapter_dir), repo_id=hub_repo)
                api.upload_file(path_or_fileobj=str(out_dir / "summary.json"), path_in_repo="summary.json", repo_id=hub_repo)
                hub_url = f"https://huggingface.co/{hub_repo}"
        except Exception as err:  # noqa: BLE001
            print(f"hub push failed: {err}", flush=True)

    return {
        "ok": True,
        "compat": "finetune-qwen-lora",
        "device_name": device_name,
        "sample": sample,
        "model_id": model_id,
        "before_accuracy": before["accuracy"],
        "after_accuracy": after["accuracy"],
        "eval_loss": eval_after_loss,
        "trainable": trainable,
        "metrics": metrics,
        "pairs_before": before["pairs"][:5],
        "pairs_after": after["pairs"][:5],
        "artifact_dir": str(out_dir),
        "hub_url": hub_url,
        "dataset_url": dataset_url,
        "seconds": summary["seconds"],
    }


@app.function(gpu="L4", image=image, timeout=3600)
def train(
    sample: bool = False,
    model_id: str = DEFAULT_MODEL,
    epochs: int = 0,
    batch_size: int = 2,
    max_steps: int = 0,
    lora_rank: int = 16,
    seed: int = 0,
    push_to_hub: bool = False,
    hub_repo: str = DEFAULT_HUB_REPO,
    dataset_repo: str = DEFAULT_DATASET_REPO,
    push_dataset: bool = False,
) -> dict:
    return _train_impl(
        sample=sample,
        model_id=model_id,
        epochs=epochs,
        batch_size=batch_size,
        max_steps=max_steps,
        lora_rank=lora_rank,
        seed=seed,
        push_to_hub=push_to_hub,
        hub_repo=hub_repo,
        dataset_repo=dataset_repo,
        push_dataset=push_dataset,
    )


@app.function(gpu="L4", image=image, timeout=3600, secrets=[hf_secret])
def train_and_push(
    sample: bool = False,
    model_id: str = DEFAULT_MODEL,
    epochs: int = 0,
    batch_size: int = 2,
    max_steps: int = 0,
    lora_rank: int = 16,
    seed: int = 0,
    push_to_hub: bool = True,
    hub_repo: str = DEFAULT_HUB_REPO,
    dataset_repo: str = DEFAULT_DATASET_REPO,
    push_dataset: bool = True,
) -> dict:
    return _train_impl(
        sample=sample,
        model_id=model_id,
        epochs=epochs,
        batch_size=batch_size,
        max_steps=max_steps,
        lora_rank=lora_rank,
        seed=seed,
        push_to_hub=push_to_hub,
        hub_repo=hub_repo,
        dataset_repo=dataset_repo,
        push_dataset=push_dataset,
    )


if __name__ == "__main__":
    rows = build_examples(0)
    assert len(rows) >= 400
    assert all("gpu:" in r["assistant"] for r in rows)
    print("examples", len(rows), "eval", len(EVAL_PROMPTS))
    print(rows[0]["user"])
    print(rows[0]["assistant"])
