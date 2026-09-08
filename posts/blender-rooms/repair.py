"""One-retry validation experiment: generic retry versus measured Blender feedback."""

from __future__ import annotations

import copy
import hashlib
import importlib.metadata
import json
import tempfile
import time
from pathlib import Path

from export import unpack_archive
from inference import batch_tokens, generate_batched
from native_export import compact_result
from pipeline import dump
from render import EVALUATOR_REVISION, render_code
from rooms import FAMILIES, MODEL, REVISION

BATCH_SIZE = 4
MAX_NEW_TOKENS = 4096


def selected_rows(rows):
    by_id = {row["id"]: row for row in rows}
    return [
        copy.deepcopy(by_id[f"{family}-l6-p{palette}"])
        for family in FAMILIES
        for palette in range(2, 6)
    ]


def measured_feedback(metrics, directory):
    evidence = {"failures": metrics.get("failures", []), "token_limit": metrics["token_limit"]}
    log = directory / "blender.log"
    if not metrics.get("executed") and log.exists():
        text = log.read_text(errors="replace")
        start = text.rfind("Traceback (most recent call last)")
        evidence["blender_error"] = (text[start:] if start >= 0 else text)[-1800:]
    scene = directory / "scene.json"
    if scene.exists():
        failed_names = {
            reason.split(":", 1)[1] for reason in metrics.get("failures", []) if ":" in reason
        }
        evidence["measured_bounds"] = [
            obj for obj in json.loads(scene.read_text()) if obj["name"] in failed_names
        ][:24]
    text = json.dumps(evidence, ensure_ascii=True)
    if len(text) > 8000:
        raise ValueError(
            "Feedback exceeds the predeclared limit; do not silently truncate evidence"
        )
    return text


def repair_messages(row, raw, metrics, directory, condition):
    instruction = (
        "Your previous program did not pass the supplied Blender evaluator. "
        "Revise it to satisfy the original room brief and constraints. Return the entire corrected "
        "Python program only, with no explanation. You have one attempt and a 4096-token output budget."
    )
    if condition == "feedback":
        instruction += "\nThe evaluator reported:\n" + measured_feedback(metrics, directory)
    elif condition != "retry":
        raise ValueError("Unknown repair condition")
    return [
        *copy.deepcopy(row["messages"][:2]),
        {"role": "assistant", "content": raw},
        {"role": "user", "content": instruction},
    ]


def select_final(first, second):
    if first["valid"] or second is None:
        return first
    if second["valid"] or (second.get("executed") and not first.get("executed")):
        return second
    return first


def attempts(model, tokenizer, rows, first, output, policy, condition):
    originals = {r["id"]: r for r in first}
    failed = [r for r in rows if not originals[r["id"]]["valid"]]
    results = []
    label = policy + "-" + condition
    for start in range(0, len(failed), BATCH_SIZE):
        group = failed[start : start + BATCH_SIZE]
        chats = []
        for row in group:
            source = output / (policy + "-first") / row["id"]
            chats.append(
                repair_messages(
                    row, (source / "raw.txt").read_text(), originals[row["id"]], source, condition
                )
            )
        prompts = [
            tokenizer.apply_chat_template(
                chat, tokenize=False, add_generation_prompt=True, enable_thinking=False
            )
            for chat in chats
        ]
        input_tokens = [len(tokenizer(p, add_special_tokens=False)["input_ids"]) for p in prompts]
        if max(input_tokens) + MAX_NEW_TOKENS > 16384:
            raise ValueError("Repair conversation exceeds the declared 16384-token context bound")
        started = time.monotonic()
        outputs = batch_tokens(model, tokenizer, prompts, MAX_NEW_TOKENS)
        generation_seconds = time.monotonic() - started
        for row, chat, tokens, input_count in zip(group, chats, outputs, input_tokens, strict=True):
            directory = output / label / row["id"]
            directory.mkdir(parents=True)
            raw = tokenizer.decode(tokens, skip_special_tokens=True)
            (directory / "raw.txt").write_text(raw)
            dump(directory / "messages.json", chat)
            exhausted = len(tokens) >= MAX_NEW_TOKENS and tokens[-1] != tokenizer.eos_token_id
            started = time.monotonic()
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
                id=row["id"],
                family=row["family"],
                split="validation",
                label=label,
                output_tokens=len(tokens),
                input_tokens=input_count,
                token_limit=exhausted,
                seconds=generation_seconds / len(group) + time.monotonic() - started,
                generation_batch_size=len(group),
                batch_ids=[r["id"] for r in group],
                raw_sha256=hashlib.sha256(raw.encode()).hexdigest(),
            )
            dump(directory / "metrics.json", metrics)
            results.append(metrics)
            dump(output / (label + "-results.json"), results)
            print(json.dumps({"phase": label, **metrics}), flush=True)
    dump(output / (label + "-results.json"), results)
    return results


def summarize(first, second):
    by_id = {r["id"]: r for r in second}
    final = [select_final(r, by_id.get(r["id"])) for r in first]
    return {
        "count": len(first),
        "first_valid": sum(r["valid"] for r in first),
        "attempted_repairs": len(second),
        "repair_valid": sum(r["valid"] for r in second),
        "within_two_valid": sum(r["valid"] for r in final),
        "within_two_rendered": sum(bool(r.get("executed")) for r in final),
        "total_output_tokens": sum(r["output_tokens"] for r in first + second),
        "selected": {r["id"]: r["label"] for r in final},
    }


def repair_from_archive(*, training_archive_path, training_sha256, training_run_id, count=16):
    import torch
    from peft import PeftModel
    from transformers import AutoModelForImageTextToText, AutoTokenizer, set_seed

    if count != 16 or not torch.cuda.is_available():
        raise ValueError("This frozen follow-up requires 16 validation cases and a disposable GPU")
    root = Path(tempfile.mkdtemp(prefix="blender-repair-"))
    artifact = root / "training"
    identity = unpack_archive(Path(training_archive_path).read_bytes(), training_sha256, artifact)
    config = json.loads((artifact / "run-config.json").read_text())
    if (config["model"], config["revision"], config["mode"]) != (MODEL, REVISION, "sft"):
        raise ValueError("Expected the pinned original full SFT artifact")
    data = artifact / "data/validation.jsonl"
    if (
        hashlib.sha256(data.read_bytes()).hexdigest()
        != config["dataset"]["splits"]["validation"]["sha256"]
    ):
        raise ValueError("Validation split identity mismatch")
    rows = selected_rows([json.loads(line) for line in data.read_text().splitlines()])
    if any(r["id"] in config["eval_ids"] for r in rows):
        raise ValueError("Follow-up overlaps previously generated validation cases")
    output = root / "repair"
    output.mkdir()
    dump(
        output / "prompts.json",
        [
            {k: v for k, v in row.items() if k != "messages"} | {"messages": row["messages"][:2]}
            for row in rows
        ],
    )
    dump(
        output / "evaluation-config.json",
        {
            "mode": "validation_one_retry_vs_measured_feedback",
            "model": MODEL,
            "revision": REVISION,
            "training_run_id": training_run_id,
            "training_export_verified": identity,
            "adapter_sha256": hashlib.sha256(
                (artifact / "adapter/adapter_model.safetensors").read_bytes()
            ).hexdigest(),
            "ids": [r["id"] for r in rows],
            "split": "validation",
            "seed": 42,
            "max_new_tokens": MAX_NEW_TOKENS,
            "batch_size": BATCH_SIZE,
            "max_context_tokens": 16384,
            "decoding": "greedy",
            "max_attempts_per_condition": 2,
            "conditions": ["retry", "feedback"],
            "evaluator_revision": EVALUATOR_REVISION,
            "weights_changed": False,
            "primary_test_repeated": False,
            "selection": "Keep a passing first attempt. Otherwise accept a passing revision, or a first render when none existed. Retain original on remaining failures.",
            "limitation": "Validation-only development experiment. Four layout groups, four correlated palette variants each. Retry uses more inference; no visual preference score.",
            "versions": {
                pkg: importlib.metadata.version(pkg)
                for pkg in ["torch", "transformers", "peft", "accelerate", "bpy", "fla-core"]
            },
            "source_sha256": {
                p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                for p in Path(__file__).parent.glob("*.py")
            },
        },
    )
    set_seed(42)
    tokenizer = AutoTokenizer.from_pretrained(MODEL, revision=REVISION, trust_remote_code=False)
    model = AutoModelForImageTextToText.from_pretrained(
        MODEL,
        revision=REVISION,
        dtype=torch.bfloat16,
        attn_implementation="sdpa",
        trust_remote_code=False,
    ).to("cuda")
    summary = {}
    for policy in ["base", "sft"]:
        if policy == "sft":
            model = PeftModel.from_pretrained(model, artifact / "adapter")
        model.eval()
        first = generate_batched(
            model, tokenizer, rows, output, policy + "-first", batch_size=BATCH_SIZE
        )
        summary[policy] = {
            "first": {
                "count": len(first),
                "valid": sum(r["valid"] for r in first),
                "rendered": sum(bool(r.get("executed")) for r in first),
            }
        }
        for condition in ["retry", "feedback"]:
            second = attempts(model, tokenizer, rows, first, output, policy, condition)
            summary[policy][condition] = summarize(first, second)
            dump(output / "summary.json", summary)
    del model
    torch.cuda.empty_cache()
    # These are frozen original test outputs, independent of follow-up successes.
    from media_frames import render_media

    media = render_media(output / "media")
    dump(output / "media.json", media)
    return compact_result(
        output,
        {
            "mode": "validation_repair",
            "training_run_id": training_run_id,
            "comparison": summary,
            "media": media,
        },
    )
