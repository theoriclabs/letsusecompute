"""Final base/SFT test comparison from an exact downloaded training artifact."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from inference import generate_batched
from pipeline import dump, paired_review, select_balanced
from rooms import MODEL, REVISION


def evaluate(
    artifact: Path, output: Path, count=32, *, replay=False, split="test", system_suffix=""
):
    import importlib.metadata

    import torch
    from peft import PeftModel
    from transformers import AutoModelForImageTextToText, AutoTokenizer, set_seed

    from render import EVALUATOR_REVISION

    if not torch.cuda.is_available():
        raise RuntimeError("Run final evaluation on a disposable NVIDIA GPU worker")
    config = json.loads((artifact / "run-config.json").read_text())
    if config["model"] != MODEL or config["revision"] != REVISION or config["mode"] != "sft":
        raise ValueError("Expected a full SFT artifact from the pinned Qwen revision")
    hashes = json.loads((artifact / "artifact-hashes.json").read_text())
    for name, digest in hashes.items():
        path = (artifact / name).resolve()
        if not path.is_relative_to(artifact.resolve()):
            raise ValueError("Invalid artifact manifest path")
        if name.startswith("adapter/") or name.startswith("data/"):
            if hashlib.sha256(path.read_bytes()).hexdigest() != digest:
                raise ValueError(f"Artifact hash mismatch: {name}")
    if split not in {"test", "validation"}:
        raise ValueError("Only held-out test or validation splits are allowed")
    test_path = artifact / ("data/" + split + ".jsonl")
    expected = config["dataset"]["splits"][split]["sha256"]
    if hashlib.sha256(test_path.read_bytes()).hexdigest() != expected:
        raise ValueError("Test split differs from the training artifact identity")
    rows = [json.loads(line) for line in test_path.read_text().splitlines()]
    if any(row["split"] != split for row in rows):
        raise ValueError("Final evaluation requires exclusively test rows")
    rows = select_balanced(rows, count)
    for row in rows:
        row["messages"][0]["content"] += system_suffix
    output.mkdir(parents=True, exist_ok=False)
    if replay:
        replay_validation(artifact, output / "validation-replay")
    set_seed(42)
    dump(
        output / "evaluation-config.json",
        {
            "model": MODEL,
            "revision": REVISION,
            "test_sha256": expected if split == "test" else None,
            "dataset_sha256": expected,
            "split": split,
            "system_suffix": system_suffix,
            "test_ids": [r["id"] for r in rows],
            "training_artifact": str(artifact.resolve()),
            "decoding": "greedy",
            "generation_batch_size": 8,
            "batch_note": "Both test policies use identical left-padded batches. Validation retains the original single-prompt generations.",
            "max_new_tokens": 4096,
            "evaluator_revision": EVALUATOR_REVISION,
            "gpu": torch.cuda.get_device_name(0),
            "versions": {
                pkg: importlib.metadata.version(pkg)
                for pkg in ["torch", "transformers", "peft", "accelerate", "bpy", "fla-core"]
            },
            "source_sha256": {
                p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                for p in Path(__file__).parent.glob("*.py")
            },
            "renderer": {"resolution": 384, "samples": 12, "engine": "CYCLES", "device": "CPU"},
        },
    )
    tokenizer = AutoTokenizer.from_pretrained(MODEL, revision=REVISION, trust_remote_code=False)
    model = AutoModelForImageTextToText.from_pretrained(
        MODEL,
        revision=REVISION,
        dtype=torch.bfloat16,
        attn_implementation="sdpa",
        trust_remote_code=False,
    ).to("cuda")
    baseline = generate_batched(model, tokenizer, rows, output, "base")
    model = PeftModel.from_pretrained(model, artifact / "adapter")
    trained = generate_batched(model, tokenizer, rows, output, "sft")
    return paired_review(output, rows, baseline, trained)


def replay_validation(artifact: Path, output: Path):
    """Re-render identical saved base/SFT programs under the corrected harness."""
    import time

    from render import EVALUATOR_REVISION, render_code

    config = json.loads((artifact / "run-config.json").read_text())
    by_id = {
        r["id"]: r
        for r in map(json.loads, (artifact / "data/validation.jsonl").read_text().splitlines())
    }
    rows = [by_id[identity] for identity in config["eval_ids"]]
    output.mkdir(parents=True, exist_ok=False)
    results = {}
    for label in ("base", "sft"):
        originals = {
            r["id"]: r for r in json.loads((artifact / (label + "-results.json")).read_text())
        }
        results[label] = []
        for row in rows:
            previous = originals[row["id"]]
            raw = (artifact / label / row["id"] / "raw.txt").read_text()
            destination = output / label / row["id"]
            destination.mkdir(parents=True)
            (destination / "raw.txt").write_text(raw)
            started = time.monotonic()
            metrics = (
                {
                    "valid": False,
                    "executed": False,
                    "coverage": 0.0,
                    "failures": ["generation_token_limit"],
                }
                if previous["token_limit"]
                else render_code(raw, destination, row["required"], resolution=384, samples=12)
            )
            metrics.update(
                {
                    "id": row["id"],
                    "family": row["family"],
                    "split": "validation",
                    "label": label,
                    "output_tokens": previous["output_tokens"],
                    "token_limit": previous["token_limit"],
                    "seconds": time.monotonic() - started,
                    "original_total_seconds": previous["seconds"],
                    "evaluator_revision": EVALUATOR_REVISION,
                    "raw_sha256": hashlib.sha256(raw.encode()).hexdigest(),
                }
            )
            dump(destination / "metrics.json", metrics)
            results[label].append(metrics)
            print(json.dumps({"phase": "validation_replay", **metrics}), flush=True)
        dump(output / (label + "-results.json"), results[label])
    dump(
        output / "evaluation-correction.json",
        {
            "revision": EVALUATOR_REVISION,
            "reason": "Allow ordinary mesh/object/transform operators equivalent to already allowed data APIs; see the frozen renderer source.",
            "generation_changed": False,
            "geometry_checks_changed": False,
            "applied_to": ["base", "sft"],
            "ids": [r["id"] for r in rows],
        },
    )
    return paired_review(output, rows, results["base"], results["sft"])


def evaluate_from_archive(
    *, training_archive_path, training_sha256, training_run_id, count=32, api_hint_control=False
):
    import tempfile

    from export import bundle_result, unpack_archive

    root = Path(tempfile.mkdtemp(prefix="blender-test-"))
    verified = unpack_archive(
        Path(training_archive_path).read_bytes(), training_sha256, root / "training"
    )
    hint = (
        """
Blender 4.5 API compatibility notes: Material.diffuse_color and the Principled BSDF Base Color input require four components (R, G, B, A), with A=1.0 for opaque colors. Do not assign an RGB triple to either field. Add bevels through obj.modifiers.new(name="edge", type="BEVEL"); MeshEdge.bevel_weight is unavailable. Define every helper variable explicitly. Leave the world, camera, lights and render settings entirely to the supplied evaluator: do not read, create or configure them. These notes do not change the requested furniture or geometry constraints.
"""
        if api_hint_control
        else ""
    )
    comparison = evaluate(
        root / "training",
        root / "test",
        count,
        replay=not api_hint_control,
        split="validation" if api_hint_control else "test",
        system_suffix=hint,
    )
    corrected_validation = (
        None
        if api_hint_control
        else json.loads((root / "test/validation-replay/comparison.json").read_text())
    )
    dump(
        root / "test/training-identity.json",
        {
            "run_id": training_run_id,
            "export_verified": verified,
        },
    )
    return bundle_result(
        root / "test",
        {
            "mode": "validation_api_hint_control" if api_hint_control else "test",
            "training_run_id": training_run_id,
            "comparison": comparison,
            "corrected_validation": corrected_validation,
            "count": count,
        },
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--count", type=int, default=32)
    args = parser.parse_args()
    print(json.dumps(evaluate(args.artifact, args.output, args.count), indent=2))
