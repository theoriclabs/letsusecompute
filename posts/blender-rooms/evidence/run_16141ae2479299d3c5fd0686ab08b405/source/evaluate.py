"""Final base/SFT test comparison from an exact downloaded training artifact."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from pipeline import dump, generate, paired_review, select_balanced
from rooms import MODEL, REVISION


def evaluate(artifact: Path, output: Path, count=32):
    import torch
    from peft import PeftModel
    from transformers import AutoModelForImageTextToText, AutoTokenizer, set_seed

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
    test_path = artifact / "data/test.jsonl"
    expected = config["dataset"]["splits"]["test"]["sha256"]
    if hashlib.sha256(test_path.read_bytes()).hexdigest() != expected:
        raise ValueError("Test split differs from the training artifact identity")
    rows = [json.loads(line) for line in test_path.read_text().splitlines()]
    if any(row["split"] != "test" for row in rows):
        raise ValueError("Final evaluation requires exclusively test rows")
    rows = select_balanced(rows, count)
    output.mkdir(parents=True, exist_ok=False)
    set_seed(42)
    dump(
        output / "evaluation-config.json",
        {
            "model": MODEL,
            "revision": REVISION,
            "test_sha256": expected,
            "test_ids": [r["id"] for r in rows],
            "training_artifact": str(artifact.resolve()),
            "decoding": "greedy",
            "max_new_tokens": 4096,
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
    baseline = generate(model, tokenizer, rows, output, "base")
    model = PeftModel.from_pretrained(model, artifact / "adapter")
    trained = generate(model, tokenizer, rows, output, "sft")
    return paired_review(output, rows, baseline, trained)


def evaluate_from_archive(*, training_archive_path, training_sha256, training_run_id, count=32):
    import tempfile

    from export import bundle_result, unpack_archive

    root = Path(tempfile.mkdtemp(prefix="blender-test-"))
    verified = unpack_archive(
        Path(training_archive_path).read_bytes(), training_sha256, root / "training"
    )
    comparison = evaluate(root / "training", root / "test", count)
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
            "mode": "test",
            "training_run_id": training_run_id,
            "comparison": comparison,
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
