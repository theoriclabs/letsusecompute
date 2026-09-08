"""Build publication artifacts solely from downloaded experiment evidence."""

from __future__ import annotations

import argparse
import hashlib
import html
import json
import shutil
from collections import Counter
from pathlib import Path


def summarize(root):
    output = {}
    for label in ("base", "sft"):
        rows = json.loads((root / (label + "-results.json")).read_text())
        failures = Counter(
            reason.split(":")[0] for row in rows for reason in row.get("failures", [])
        )
        output[label] = {
            "count": len(rows),
            "valid": sum(bool(r["valid"]) for r in rows),
            "executed": sum(bool(r.get("executed")) for r in rows),
            "token_limit": sum(bool(r.get("token_limit")) for r in rows),
            "ordinary_operator_unmeasured": [
                r["id"]
                for r in rows
                if any("bpy.ops.object.material_slot_add" in f for f in r.get("failures", []))
            ],
            "mean_coverage": sum(r.get("coverage", 0) for r in rows) / len(rows),
            "failure_categories": dict(failures),
            "seconds": sum(r["seconds"] for r in rows),
            "by_family": {
                family: {
                    "count": sum(r["family"] == family for r in rows),
                    "valid": sum(r["family"] == family and r["valid"] for r in rows),
                    "executed": sum(
                        r["family"] == family and bool(r.get("executed")) for r in rows
                    ),
                }
                for family in sorted({r["family"] for r in rows})
            },
        }
    return output


def build(training: Path, output: Path, test: Path | None = None, control: Path | None = None):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output.mkdir(parents=True, exist_ok=True)
    history = json.loads((training / "history.json").read_text())
    train = [r for r in history if "loss" in r]
    validation = [r for r in history if "eval_loss" in r]
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11})
    fig, ax = plt.subplots(figsize=(8, 3.8), layout="constrained")
    ax.plot(
        [r["step"] for r in train],
        [float(r["loss"]) for r in train],
        color="#bb5428",
        label="Training completion loss",
    )
    ax.plot(
        [r["step"] for r in validation],
        [float(r["eval_loss"]) for r in validation],
        "o-",
        color="#326e7b",
        label="Validation completion loss",
    )
    ax.set(
        xlabel="Optimizer step",
        ylabel="Cross-entropy",
        title="Qwen3.5-9B · rank-4 LoRA · 192 training programs",
    )
    ax.grid(alpha=0.2)
    ax.legend(frameon=False)
    fig.savefig(output / "loss.png", dpi=180)
    fig.savefig(output / "loss.svg")
    plt.close(fig)

    validation_root = (
        test / "validation-replay" if test and (test / "validation-replay").is_dir() else training
    )
    summary = {
        "validation": summarize(validation_root),
        "test": summarize(test) if test else None,
        "api_hint_control": summarize(control) if control else None,
        "diagnostic_caveat": "Original validation used batch 1; the API-hint diagnostic used batch 8 and an appended system instruction. The effects are not isolated." if control else None,
        "training": json.loads((training / "training.json").read_text()),
        "run_config": json.loads((training / "run-config.json").read_text()),
        "limitation": "Procedural-data control; correlated palette variants and shared primitives. No independent blind aesthetic judgment.",
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    sections = []
    provenance = []
    for split, root in [
        ("validation", validation_root),
        ("test", test),
        ("api_hint_control", control),
    ]:
        if root is None:
            continue
        data_split = "validation" if split == "api_hint_control" else split
        prompts = {
            r["id"]: r["messages"][1]["content"]
            for r in map(
                json.loads, (training / ("data/" + data_split + ".jsonl")).read_text().splitlines()
            )
        }
        metrics = {
            label: {r["id"]: r for r in json.loads((root / (label + "-results.json")).read_text())}
            for label in ["base", "sft"]
        }
        identities = list(metrics["base"])
        assert identities == list(metrics["sft"]), "Pair IDs differ"
        title = (
            "Validation with API hints (post-hoc control)"
            if split == "api_hint_control"
            else split.title()
        )
        sections.append(f"<h2>{title} · all {len(identities)} pairs</h2>")
        if split == "api_hint_control":
            sections.append(
                '<p class="note">Post-hoc validation diagnostic: identical API hints were appended for both policies. Original validation used batch 1; this diagnostic used batch 8. It does not isolate instructions from batching. The primary test was unchanged.</p>'
            )
        for identity in identities:
            sections.append(
                f'<section><h3>{html.escape(identity)}</h3><details><summary>Room brief</summary><p style="white-space:pre-wrap">{html.escape(prompts[identity])}</p></details><div class="pair">'
            )
            for label in ["base", "sft"]:
                row = metrics[label][identity]
                source = root / label / identity
                image = source / "render.png"
                name = f"{split}-{identity}-{label}.png"
                status = (
                    "Geometry valid"
                    if row["valid"]
                    else "Failed: " + ", ".join(row.get("failures", []))
                )
                sections.append(
                    f"<figure><figcaption>{label.upper()} · {html.escape(status)}</figcaption>"
                )
                if image.exists():
                    shutil.copy2(image, output / name)
                    sections.append(
                        f'<img src="{name}" width="384" height="384" alt="{label} {html.escape(identity)}" loading="lazy">'
                    )
                    provenance.append(
                        {
                            "file": name,
                            "source": f"{label}/{identity}/render.png",
                            "split": split,
                            "sha256": hashlib.sha256(image.read_bytes()).hexdigest(),
                        }
                    )
                else:
                    sections.append('<div class="missing">No render</div>')
                text_name = f"{split}-{identity}-{label}.txt"
                shutil.copy2(source / "raw.txt", output / text_name)
                sections.append(
                    f'<p><a href="{text_name}">Raw model output</a> · {row["output_tokens"]} tokens</p></figure>'
                )
            sections.append("</div></section>")
    document = """<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><meta name="robots" content="noindex"><title>Blender rooms — complete base/SFT comparison</title><style>
body{font:16px/1.5 Georgia,serif;background:#f7f4ef;color:#1a1a1a;margin:0;padding:2rem;max-width:1000px;margin:auto}h1{line-height:1.15}.pair{display:grid;grid-template-columns:1fr 1fr;gap:1rem}figure{margin:0;min-width:0}figcaption{min-height:3rem;font-size:14px;overflow-wrap:anywhere}img{width:100%;height:auto;display:block;border-radius:8px}.missing{display:grid;place-items:center;aspect-ratio:1;background:#e6dfd3;color:#625a4e}section{border-top:1px solid #d8cdbd;padding:1rem 0}a{color:#a34e27}p{font-size:15px}.note{background:white;padding:1rem}@media(max-width:600px){body{padding:1rem}.pair{gap:.6rem}}
</style></head><body><h1>Every base/SFT comparison</h1><p class="note">Same prompts, greedy decoding, 4,096-token output budget, fixed renderer. Every selected prompt is shown, including missing or invalid renders. Geometry is an approximate diagnostic; these labeled images are not a blinded aesthetic evaluation. Programs are untrusted text and must run only on disposable workers.</p>"""
    (output / "gallery.html").write_text(document + "".join(sections) + "</body></html>")
    (output / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
    from review import build_review

    build_review((test or validation_root) / "review", output / "review")
    if control:
        build_review(control / "review", output / "control-review")
    print(json.dumps({k: summary[k] for k in ["validation", "test"]}, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--training", type=Path, required=True)
    parser.add_argument("--test", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--control", type=Path)
    args = parser.parse_args()
    build(args.training, args.output, args.test, args.control)
