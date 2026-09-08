"""Create a labeled scientific contact sheet from training demonstrations."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

from rooms import FAMILIES


def build(evidence: Path, output: Path):
    output.mkdir(parents=True, exist_ok=True)
    training = {
        r["id"]: r
        for r in map(json.loads, (evidence / "data/train.jsonl").read_text().splitlines())
    }
    sheet = Image.new("RGB", (1040, 720), "#f7f4ef")
    draw = ImageDraw.Draw(sheet)
    font = ImageFont.load_default(size=22)
    small = ImageFont.load_default(size=15)
    ledger = []
    for i, family in enumerate(FAMILIES):
        identity = f"{family}-l0-p{i}"
        assert identity in training
        source = evidence / "teacher" / identity
        metrics = json.loads((source / "metrics.json").read_text())
        if not metrics["valid"]:
            raise ValueError(f"Invalid teacher scene: {identity}")
        image = Image.open(source / "render.png").convert("RGB")
        image.thumbnail((285, 285))
        x = (i % 2) * 520
        y = (i // 2) * 360
        sheet.paste(image, (x + (520 - image.width) // 2, y))
        draw.text((x + 28, y + 284), family.replace("_", " ").title(), font=font, fill="#1a1a1a")
        draw.text(
            (x + 28, y + 311), "Teacher demonstration · Blender 4.5.3", font=small, fill="#5c5c5c"
        )
        ledger.append(
            {
                "id": identity,
                "split": "train",
                "provenance": training[identity]["provenance"],
                "program_sha256": training[identity]["code_sha256"],
                "render_sha256": hashlib.sha256((source / "render.png").read_bytes()).hexdigest(),
            }
        )
        (output / (family + ".py")).write_text(training[identity]["messages"][-1]["content"])
    sheet.save(output / "teacher-grid.png")
    (output / "teacher-grid.provenance.json").write_text(json.dumps(ledger, indent=2) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evidence", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("assets"))
    args = parser.parse_args()
    build(args.evidence, args.output)
