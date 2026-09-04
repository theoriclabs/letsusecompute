#!/usr/bin/env python3
"""Build deterministic, attributed article assets from the local train split."""

from __future__ import annotations

import argparse
import hashlib
import html
import json
from pathlib import Path

from datasets import load_from_disk
from PIL import Image, ImageDraw, ImageFont


CLASS_NAMES = ("no face", "face")
SAMPLE_COUNT_PER_CLASS = 4
CC_BY_2 = "https://creativecommons.org/licenses/by/2.0/"


def _font(size: int, *, bold: bool = False):
    candidates = [
        "/System/Library/Fonts/Supplemental/Arial Bold.ttf" if bold else "/System/Library/Fonts/Supplemental/Arial.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf" if bold else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    ]
    for candidate in candidates:
        if Path(candidate).exists():
            return ImageFont.truetype(candidate, size=size)
    return ImageFont.load_default()


def _pick_training_rows(split) -> list[dict]:
    ranked: dict[int, list[tuple[str, int]]] = {0: [], 1: []}
    for index, (label, source_id) in enumerate(
        zip(split["label"], split["source_image_id"])
    ):
        digest = hashlib.sha256(f"article-sample:{source_id}".encode()).hexdigest()
        ranked[int(label)].append((digest, index))
    selected = []
    for label in (0, 1):
        for _digest, index in sorted(ranked[label])[:SAMPLE_COUNT_PER_CLASS]:
            selected.append(split[index])
    return selected


def _cover(image: Image.Image, size: tuple[int, int]) -> Image.Image:
    image = image.convert("RGB")
    target_w, target_h = size
    ratio = max(target_w / image.width, target_h / image.height)
    resized = image.resize(
        (round(image.width * ratio), round(image.height * ratio)), Image.Resampling.LANCZOS
    )
    left = (resized.width - target_w) // 2
    top = (resized.height - target_h) // 2
    return resized.crop((left, top, left + target_w, top + target_h))


def write_dataset_grid(rows: list[dict], output: Path) -> list[dict]:
    width, height = 1440, 1060
    canvas = Image.new("RGB", (width, height), "#f5f1e8")
    draw = ImageDraw.Draw(canvas)
    title_font = _font(54, bold=True)
    body_font = _font(23)
    label_font = _font(20, bold=True)
    meta_font = _font(16)
    draw.text((60, 45), "Face or not?", fill="#17201e", font=title_font)
    draw.text(
        (62, 112),
        "Balanced Open Images V7 training crops • deterministic sample",
        fill="#53605b",
        font=body_font,
    )

    margin_x, gap_x, gap_y = 60, 24, 42
    card_w = (width - margin_x * 2 - gap_x * 3) // 4
    image_h, caption_h = 300, 76
    card_h = image_h + caption_h
    top = 178
    ledger_rows = []
    for position, row in enumerate(rows):
        column = position % 4
        row_number = position // 4
        x = margin_x + column * (card_w + gap_x)
        y = top + row_number * (card_h + gap_y)
        draw.rounded_rectangle(
            (x, y, x + card_w, y + card_h), radius=16, fill="#ffffff", outline="#d4d8d2", width=2
        )
        crop = _cover(row["image"], (card_w - 16, image_h - 16))
        canvas.paste(crop, (x + 8, y + 8))
        label = int(row["label"])
        color = "#156b4a" if label else "#9a3d28"
        draw.text(
            (x + 14, y + image_h + 10),
            CLASS_NAMES[label].upper(),
            fill=color,
            font=label_font,
        )
        draw.text(
            (x + 14, y + image_h + 42),
            f"Open Images {row['source_image_id'][:10]}…",
            fill="#67706c",
            font=meta_font,
        )
        ledger_rows.append(
            {
                "asset": output.name,
                "sample_position": position + 1,
                "asset_box_px": [x + 8, y + 8, x + card_w - 8, y + image_h - 8],
                "split": "train",
                "label": label,
                "label_name": CLASS_NAMES[label].replace(" ", "_"),
                "source_image_id": row["source_image_id"],
                "source_image_url": row["source_image_url"],
                "original_url": row["original_url"],
                "original_landing_url": row["original_landing_url"],
                "license": row["license"],
                "author": row["author"],
                "author_profile_url": row["author_profile_url"],
                "title": row["title"],
                "crop_norm_json": row["crop_norm_json"],
                "crop_strategy": row["crop_strategy"],
                "annotation_source": row["annotation_source"],
                "pixel_sha256": hashlib.sha256(row["image"].convert("RGB").tobytes()).hexdigest(),
            }
        )

    draw.text(
        (62, height - 46),
        "Images: CC BY 2.0 • full per-image attribution in assets/ATTRIBUTION.md",
        fill="#53605b",
        font=meta_font,
    )
    canvas.save(output, format="PNG", optimize=True)
    return ledger_rows


def write_attribution(rows: list[dict], output_dir: Path) -> None:
    jsonl = output_dir / "attribution.jsonl"
    with jsonl.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True, ensure_ascii=False) + "\n")

    lines = [
        "# Asset attribution",
        "",
        "`dataset_samples.png` contains deterministic examples from the published dataset's training split. No validation or test image is used. Open Images lists each source image under [CC BY 2.0](https://creativecommons.org/licenses/by/2.0/). The crop annotations are from Open Images and are licensed CC BY 4.0; see the dataset card for the source caveat and complete build method.",
        "",
        "| # | label | title | author | source | license | Open Images ID |",
        "|---:|---|---|---|---|---|---|",
    ]
    for row in rows:
        clean = lambda value: str(value or "").replace("|", "\\|").replace("\n", " ")
        lines.append(
            f"| {row['sample_position']} | {row['label_name']} | {clean(row['title'])} | "
            f"[{clean(row['author'])}]({row['author_profile_url']}) | "
            f"[original]({row['original_landing_url']}) | [CC BY 2.0]({row['license']}) | "
            f"`{row['source_image_id']}` |"
        )
    lines.extend(
        [
            "",
            "The machine-readable ledger is [`attribution.jsonl`](./attribution.jsonl). Its `pixel_sha256` hashes the decoded 128x128 RGB crop bytes, and `asset_box_px` locates that crop inside the generated grid.",
            "",
        ]
    )
    (output_dir / "ATTRIBUTION.md").write_text("\n".join(lines), encoding="utf-8")


def _svg_document(width: int, height: int, body: str) -> str:
    return f'''<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}" role="img">
  <rect width="100%" height="100%" fill="#f5f1e8"/>
  <style>
    .title {{ font: 700 36px system-ui, sans-serif; fill: #17201e; }}
    .head {{ font: 700 19px system-ui, sans-serif; fill: #17201e; }}
    .body {{ font: 15px system-ui, sans-serif; fill: #53605b; }}
    .small {{ font: 13px system-ui, sans-serif; fill: #53605b; }}
    .arrow {{ stroke: #6c7772; stroke-width: 3; fill: none; marker-end: url(#arrow); }}
  </style>
  <defs><marker id="arrow" markerWidth="10" markerHeight="10" refX="8" refY="3" orient="auto"><path d="M0,0 L0,6 L9,3 z" fill="#6c7772"/></marker></defs>
{body}
</svg>
'''


def write_task_flow(output: Path) -> None:
    boxes = [
        ("4,000 crops", "balanced + source-disjoint", "#fff", "#436b58"),
        ("Train", "augment 3,200 rows", "#fff", "#436b58"),
        ("Validation", "select checkpoint", "#fff", "#b56b39"),
        ("Held-out test", "one pass per model", "#fff", "#8c4b43"),
        ("Artifact + gate", "publish only at ≥95%", "#fff", "#5b5a83"),
    ]
    body = ['  <text x="60" y="62" class="title">Leakage-resistant evaluation flow</text>']
    start_x, y, box_w, box_h, gap = 60, 125, 205, 130, 34
    for index, (head, detail, fill, stroke) in enumerate(boxes):
        x = start_x + index * (box_w + gap)
        body.append(f'  <rect x="{x}" y="{y}" width="{box_w}" height="{box_h}" rx="18" fill="{fill}" stroke="{stroke}" stroke-width="3"/>')
        body.append(f'  <text x="{x + 18}" y="{y + 48}" class="head">{html.escape(head)}</text>')
        body.append(f'  <text x="{x + 18}" y="{y + 80}" class="body">{html.escape(detail)}</text>')
        if index < len(boxes) - 1:
            body.append(f'  <path d="M{x + box_w + 7} {y + box_h/2} H{x + box_w + gap - 8}" class="arrow"/>')
    body.extend(
        [
            '  <path d="M538 278 V336 H299 V278" stroke="#b56b39" stroke-width="3" stroke-dasharray="7 6" fill="none"/>',
            '  <text x="418" y="367" text-anchor="middle" class="body">training decisions use validation only</text>',
            '  <text x="60" y="438" class="small">Correction chosen from train/validation curves; its test pass is the second model measurement on this benchmark.</text>',
        ]
    )
    output.write_text(_svg_document(1300, 490, "\n".join(body)), encoding="utf-8")


def write_architecture(output: Path) -> None:
    body = [
        '  <text x="60" y="62" class="title">Staged ResNet18 transfer learning</text>',
        '  <text x="60" y="95" class="body">128×128 RGB input • torchvision IMAGENET1K_V1 initialization</text>',
    ]
    layers = [
        ("stem + layer1", "frozen", 210, "#dfe6e1", "#708078"),
        ("layer2", "frozen", 190, "#dfe6e1", "#708078"),
        ("layer3", "frozen", 190, "#dfe6e1", "#708078"),
        ("layer4", "epoch 5+: 1e-4", 230, "#f4dccb", "#b56b39"),
        ("2-class head", "warm 1e-3 → 5e-4", 230, "#d5e8dd", "#2f7654"),
    ]
    x, y, gap, height = 60, 150, 25, 145
    for index, (name, detail, width, fill, stroke) in enumerate(layers):
        body.append(f'  <rect x="{x}" y="{y}" width="{width}" height="{height}" rx="18" fill="{fill}" stroke="{stroke}" stroke-width="3"/>')
        body.append(f'  <text x="{x + 18}" y="{y + 54}" class="head">{html.escape(name)}</text>')
        body.append(f'  <text x="{x + 18}" y="{y + 88}" class="body">{html.escape(detail)}</text>')
        if index < len(layers) - 1:
            body.append(f'  <path d="M{x + width + 7} {y + height/2} H{x + width + gap - 8}" class="arrow"/>')
        x += width + gap
    body.extend(
        [
            '  <rect x="60" y="340" width="1160" height="112" rx="18" fill="#ffffff" stroke="#cfd5d1" stroke-width="2"/>',
            '  <text x="85" y="382" class="head">Two phases, one validation protocol</text>',
            '  <text x="85" y="416" class="body">Epochs 1–4 train the head. Epoch 5 onward trains layer4 + head with AdamW and cosine decay.</text>',
            '  <text x="85" y="440" class="body">Choose on validation. One test pass per model; corrected ResNet18 is the second model measured.</text>',
        ]
    )
    output.write_text(_svg_document(1280, 510, "\n".join(body)), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, default=Path("dataset/face-or-not-dataset"))
    parser.add_argument("--output", type=Path, default=Path("assets"))
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    dataset = load_from_disk(str(args.dataset))
    rows = _pick_training_rows(dataset["train"])
    if len(rows) != SAMPLE_COUNT_PER_CLASS * 2:
        raise RuntimeError("could not select the balanced training sample")
    if any(row["license"] != CC_BY_2 for row in rows):
        raise RuntimeError("every displayed source image must be listed as CC BY 2.0")
    ledger = write_dataset_grid(rows, args.output / "dataset_samples.png")
    write_attribution(ledger, args.output)
    write_task_flow(args.output / "task_flow.svg")
    write_architecture(args.output / "architecture.svg")
    print(
        json.dumps(
            {
                "dataset_samples": len(ledger),
                "labels": {name: sum(row["label"] == i for row in ledger) for i, name in enumerate(CLASS_NAMES)},
                "split": "train",
                "output": str(args.output),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
