#!/usr/bin/env python3
"""Render final article assets from one completed, held-out Compute result."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

from datasets import load_from_disk
from PIL import Image, ImageDraw, ImageFont


CLASS_NAMES = ("no_face", "face")


def _font(size: int, *, bold: bool = False):
    candidates = [
        "/System/Library/Fonts/Supplemental/Arial Bold.ttf" if bold else "/System/Library/Fonts/Supplemental/Arial.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf" if bold else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    ]
    for candidate in candidates:
        if Path(candidate).exists():
            return ImageFont.truetype(candidate, size=size)
    return ImageFont.load_default()


def _read_inputs(artifact: Path):
    history = json.loads((artifact / "history.json").read_text(encoding="utf-8"))
    metrics = json.loads((artifact / "metrics.json").read_text(encoding="utf-8"))
    predictions = [
        json.loads(line)
        for line in (artifact / "test_predictions.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    required_history = {"epoch", "train_loss", "validation_loss", "validation_accuracy"}
    if not history or any(not required_history <= set(row) for row in history):
        raise ValueError("history.json is empty or missing required fields")
    if not predictions:
        raise ValueError("test_predictions.jsonl is empty")
    tn = sum(row["true_label"] == 0 and row["predicted_label"] == 0 for row in predictions)
    fp = sum(row["true_label"] == 0 and row["predicted_label"] == 1 for row in predictions)
    fn = sum(row["true_label"] == 1 and row["predicted_label"] == 0 for row in predictions)
    tp = sum(row["true_label"] == 1 and row["predicted_label"] == 1 for row in predictions)
    matrix = [[tn, fp], [fn, tp]]
    if matrix != metrics["test"]["confusion_matrix"]:
        raise ValueError("prediction records do not match metrics confusion matrix")
    accuracy = (tn + tp) / len(predictions)
    if not math.isclose(accuracy, metrics["test"]["accuracy"], abs_tol=1e-12):
        raise ValueError("prediction records do not match metrics accuracy")
    return history, metrics, predictions


def _polyline(values, *, x0, y0, width, height, low, high):
    points = []
    for index, value in enumerate(values):
        x = x0 + width * index / max(len(values) - 1, 1)
        y = y0 + height * (1 - (value - low) / max(high - low, 1e-12))
        points.append(f"{x:.1f},{y:.1f}")
    return " ".join(points)


def write_training_curves(history: list[dict], metrics: dict, output: Path) -> None:
    width, height = 1200, 620
    panel_w, panel_h, top = 475, 390, 130
    left_x, right_x = 85, 650
    train_loss = [float(row["train_loss"]) for row in history]
    val_loss = [float(row["validation_loss"]) for row in history]
    val_accuracy = [float(row["validation_accuracy"]) for row in history]
    loss_low = min(train_loss + val_loss)
    loss_high = max(train_loss + val_loss)
    loss_pad = max((loss_high - loss_low) * 0.08, 0.01)
    loss_low = max(0.0, loss_low - loss_pad)
    loss_high += loss_pad
    body = [
        '<rect width="100%" height="100%" fill="#f5f1e8"/>',
        '<style>.title{font:700 38px system-ui,sans-serif;fill:#17201e}.head{font:700 20px system-ui,sans-serif;fill:#17201e}.body{font:15px system-ui,sans-serif;fill:#53605b}.axis{stroke:#89938e;stroke-width:1}.grid{stroke:#d7dbd8;stroke-width:1}</style>',
        '<text x="65" y="58" class="title">Training and validation history</text>',
        f'<text x="{left_x}" y="105" class="head">Loss</text>',
        f'<text x="{right_x}" y="105" class="head">Validation accuracy</text>',
    ]
    for tick in range(6):
        frac = tick / 5
        y = top + panel_h * frac
        loss_value = loss_high - (loss_high - loss_low) * frac
        accuracy_value = 1 - 0.5 * frac
        body.extend(
            [
                f'<line x1="{left_x}" y1="{y:.1f}" x2="{left_x + panel_w}" y2="{y:.1f}" class="grid"/>',
                f'<text x="{left_x - 12}" y="{y + 5:.1f}" text-anchor="end" class="body">{loss_value:.2f}</text>',
                f'<line x1="{right_x}" y1="{y:.1f}" x2="{right_x + panel_w}" y2="{y:.1f}" class="grid"/>',
                f'<text x="{right_x - 12}" y="{y + 5:.1f}" text-anchor="end" class="body">{accuracy_value:.0%}</text>',
            ]
        )
    for x0 in (left_x, right_x):
        body.append(f'<line x1="{x0}" y1="{top + panel_h}" x2="{x0 + panel_w}" y2="{top + panel_h}" class="axis"/>')
        body.append(f'<text x="{x0 + panel_w/2}" y="{top + panel_h + 45}" text-anchor="middle" class="body">epoch</text>')
    body.extend(
        [
            f'<polyline points="{_polyline(train_loss,x0=left_x,y0=top,width=panel_w,height=panel_h,low=loss_low,high=loss_high)}" fill="none" stroke="#3d6b8a" stroke-width="4"/>',
            f'<polyline points="{_polyline(val_loss,x0=left_x,y0=top,width=panel_w,height=panel_h,low=loss_low,high=loss_high)}" fill="none" stroke="#c45c26" stroke-width="4"/>',
            f'<polyline points="{_polyline(val_accuracy,x0=right_x,y0=top,width=panel_w,height=panel_h,low=0.5,high=1.0)}" fill="none" stroke="#237a48" stroke-width="4"/>',
            f'<line x1="{right_x}" y1="{top + panel_h * 0.1:.1f}" x2="{right_x + panel_w}" y2="{top + panel_h * 0.1:.1f}" stroke="#8c4b43" stroke-width="2" stroke-dasharray="8 6"/>',
            f'<text x="{right_x + panel_w - 4}" y="{top + panel_h * 0.1 - 9:.1f}" text-anchor="end" class="body">95% target</text>',
            '<line x1="85" y1="582" x2="125" y2="582" stroke="#3d6b8a" stroke-width="4"/><text x="135" y="587" class="body">train loss</text>',
            '<line x1="245" y1="582" x2="285" y2="582" stroke="#c45c26" stroke-width="4"/><text x="295" y="587" class="body">validation loss</text>',
            '<line x1="650" y1="582" x2="690" y2="582" stroke="#237a48" stroke-width="4"/><text x="700" y="587" class="body">validation accuracy</text>',
        ]
    )
    best_epoch = int(metrics["best_epoch"])
    best_index = next(
        index
        for index, row in enumerate(history)
        if int(float(row["epoch"])) == best_epoch
    )
    best_x_offset = panel_w * best_index / max(len(history) - 1, 1)
    for panel_x in (left_x, right_x):
        best_x = panel_x + best_x_offset
        body.append(
            f'<line x1="{best_x:.1f}" y1="{top}" x2="{best_x:.1f}" '
            'y2="520" stroke="#5b5a83" stroke-width="2" stroke-dasharray="4 5"/>'
        )
        body.append(
            f'<text x="{best_x + 5:.1f}" y="{top + 18}" class="body">best epoch {best_epoch}</text>'
        )

    phase_index = next(
        (
            index
            for index in range(1, len(history))
            if history[index].get("phase") != history[index - 1].get("phase")
        ),
        None,
    )
    if phase_index is not None:
        phase_epoch = int(float(history[phase_index]["epoch"]))
        phase_x_offset = panel_w * phase_index / max(len(history) - 1, 1)
        for panel_x in (left_x, right_x):
            phase_x = panel_x + phase_x_offset
            body.append(
                f'<line x1="{phase_x:.1f}" y1="{top}" x2="{phase_x:.1f}" '
                'y2="520" stroke="#b56b39" stroke-width="2" stroke-dasharray="9 6"/>'
            )
            body.append(
                f'<text x="{phase_x + 5:.1f}" y="{top + 40}" class="body">layer4 starts, epoch {phase_epoch}</text>'
            )
    output.write_text(
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}" role="img">\n' + "\n".join(body) + "\n</svg>\n",
        encoding="utf-8",
    )


def write_confusion_matrix(metrics: dict, output: Path) -> None:
    matrix = metrics["test"]["confusion_matrix"]
    maximum = max(value for row in matrix for value in row) or 1
    cells = []
    labels = (("true negative", 0, 0), ("false positive", 0, 1), ("false negative", 1, 0), ("true positive", 1, 1))
    for label, row, column in labels:
        value = matrix[row][column]
        x, y = 225 + column * 255, 160 + row * 175
        percent = value / max(sum(matrix[row]), 1)
        opacity = 0.14 + 0.72 * value / maximum
        cells.append(f'<rect x="{x}" y="{y}" width="225" height="145" rx="18" fill="#35618f" fill-opacity="{opacity:.3f}"/>')
        cells.append(f'<text x="{x + 112.5}" y="{y + 64}" text-anchor="middle" class="value">{value}</text>')
        cells.append(f'<text x="{x + 112.5}" y="{y + 95}" text-anchor="middle" class="body">{percent:.1%} of row</text>')
        cells.append(f'<text x="{x + 112.5}" y="{y + 122}" text-anchor="middle" class="small">{label}</text>')
    accuracy = metrics["test"]["accuracy"]
    svg = f'''<svg xmlns="http://www.w3.org/2000/svg" width="900" height="590" viewBox="0 0 900 590" role="img">
<rect width="100%" height="100%" fill="#f5f1e8"/>
<style>.title{{font:700 36px system-ui,sans-serif;fill:#17201e}}.head{{font:700 18px system-ui,sans-serif;fill:#17201e}}.value{{font:700 42px system-ui,sans-serif;fill:#17201e}}.body{{font:16px system-ui,sans-serif;fill:#35413d}}.small{{font:13px system-ui,sans-serif;fill:#35413d}}</style>
<text x="55" y="58" class="title">Held-out confusion matrix</text>
<text x="55" y="92" class="body">accuracy {accuracy:.1%} • {sum(sum(row) for row in matrix)} test crops • this model's one test pass</text>
<text x="465" y="128" text-anchor="middle" class="head">Predicted label</text>
<text x="337" y="150" text-anchor="middle" class="body">no face</text><text x="592" y="150" text-anchor="middle" class="body">face</text>
<text x="105" y="320" transform="rotate(-90 105 320)" text-anchor="middle" class="head">True label</text>
<text x="205" y="238" text-anchor="end" class="body">no face</text><text x="205" y="413" text-anchor="end" class="body">face</text>
{''.join(cells)}
</svg>
'''
    output.write_text(svg, encoding="utf-8")


def _selected_predictions(predictions: list[dict], limit: int = 16) -> list[dict]:
    mistakes = sorted(
        (row for row in predictions if not row["correct"]),
        key=lambda row: (-abs(float(row["face_probability"]) - 0.5), row["source_image_id"]),
    )
    successes = sorted(
        (row for row in predictions if row["correct"]),
        key=lambda row: (abs(float(row["face_probability"]) - 0.5), row["source_image_id"]),
    )
    count = min(len(mistakes), limit // 2)
    result = mistakes[:count] + successes[: limit - count]
    if len(result) < limit:
        result.extend(mistakes[count : count + limit - len(result)])
    return result


def write_predictions(
    predictions: list[dict],
    test_split,
    output: Path,
    ledger_path: Path,
    attribution_path: Path,
    model_label: str,
) -> None:
    selected = _selected_predictions(predictions)
    indexes = {source_id: index for index, source_id in enumerate(test_split["source_image_id"])}
    width, height = 1280, 1245
    canvas = Image.new("RGB", (width, height), "#f5f1e8")
    draw = ImageDraw.Draw(canvas)
    draw.text(
        (52, 38),
        f"Held-out predictions — {model_label}",
        fill="#17201e",
        font=_font(42, bold=True),
    )
    draw.text((54, 99), "Errors first, then closest correct calls • selected after this model's test pass", fill="#53605b", font=_font(20))
    margin, gap, top, card_w, card_h = 52, 20, 154, 279, 252
    image_size = 174
    ledger = []
    for position, prediction in enumerate(selected):
        row = test_split[indexes[prediction["source_image_id"]]]
        column, row_number = position % 4, position // 4
        x, y = margin + column * (card_w + gap), top + row_number * (card_h + gap)
        border = "#237a48" if prediction["correct"] else "#a32626"
        draw.rounded_rectangle((x, y, x + card_w, y + card_h), radius=14, fill="#fff", outline=border, width=4)
        image = row["image"].convert("RGB").resize(
            (image_size, image_size), Image.Resampling.LANCZOS
        )
        image_x = x + (card_w - image_size) // 2
        canvas.paste(image, (image_x, y + 7))
        draw.text((x + 10, y + 190), f"true {prediction['true_label_name']}  →  {prediction['predicted_label_name']}", fill=border, font=_font(17, bold=True))
        draw.text((x + 10, y + 220), f"p(face) {float(prediction['face_probability']):.3f}", fill="#53605b", font=_font(16))
        ledger.append(
            {
                "asset": output.name,
                "sample_position": position + 1,
                "asset_box_px": [
                    image_x,
                    y + 7,
                    image_x + image_size,
                    y + 7 + image_size,
                ],
                "split": "test",
                "source_image_id": row["source_image_id"],
                "true_label": prediction["true_label_name"],
                "predicted_label": prediction["predicted_label_name"],
                "face_probability": prediction["face_probability"],
                "correct": prediction["correct"],
                "author": row["author"],
                "author_profile_url": row["author_profile_url"],
                "title": row["title"],
                "original_landing_url": row["original_landing_url"],
                "license": row["license"],
                "crop_norm_json": row["crop_norm_json"],
                "pixel_sha256": hashlib.sha256(row["image"].convert("RGB").tobytes()).hexdigest(),
            }
        )
    canvas.save(output, format="PNG", optimize=True)
    with ledger_path.open("w", encoding="utf-8") as handle:
        for row in ledger:
            handle.write(json.dumps(row, sort_keys=True, ensure_ascii=False) + "\n")
    lines = [
        f"# Prediction-grid attribution — {model_label}",
        "",
        f"These held-out examples were selected automatically after the {model_label} test pass for reporting only. The label identifies the model that produced this grid; these images must not be attributed to another checkpoint. They were not used to choose the corrected architecture. Open Images lists every displayed source image under [CC BY 2.0](https://creativecommons.org/licenses/by/2.0/).",
        "",
        "| # | true → predicted | title | author | source | license | Open Images ID |",
        "|---:|---|---|---|---|---|---|",
    ]
    for row in ledger:
        title = str(row["title"] or "").replace("|", "\\|").replace("\n", " ")
        author = str(row["author"] or "").replace("|", "\\|").replace("\n", " ")
        lines.append(
            f"| {row['sample_position']} | `{row['true_label']} → {row['predicted_label']}` | "
            f"{title} | [{author}]({row['author_profile_url']}) | "
            f"[original]({row['original_landing_url']}) | [CC BY 2.0]({row['license']}) | "
            f"`{row['source_image_id']}` |"
        )
    lines.extend(
        [
            "",
            "Machine-readable attribution, probabilities, outcomes, crop coordinates, and decoded-pixel hashes are in [`prediction_attribution.jsonl`](./prediction_attribution.jsonl).",
            "",
        ]
    )
    attribution_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("artifact", type=Path)
    parser.add_argument("--dataset", type=Path, default=Path("dataset/face-or-not-dataset"))
    parser.add_argument("--output", type=Path, default=Path("assets/results"))
    parser.add_argument("--model-label", default="model")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    history, metrics, predictions = _read_inputs(args.artifact)
    dataset = load_from_disk(str(args.dataset))
    if len(predictions) != len(dataset["test"]):
        raise ValueError("prediction count does not match held-out dataset")
    if set(row["source_image_id"] for row in predictions) != set(dataset["test"]["source_image_id"]):
        raise ValueError("prediction IDs do not match the held-out dataset")
    write_training_curves(history, metrics, args.output / "training_curves.svg")
    write_confusion_matrix(metrics, args.output / "confusion_matrix.svg")
    write_predictions(
        predictions,
        dataset["test"],
        args.output / "predictions.png",
        args.output / "prediction_attribution.jsonl",
        args.output / "PREDICTION_ATTRIBUTION.md",
        args.model_label,
    )
    summary = {
        "artifact": str(args.artifact),
        "best_epoch": metrics["best_epoch"],
        "best_validation_accuracy": metrics["best_validation_accuracy"],
        "test": metrics["test"],
        "target_met": metrics["target_met"],
        "model_label": args.model_label,
        "rendered_files": [
            "training_curves.svg",
            "confusion_matrix.svg",
            "predictions.png",
            "prediction_attribution.jsonl",
            "PREDICTION_ATTRIBUTION.md",
        ],
    }
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, sort_keys=True))


if __name__ == "__main__":
    main()
