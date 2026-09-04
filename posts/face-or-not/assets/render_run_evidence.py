#!/usr/bin/env python3
"""Render the final assets recoverable from archived Compute logs and result JSON."""

from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path

from render_results import write_confusion_matrix, write_training_curves


EPOCH_PATTERN = re.compile(
    r"^epoch (?P<epoch>\d+)/\d+ "
    r"train_loss=(?P<train_loss>\d+\.\d+) "
    r"validation_loss=(?P<validation_loss>\d+\.\d+) "
    r"validation_accuracy=(?P<validation_accuracy>\d+\.\d+)$",
    re.MULTILINE,
)
UNFREEZE_PATTERN = re.compile(r"^unfreezing layer4 at epoch (?P<epoch>\d+);", re.MULTILINE)


def parse_history(logs: str) -> list[dict]:
    unfreeze_match = UNFREEZE_PATTERN.search(logs)
    if not unfreeze_match:
        raise ValueError("logs do not contain the layer4 phase boundary")
    unfreeze_epoch = int(unfreeze_match.group("epoch"))
    history = []
    for match in EPOCH_PATTERN.finditer(logs):
        epoch = int(match.group("epoch"))
        history.append(
            {
                "epoch": epoch,
                "phase": "head_warmup" if epoch < unfreeze_epoch else "layer4_finetune",
                "train_loss": float(match.group("train_loss")),
                "validation_loss": float(match.group("validation_loss")),
                "validation_accuracy": float(match.group("validation_accuracy")),
            }
        )
    if not history or [row["epoch"] for row in history] != list(range(1, len(history) + 1)):
        raise ValueError("archived logs do not contain a contiguous epoch history")
    return history


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("evidence", type=Path)
    parser.add_argument("--output", type=Path, default=Path("assets/final-results"))
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    logs = (args.evidence / "logs.txt").read_text(encoding="utf-8")
    result_document = json.loads((args.evidence / "result.json").read_text(encoding="utf-8"))
    if result_document.get("status") != "succeeded":
        raise ValueError("result evidence is not terminal success")
    result = result_document["result"]
    history = parse_history(logs)
    best_epoch = int(result["best_epoch"])
    best_row = history[best_epoch - 1]
    if not math.isclose(
        best_row["validation_accuracy"],
        float(result["best_validation_accuracy"]),
        abs_tol=5e-5,
    ):
        raise ValueError("archived log accuracy does not match the structured result")
    if not math.isclose(
        best_row["validation_loss"],
        float(result["best_validation_loss"]),
        abs_tol=5e-5,
    ):
        raise ValueError("archived log loss does not match the structured result")

    metrics = {
        "best_epoch": best_epoch,
        "best_validation_accuracy": result["best_validation_accuracy"],
        "best_validation_loss": result["best_validation_loss"],
        "test": result["test_metrics"],
        "target_accuracy": result["target_accuracy"],
        "target_met": result["target_met"],
    }
    write_training_curves(history, metrics, args.output / "training_curves.svg")
    write_confusion_matrix(metrics, args.output / "confusion_matrix.svg")
    (args.output / "history.from-logs.json").write_text(
        json.dumps(history, indent=2) + "\n", encoding="utf-8"
    )
    (args.output / "metrics.from-result.json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8"
    )
    summary = {
        "run_id": result_document["run_id"],
        "source": "archived Compute logs and structured result",
        "history_precision": "four decimal places, as emitted in logs",
        "best_epoch": best_epoch,
        "best_validation_accuracy": result["best_validation_accuracy"],
        "test": result["test_metrics"],
        "target_met": result["target_met"],
        "rendered_files": [
            "training_curves.svg",
            "confusion_matrix.svg",
            "history.from-logs.json",
            "metrics.from-result.json",
        ],
        "unavailable_files": {
            "predictions.png": "Compute returned no artifact or prediction records",
            "PREDICTION_ATTRIBUTION.md": "Compute returned no artifact or prediction records",
            "model.pt": "Compute returned no completed artifact",
        },
    }
    (args.output / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, sort_keys=True))


if __name__ == "__main__":
    main()
