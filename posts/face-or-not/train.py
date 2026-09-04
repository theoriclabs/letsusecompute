"""Fine-tune a compact ResNet18 to classify 128x128 crops as face or no face.

Dataset: ``theoriclabs/face-or-not`` on Hugging Face. Each run evaluates the
``test`` split once after choosing the best epoch on ``validation``. The staged
ResNet18 is the second model measured on this benchmark after a random-CNN
baseline; its architecture was chosen from training/validation curves without
inspecting held-out identities or images.

    compute run train.py::train --gpu cheap --dry-run
    compute run train.py::train --gpu cheap --timeout 1800 --wait

This is image classification. It does not locate faces or return boxes.
"""

from __future__ import annotations

import json
import math
import os
import tempfile
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import compute

app = compute.App("face-or-not")
image = compute.Image.cuda_pytorch().pip_install(
    "--extra-index-url",
    "https://download.pytorch.org/whl/cu124",
    "torch==2.5.1+cu124",
    "torchvision==0.20.1+cu124",
    "datasets==3.6.0",
    "huggingface_hub>=0.33,<1",
    "Pillow>=10,<12",
)
hf_secret = compute.Secret.from_name("hf")

DEFAULT_DATASET = "theoriclabs/face-or-not"
DEFAULT_DATASET_REVISION = "7c81a2453be2cb48bb0b48192db91a826108e3de"
DEFAULT_DATASET_MANIFEST_SHA256 = (
    "8026d6deb5fcf6f9d9945c996d73a7dc69322ff466b05269591cd95d3105e8fa"
)
DEFAULT_HUB_REPO = "theoriclabs/face-or-not-cnn"
CLASS_NAMES = ("no_face", "face")
TARGET_ACCURACY = 0.95
WORKLOAD_SUBDIR = "face-or-not"
ARTIFACT_NAME = "face-or-not-cnn"
ARTIFACT_MARKER = ".compute-artifact.json"
DEFAULT_ARTIFACT_FALLBACK = Path("/tmp/compute-face-or-not")


def _bridge_hf_token() -> None:
    token = (
        os.environ.get("HF_TOKEN")
        or os.environ.get("HUGGING_FACE_HUB_TOKEN")
        or os.environ.get("hf")
    )
    if token:
        os.environ.setdefault("HF_TOKEN", token)
        os.environ.setdefault("HUGGING_FACE_HUB_TOKEN", token)


def resolve_artifact_dir(env: Mapping[str, str] | None = None) -> Path:
    environ = os.environ if env is None else env
    base = environ.get("COMPUTE_ARTIFACT_DIR")
    root = Path(base) / WORKLOAD_SUBDIR if base else DEFAULT_ARTIFACT_FALLBACK
    root.mkdir(parents=True, exist_ok=True)
    return root


def write_artifact_marker(
    directory: Path | str,
    *,
    name: str,
    compatibility_key: str,
    metadata: dict[str, Any],
) -> Path:
    """Atomically write Compute's artifact marker."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    marker_path = directory / ARTIFACT_MARKER
    payload = {
        "compatibility_key": compatibility_key,
        "kind": "output",
        "metadata": metadata,
        "name": name,
        "version": 1,
    }
    data = json.dumps(payload, sort_keys=True, indent=2) + "\n"
    fd, temporary = tempfile.mkstemp(
        prefix=".compute-artifact.", suffix=".tmp", dir=str(directory)
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, marker_path)
        directory_fd = os.open(str(directory), os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    except Exception:
        try:
            os.unlink(temporary)
        except OSError:
            pass
        raise
    return marker_path


def write_training_curves(directory: Path, history: list[dict[str, Any]]) -> None:
    """Write history JSON and dependency-free loss/accuracy SVG panels."""
    (directory / "history.json").write_text(
        json.dumps(history, indent=2) + "\n", encoding="utf-8"
    )
    width, height, pad = 720, 440, 42
    panel_height = 160
    losses = [row[key] for row in history for key in ("train_loss", "validation_loss")]
    low, high = (min(losses), max(losses)) if losses else (0.0, 1.0)
    if high <= low:
        high = low + 1e-6

    def point(index: int, value: float, *, top: int, bottom: int) -> str:
        x = pad + (width - 2 * pad) * (index / max(len(history) - 1, 1))
        y = top + (bottom - top) * (1 - (value - low) / (high - low))
        return f"{x:.1f},{y:.1f}"

    def line(key: str) -> str:
        return " ".join(
            point(index, row[key], top=42, bottom=42 + panel_height)
            for index, row in enumerate(history)
        )

    def accuracy_line() -> str:
        return " ".join(
            f"{pad + (width - 2 * pad) * (index / max(len(history) - 1, 1)):.1f},"
            f"{246 + panel_height * (1 - row['validation_accuracy']):.1f}"
            for index, row in enumerate(history)
        )

    svg = f"""<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">
  <rect width="100%" height="100%" fill="#fff"/>
  <text x="{pad}" y="22" font-size="13" fill="#333">train loss (steel) vs validation loss (orange)</text>
  <polyline fill="none" stroke="#3d6b8a" stroke-width="2" points="{line('train_loss')}"/>
  <polyline fill="none" stroke="#c45c26" stroke-width="2" points="{line('validation_loss')}"/>
  <text x="{pad}" y="220" font-size="11" fill="#666">loss min {low:.3f} · max {high:.3f}</text>
  <text x="{pad}" y="238" font-size="13" fill="#333">validation accuracy (green)</text>
  <line x1="{pad}" y1="{246 + panel_height * 0.05:.1f}" x2="{width - pad}" y2="{246 + panel_height * 0.05:.1f}" stroke="#999" stroke-dasharray="5 4"/>
  <polyline fill="none" stroke="#237a48" stroke-width="2" points="{accuracy_line()}"/>
  <text x="{pad}" y="{height - 8}" font-size="11" fill="#666">dashed target 0.95 · {len(history)} epochs</text>
</svg>
"""
    (directory / "training_curves.svg").write_text(svg, encoding="utf-8")


def _resnet18_transfer_model(num_classes: int = 2, *, pretrained: bool = True):
    import torch.nn as nn
    from torchvision.models import ResNet18_Weights, resnet18

    weights = ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
    model = resnet18(weights=weights)
    for parameter in model.parameters():
        parameter.requires_grad = False
    in_features = model.fc.in_features
    model.fc = nn.Sequential(nn.Dropout(p=0.2), nn.Linear(in_features, num_classes))
    return model


def _set_transfer_phase(model, *, fine_tune_layer4: bool) -> None:
    """Freeze early features and control BatchNorm state for the active phase."""
    for parameter in model.parameters():
        parameter.requires_grad = False
    for parameter in model.fc.parameters():
        parameter.requires_grad = True
    if fine_tune_layer4:
        for parameter in model.layer4.parameters():
            parameter.requires_grad = True

    model.train()
    model.conv1.eval()
    model.bn1.eval()
    model.layer1.eval()
    model.layer2.eval()
    model.layer3.eval()
    if fine_tune_layer4:
        model.layer4.train()
    else:
        model.layer4.eval()
    model.fc.train()


def _load_dataset(
    dataset_id: str,
    dataset_revision: str,
    seed: int,
    max_train: int,
    max_validation: int,
    max_test: int,
):
    from datasets import load_dataset

    if dataset_id == DEFAULT_DATASET and dataset_revision != DEFAULT_DATASET_REVISION:
        raise ValueError(
            "the published face-or-not dataset must use its verified revision "
            f"{DEFAULT_DATASET_REVISION}"
        )
    dataset = load_dataset(dataset_id, revision=dataset_revision)
    missing = {"train", "validation", "test"} - set(dataset)
    if missing:
        raise ValueError(f"dataset {dataset_id!r} is missing splits: {sorted(missing)}")

    features = dataset["train"].features
    label_feature = features.get("label")
    names = tuple(getattr(label_feature, "names", ()))
    if names and names != CLASS_NAMES:
        raise ValueError(f"expected label names {CLASS_NAMES}, got {names}")
    for split in ("train", "validation", "test"):
        if "source_image_id" not in dataset[split].column_names:
            raise ValueError(f"{split} is missing source_image_id needed for leakage check")
        source_ids = dataset[split]["source_image_id"]
        if len(source_ids) != len(set(source_ids)):
            raise ValueError(f"{split} contains duplicate source_image_id values")
        labels = [int(label) for label in dataset[split]["label"]]
        counts = {label: labels.count(label) for label in (0, 1)}
        if set(labels) != {0, 1} or counts[0] != counts[1]:
            raise ValueError(f"{split} must be balanced across labels 0 and 1, got {counts}")

    source_sets = {split: set(dataset[split]["source_image_id"]) for split in dataset}
    if source_sets["train"] & source_sets["validation"]:
        raise ValueError("source-image leakage between train and validation")
    if source_sets["train"] & source_sets["test"]:
        raise ValueError("source-image leakage between train and test")
    if source_sets["validation"] & source_sets["test"]:
        raise ValueError("source-image leakage between validation and test")

    limits = {"train": max_train, "validation": max_validation, "test": max_test}
    for split, limit in limits.items():
        if limit <= 0:
            raise ValueError(f"{split} limit must be positive")
        if limit > len(dataset[split]):
            raise ValueError(
                f"{split} has {len(dataset[split])} rows, fewer than requested {limit}"
            )
        if limit < len(dataset[split]):
            dataset[split] = _balanced_subset(dataset[split], limit, seed)
    return dataset


def _balanced_subset(split, limit: int, seed: int):
    import hashlib

    if limit % 2:
        raise ValueError(f"balanced subset limit must be even, got {limit}")
    per_label = limit // 2
    selected: list[int] = []
    labels = split["label"]
    source_ids = split["source_image_id"]
    for label in (0, 1):
        indexes = [index for index, value in enumerate(labels) if int(value) == label]
        indexes.sort(
            key=lambda index: hashlib.sha256(
                f"{seed}:{source_ids[index]}".encode()
            ).hexdigest()
        )
        selected.extend(indexes[:per_label])
    if len(selected) != per_label * 2:
        raise ValueError(f"cannot make balanced subset of {limit} rows")
    return split.select(sorted(selected))


def _metrics_from_counts(tn: int, fp: int, fn: int, tp: int) -> dict[str, Any]:
    total = tn + fp + fn + tp
    accuracy = (tn + tp) / max(total, 1)
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-12)
    return {
        "accuracy": accuracy,
        "face_precision": precision,
        "face_recall": recall,
        "face_f1": f1,
        "confusion_matrix": [[tn, fp], [fn, tp]],
    }


def _evaluate(
    model,
    loader,
    loss_fn,
    device,
    *,
    collect_predictions: bool = False,
) -> dict[str, Any]:
    import torch

    model.eval()
    loss_sum = 0.0
    total = 0
    tn = fp = fn = tp = 0
    prediction_records: list[dict[str, Any]] = []
    with torch.no_grad():
        for inputs, labels, source_ids in loader:
            inputs = inputs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            logits = model(inputs)
            loss_sum += float(loss_fn(logits, labels).detach()) * labels.size(0)
            predictions = logits.argmax(dim=1)
            tn += int(((labels == 0) & (predictions == 0)).sum().item())
            fp += int(((labels == 0) & (predictions == 1)).sum().item())
            fn += int(((labels == 1) & (predictions == 0)).sum().item())
            tp += int(((labels == 1) & (predictions == 1)).sum().item())
            total += labels.size(0)
            if collect_predictions:
                face_probabilities = torch.softmax(logits, dim=1)[:, 1]
                for source_id, true, predicted, probability in zip(
                    source_ids,
                    labels.detach().cpu().tolist(),
                    predictions.detach().cpu().tolist(),
                    face_probabilities.detach().cpu().tolist(),
                ):
                    prediction_records.append(
                        {
                            "source_image_id": source_id,
                            "true_label": int(true),
                            "true_label_name": CLASS_NAMES[int(true)],
                            "predicted_label": int(predicted),
                            "predicted_label_name": CLASS_NAMES[int(predicted)],
                            "face_probability": float(probability),
                            "correct": int(true) == int(predicted),
                        }
                    )
    result = {
        "loss": loss_sum / max(total, 1),
        **_metrics_from_counts(tn, fp, fn, tp),
    }
    if collect_predictions:
        result["predictions"] = prediction_records
    return result


def write_prediction_records(
    directory: Path,
    records: list[dict[str, Any]],
) -> Path:
    path = directory / "test_predictions.jsonl"
    with path.open("w", encoding="utf-8") as handle:
        for row in records:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
    return path


def _select_prediction_examples(
    records: list[dict[str, Any]],
    limit: int = 16,
) -> list[dict[str, Any]]:
    mistakes = sorted(
        (row for row in records if not row["correct"]),
        key=lambda row: (
            -abs(row["face_probability"] - 0.5),
            row["source_image_id"],
        ),
    )
    successes = sorted(
        (row for row in records if row["correct"]),
        key=lambda row: (
            abs(row["face_probability"] - 0.5),
            row["source_image_id"],
        ),
    )
    mistake_count = min(len(mistakes), limit // 2)
    selected = mistakes[:mistake_count] + successes[: limit - mistake_count]
    if len(selected) < limit:
        selected.extend(mistakes[mistake_count : mistake_count + limit - len(selected)])
    return selected


def write_prediction_grid(
    directory: Path,
    records: list[dict[str, Any]],
    test_split,
) -> Path:
    from PIL import Image, ImageDraw

    selected = _select_prediction_examples(records)
    if not selected:
        raise RuntimeError("cannot make a prediction grid without predictions")
    indexes = {
        source_id: index
        for index, source_id in enumerate(test_split["source_image_id"])
    }
    columns = 4
    image_size = 128
    caption_height = 42
    cell_height = image_size + caption_height
    rows = math.ceil(len(selected) / columns)
    canvas = Image.new("RGB", (columns * image_size, rows * cell_height), "white")
    draw = ImageDraw.Draw(canvas)
    for position, record in enumerate(selected):
        x = position % columns * image_size
        y = position // columns * cell_height
        image = test_split[indexes[record["source_image_id"]]]["image"].convert("RGB")
        canvas.paste(image.resize((image_size, image_size)), (x, y))
        color = "#176b35" if record["correct"] else "#a32626"
        draw.rectangle((x, y + image_size, x + image_size, y + cell_height), fill="#ffffff")
        draw.text(
            (x + 3, y + image_size + 3),
            f"t:{record['true_label_name']} p:{record['predicted_label_name']}",
            fill=color,
        )
        draw.text(
            (x + 3, y + image_size + 20),
            f"p(face)={record['face_probability']:.3f} | "
            f"{'OK' if record['correct'] else 'WRONG'}",
            fill=color,
        )
    path = directory / "prediction_grid.png"
    canvas.save(path, format="PNG", optimize=True)
    return path


def write_confusion_matrix(directory: Path, matrix: list[list[int]]) -> Path:
    tn, fp = matrix[0]
    fn, tp = matrix[1]
    values = [tn, fp, fn, tp]
    maximum = max(values) or 1

    def cell(x: int, y: int, value: int, label: str) -> str:
        opacity = 0.15 + 0.75 * value / maximum
        return f"""  <rect x="{x}" y="{y}" width="180" height="110" rx="10" fill="#35618f" fill-opacity="{opacity:.3f}"/>
  <text x="{x + 90}" y="{y + 48}" text-anchor="middle" font-size="30" fill="#111">{value}</text>
  <text x="{x + 90}" y="{y + 78}" text-anchor="middle" font-size="13" fill="#222">{label}</text>"""

    svg = f"""<svg xmlns="http://www.w3.org/2000/svg" width="540" height="340" viewBox="0 0 540 340">
  <rect width="100%" height="100%" fill="#fff"/>
  <text x="300" y="24" text-anchor="middle" font-size="15" fill="#333">Predicted label</text>
  <text x="245" y="48" text-anchor="middle" font-size="13" fill="#555">no_face</text>
  <text x="435" y="48" text-anchor="middle" font-size="13" fill="#555">face</text>
  <text x="24" y="175" transform="rotate(-90 24 175)" text-anchor="middle" font-size="15" fill="#333">True label</text>
  <text x="100" y="122" text-anchor="middle" font-size="13" fill="#555">no_face</text>
  <text x="100" y="242" text-anchor="middle" font-size="13" fill="#555">face</text>
{cell(155, 60, tn, 'true negative')}
{cell(345, 60, fp, 'false positive')}
{cell(155, 180, fn, 'false negative')}
{cell(345, 180, tp, 'true positive')}
</svg>
"""
    path = directory / "confusion_matrix.svg"
    path.write_text(svg, encoding="utf-8")
    return path


def _write_model_card(
    directory: Path,
    metrics: dict[str, Any],
    dataset_id: str,
    dataset_revision: str,
    dataset_manifest_sha256: str,
    param_count: int,
    trainable_param_count: int,
) -> None:
    text = f"""---
license: other
library_name: pytorch
pipeline_tag: image-classification
datasets:
- {dataset_id}
---

# Face or Not ResNet18

A {param_count:,}-parameter ResNet18 initialized from torchvision's
`IMAGENET1K_V1` weights and adapted on `{dataset_id}`. The linear head is warmed
first; then only `layer4` and the head are fine-tuned ({trainable_param_count:,}
parameters in the second stage). Earlier layers stay frozen.
It classifies a 128x128 crop as `no_face` or `face`; it does not locate,
identify, or recognize people.

Dataset revision: `{dataset_revision}`

Dataset manifest SHA-256: `{dataset_manifest_sha256}`

Held-out test accuracy: {metrics['accuracy']:.4f}

Face precision: {metrics['face_precision']:.4f}

Face recall: {metrics['face_recall']:.4f}

Face F1: {metrics['face_f1']:.4f}

The training images are derived from Open Images V7. See the dataset card for
per-image attribution, licenses, intended use, and limitations.

No permissive checkpoint license is asserted here. Any public release must
account for the terms associated with torchvision's ImageNet-pretrained weights,
the dataset's per-image CC BY 2.0 terms, and its CC BY 4.0 annotations.
"""
    (directory / "README.md").write_text(text, encoding="utf-8")


def _train_impl(
    *,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    weight_decay: float,
    image_size: int,
    max_train: int,
    max_validation: int,
    max_test: int,
    dataset_id: str,
    dataset_revision: str,
    seed: int,
    patience: int,
    warmup_epochs: int,
    backbone_learning_rate: float,
    push_to_hub: bool,
    hub_repo: str,
) -> dict[str, Any]:
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, Dataset
    from torchvision import transforms

    if image_size != 128:
        raise ValueError("this issue is scoped to 128x128 crops")
    if epochs <= 0 or batch_size <= 0 or patience <= 0:
        raise ValueError("epochs, batch_size, and patience must be positive")
    if warmup_epochs <= 0 or warmup_epochs >= epochs:
        raise ValueError("warmup_epochs must be positive and less than epochs")
    if learning_rate <= 0 or backbone_learning_rate <= 0 or weight_decay < 0:
        raise ValueError("learning rates must be positive and weight decay non-negative")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required (torch.cuda.is_available() is False)")

    _bridge_hf_token()
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    device = torch.device("cuda")
    dataset_manifest_sha256 = (
        DEFAULT_DATASET_MANIFEST_SHA256 if dataset_id == DEFAULT_DATASET else "unknown"
    )
    dataset = _load_dataset(
        dataset_id,
        dataset_revision,
        seed,
        max_train,
        max_validation,
        max_test,
    )

    train_transform = transforms.Compose(
        [
            transforms.RandomResizedCrop(image_size, scale=(0.85, 1.0), ratio=(0.9, 1.1)),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.1),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )
    eval_transform = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )

    class VisionSplit(Dataset):
        def __init__(self, split, transform) -> None:
            self.split = split
            self.transform = transform

        def __len__(self) -> int:
            return len(self.split)

        def __getitem__(self, index: int):
            row = self.split[index]
            return (
                self.transform(row["image"].convert("RGB")),
                int(row["label"]),
                row["source_image_id"],
            )

    generator = torch.Generator().manual_seed(seed)
    loaders = {
        "train": DataLoader(
            VisionSplit(dataset["train"], train_transform),
            batch_size=batch_size,
            shuffle=True,
            num_workers=2,
            pin_memory=True,
            generator=generator,
        ),
        "validation": DataLoader(
            VisionSplit(dataset["validation"], eval_transform),
            batch_size=batch_size,
            shuffle=False,
            num_workers=2,
            pin_memory=True,
        ),
        "test": DataLoader(
            VisionSplit(dataset["test"], eval_transform),
            batch_size=batch_size,
            shuffle=False,
            num_workers=2,
            pin_memory=True,
        ),
    }

    model = _resnet18_transfer_model().to(device)
    param_count = sum(parameter.numel() for parameter in model.parameters())
    _set_transfer_phase(model, fine_tune_layer4=False)
    optimizer = torch.optim.AdamW(
        model.fc.parameters(), lr=learning_rate, weight_decay=weight_decay
    )
    scheduler = None
    loss_fn = nn.CrossEntropyLoss(label_smoothing=0.05)

    history: list[dict[str, Any]] = []
    best_validation = -math.inf
    best_validation_loss = math.inf
    best_epoch = 0
    best_state: dict[str, Any] | None = None
    epochs_without_improvement = 0

    for epoch in range(1, epochs + 1):
        if epoch == warmup_epochs + 1:
            _set_transfer_phase(model, fine_tune_layer4=True)
            optimizer = torch.optim.AdamW(
                [
                    {"params": model.fc.parameters(), "lr": learning_rate * 0.5},
                    {"params": model.layer4.parameters(), "lr": backbone_learning_rate},
                ],
                weight_decay=weight_decay,
            )
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=epochs - warmup_epochs
            )
            epochs_without_improvement = 0
            print(
                f"unfreezing layer4 at epoch {epoch}; "
                f"head_lr={learning_rate * 0.5:g} "
                f"backbone_lr={backbone_learning_rate:g}",
                flush=True,
            )
        _set_transfer_phase(model, fine_tune_layer4=epoch > warmup_epochs)
        train_loss_sum = 0.0
        seen = 0
        for inputs, labels, _source_ids in loaders["train"]:
            inputs = inputs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            logits = model(inputs)
            loss = loss_fn(logits, labels)
            loss.backward()
            optimizer.step()
            train_loss_sum += float(loss.detach()) * labels.size(0)
            seen += labels.size(0)
        validation = _evaluate(model, loaders["validation"], loss_fn, device)
        row = {
            "epoch": float(epoch),
            "phase": "head_warmup" if epoch <= warmup_epochs else "layer4_finetune",
            "head_learning_rate": float(optimizer.param_groups[0]["lr"]),
            "backbone_learning_rate": (
                0.0 if epoch <= warmup_epochs else float(optimizer.param_groups[1]["lr"])
            ),
            "train_loss": train_loss_sum / max(seen, 1),
            "validation_loss": float(validation["loss"]),
            "validation_accuracy": float(validation["accuracy"]),
            "validation_face_f1": float(validation["face_f1"]),
        }
        history.append(row)
        print(
            f"epoch {epoch}/{epochs} train_loss={row['train_loss']:.4f} "
            f"validation_loss={row['validation_loss']:.4f} "
            f"validation_accuracy={row['validation_accuracy']:.4f}",
            flush=True,
        )

        validation_accuracy = float(validation["accuracy"])
        validation_loss = float(validation["loss"])
        improved = validation_accuracy > best_validation or (
            validation_accuracy == best_validation
            and validation_loss < best_validation_loss
        )
        if improved:
            best_validation = validation_accuracy
            best_validation_loss = validation_loss
            best_epoch = epoch
            best_state = {
                name: tensor.detach().cpu().clone()
                for name, tensor in model.state_dict().items()
            }
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
        if scheduler is not None:
            scheduler.step()
        if epoch >= warmup_epochs + patience and epochs_without_improvement >= patience:
            print(f"early stopping at epoch {epoch}; best epoch was {best_epoch}", flush=True)
            break

    if best_state is None:
        raise RuntimeError("training did not produce a checkpoint")
    model.load_state_dict(best_state)
    model.to(device)
    test_evaluation = _evaluate(
        model,
        loaders["test"],
        loss_fn,
        device,
        collect_predictions=True,
    )
    prediction_records = test_evaluation.pop("predictions")
    test_metrics = test_evaluation
    target_met = float(test_metrics["accuracy"]) >= TARGET_ACCURACY
    print(
        f"held-out test accuracy={test_metrics['accuracy']:.4f} "
        f"face_f1={test_metrics['face_f1']:.4f} target_met={target_met}",
        flush=True,
    )

    output = resolve_artifact_dir()
    checkpoint = {
        "state_dict": best_state,
        "class_names": list(CLASS_NAMES),
        "image_size": image_size,
        "param_count": param_count,
        "architecture": "resnet18",
        "pretrained_weights": "IMAGENET1K_V1",
        "warmup_epochs": warmup_epochs,
        "backbone_learning_rate": backbone_learning_rate,
        "dataset_id": dataset_id,
        "dataset_revision": dataset_revision,
        "dataset_manifest_sha256": dataset_manifest_sha256,
        "best_epoch": best_epoch,
        "test_metrics": test_metrics,
    }
    torch.save(checkpoint, output / "model.pt")
    metrics_document = {
        "dataset_id": dataset_id,
        "dataset_revision": dataset_revision,
        "dataset_manifest_sha256": dataset_manifest_sha256,
        "architecture": "resnet18",
        "pretrained_weights": "IMAGENET1K_V1",
        "param_count": param_count,
        "warmup_epochs": warmup_epochs,
        "backbone_learning_rate": backbone_learning_rate,
        "split_sizes": {split: len(dataset[split]) for split in dataset},
        "best_epoch": best_epoch,
        "best_validation_accuracy": best_validation,
        "best_validation_loss": best_validation_loss,
        "test": test_metrics,
        "target_accuracy": TARGET_ACCURACY,
        "target_met": target_met,
        "seed": seed,
    }
    (output / "metrics.json").write_text(
        json.dumps(metrics_document, indent=2) + "\n", encoding="utf-8"
    )
    write_training_curves(output, history)
    write_prediction_records(output, prediction_records)
    write_prediction_grid(output, prediction_records, dataset["test"])
    write_confusion_matrix(output, test_metrics["confusion_matrix"])
    trainable_param_count = sum(
        parameter.numel() for parameter in model.parameters() if parameter.requires_grad
    )
    _write_model_card(
        output,
        test_metrics,
        dataset_id,
        dataset_revision,
        dataset_manifest_sha256,
        param_count,
        trainable_param_count,
    )
    write_artifact_marker(
        output,
        name=ARTIFACT_NAME,
        compatibility_key=f"face-or-not-resnet18-v2-{image_size}",
        metadata={
            "filename": "model.pt",
            "dataset_id": dataset_id,
            "dataset_revision": dataset_revision,
            "dataset_manifest_sha256": dataset_manifest_sha256,
            "class_names": list(CLASS_NAMES),
            "image_size": image_size,
            "param_count": param_count,
            "trainable_param_count": trainable_param_count,
            "architecture": "resnet18",
            "pretrained_weights": "IMAGENET1K_V1",
            "best_epoch": best_epoch,
            "test_accuracy": test_metrics["accuracy"],
            "target_met": target_met,
            "files": [
                "model.pt",
                "metrics.json",
                "history.json",
                "training_curves.svg",
                "test_predictions.jsonl",
                "prediction_grid.png",
                "confusion_matrix.svg",
                "README.md",
            ],
        },
    )

    hub_url = None
    publish_skipped_reason = None
    if push_to_hub and not target_met:
        publish_skipped_reason = (
            f"held-out accuracy {test_metrics['accuracy']:.4f} is below "
            f"the {TARGET_ACCURACY:.2f} publication target"
        )
        print(f"skipping Hugging Face publish: {publish_skipped_reason}", flush=True)
    elif push_to_hub:
        from huggingface_hub import HfApi

        api = HfApi()
        api.create_repo(hub_repo, repo_type="model", exist_ok=True, private=False)
        api.upload_folder(folder_path=str(output), repo_id=hub_repo, repo_type="model")
        hub_url = f"https://huggingface.co/{hub_repo}"

    return {
        "ok": True,
        "compat": "face-or-not-resnet18-v2",
        "device": str(device),
        "device_name": torch.cuda.get_device_name(0),
        "torch": torch.__version__,
        "dataset_id": dataset_id,
        "dataset_revision": dataset_revision,
        "dataset_manifest_sha256": dataset_manifest_sha256,
        "split_sizes": metrics_document["split_sizes"],
        "epochs_completed": len(history),
        "best_epoch": best_epoch,
        "best_validation_accuracy": best_validation,
        "best_validation_loss": best_validation_loss,
        "test_metrics": test_metrics,
        "target_accuracy": TARGET_ACCURACY,
        "target_met": target_met,
        "param_count": param_count,
        "trainable_param_count": trainable_param_count,
        "architecture": "resnet18",
        "pretrained_weights": "IMAGENET1K_V1",
        "artifact_dir": str(output),
        "hub_url": hub_url,
        "publish_skipped_reason": publish_skipped_reason,
    }


@app.function(gpu="RTX-3090", image=image, timeout=1800)
def train(
    epochs: int = 30,
    batch_size: int = 64,
    learning_rate: float = 1e-3,
    backbone_learning_rate: float = 1e-4,
    weight_decay: float = 1e-4,
    image_size: int = 128,
    max_train: int = 3200,
    max_validation: int = 400,
    max_test: int = 400,
    dataset_id: str = DEFAULT_DATASET,
    dataset_revision: str = DEFAULT_DATASET_REVISION,
    seed: int = 20260904,
    patience: int = 6,
    warmup_epochs: int = 4,
) -> dict[str, Any]:
    """Train and preserve weights as a Compute artifact; no HF secret needed."""
    return _train_impl(
        epochs=epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        image_size=image_size,
        max_train=max_train,
        max_validation=max_validation,
        max_test=max_test,
        dataset_id=dataset_id,
        dataset_revision=dataset_revision,
        seed=seed,
        patience=patience,
        warmup_epochs=warmup_epochs,
        backbone_learning_rate=backbone_learning_rate,
        push_to_hub=False,
        hub_repo=DEFAULT_HUB_REPO,
    )


@app.function(gpu="RTX-3090", image=image, timeout=1800, secrets=[hf_secret])
def train_and_push(
    epochs: int = 30,
    batch_size: int = 64,
    learning_rate: float = 1e-3,
    backbone_learning_rate: float = 1e-4,
    weight_decay: float = 1e-4,
    image_size: int = 128,
    max_train: int = 3200,
    max_validation: int = 400,
    max_test: int = 400,
    dataset_id: str = DEFAULT_DATASET,
    dataset_revision: str = DEFAULT_DATASET_REVISION,
    seed: int = 20260904,
    patience: int = 6,
    warmup_epochs: int = 4,
    hub_repo: str = DEFAULT_HUB_REPO,
) -> dict[str, Any]:
    """Train, preserve artifacts, then publish the model with the stored HF secret."""
    return _train_impl(
        epochs=epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        image_size=image_size,
        max_train=max_train,
        max_validation=max_validation,
        max_test=max_test,
        dataset_id=dataset_id,
        dataset_revision=dataset_revision,
        seed=seed,
        patience=patience,
        warmup_epochs=warmup_epochs,
        backbone_learning_rate=backbone_learning_rate,
        push_to_hub=True,
        hub_repo=hub_repo,
    )
