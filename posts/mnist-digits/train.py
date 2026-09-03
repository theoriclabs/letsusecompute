"""Train a tiny from-scratch CNN on MNIST handwritten digits.

Dataset: ``torchvision.datasets.MNIST`` — downloaded on the machine during
the run, not uploaded by you.

Weights land under ``$COMPUTE_ARTIFACT_DIR`` with a ``.compute-artifact.json``
marker so ``compute artifacts get`` works after teardown. Optional Hub push
when ``push_to_hub=True`` and an HF write token is available.

    compute run train.py::train --gpu cheap --dry-run
    compute run train.py::train --gpu cheap --timeout 900 --wait
"""

from __future__ import annotations

import json
import os
import tempfile
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import compute

app = compute.App("mnist-digits")
# Pin cu124 wheels: many Vast consumer hosts still expose driver 12.4
# (CUDA_VERSION 12040). Unpinned torchvision pulls a newer torch that
# refuses to initialize on that driver.
image = compute.Image.cuda_pytorch().pip_install(
    "--extra-index-url",
    "https://download.pytorch.org/whl/cu124",
    "torch==2.5.1+cu124",
    "torchvision==0.20.1+cu124",
    "huggingface_hub",
)

# Stored with `compute secrets set hf`. Injected only when a function lists
# `secrets=[hf_secret]`. Training does not need it.
hf_secret = compute.Secret.from_name("hf")

WORKLOAD_SUBDIR = "mnist-digits"
ARTIFACT_NAME = "mnist-cnn"
ARTIFACT_MARKER = ".compute-artifact.json"
DEFAULT_ARTIFACT_FALLBACK = Path("/tmp/compute-mnist-digits")
CLASS_NAMES = tuple(str(i) for i in range(10))


def _bridge_hf_token() -> None:
    token = (
        os.environ.get("HF_TOKEN")
        or os.environ.get("HUGGING_FACE_HUB_TOKEN")
        or os.environ.get("hf")
    )
    if token:
        os.environ.setdefault("HF_TOKEN", token)
        os.environ.setdefault("HUGGING_FACE_HUB_TOKEN", token)


def write_artifact_marker(
    directory: Path | str,
    *,
    name: str,
    kind: str,
    compatibility_key: str,
    metadata: dict[str, Any],
) -> Path:
    """Atomically write ``.compute-artifact.json`` (temp + flush + fsync + replace)."""
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
    fd, tmp_name = tempfile.mkstemp(
        prefix=".compute-artifact.",
        suffix=".tmp",
        dir=str(directory),
    )
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


def write_loss_curve(directory: Path, history: list[dict[str, float]]) -> Path:
    """Write JSON + a tiny SVG of train/test loss. No extra deps."""
    directory.mkdir(parents=True, exist_ok=True)
    json_path = directory / "loss_curve.json"
    json_path.write_text(json.dumps(history, indent=2) + "\n", encoding="utf-8")

    width, height, pad = 720, 280, 36
    losses = [row["train_loss"] for row in history] + [row["test_loss"] for row in history]
    lo = min(losses) if losses else 0.0
    hi = max(losses) if losses else 1.0
    if hi <= lo:
        hi = lo + 1e-6

    def xy(index: int, value: float, count: int) -> str:
        x = pad + (width - 2 * pad) * (index / max(count - 1, 1))
        y = pad + (height - 2 * pad) * (1 - (value - lo) / (hi - lo))
        return f"{x:.1f},{y:.1f}"

    def polyline(key: str) -> str:
        count = len(history)
        pts = " ".join(xy(i, row[key], count) for i, row in enumerate(history))
        return pts

    svg = f"""<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">
  <rect width="100%" height="100%" fill="#fff"/>
  <text x="{pad}" y="20" font-size="13" fill="#333">train loss (steel) vs test loss (orange)</text>
  <polyline fill="none" stroke="#3d6b8a" stroke-width="2" points="{polyline("train_loss")}"/>
  <polyline fill="none" stroke="#c45c26" stroke-width="2" points="{polyline("test_loss")}"/>
  <text x="{pad}" y="{height - 8}" font-size="11" fill="#666">min {lo:.3f} · max {hi:.3f} · {len(history)} epochs</text>
</svg>
"""
    (directory / "loss_curve.svg").write_text(svg, encoding="utf-8")
    return json_path


def _tiny_cnn(num_classes: int = 10):
    import torch.nn as nn

    # ~106k params. Random init — no pretrained weights.
    return nn.Sequential(
        nn.Conv2d(1, 16, kernel_size=3, padding=1),
        nn.ReLU(inplace=True),
        nn.MaxPool2d(2),  # 28x28 -> 14x14
        nn.Conv2d(16, 32, kernel_size=3, padding=1),
        nn.ReLU(inplace=True),
        nn.MaxPool2d(2),  # 14x14 -> 7x7
        nn.Flatten(),
        nn.Linear(32 * 7 * 7, 64),
        nn.ReLU(inplace=True),
        nn.Linear(64, num_classes),
    )


def _train_impl(
    *,
    epochs: int,
    batch_size: int,
    lr: float,
    max_train: int | None,
    max_test: int | None,
    seed: int,
    push_to_hub: bool,
    hub_repo: str,
) -> dict:
    """Train the MNIST digits CNN and write weights as a compute artifact."""
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, Subset
    from torchvision import datasets, transforms

    _bridge_hf_token()
    torch.manual_seed(seed)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required (torch.cuda.is_available() is False)")
    device = torch.device("cuda")
    device_name = torch.cuda.get_device_name(0)

    data_dir = Path(os.environ.get("COMPUTE_DATA_DIR", "/tmp/mnist-data"))
    tfm = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.1307], std=[0.3081]),
        ]
    )
    train_full = datasets.MNIST(root=str(data_dir), train=True, download=True, transform=tfm)
    test_full = datasets.MNIST(root=str(data_dir), train=False, download=True, transform=tfm)
    train_ds = Subset(train_full, range(min(max_train, len(train_full)))) if max_train else train_full
    test_ds = Subset(test_full, range(min(max_test, len(test_full)))) if max_test else test_full

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=2)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=2)

    model = _tiny_cnn(num_classes=10).to(device)
    param_count = sum(p.numel() for p in model.parameters())
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()

    history: list[dict[str, float]] = []
    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        n = 0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            logits = model(xb)
            loss = loss_fn(logits, yb)
            loss.backward()
            opt.step()
            total_loss += float(loss.detach()) * xb.size(0)
            n += xb.size(0)
        train_loss = total_loss / max(n, 1)

        model.eval()
        correct = 0
        total = 0
        test_loss_sum = 0.0
        with torch.no_grad():
            for xb, yb in test_loader:
                xb, yb = xb.to(device), yb.to(device)
                logits = model(xb)
                test_loss_sum += float(loss_fn(logits, yb).detach()) * xb.size(0)
                pred = logits.argmax(dim=1)
                correct += int((pred == yb).sum().item())
                total += yb.size(0)
        test_loss = test_loss_sum / max(total, 1)
        acc = correct / max(total, 1)
        history.append(
            {
                "epoch": epoch + 1,
                "train_loss": train_loss,
                "test_loss": test_loss,
                "test_accuracy": acc,
            }
        )
        print(
            f"epoch {epoch + 1}/{epochs} "
            f"train_loss={train_loss:.4f} test_loss={test_loss:.4f} acc={acc:.4f}",
            flush=True,
        )

    out_dir = resolve_artifact_dirs()
    weights_path = out_dir / "model.pt"
    torch.save(
        {
            "state_dict": model.state_dict(),
            "class_names": list(CLASS_NAMES),
            "param_count": param_count,
        },
        weights_path,
    )
    write_loss_curve(out_dir, history)
    write_artifact_marker(
        out_dir,
        name=ARTIFACT_NAME,
        kind="output",
        compatibility_key="mnist-cnn-v1",
        metadata={
            "filename": weights_path.name,
            "param_count": param_count,
            "epochs": epochs,
            "test_accuracy": history[-1]["test_accuracy"] if history else None,
            "class_names": list(CLASS_NAMES),
        },
    )

    hub_url = None
    if push_to_hub:
        try:
            from huggingface_hub import HfApi

            api = HfApi()
            api.create_repo(hub_repo, exist_ok=True, private=False)
            api.upload_file(
                path_or_fileobj=str(weights_path),
                path_in_repo="model.pt",
                repo_id=hub_repo,
            )
            hub_url = f"https://huggingface.co/{hub_repo}"
        except Exception as err:  # noqa: BLE001 — publish is optional; keep weights via artifacts
            print(f"push_to_hub failed: {err}", flush=True)
            hub_url = None

    return {
        "ok": True,
        "compat": "mnist-cnn",
        "device": str(device),
        "device_name": device_name,
        "epochs": epochs,
        "param_count": param_count,
        "train_size": len(train_ds),
        "test_size": len(test_ds),
        "history": history,
        "test_accuracy": history[-1]["test_accuracy"] if history else None,
        "artifact_dir": str(out_dir),
        "weights_file": weights_path.name,
        "hub_url": hub_url,
        "torch": torch.__version__,
    }


@app.function(
    gpu="RTX-3090",
    image=image,
    timeout=900,
)
def train(
    epochs: int = 8,
    batch_size: int = 128,
    lr: float = 1e-3,
    max_train: int | None = None,
    max_test: int | None = None,
    seed: int = 0,
    push_to_hub: bool = False,
    hub_repo: str = "theoriclabs/mnist-cnn",
) -> dict:
    """Train only. No Hugging Face token required."""
    return _train_impl(
        epochs=epochs,
        batch_size=batch_size,
        lr=lr,
        max_train=max_train,
        max_test=max_test,
        seed=seed,
        push_to_hub=push_to_hub,
        hub_repo=hub_repo,
    )


@app.function(
    gpu="RTX-3090",
    image=image,
    timeout=900,
    secrets=[hf_secret],
)
def train_and_push(
    epochs: int = 8,
    batch_size: int = 128,
    lr: float = 1e-3,
    max_train: int | None = None,
    max_test: int | None = None,
    seed: int = 0,
    push_to_hub: bool = True,
    hub_repo: str = "theoriclabs/mnist-cnn",
) -> dict:
    """Same train, then upload ``model.pt`` using the stored ``hf`` secret."""
    return _train_impl(
        epochs=epochs,
        batch_size=batch_size,
        lr=lr,
        max_train=max_train,
        max_test=max_test,
        seed=seed,
        push_to_hub=push_to_hub,
        hub_repo=hub_repo,
    )


if __name__ == "__main__":
    with app.run():
        print(train.remote(epochs=1, max_train=256, max_test=256))
