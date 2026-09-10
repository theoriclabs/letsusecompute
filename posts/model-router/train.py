"""Train a small prompt router: cheap model vs strong model.

Dataset: ``routellm/gpt4_dataset`` on Hugging Face (Mixtral 8x7B scores vs
GPT-4 Turbo). Downloaded on the machine, not uploaded by you.

A prompt is labeled ``cheap_suffices`` when Mixtral's judge score is at least
4 / 5 — the RouteLLM "weak model is good enough" cut. The job trains a TF-IDF
logistic baseline and a DistilBERT classifier, then draws the cost-vs-quality
curve using public list prices for the two models.

    compute run train.py::train --gpu cheap --dry-run
    compute run train.py::train --gpu cheap --timeout 3600 --wait
"""

from __future__ import annotations

import json
import os
import tempfile
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import compute

app = compute.App("model-router")
# Pin cu124 wheels: many Vast consumer hosts still expose driver 12.4.
image = compute.Image.cuda_pytorch().pip_install(
    "--extra-index-url",
    "https://download.pytorch.org/whl/cu124",
    "torch==2.5.1+cu124",
    "transformers==4.46.3",
    "datasets==3.6.0",
    "scikit-learn==1.6.1",
    "huggingface_hub>=0.33,<1",
)

hf_secret = compute.Secret.from_name("hf")

DEFAULT_DATASET = "routellm/gpt4_dataset"
DEFAULT_ENCODER = "distilbert-base-uncased"
DEFAULT_HUB_REPO = "theoriclabs/prompt-router-small"
WORKLOAD_SUBDIR = "model-router"
ARTIFACT_NAME = "prompt-router-small"
ARTIFACT_MARKER = ".compute-artifact.json"
DEFAULT_ARTIFACT_FALLBACK = Path("/tmp/compute-model-router")
CLASS_NAMES = ("need_strong", "cheap_suffices")

# Public list prices matching the models in the dataset, USD / 1M tokens.
# Mixtral 8x7B Instruct: Together AI list (2024). GPT-4 Turbo (gpt-4-1106-preview):
# OpenAI list. Readers can swap these for their own providers.
CHEAP_INPUT_PER_M = 0.24
CHEAP_OUTPUT_PER_M = 0.24
STRONG_INPUT_PER_M = 10.00
STRONG_OUTPUT_PER_M = 30.00
CHARS_PER_TOKEN = 4.0


def _bridge_hf_token() -> None:
    token = (
        os.environ.get("HF_TOKEN")
        or os.environ.get("HUGGING_FACE_HUB_TOKEN")
        or os.environ.get("hf")
    )
    if token:
        os.environ.setdefault("HF_TOKEN", token)
        os.environ.setdefault("HUGGING_FACE_HUB_TOKEN", token)


def _py(value: Any) -> Any:
    """JSON-safe scalars (numpy / torch sneak into metrics otherwise)."""
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            pass
    if isinstance(value, float):
        return float(value)
    if isinstance(value, int):
        return int(value)
    return value


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


def estimate_tokens(text: str) -> float:
    return max(len(text), 1) / CHARS_PER_TOKEN


def example_costs(prompt: str, cheap_response: str, strong_response: str) -> tuple[float, float]:
    prompt_tokens = estimate_tokens(prompt)
    cheap = (
        prompt_tokens * CHEAP_INPUT_PER_M
        + estimate_tokens(cheap_response) * CHEAP_OUTPUT_PER_M
    ) / 1_000_000
    strong = (
        prompt_tokens * STRONG_INPUT_PER_M
        + estimate_tokens(strong_response) * STRONG_OUTPUT_PER_M
    ) / 1_000_000
    return cheap, strong


def load_routing_rows(
    dataset_id: str,
    *,
    max_train: int,
    max_eval: int,
    cheap_ok_min: int,
    seed: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    from datasets import load_dataset

    ds = load_dataset(dataset_id)
    train_split = ds["train"].shuffle(seed=seed)
    eval_split = ds["validation"].shuffle(seed=seed)
    if max_train > 0:
        train_split = train_split.select(range(min(max_train, len(train_split))))
    if max_eval > 0:
        eval_split = eval_split.select(range(min(max_eval, len(eval_split))))

    def rows_from(split) -> list[dict[str, Any]]:
        out: list[dict[str, Any]] = []
        for row in split:
            prompt = str(row["prompt"])
            cheap_response = str(row["mixtral_response"])
            strong_response = str(row["gpt4_response"])
            score = int(row["mixtral_score"])
            cheap_cost, strong_cost = example_costs(prompt, cheap_response, strong_response)
            out.append(
                {
                    "prompt": prompt,
                    "label": 1 if score >= cheap_ok_min else 0,
                    "mixtral_score": score,
                    "mixtral_quality": score / 5.0,
                    "cheap_cost": cheap_cost,
                    "strong_cost": strong_cost,
                    "source": list(row["source"]) if row.get("source") is not None else [],
                }
            )
        return out

    train_rows = rows_from(train_split)
    eval_rows = rows_from(eval_split)
    stats = {
        "dataset_id": dataset_id,
        "train_size": len(train_rows),
        "eval_size": len(eval_rows),
        "train_cheap_ok_frac": sum(r["label"] for r in train_rows) / max(len(train_rows), 1),
        "eval_cheap_ok_frac": sum(r["label"] for r in eval_rows) / max(len(eval_rows), 1),
        "cheap_ok_min": cheap_ok_min,
    }
    return train_rows, eval_rows, stats


def train_tfidf(train_rows: list[dict[str, Any]], eval_rows: list[dict[str, Any]]) -> dict[str, Any]:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression

    vectorizer = TfidfVectorizer(max_features=50_000, ngram_range=(1, 2), min_df=2)
    x_train = vectorizer.fit_transform([r["prompt"] for r in train_rows])
    y_train = [r["label"] for r in train_rows]
    clf = LogisticRegression(max_iter=400, class_weight="balanced", n_jobs=1)
    clf.fit(x_train, y_train)
    x_eval = vectorizer.transform([r["prompt"] for r in eval_rows])
    probs = clf.predict_proba(x_eval)[:, 1].tolist()
    preds = [1 if p >= 0.5 else 0 for p in probs]
    acc = sum(int(p == r["label"]) for p, r in zip(preds, eval_rows)) / max(len(eval_rows), 1)
    return {
        "name": "tfidf_logreg",
        "probs": probs,
        "accuracy": acc,
        "vectorizer": vectorizer,
        "classifier": clf,
        "feature_count": int(len(vectorizer.vocabulary_)),
    }


def train_distilbert(
    train_rows: list[dict[str, Any]],
    eval_rows: list[dict[str, Any]],
    *,
    encoder: str,
    epochs: int,
    batch_size: int,
    lr: float,
    max_length: int,
    seed: int,
) -> dict[str, Any]:
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, Dataset
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    torch.manual_seed(seed)
    device = torch.device("cuda")
    tokenizer = AutoTokenizer.from_pretrained(encoder)

    class PromptSet(Dataset):
        def __init__(self, rows: list[dict[str, Any]]) -> None:
            self.rows = rows

        def __len__(self) -> int:
            return len(self.rows)

        def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
            row = self.rows[idx]
            enc = tokenizer(
                row["prompt"],
                truncation=True,
                padding="max_length",
                max_length=max_length,
                return_tensors="pt",
            )
            return {
                "input_ids": enc["input_ids"].squeeze(0),
                "attention_mask": enc["attention_mask"].squeeze(0),
                "labels": torch.tensor(row["label"], dtype=torch.long),
            }

    n_pos = sum(r["label"] for r in train_rows)
    n_neg = len(train_rows) - n_pos
    # Inverse-frequency weights so the rarer "needs strong" class is not ignored.
    weights = torch.tensor(
        [len(train_rows) / max(2 * n_neg, 1), len(train_rows) / max(2 * n_pos, 1)],
        dtype=torch.float32,
        device=device,
    )
    model = AutoModelForSequenceClassification.from_pretrained(encoder, num_labels=2).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss(weight=weights)
    loader = DataLoader(PromptSet(train_rows), batch_size=batch_size, shuffle=True, num_workers=0)
    eval_loader = DataLoader(PromptSet(eval_rows), batch_size=batch_size, shuffle=False, num_workers=0)

    history: list[dict[str, float]] = []
    for epoch in range(epochs):
        model.train()
        running = 0.0
        seen = 0
        for batch in loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            opt.zero_grad()
            logits = model(input_ids=input_ids, attention_mask=attention_mask).logits
            loss = loss_fn(logits, labels)
            loss.backward()
            opt.step()
            running += float(loss.detach()) * labels.size(0)
            seen += labels.size(0)
        train_loss = running / max(seen, 1)

        model.eval()
        correct = 0
        total = 0
        eval_loss_sum = 0.0
        with torch.no_grad():
            for batch in eval_loader:
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                labels = batch["labels"].to(device)
                logits = model(input_ids=input_ids, attention_mask=attention_mask).logits
                eval_loss_sum += float(loss_fn(logits, labels).detach()) * labels.size(0)
                pred = logits.argmax(dim=1)
                correct += int((pred == labels).sum().item())
                total += labels.size(0)
        eval_loss = eval_loss_sum / max(total, 1)
        acc = correct / max(total, 1)
        history.append(
            {
                "epoch": epoch + 1,
                "train_loss": train_loss,
                "eval_loss": eval_loss,
                "eval_accuracy": acc,
            }
        )
        print(
            f"epoch {epoch + 1}/{epochs} "
            f"train_loss={train_loss:.4f} eval_loss={eval_loss:.4f} acc={acc:.4f}",
            flush=True,
        )

    model.eval()
    probs: list[float] = []
    with torch.no_grad():
        for batch in eval_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            logits = model(input_ids=input_ids, attention_mask=attention_mask).logits
            batch_probs = torch.softmax(logits, dim=1)[:, 1].detach().cpu().tolist()
            probs.extend(float(p) for p in batch_probs)

    preds = [1 if p >= 0.5 else 0 for p in probs]
    acc = sum(int(p == r["label"]) for p, r in zip(preds, eval_rows)) / max(len(eval_rows), 1)
    param_count = sum(p.numel() for p in model.parameters())
    return {
        "name": "distilbert",
        "model": model,
        "tokenizer": tokenizer,
        "probs": probs,
        "accuracy": acc,
        "history": history,
        "param_count": int(param_count),
        "class_weights": [float(w) for w in weights.detach().cpu().tolist()],
    }


def sweep_curve(
    rows: list[dict[str, Any]],
    probs: list[float],
    *,
    name: str,
) -> list[dict[str, float]]:
    thresholds = [i / 40 for i in range(41)]
    curve: list[dict[str, float]] = []
    for threshold in thresholds:
        quality_sum = 0.0
        cost_sum = 0.0
        cheap_n = 0
        for row, prob in zip(rows, probs):
            use_cheap = prob >= threshold
            quality_sum += row["mixtral_quality"] if use_cheap else 1.0
            cost_sum += row["cheap_cost"] if use_cheap else row["strong_cost"]
            cheap_n += int(use_cheap)
        n = max(len(rows), 1)
        curve.append(
            {
                "router": name,
                "threshold": threshold,
                "quality": quality_sum / n,
                "mean_cost_usd": cost_sum / n,
                "cheap_frac": cheap_n / n,
            }
        )
    return curve


def constant_policy(rows: list[dict[str, Any]], *, cheap: bool, name: str) -> dict[str, float]:
    n = max(len(rows), 1)
    if cheap:
        quality = sum(r["mixtral_quality"] for r in rows) / n
        cost = sum(r["cheap_cost"] for r in rows) / n
        cheap_frac = 1.0
    else:
        quality = 1.0
        cost = sum(r["strong_cost"] for r in rows) / n
        cheap_frac = 0.0
    return {
        "router": name,
        "threshold": 1.0 if cheap else 0.0,
        "quality": quality,
        "mean_cost_usd": cost,
        "cheap_frac": cheap_frac,
    }


def oracle_policy(rows: list[dict[str, Any]]) -> dict[str, float]:
    n = max(len(rows), 1)
    quality = 0.0
    cost = 0.0
    cheap_n = 0
    for row in rows:
        use_cheap = row["label"] == 1
        quality += row["mixtral_quality"] if use_cheap else 1.0
        cost += row["cheap_cost"] if use_cheap else row["strong_cost"]
        cheap_n += int(use_cheap)
    return {
        "router": "oracle",
        "threshold": None,
        "quality": quality / n,
        "mean_cost_usd": cost / n,
        "cheap_frac": cheap_n / n,
    }


def pick_operating_point(
    curve: list[dict[str, float]],
    *,
    quality_floor: float,
) -> dict[str, float] | None:
    feasible = [point for point in curve if point["quality"] + 1e-12 >= quality_floor]
    if not feasible:
        return None
    return min(feasible, key=lambda point: (point["mean_cost_usd"], -point["cheap_frac"]))


def write_curve_svg(
    directory: Path,
    *,
    tfidf_curve: list[dict[str, float]],
    bert_curve: list[dict[str, float]],
    always_cheap: dict[str, float],
    always_strong: dict[str, float],
    oracle: dict[str, float],
    bert_op: dict[str, float] | None,
    quality_floor: float,
) -> Path:
    width, height, pad = 760, 420, 56
    points = tfidf_curve + bert_curve + [always_cheap, always_strong, oracle]
    xs = [p["mean_cost_usd"] for p in points]
    ys = [p["quality"] for p in points]
    x_lo, x_hi = min(xs), max(xs)
    y_lo, y_hi = min(0.70, min(ys) - 0.02), 1.0
    if x_hi <= x_lo:
        x_hi = x_lo + 1e-6
    if y_hi <= y_lo:
        y_hi = y_lo + 1e-6

    def xy(cost: float, quality: float) -> tuple[float, float]:
        x = pad + (width - 2 * pad) * (cost - x_lo) / (x_hi - x_lo)
        y = pad + (height - 2 * pad) * (1 - (quality - y_lo) / (y_hi - y_lo))
        return x, y

    def polyline(curve: list[dict[str, float]]) -> str:
        return " ".join(f"{xy(p['mean_cost_usd'], p['quality'])[0]:.1f},{xy(p['mean_cost_usd'], p['quality'])[1]:.1f}" for p in curve)

    cheap_x, cheap_y = xy(always_cheap["mean_cost_usd"], always_cheap["quality"])
    strong_x, strong_y = xy(always_strong["mean_cost_usd"], always_strong["quality"])
    oracle_x, oracle_y = xy(oracle["mean_cost_usd"], oracle["quality"])
    floor_x1, floor_y = xy(x_lo, quality_floor)
    floor_x2, _ = xy(x_hi, quality_floor)
    op_dot = ""
    if bert_op is not None:
        op_x, op_y = xy(bert_op["mean_cost_usd"], bert_op["quality"])
        op_dot = (
            f'<circle cx="{op_x:.1f}" cy="{op_y:.1f}" r="5.5" fill="#c45c26" />'
            f'<text x="{op_x + 8:.1f}" y="{op_y - 8:.1f}" font-size="11" fill="#c45c26">'
            f"DistilBERT @ {quality_floor:.0%} quality</text>"
        )

    svg = f"""<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">
  <rect width="100%" height="100%" fill="#fff"/>
  <text x="{pad}" y="22" font-size="14" fill="#1a1a1a">Cost vs quality on held-out prompts</text>
  <line x1="{floor_x1:.1f}" y1="{floor_y:.1f}" x2="{floor_x2:.1f}" y2="{floor_y:.1f}" stroke="#c45c26" stroke-dasharray="4 4" stroke-width="1"/>
  <text x="{floor_x2 - 4:.1f}" y="{floor_y - 6:.1f}" font-size="11" fill="#c45c26" text-anchor="end">{quality_floor:.0%} of always-strong</text>
  <polyline fill="none" stroke="#7a8b99" stroke-width="2" points="{polyline(tfidf_curve)}"/>
  <polyline fill="none" stroke="#3d6b8a" stroke-width="2.4" points="{polyline(bert_curve)}"/>
  <circle cx="{cheap_x:.1f}" cy="{cheap_y:.1f}" r="4.5" fill="#3d6b4f"/>
  <circle cx="{strong_x:.1f}" cy="{strong_y:.1f}" r="4.5" fill="#1a1a1a"/>
  <circle cx="{oracle_x:.1f}" cy="{oracle_y:.1f}" r="4.5" fill="#8a5a2b"/>
  {op_dot}
  <text x="{pad}" y="{height - 14}" font-size="11" fill="#5c5c5c">x = mean USD / prompt (list prices) · green = always cheap · black = always strong · brown = oracle · gray = TF-IDF · steel = DistilBERT</text>
</svg>
"""
    path = directory / "cost_quality_curve.svg"
    path.write_text(svg, encoding="utf-8")
    return path


def pick_examples(
    rows: list[dict[str, Any]],
    probs: list[float],
    *,
    threshold: float,
    limit: int = 3,
) -> dict[str, list[dict[str, Any]]]:
    cheap: list[dict[str, Any]] = []
    strong: list[dict[str, Any]] = []
    ranked = sorted(zip(rows, probs), key=lambda item: item[1], reverse=True)
    for row, prob in ranked:
        snippet = row["prompt"].replace("\n", " ")
        if len(snippet) > 220:
            snippet = snippet[:217] + "..."
        item = {
            "prompt": snippet,
            "prob_cheap_ok": round(float(prob), 4),
            "mixtral_score": row["mixtral_score"],
            "label": row["label"],
        }
        if prob >= threshold and len(cheap) < limit:
            cheap.append(item)
        if prob < threshold and len(strong) < limit:
            strong.append(item)
        if len(cheap) >= limit and len(strong) >= limit:
            break
    return {"routed_cheap": cheap, "routed_strong": strong}


def _train_impl(
    *,
    epochs: int,
    batch_size: int,
    lr: float,
    max_length: int,
    max_train: int,
    max_eval: int,
    cheap_ok_min: int,
    quality_target: float,
    dataset_id: str,
    encoder: str,
    seed: int,
    push_to_hub: bool,
    hub_repo: str,
) -> dict:
    import torch

    _bridge_hf_token()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required (torch.cuda.is_available() is False)")
    device_name = torch.cuda.get_device_name(0)

    train_rows, eval_rows, data_stats = load_routing_rows(
        dataset_id,
        max_train=max_train,
        max_eval=max_eval,
        cheap_ok_min=cheap_ok_min,
        seed=seed,
    )
    print(
        f"loaded {data_stats['train_size']} train / {data_stats['eval_size']} eval · "
        f"eval cheap-ok frac={data_stats['eval_cheap_ok_frac']:.3f}",
        flush=True,
    )

    tfidf = train_tfidf(train_rows, eval_rows)
    print(f"tfidf accuracy={tfidf['accuracy']:.4f} features={tfidf['feature_count']}", flush=True)
    bert = train_distilbert(
        train_rows,
        eval_rows,
        encoder=encoder,
        epochs=epochs,
        batch_size=batch_size,
        lr=lr,
        max_length=max_length,
        seed=seed,
    )
    print(f"distilbert accuracy={bert['accuracy']:.4f} params={bert['param_count']}", flush=True)

    always_cheap = constant_policy(eval_rows, cheap=True, name="always_cheap")
    always_strong = constant_policy(eval_rows, cheap=False, name="always_strong")
    oracle = oracle_policy(eval_rows)
    tfidf_curve = sweep_curve(eval_rows, tfidf["probs"], name="tfidf")
    bert_curve = sweep_curve(eval_rows, bert["probs"], name="distilbert")
    quality_floor = quality_target * always_strong["quality"]
    tfidf_op = pick_operating_point(tfidf_curve, quality_floor=quality_floor)
    bert_op = pick_operating_point(bert_curve, quality_floor=quality_floor)

    def reduction(point: dict[str, float] | None) -> float | None:
        if point is None:
            return None
        return 1.0 - (point["mean_cost_usd"] / always_strong["mean_cost_usd"])

    print(
        f"always_cheap quality={always_cheap['quality']:.4f} cost={always_cheap['mean_cost_usd']:.6f}",
        flush=True,
    )
    print(
        f"always_strong quality={always_strong['quality']:.4f} cost={always_strong['mean_cost_usd']:.6f}",
        flush=True,
    )
    print(
        f"oracle quality={oracle['quality']:.4f} cost={oracle['mean_cost_usd']:.6f} "
        f"cheap_frac={oracle['cheap_frac']:.3f}",
        flush=True,
    )
    if bert_op:
        print(
            f"distilbert@{quality_floor:.0%} quality={bert_op['quality']:.4f} "
            f"cheap_frac={bert_op['cheap_frac']:.3f} "
            f"cost={bert_op['mean_cost_usd']:.6f} "
            f"save={reduction(bert_op):.3f}",
            flush=True,
        )
    if tfidf_op:
        print(
            f"tfidf@{quality_floor:.0%} quality={tfidf_op['quality']:.4f} "
            f"cheap_frac={tfidf_op['cheap_frac']:.3f} "
            f"cost={tfidf_op['mean_cost_usd']:.6f} "
            f"save={reduction(tfidf_op):.3f}",
            flush=True,
        )

    out_dir = resolve_artifact_dirs()
    # Keep the Compute artifact small and single-file-friendly. A full
    # save_pretrained folder (~250MB) plus a 50k-feature joblib dump
    # previously failed artifact PUT with HTTP 411 (run_4c1a20…).
    # The Hub path writes the full folder to a temp dir instead.
    head_keys = [key for key in bert["model"].state_dict() if key.startswith(("classifier.", "pre_classifier."))]
    head_state = {key: bert["model"].state_dict()[key].detach().cpu().half() for key in head_keys}
    torch.save(
        {
            "encoder": encoder,
            "class_names": list(CLASS_NAMES),
            "state_dict": head_state,
            "threshold": bert_op["threshold"] if bert_op else 0.5,
            "max_length": max_length,
            "param_count": bert["param_count"],
        },
        out_dir / "router.pt",
    )

    examples = pick_examples(
        eval_rows,
        bert["probs"],
        threshold=bert_op["threshold"] if bert_op else 0.5,
    )
    curve_payload = {
        "always_cheap": always_cheap,
        "always_strong": always_strong,
        "oracle": oracle,
        "tfidf_curve": tfidf_curve,
        "distilbert_curve": bert_curve,
        "tfidf_operating_point": tfidf_op,
        "distilbert_operating_point": bert_op,
        "quality_floor": quality_floor,
        "prices": {
            "cheap_model": "mistralai/Mixtral-8x7B-Instruct-v0.1",
            "strong_model": "gpt-4-1106-preview",
            "cheap_input_per_m": CHEAP_INPUT_PER_M,
            "cheap_output_per_m": CHEAP_OUTPUT_PER_M,
            "strong_input_per_m": STRONG_INPUT_PER_M,
            "strong_output_per_m": STRONG_OUTPUT_PER_M,
            "chars_per_token": CHARS_PER_TOKEN,
        },
    }
    (out_dir / "curve.json").write_text(json.dumps(curve_payload, indent=2) + "\n", encoding="utf-8")
    (out_dir / "examples.json").write_text(json.dumps(examples, indent=2) + "\n", encoding="utf-8")
    (out_dir / "history.json").write_text(json.dumps(bert["history"], indent=2) + "\n", encoding="utf-8")
    write_curve_svg(
        out_dir,
        tfidf_curve=tfidf_curve,
        bert_curve=bert_curve,
        always_cheap=always_cheap,
        always_strong=always_strong,
        oracle=oracle,
        bert_op=bert_op,
        quality_floor=quality_floor,
    )

    summary = {
        "tfidf_accuracy": _py(tfidf["accuracy"]),
        "distilbert_accuracy": _py(bert["accuracy"]),
        "always_cheap": {k: _py(v) for k, v in always_cheap.items()},
        "always_strong": {k: _py(v) for k, v in always_strong.items()},
        "oracle": {k: _py(v) for k, v in oracle.items() if k != "threshold"},
        "distilbert_operating_point": {k: _py(v) for k, v in bert_op.items()} if bert_op else None,
        "tfidf_operating_point": {k: _py(v) for k, v in tfidf_op.items()} if tfidf_op else None,
        "cost_reduction_vs_always_strong": _py(reduction(bert_op)) if bert_op else None,
        "quality_floor": _py(quality_floor),
        "beats_always_cheap_quality": bool(
            bert_op is not None and bert_op["quality"] > always_cheap["quality"] + 1e-12
        ),
        "beats_always_strong_cost": bool(
            bert_op is not None and bert_op["mean_cost_usd"] < always_strong["mean_cost_usd"] - 1e-12
        ),
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    write_artifact_marker(
        out_dir,
        name=ARTIFACT_NAME,
        kind="output",
        compatibility_key=f"prompt-router-distilbert-v1-{encoder}",
        metadata={
            "filename": "router.pt",
            "encoder": encoder,
            "param_count": bert["param_count"],
            "epochs": epochs,
            "distilbert_accuracy": summary["distilbert_accuracy"],
            "tfidf_accuracy": summary["tfidf_accuracy"],
            "cheap_frac_at_target": summary["distilbert_operating_point"]["cheap_frac"]
            if summary["distilbert_operating_point"]
            else None,
            "cost_reduction_vs_always_strong": summary["cost_reduction_vs_always_strong"],
            "dataset_id": dataset_id,
            "class_names": list(CLASS_NAMES),
        },
    )

    hub_url = None
    if push_to_hub:
        try:
            from huggingface_hub import HfApi

            hub_dir = Path(tempfile.mkdtemp(prefix="prompt-router-hub-"))
            bert["model"].save_pretrained(hub_dir)
            bert["tokenizer"].save_pretrained(hub_dir)
            api = HfApi()
            api.create_repo(hub_repo, exist_ok=True, private=False)
            api.upload_folder(folder_path=str(hub_dir), repo_id=hub_repo)
            hub_url = f"https://huggingface.co/{hub_repo}"
        except Exception as err:  # noqa: BLE001 — publish is optional; keep weights via artifacts
            print(f"push_to_hub failed: {err}", flush=True)
            hub_url = None

    return {
        "ok": True,
        "compat": "prompt-router-small",
        "device": "cuda",
        "device_name": device_name,
        "dataset_id": dataset_id,
        "encoder": encoder,
        "epochs": epochs,
        "param_count": bert["param_count"],
        "train_size": data_stats["train_size"],
        "eval_size": data_stats["eval_size"],
        "train_cheap_ok_frac": _py(data_stats["train_cheap_ok_frac"]),
        "eval_cheap_ok_frac": _py(data_stats["eval_cheap_ok_frac"]),
        "history": bert["history"],
        "tfidf_accuracy": summary["tfidf_accuracy"],
        "distilbert_accuracy": summary["distilbert_accuracy"],
        "always_cheap": summary["always_cheap"],
        "always_strong": summary["always_strong"],
        "oracle": summary["oracle"],
        "distilbert_operating_point": summary["distilbert_operating_point"],
        "tfidf_operating_point": summary["tfidf_operating_point"],
        "cost_reduction_vs_always_strong": summary["cost_reduction_vs_always_strong"],
        "quality_floor": summary["quality_floor"],
        "beats_always_cheap_quality": summary["beats_always_cheap_quality"],
        "beats_always_strong_cost": summary["beats_always_strong_cost"],
        "examples": examples,
        "artifact_dir": str(out_dir),
        "hub_url": hub_url,
        "torch": torch.__version__,
    }


@app.function(
    gpu="RTX-3090",
    image=image,
    timeout=3600,
)
def train(
    epochs: int = 2,
    batch_size: int = 32,
    lr: float = 2e-5,
    max_length: int = 256,
    max_train: int = 0,
    max_eval: int = 0,
    cheap_ok_min: int = 4,
    quality_target: float = 0.95,
    dataset_id: str = DEFAULT_DATASET,
    encoder: str = DEFAULT_ENCODER,
    seed: int = 0,
    push_to_hub: bool = False,
    hub_repo: str = DEFAULT_HUB_REPO,
) -> dict:
    """Train the router. No Hugging Face token required."""
    return _train_impl(
        epochs=epochs,
        batch_size=batch_size,
        lr=lr,
        max_length=max_length,
        max_train=max_train,
        max_eval=max_eval,
        cheap_ok_min=cheap_ok_min,
        quality_target=quality_target,
        dataset_id=dataset_id,
        encoder=encoder,
        seed=seed,
        push_to_hub=push_to_hub,
        hub_repo=hub_repo,
    )


@app.function(
    gpu="RTX-3090",
    image=image,
    timeout=3600,
    secrets=[hf_secret],
)
def train_and_push(
    epochs: int = 2,
    batch_size: int = 32,
    lr: float = 2e-5,
    max_length: int = 256,
    max_train: int = 0,
    max_eval: int = 0,
    cheap_ok_min: int = 4,
    quality_target: float = 0.95,
    dataset_id: str = DEFAULT_DATASET,
    encoder: str = DEFAULT_ENCODER,
    seed: int = 0,
    push_to_hub: bool = True,
    hub_repo: str = DEFAULT_HUB_REPO,
) -> dict:
    """Same train, then upload the DistilBERT folder using the stored ``hf`` secret."""
    return _train_impl(
        epochs=epochs,
        batch_size=batch_size,
        lr=lr,
        max_length=max_length,
        max_train=max_train,
        max_eval=max_eval,
        cheap_ok_min=cheap_ok_min,
        quality_target=quality_target,
        dataset_id=dataset_id,
        encoder=encoder,
        seed=seed,
        push_to_hub=push_to_hub,
        hub_repo=hub_repo,
    )


if __name__ == "__main__":
    with app.run():
        print(train.remote(epochs=1, max_train=256, max_eval=128))
