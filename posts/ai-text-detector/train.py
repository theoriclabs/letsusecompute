"""Train a small human-vs-AI text detector.

Dataset: ``Hello-SimpleAI/HC3`` (English). Downloaded on the machine, not
uploaded by you. Each question contributes one human answer and one ChatGPT
answer. Medicine is held out as a domain-shift set. A streamed RAID slice
(GPT-4 / Llama-chat / human, no adversarial attacks) is the different-generator
check.

A TF-IDF logistic baseline and DistilBERT share the same splits. The number
that matters is the false-positive rate on human text.

    compute run train.py::train --gpu cheap --dry-run
    compute run train.py::train --gpu cheap --timeout 2400 --wait
"""

from __future__ import annotations

import json
import os
import random
import tempfile
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import compute

app = compute.App("ai-text-detector")
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

DEFAULT_DATASET = "Hello-SimpleAI/HC3"
DEFAULT_DATASET_CONFIG = "all"
DEFAULT_ENCODER = "distilbert-base-uncased"
DEFAULT_HUB_REPO = "theoriclabs/ai-text-detector-small"
DEFAULT_DOMAIN_HOLDOUT = "medicine"
DEFAULT_OOD_DATASET = "liamdugan/raid"
WORKLOAD_SUBDIR = "ai-text-detector"
ARTIFACT_NAME = "ai-text-detector-small"
ARTIFACT_MARKER = ".compute-artifact.json"
DEFAULT_ARTIFACT_FALLBACK = Path("/tmp/compute-ai-text-detector")
CLASS_NAMES = ("human", "ai")
MAX_TEXT_CHARS = 2_000


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


def _first_answer(answers: Any) -> str:
    if answers is None:
        return ""
    if isinstance(answers, str):
        return answers.strip()
    for item in answers:
        text = str(item).strip()
        if text:
            return text
    return ""


def _clip(text: str) -> str:
    text = " ".join(text.split())
    if len(text) > MAX_TEXT_CHARS:
        return text[: MAX_TEXT_CHARS - 3] + "..."
    return text


def _explode(questions: list[dict[str, str]], split: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for question in questions:
        rows.append(
            {
                "text": question["human"],
                "label": 0,
                "source": question["source"],
                "split": split,
            }
        )
        rows.append(
            {
                "text": question["ai"],
                "label": 1,
                "source": question["source"],
                "split": split,
            }
        )
    return rows


def load_hc3_rows(
    dataset_id: str,
    *,
    config: str,
    domain_holdout: str,
    eval_frac: float,
    max_train: int,
    max_eval: int,
    seed: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    from datasets import load_dataset

    raw = load_dataset(dataset_id, config)["train"]
    questions: list[dict[str, str]] = []
    skipped = 0
    for row in raw:
        human = _clip(_first_answer(row.get("human_answers")))
        ai = _clip(_first_answer(row.get("chatgpt_answers")))
        source = str(row.get("source") or "unknown")
        if len(human) < 40 or len(ai) < 40:
            skipped += 1
            continue
        questions.append({"human": human, "ai": ai, "source": source})

    holdout_q = [q for q in questions if q["source"] == domain_holdout]
    rest = [q for q in questions if q["source"] != domain_holdout]
    rng = random.Random(seed)
    rng.shuffle(rest)
    n_eval = max(1, int(len(rest) * eval_frac))
    eval_q = rest[:n_eval]
    train_q = rest[n_eval:]
    if max_train > 0:
        train_q = train_q[: max(1, max_train // 2)]
    if max_eval > 0:
        eval_q = eval_q[: max(1, max_eval // 2)]
        holdout_q = holdout_q[: max(1, max_eval // 2)]

    train_rows = _explode(train_q, "train")
    eval_rows = _explode(eval_q, "id_holdout")
    domain_rows = _explode(holdout_q, "domain_holdout")
    sources = sorted({q["source"] for q in rest})
    stats = {
        "dataset_id": dataset_id,
        "config": config,
        "questions": len(questions),
        "skipped_short": skipped,
        "train_questions": len(train_q),
        "eval_questions": len(eval_q),
        "domain_questions": len(holdout_q),
        "train_size": len(train_rows),
        "eval_size": len(eval_rows),
        "domain_size": len(domain_rows),
        "train_ai_frac": sum(r["label"] for r in train_rows) / max(len(train_rows), 1),
        "eval_ai_frac": sum(r["label"] for r in eval_rows) / max(len(eval_rows), 1),
        "domain_holdout": domain_holdout,
        "train_sources": sources,
    }
    return train_rows, eval_rows, domain_rows, stats


def load_raid_ood(
    dataset_id: str,
    *,
    max_per_model: int,
    max_scan: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    from datasets import load_dataset

    wanted = {"human": 0, "gpt4": 1, "llama-chat": 1}
    counts = {name: 0 for name in wanted}
    rows: list[dict[str, Any]] = []
    try:
        stream = load_dataset(dataset_id, split="train", streaming=True)
    except Exception as err:  # noqa: BLE001 — OOD is optional; HC3 still stands
        return [], {"ok": False, "error": str(err), "dataset_id": dataset_id}

    scanned = 0
    try:
        for row in stream:
            scanned += 1
            if scanned > max_scan:
                break
            if all(counts[name] >= max_per_model for name in wanted):
                break
            model = str(row.get("model") or "")
            attack = row.get("attack") or "none"
            if model not in wanted or attack not in (None, "none"):
                continue
            if counts[model] >= max_per_model:
                continue
            text = _clip(str(row.get("generation") or ""))
            if len(text) < 40:
                continue
            rows.append(
                {
                    "text": text,
                    "label": wanted[model],
                    "source": f"raid:{model}",
                    "split": "ood_raid",
                }
            )
            counts[model] += 1
    except Exception as err:  # noqa: BLE001 — keep the rows collected so far
        return rows, {
            "ok": False,
            "error": str(err),
            "dataset_id": dataset_id,
            "counts": counts,
            "scanned": scanned,
            "size": len(rows),
        }

    return rows, {
        "ok": True,
        "dataset_id": dataset_id,
        "counts": counts,
        "scanned": scanned,
        "size": len(rows),
    }


def classification_metrics(
    rows: list[dict[str, Any]],
    probs: list[float],
    *,
    threshold: float,
) -> dict[str, float]:
    tp = fp = tn = fn = 0
    for row, prob in zip(rows, probs):
        pred = 1 if prob >= threshold else 0
        if row["label"] == 1 and pred == 1:
            tp += 1
        elif row["label"] == 0 and pred == 1:
            fp += 1
        elif row["label"] == 0 and pred == 0:
            tn += 1
        else:
            fn += 1
    n = max(len(rows), 1)
    human_n = tn + fp
    ai_n = tp + fn
    return {
        "threshold": threshold,
        "n": float(len(rows)),
        "accuracy": (tp + tn) / n,
        "human_fpr": fp / max(human_n, 1),
        "ai_recall": tp / max(ai_n, 1),
        "ai_precision": tp / max(tp + fp, 1),
        "human_n": float(human_n),
        "ai_n": float(ai_n),
    }


def score_rows(
    predict_probs,
    rows: list[dict[str, Any]],
    *,
    name: str,
) -> dict[str, Any]:
    if not rows:
        return {"name": name, "n": 0, "at_0_5": None, "at_5pct_fpr": None, "probs": []}
    probs = predict_probs(rows)
    at_half = classification_metrics(rows, probs, threshold=0.5)
    # Lowest threshold that still keeps human FPR at or under 5%.
    at_fpr = None
    for step in range(0, 41):
        threshold = step / 40
        metrics = classification_metrics(rows, probs, threshold=threshold)
        if metrics["human_fpr"] <= 0.05 + 1e-12:
            at_fpr = metrics
            break
    return {
        "name": name,
        "n": len(rows),
        "at_0_5": at_half,
        "at_5pct_fpr": at_fpr,
        "probs": probs,
    }


def train_tfidf(train_rows: list[dict[str, Any]]):
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression

    vectorizer = TfidfVectorizer(max_features=50_000, ngram_range=(1, 2), min_df=2)
    x_train = vectorizer.fit_transform([r["text"] for r in train_rows])
    y_train = [r["label"] for r in train_rows]
    clf = LogisticRegression(max_iter=400, class_weight="balanced", n_jobs=1)
    clf.fit(x_train, y_train)

    def predict_probs(rows: list[dict[str, Any]]) -> list[float]:
        x = vectorizer.transform([r["text"] for r in rows])
        return clf.predict_proba(x)[:, 1].tolist()

    return {
        "name": "tfidf_logreg",
        "predict_probs": predict_probs,
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

    class TextSet(Dataset):
        def __init__(self, rows: list[dict[str, Any]]) -> None:
            self.rows = rows

        def __len__(self) -> int:
            return len(self.rows)

        def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
            row = self.rows[idx]
            enc = tokenizer(
                row["text"],
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
    weights = torch.tensor(
        [len(train_rows) / max(2 * n_neg, 1), len(train_rows) / max(2 * n_pos, 1)],
        dtype=torch.float32,
        device=device,
    )
    model = AutoModelForSequenceClassification.from_pretrained(encoder, num_labels=2).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss(weight=weights)
    loader = DataLoader(TextSet(train_rows), batch_size=batch_size, shuffle=True, num_workers=0)
    eval_loader = DataLoader(TextSet(eval_rows), batch_size=batch_size, shuffle=False, num_workers=0)

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

    def predict_probs(rows: list[dict[str, Any]]) -> list[float]:
        if not rows:
            return []
        model.eval()
        loader_pred = DataLoader(TextSet(rows), batch_size=batch_size, shuffle=False, num_workers=0)
        out: list[float] = []
        with torch.no_grad():
            for batch in loader_pred:
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                logits = model(input_ids=input_ids, attention_mask=attention_mask).logits
                out.extend(torch.softmax(logits, dim=1)[:, 1].detach().cpu().tolist())
        return [float(p) for p in out]

    param_count = sum(p.numel() for p in model.parameters())
    return {
        "name": "distilbert",
        "model": model,
        "tokenizer": tokenizer,
        "predict_probs": predict_probs,
        "history": history,
        "param_count": int(param_count),
        "class_weights": [float(w) for w in weights.detach().cpu().tolist()],
    }


def pick_examples(
    rows: list[dict[str, Any]],
    probs: list[float],
    *,
    threshold: float,
    limit: int = 3,
) -> dict[str, list[dict[str, Any]]]:
    false_pos: list[dict[str, Any]] = []
    false_neg: list[dict[str, Any]] = []
    true_ai: list[dict[str, Any]] = []
    ranked = sorted(zip(rows, probs), key=lambda item: item[1], reverse=True)
    for row, prob in ranked:
        snippet = row["text"]
        if len(snippet) > 220:
            snippet = snippet[:217] + "..."
        item = {
            "text": snippet,
            "prob_ai": round(float(prob), 4),
            "label": CLASS_NAMES[row["label"]],
            "source": row["source"],
        }
        pred = 1 if prob >= threshold else 0
        if row["label"] == 0 and pred == 1 and len(false_pos) < limit:
            false_pos.append(item)
        if row["label"] == 1 and pred == 0 and len(false_neg) < limit:
            false_neg.append(item)
        if row["label"] == 1 and pred == 1 and len(true_ai) < limit:
            true_ai.append(item)
        if len(false_pos) >= limit and len(false_neg) >= limit and len(true_ai) >= limit:
            break
    return {
        "human_called_ai": false_pos,
        "ai_called_human": false_neg,
        "ai_caught": true_ai,
    }


def write_metrics_svg(
    directory: Path,
    *,
    splits: list[tuple[str, dict[str, Any] | None, dict[str, Any] | None]],
) -> Path:
    width, height, pad = 760, 360, 56
    usable = [(label, tfidf, bert) for label, tfidf, bert in splits if tfidf and bert]
    n = max(len(usable), 1)
    gap = (width - 2 * pad) / n
    bar_w = min(28.0, gap / 5)

    def y_of(value: float) -> float:
        return pad + (height - 2 * pad) * (1 - value)

    bars = []
    labels = []
    for i, (label, tfidf, bert) in enumerate(usable):
        x0 = pad + gap * i + gap / 2
        tfidf_acc = tfidf["accuracy"]
        bert_acc = bert["accuracy"]
        tfidf_fpr = tfidf["human_fpr"]
        bert_fpr = bert["human_fpr"]
        bars.append(
            f'<rect x="{x0 - 2 * bar_w:.1f}" y="{y_of(tfidf_acc):.1f}" '
            f'width="{bar_w:.1f}" height="{y_of(0) - y_of(tfidf_acc):.1f}" fill="#7a8b99"/>'
        )
        bars.append(
            f'<rect x="{x0 - bar_w:.1f}" y="{y_of(bert_acc):.1f}" '
            f'width="{bar_w:.1f}" height="{y_of(0) - y_of(bert_acc):.1f}" fill="#3d6b8a"/>'
        )
        bars.append(
            f'<rect x="{x0 + 4:.1f}" y="{y_of(tfidf_fpr):.1f}" '
            f'width="{bar_w:.1f}" height="{y_of(0) - y_of(tfidf_fpr):.1f}" fill="#c4a484"/>'
        )
        bars.append(
            f'<rect x="{x0 + 4 + bar_w:.1f}" y="{y_of(bert_fpr):.1f}" '
            f'width="{bar_w:.1f}" height="{y_of(0) - y_of(bert_fpr):.1f}" fill="#c45c26"/>'
        )
        labels.append(
            f'<text x="{x0:.1f}" y="{height - 18}" font-size="12" text-anchor="middle" fill="#5c5c5c">{label}</text>'
        )

    svg = f"""<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">
  <rect width="100%" height="100%" fill="#fff"/>
  <text x="{pad}" y="22" font-size="14" fill="#1a1a1a">Accuracy vs human false-positive rate</text>
  <line x1="{pad}" y1="{y_of(0):.1f}" x2="{width - pad}" y2="{y_of(0):.1f}" stroke="#ddd4c6"/>
  {"".join(bars)}
  {"".join(labels)}
  <text x="{pad}" y="{height - 4}" font-size="11" fill="#5c5c5c">steel = DistilBERT acc · gray = TF-IDF acc · orange = DistilBERT human FPR · tan = TF-IDF human FPR</text>
</svg>
"""
    path = directory / "metrics.svg"
    path.write_text(svg, encoding="utf-8")
    return path


def _clean_split(result: dict[str, Any]) -> dict[str, Any]:
    return {
        "name": result["name"],
        "n": result["n"],
        "at_0_5": {k: _py(v) for k, v in result["at_0_5"].items()} if result.get("at_0_5") else None,
        "at_5pct_fpr": {k: _py(v) for k, v in result["at_5pct_fpr"].items()}
        if result.get("at_5pct_fpr")
        else None,
    }


def _train_impl(
    *,
    epochs: int,
    batch_size: int,
    lr: float,
    max_length: int,
    max_train: int,
    max_eval: int,
    eval_frac: float,
    domain_holdout: str,
    ood_per_model: int,
    ood_scan: int,
    dataset_id: str,
    dataset_config: str,
    ood_dataset_id: str,
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

    train_rows, eval_rows, domain_rows, data_stats = load_hc3_rows(
        dataset_id,
        config=dataset_config,
        domain_holdout=domain_holdout,
        eval_frac=eval_frac,
        max_train=max_train,
        max_eval=max_eval,
        seed=seed,
    )
    print(
        f"loaded HC3 {data_stats['train_size']} train / {data_stats['eval_size']} id "
        f"/ {data_stats['domain_size']} {domain_holdout}",
        flush=True,
    )
    ood_rows, ood_stats = load_raid_ood(
        ood_dataset_id,
        max_per_model=ood_per_model,
        max_scan=ood_scan,
    )
    print(f"ood raid {ood_stats}", flush=True)

    tfidf = train_tfidf(train_rows)
    print(f"tfidf features={tfidf['feature_count']}", flush=True)
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
    print(f"distilbert params={bert['param_count']}", flush=True)

    def eval_both(rows: list[dict[str, Any]], name: str) -> tuple[dict[str, Any], dict[str, Any]]:
        tfidf_res = score_rows(tfidf["predict_probs"], rows, name=f"tfidf_{name}")
        bert_res = score_rows(bert["predict_probs"], rows, name=f"distilbert_{name}")
        if tfidf_res["at_0_5"]:
            print(
                f"tfidf {name} acc={tfidf_res['at_0_5']['accuracy']:.4f} "
                f"human_fpr={tfidf_res['at_0_5']['human_fpr']:.4f}",
                flush=True,
            )
        if bert_res["at_0_5"]:
            print(
                f"distilbert {name} acc={bert_res['at_0_5']['accuracy']:.4f} "
                f"human_fpr={bert_res['at_0_5']['human_fpr']:.4f}",
                flush=True,
            )
        return tfidf_res, bert_res

    tfidf_id, bert_id = eval_both(eval_rows, "id")
    tfidf_domain, bert_domain = eval_both(domain_rows, "medicine")
    tfidf_ood, bert_ood = eval_both(ood_rows, "raid")

    out_dir = resolve_artifact_dirs()
    # Keep the Compute artifact small. A full save_pretrained folder previously
    # failed artifact PUT with HTTP 411 on this account.
    head_keys = [
        key
        for key in bert["model"].state_dict()
        if key.startswith(("classifier.", "pre_classifier."))
    ]
    head_state = {key: bert["model"].state_dict()[key].detach().cpu().half() for key in head_keys}
    torch.save(
        {
            "encoder": encoder,
            "class_names": list(CLASS_NAMES),
            "state_dict": head_state,
            "threshold": 0.5,
            "max_length": max_length,
            "param_count": bert["param_count"],
        },
        out_dir / "detector.pt",
    )

    examples = pick_examples(eval_rows, bert_id["probs"], threshold=0.5)
    (out_dir / "examples.json").write_text(json.dumps(examples, indent=2) + "\n", encoding="utf-8")
    (out_dir / "history.json").write_text(json.dumps(bert["history"], indent=2) + "\n", encoding="utf-8")
    write_metrics_svg(
        out_dir,
        splits=[
            ("HC3 holdout", tfidf_id.get("at_0_5"), bert_id.get("at_0_5")),
            ("medicine", tfidf_domain.get("at_0_5"), bert_domain.get("at_0_5")),
            ("RAID OOD", tfidf_ood.get("at_0_5"), bert_ood.get("at_0_5")),
        ],
    )

    id_half = bert_id["at_0_5"] or {}
    tfidf_half = tfidf_id["at_0_5"] or {}
    summary = {
        "tfidf_id": _clean_split(tfidf_id),
        "distilbert_id": _clean_split(bert_id),
        "tfidf_medicine": _clean_split(tfidf_domain),
        "distilbert_medicine": _clean_split(bert_domain),
        "tfidf_raid": _clean_split(tfidf_ood),
        "distilbert_raid": _clean_split(bert_ood),
        "beats_baseline_id_accuracy": bool(
            id_half and tfidf_half and id_half["accuracy"] > tfidf_half["accuracy"] + 1e-12
        ),
        "beats_baseline_id_fpr": bool(
            id_half and tfidf_half and id_half["human_fpr"] < tfidf_half["human_fpr"] - 1e-12
        ),
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    write_artifact_marker(
        out_dir,
        name=ARTIFACT_NAME,
        kind="output",
        compatibility_key=f"ai-text-detector-distilbert-v1-{encoder}",
        metadata={
            "filename": "detector.pt",
            "encoder": encoder,
            "param_count": bert["param_count"],
            "epochs": epochs,
            "id_accuracy": id_half.get("accuracy"),
            "id_human_fpr": id_half.get("human_fpr"),
            "dataset_id": dataset_id,
            "class_names": list(CLASS_NAMES),
        },
    )

    hub_url = None
    if push_to_hub:
        try:
            from huggingface_hub import HfApi

            hub_dir = Path(tempfile.mkdtemp(prefix="ai-text-detector-hub-"))
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
        "compat": "ai-text-detector-small",
        "device": "cuda",
        "device_name": device_name,
        "dataset_id": dataset_id,
        "encoder": encoder,
        "epochs": epochs,
        "param_count": bert["param_count"],
        "data": {k: _py(v) if not isinstance(v, list) else v for k, v in data_stats.items()},
        "ood": ood_stats,
        "history": bert["history"],
        "tfidf_id": summary["tfidf_id"],
        "distilbert_id": summary["distilbert_id"],
        "tfidf_medicine": summary["tfidf_medicine"],
        "distilbert_medicine": summary["distilbert_medicine"],
        "tfidf_raid": summary["tfidf_raid"],
        "distilbert_raid": summary["distilbert_raid"],
        "beats_baseline_id_accuracy": summary["beats_baseline_id_accuracy"],
        "beats_baseline_id_fpr": summary["beats_baseline_id_fpr"],
        "examples": examples,
        "artifact_dir": str(out_dir),
        "hub_url": hub_url,
        "torch": torch.__version__,
    }


@app.function(
    gpu="A100-PCIe-80GB",
    image=image,
    timeout=2400,
)
def train(
    epochs: int = 2,
    batch_size: int = 32,
    lr: float = 2e-5,
    max_length: int = 256,
    max_train: int = 0,
    max_eval: int = 0,
    eval_frac: float = 0.10,
    domain_holdout: str = DEFAULT_DOMAIN_HOLDOUT,
    ood_per_model: int = 250,
    ood_scan: int = 150_000,
    dataset_id: str = DEFAULT_DATASET,
    dataset_config: str = DEFAULT_DATASET_CONFIG,
    ood_dataset_id: str = DEFAULT_OOD_DATASET,
    encoder: str = DEFAULT_ENCODER,
    seed: int = 0,
    push_to_hub: bool = False,
    hub_repo: str = DEFAULT_HUB_REPO,
) -> dict:
    """Train the detector. No Hugging Face token required."""
    return _train_impl(
        epochs=epochs,
        batch_size=batch_size,
        lr=lr,
        max_length=max_length,
        max_train=max_train,
        max_eval=max_eval,
        eval_frac=eval_frac,
        domain_holdout=domain_holdout,
        ood_per_model=ood_per_model,
        ood_scan=ood_scan,
        dataset_id=dataset_id,
        dataset_config=dataset_config,
        ood_dataset_id=ood_dataset_id,
        encoder=encoder,
        seed=seed,
        push_to_hub=push_to_hub,
        hub_repo=hub_repo,
    )


@app.function(
    gpu="A100-PCIe-80GB",
    image=image,
    timeout=2400,
    secrets=[hf_secret],
)
def train_and_push(
    epochs: int = 2,
    batch_size: int = 32,
    lr: float = 2e-5,
    max_length: int = 256,
    max_train: int = 0,
    max_eval: int = 0,
    eval_frac: float = 0.10,
    domain_holdout: str = DEFAULT_DOMAIN_HOLDOUT,
    ood_per_model: int = 250,
    ood_scan: int = 150_000,
    dataset_id: str = DEFAULT_DATASET,
    dataset_config: str = DEFAULT_DATASET_CONFIG,
    ood_dataset_id: str = DEFAULT_OOD_DATASET,
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
        eval_frac=eval_frac,
        domain_holdout=domain_holdout,
        ood_per_model=ood_per_model,
        ood_scan=ood_scan,
        dataset_id=dataset_id,
        dataset_config=dataset_config,
        ood_dataset_id=ood_dataset_id,
        encoder=encoder,
        seed=seed,
        push_to_hub=push_to_hub,
        hub_repo=hub_repo,
    )


if __name__ == "__main__":
    with app.run():
        print(train.remote(epochs=1, max_train=256, max_eval=128, ood_per_model=32, ood_scan=2_000))
