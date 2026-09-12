"""Render measured figures for the AI-text detector guide from the headline result."""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
RESULT = json.loads((ROOT / "results" / "summary.json").read_text())

BG = "#f7f4ef"
CARD = "#ffffff"
INK = "#1a1a1a"
MUTED = "#5c5c5c"
LINE = "#ddd4c6"
ACCENT = "#c45c26"
STEEL = "#3d6b8a"
GREEN = "#3d6b4f"
GRAY = "#7a8b99"
TAN = "#c4a484"


def write(path: Path, svg: str) -> None:
    path.write_text(svg, encoding="utf-8")
    print(f"wrote {path.relative_to(ROOT)}")


def pct(value: float) -> str:
    return f"{value * 100:.1f}%"


def split_bars() -> str:
    rows = [
        ("HC3 holdout", RESULT["tfidf_id"]["at_0_5"], RESULT["distilbert_id"]["at_0_5"]),
        ("Medicine", RESULT["tfidf_medicine"]["at_0_5"], RESULT["distilbert_medicine"]["at_0_5"]),
        ("RAID OOD", RESULT["tfidf_raid"]["at_0_5"], RESULT["distilbert_raid"]["at_0_5"]),
    ]
    width, height, pad = 1200, 560, 64
    gap = (width - 2 * pad) / len(rows)
    bar_w = 36

    def y_of(value: float) -> float:
        return 72 + (height - 140) * (1 - value)

    bars = []
    labels = []
    for i, (name, tfidf, bert) in enumerate(rows):
        x0 = pad + gap * i + gap / 2
        pairs = [
            (x0 - 2 * bar_w - 6, tfidf["accuracy"], GRAY),
            (x0 - bar_w - 2, bert["accuracy"], STEEL),
            (x0 + 6, tfidf["human_fpr"], TAN),
            (x0 + bar_w + 10, bert["human_fpr"], ACCENT),
        ]
        for x, value, color in pairs:
            top = y_of(value)
            bars.append(
                f'<rect x="{x:.1f}" y="{top:.1f}" width="{bar_w}" height="{y_of(0) - top:.1f}" fill="{color}"/>'
            )
            bars.append(
                f'<text x="{x + bar_w / 2:.1f}" y="{top - 8:.1f}" font-size="13" text-anchor="middle" fill="{INK}" font-family="system-ui, sans-serif">{pct(value)}</text>'
            )
        labels.append(
            f'<text x="{x0:.1f}" y="{height - 36}" font-size="18" font-weight="700" text-anchor="middle" fill="{INK}" font-family="system-ui, sans-serif">{name}</text>'
        )

    return f"""<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}" role="img" aria-labelledby="t d">
  <title id="t">Accuracy stays high on ChatGPT answers and collapses on RAID</title>
  <desc id="d">Grouped bars for TF-IDF and DistilBERT accuracy and human false-positive rate on HC3 holdout, medicine, and RAID.</desc>
  <rect width="100%" height="100%" fill="{BG}"/>
  <text x="{pad}" y="40" font-size="28" font-weight="700" fill="{INK}" font-family="system-ui, sans-serif">Same detector, three tests</text>
  <line x1="{pad}" y1="{y_of(0):.1f}" x2="{width - pad}" y2="{y_of(0):.1f}" stroke="{LINE}"/>
  {"".join(bars)}
  {"".join(labels)}
  <text x="{pad}" y="{height - 10}" font-size="14" fill="{MUTED}" font-family="system-ui, sans-serif">steel = DistilBERT acc · gray = TF-IDF acc · orange = DistilBERT human FPR · tan = TF-IDF human FPR</text>
</svg>
"""


def training_curves() -> str:
    history = RESULT["history"]
    width, height, pad = 1200, 520, 72
    xs = [row["epoch"] for row in history]
    losses = [row["train_loss"] for row in history] + [row["eval_loss"] for row in history]
    accs = [row["eval_accuracy"] for row in history]
    lo, hi = min(losses), max(losses)
    if hi <= lo:
        hi = lo + 1e-6
    a_lo, a_hi = min(0.97, min(accs) - 0.01), 1.0

    def x_of(epoch: float) -> float:
        return pad + (width - 2 * pad) * ((epoch - 1) / max(len(history) - 1, 1))

    def y_loss(value: float) -> float:
        return pad + (height - 2 * pad) * (1 - (value - lo) / (hi - lo))

    def y_acc(value: float) -> float:
        return pad + (height - 2 * pad) * (1 - (value - a_lo) / (a_hi - a_lo))

    train_pts = " ".join(f"{x_of(r['epoch']):.1f},{y_loss(r['train_loss']):.1f}" for r in history)
    eval_pts = " ".join(f"{x_of(r['epoch']):.1f},{y_loss(r['eval_loss']):.1f}" for r in history)
    acc_pts = " ".join(f"{x_of(r['epoch']):.1f},{y_acc(r['eval_accuracy']):.1f}" for r in history)
    return f"""<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}" role="img" aria-labelledby="t d">
  <title id="t">Two-epoch training curves on HC3</title>
  <desc id="d">Train and eval loss fall. Held-out accuracy rises from 99.22 percent to 99.42 percent.</desc>
  <rect width="100%" height="100%" fill="{BG}"/>
  <text x="{pad}" y="40" font-size="28" font-weight="700" fill="{INK}" font-family="system-ui, sans-serif">Two epochs on ChatGPT answers</text>
  <polyline fill="none" stroke="{STEEL}" stroke-width="4" points="{train_pts}"/>
  <polyline fill="none" stroke="{ACCENT}" stroke-width="4" points="{eval_pts}"/>
  <polyline fill="none" stroke="{GREEN}" stroke-width="3" stroke-dasharray="8 6" points="{acc_pts}"/>
  <text x="{pad}" y="{height - 16}" font-size="14" fill="{MUTED}" font-family="system-ui, sans-serif">steel = train loss · orange = eval loss · dashed green = held-out accuracy (right scale, {pct(a_lo)}–{pct(a_hi)})</text>
</svg>
"""


def ood_story() -> str:
    id_m = RESULT["distilbert_id"]["at_0_5"]
    raid = RESULT["distilbert_raid"]["at_0_5"]
    caught = raid["ai_recall"]
    missed = 1 - caught
    return f"""<svg xmlns="http://www.w3.org/2000/svg" width="1200" height="420" viewBox="0 0 1200 420" role="img" aria-labelledby="t d">
  <title id="t">The detector catches ChatGPT and misses most GPT-4 and Llama text</title>
  <desc id="d">On HC3 it flags 99.9 percent of ChatGPT answers. On RAID it flags 22.8 percent of GPT-4 and Llama answers.</desc>
  <rect width="100%" height="100%" fill="{BG}"/>
  <text x="48" y="48" font-size="28" font-weight="700" fill="{INK}" font-family="system-ui, sans-serif">Change the generator, keep the humans</text>
  <rect x="48" y="96" width="520" height="220" rx="16" fill="{CARD}" stroke="{LINE}" stroke-width="2"/>
  <text x="72" y="140" font-size="18" font-weight="700" fill="{INK}" font-family="system-ui, sans-serif">HC3 holdout · ChatGPT</text>
  <text x="72" y="196" font-size="52" font-weight="700" fill="{STEEL}" font-family="system-ui, sans-serif">{pct(id_m["ai_recall"])}</text>
  <text x="72" y="236" font-size="16" fill="{MUTED}" font-family="system-ui, sans-serif">of ChatGPT answers flagged</text>
  <text x="72" y="268" font-size="16" fill="{MUTED}" font-family="system-ui, sans-serif">{pct(id_m["human_fpr"])} of human answers flagged</text>
  <rect x="632" y="96" width="520" height="220" rx="16" fill="#f4dccb" stroke="{ACCENT}" stroke-width="3"/>
  <text x="656" y="140" font-size="18" font-weight="700" fill="{INK}" font-family="system-ui, sans-serif">RAID · GPT-4 + Llama-chat</text>
  <text x="656" y="196" font-size="52" font-weight="700" fill="{ACCENT}" font-family="system-ui, sans-serif">{pct(caught)}</text>
  <text x="656" y="236" font-size="16" fill="{MUTED}" font-family="system-ui, sans-serif">of new-generator answers flagged</text>
  <text x="656" y="268" font-size="16" fill="{MUTED}" font-family="system-ui, sans-serif">{pct(missed)} walk through · {pct(raid["human_fpr"])} human FPR</text>
  <text x="48" y="380" font-size="14" fill="{MUTED}" font-family="system-ui, sans-serif">200 RAID humans, 200 GPT-4, 200 Llama-chat. No adversarial attacks. Threshold 0.5.</text>
</svg>
"""


def social_card() -> str:
    acc = RESULT["distilbert_id"]["at_0_5"]["accuracy"]
    raid_acc = RESULT["distilbert_raid"]["at_0_5"]["accuracy"]
    return f"""<svg xmlns="http://www.w3.org/2000/svg" width="1200" height="630" viewBox="0 0 1200 630" role="img" aria-labelledby="t">
  <title id="t">A $0.38 AI-text detector</title>
  <rect width="100%" height="100%" fill="{BG}"/>
  <text x="72" y="120" font-size="22" fill="{MUTED}" font-family="system-ui, sans-serif">Let’s use compute · post #6</text>
  <text x="72" y="210" font-size="56" font-weight="700" fill="{INK}" font-family="system-ui, sans-serif">A $0.38 AI-text detector</text>
  <text x="72" y="290" font-size="28" fill="{MUTED}" font-family="system-ui, sans-serif">{pct(acc)} on ChatGPT answers. {pct(raid_acc)} when the</text>
  <text x="72" y="332" font-size="28" fill="{MUTED}" font-family="system-ui, sans-serif">generator changes to GPT-4 and Llama.</text>
  <text x="72" y="520" font-size="20" fill="{ACCENT}" font-family="system-ui, sans-serif">letsusecompute.com/posts/ai-text-detector</text>
</svg>
"""


def main() -> None:
    write(ROOT / "metrics.svg", split_bars())
    write(ROOT / "training_curves.svg", training_curves())
    write(ROOT / "ood_story.svg", ood_story())
    write(ROOT / "social_card.svg", social_card())


if __name__ == "__main__":
    main()
