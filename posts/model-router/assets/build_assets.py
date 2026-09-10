"""Render measured figures for the model-router guide from the headline result."""

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
BROWN = "#8a5a2b"
GRAY = "#7a8b99"

always_cheap = RESULT["always_cheap"]
always_strong = RESULT["always_strong"]
oracle = RESULT["oracle"]
bert = RESULT["distilbert_operating_point"]
tfidf = RESULT["tfidf_operating_point"]
history = RESULT["history"]

MILLION = 1_000_000
BILLS = [
    ("Always GPT-4", always_strong["mean_cost_usd"] * MILLION, INK, "100% quality"),
    ("TF-IDF router", tfidf["mean_cost_usd"] * MILLION, GRAY, "95.2% quality"),
    ("DistilBERT router", bert["mean_cost_usd"] * MILLION, ACCENT, "95.2% quality · 72% less"),
    ("Oracle", oracle["mean_cost_usd"] * MILLION, BROWN, "96.6% quality"),
    ("Always Mixtral", always_cheap["mean_cost_usd"] * MILLION, GREEN, "88.3% quality"),
]


def write(path: Path, svg: str) -> None:
    path.write_text(svg, encoding="utf-8")
    print(f"wrote {path.relative_to(ROOT)}")


def money(value: float) -> str:
    if value >= 100:
        return f"${value:,.0f}"
    if value >= 1:
        return f"${value:,.2f}"
    return f"${value:.4f}"


def router_flow() -> str:
    return f"""<svg xmlns="http://www.w3.org/2000/svg" width="1200" height="420" viewBox="0 0 1200 420" role="img" aria-labelledby="t d">
  <title id="t">A prompt router sends easy prompts to Mixtral and hard ones to GPT-4</title>
  <desc id="d">Prompt goes into DistilBERT. If the cheap-ok probability is at least 0.575, the job goes to Mixtral. Otherwise it goes to GPT-4.</desc>
  <rect width="100%" height="100%" fill="{BG}"/>
  <text x="48" y="48" font-size="28" font-weight="700" fill="{INK}" font-family="system-ui, sans-serif">How the router spends</text>
  <text x="48" y="80" font-size="16" fill="{MUTED}" font-family="system-ui, sans-serif">Operating point from the headline run: send Mixtral if P(cheap suffices) ≥ 0.575</text>
  <rect x="48" y="130" width="220" height="140" rx="16" fill="{CARD}" stroke="{LINE}" stroke-width="2"/>
  <text x="68" y="188" font-size="18" font-weight="700" fill="{INK}" font-family="system-ui, sans-serif">Prompt</text>
  <text x="68" y="216" font-size="14" fill="{MUTED}" font-family="system-ui, sans-serif">10,000 held-out</text>
  <text x="68" y="238" font-size="14" fill="{MUTED}" font-family="system-ui, sans-serif">English instructions</text>
  <path d="M278 200 H318" stroke="{GRAY}" stroke-width="3" fill="none" marker-end="url(#a)"/>
  <rect x="328" y="118" width="300" height="164" rx="16" fill="#f4dccb" stroke="{ACCENT}" stroke-width="3"/>
  <text x="348" y="168" font-size="18" font-weight="700" fill="{INK}" font-family="system-ui, sans-serif">DistilBERT router</text>
  <text x="348" y="196" font-size="14" fill="{MUTED}" font-family="system-ui, sans-serif">67M params · class-weighted</text>
  <text x="348" y="220" font-size="14" fill="{MUTED}" font-family="system-ui, sans-serif">threshold 0.575</text>
  <text x="348" y="250" font-size="14" font-weight="700" fill="{ACCENT}" font-family="system-ui, sans-serif">P(Mixtral is good enough)</text>
  <path d="M628 160 H678" stroke="{GREEN}" stroke-width="3" fill="none" marker-end="url(#ag)"/>
  <path d="M628 240 H678" stroke="{INK}" stroke-width="3" fill="none" marker-end="url(#ak)"/>
  <rect x="688" y="96" width="464" height="112" rx="16" fill="#d5e8dd" stroke="{GREEN}" stroke-width="3"/>
  <text x="708" y="140" font-size="18" font-weight="700" fill="{INK}" font-family="system-ui, sans-serif">Mixtral 8x7B · 67.8% of traffic</text>
  <text x="708" y="168" font-size="14" fill="{MUTED}" font-family="system-ui, sans-serif">$0.24 / 1M tokens · everyday how-to prompts</text>
  <text x="708" y="190" font-size="14" fill="{MUTED}" font-family="system-ui, sans-serif">mean quality when chosen: Mixtral score / 5</text>
  <rect x="688" y="232" width="464" height="112" rx="16" fill="{CARD}" stroke="{INK}" stroke-width="3"/>
  <text x="708" y="276" font-size="18" font-weight="700" fill="{INK}" font-family="system-ui, sans-serif">GPT-4 Turbo · 32.2% of traffic</text>
  <text x="708" y="304" font-size="14" fill="{MUTED}" font-family="system-ui, sans-serif">$10 / $30 per 1M tokens · long extraction, messy code</text>
  <text x="708" y="326" font-size="14" fill="{MUTED}" font-family="system-ui, sans-serif">quality taken as 1.0 in this dataset</text>
  <text x="48" y="390" font-size="14" fill="{MUTED}" font-family="system-ui, sans-serif">Measured on run_2538f6a593a1fedfce8b4d4ce694c3d0. Prices are 2024 public lists for the two models in the dataset.</text>
  <defs>
    <marker id="a" markerWidth="10" markerHeight="10" refX="8" refY="3" orient="auto"><path d="M0,0 L0,6 L9,3 z" fill="{GRAY}"/></marker>
    <marker id="ag" markerWidth="10" markerHeight="10" refX="8" refY="3" orient="auto"><path d="M0,0 L0,6 L9,3 z" fill="{GREEN}"/></marker>
    <marker id="ak" markerWidth="10" markerHeight="10" refX="8" refY="3" orient="auto"><path d="M0,0 L0,6 L9,3 z" fill="{INK}"/></marker>
  </defs>
</svg>
"""


def million_bill() -> str:
    width, height, pad_l, pad_r, pad_t, pad_b = 1200, 560, 280, 80, 110, 60
    inner_w = width - pad_l - pad_r
    max_bill = max(row[1] for row in BILLS)
    bar_h = 52
    gap = 28
    rows = []
    for i, (label, bill, color, note) in enumerate(BILLS):
        y = pad_t + i * (bar_h + gap)
        w = max(8, inner_w * (bill / max_bill))
        value_x = pad_l + w + 14
        anchor = "start"
        fill_value = INK
        if w > 220:
            value_x = pad_l + w - 14
            anchor = "end"
            fill_value = "#fff"
        rows.append(f"""
  <text x="{pad_l - 16}" y="{y + 22}" font-size="16" font-weight="700" fill="{INK}" text-anchor="end" font-family="system-ui, sans-serif">{label}</text>
  <text x="{pad_l - 16}" y="{y + 42}" font-size="13" fill="{MUTED}" text-anchor="end" font-family="system-ui, sans-serif">{note}</text>
  <rect x="{pad_l}" y="{y}" width="{w:.1f}" height="{bar_h}" rx="8" fill="{color}"/>
  <text x="{value_x:.1f}" y="{y + 34}" font-size="18" font-weight="700" fill="{fill_value}" text-anchor="{anchor}" font-family="system-ui, sans-serif">{money(bill)}</text>
""")
    return f"""<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}" role="img" aria-labelledby="t d">
  <title id="t">List-price bill for one million similar prompts</title>
  <desc id="d">Always GPT-4 costs about $8,416. The DistilBERT router costs about $2,356. Always Mixtral is $92 and misses the 95 percent quality target.</desc>
  <rect width="100%" height="100%" fill="{BG}"/>
  <text x="48" y="48" font-size="28" font-weight="700" fill="{INK}" font-family="system-ui, sans-serif">A million prompts, at list price</text>
  <text x="48" y="80" font-size="16" fill="{MUTED}" font-family="system-ui, sans-serif">Same 10,000-prompt mix, scaled up. Tokens estimated as characters / 4.</text>
  {''.join(rows)}
  <text x="48" y="{height - 22}" font-size="13" fill="{MUTED}" font-family="system-ui, sans-serif">Mixtral $0.24 / 1M · GPT-4 Turbo $10 / $30 per 1M · measured mean USD / prompt × 1,000,000</text>
</svg>
"""


def cost_quality() -> str:
    width, height, pad = 1200, 640, 88
    points = [
        ("Always Mixtral", always_cheap, GREEN),
        ("Oracle", oracle, BROWN),
        ("DistilBERT @ 95%", bert, ACCENT),
        ("TF-IDF @ 95%", tfidf, GRAY),
        ("Always GPT-4", always_strong, INK),
    ]
    xs = [p[1]["mean_cost_usd"] for p in points]
    x_lo, x_hi = 0.0, max(xs) * 1.10
    y_lo, y_hi = 0.86, 1.01

    def xy(cost: float, quality: float) -> tuple[float, float]:
        x = pad + (width - 2 * pad) * (cost - x_lo) / (x_hi - x_lo)
        y = pad + (height - 2 * pad) * (1 - (quality - y_lo) / (y_hi - y_lo))
        return x, y

    x_ticks = [0.0, 0.002, 0.004, 0.006, 0.008]
    y_ticks = [0.88, 0.90, 0.92, 0.94, 0.96, 0.98, 1.00]
    grid = []
    for tick in x_ticks:
        x, _ = xy(tick, y_lo)
        grid.append(
            f'<line x1="{x:.1f}" y1="{pad}" x2="{x:.1f}" y2="{height - pad}" stroke="{LINE}"/>'
            f'<text x="{x:.1f}" y="{height - pad + 28}" font-size="14" fill="{MUTED}" text-anchor="middle" font-family="system-ui, sans-serif">${tick:.3f}</text>'
        )
    for tick in y_ticks:
        _, y = xy(x_lo, tick)
        grid.append(
            f'<line x1="{pad}" y1="{y:.1f}" x2="{width - pad}" y2="{y:.1f}" stroke="{LINE}"/>'
            f'<text x="{pad - 12:.1f}" y="{y + 5:.1f}" font-size="14" fill="{MUTED}" text-anchor="end" font-family="system-ui, sans-serif">{tick:.0%}</text>'
        )
    floor_x1, floor_y = xy(x_lo, 0.95)
    floor_x2, _ = xy(x_hi, 0.95)
    labels = {
        "Always Mixtral": (12, 22, "start"),
        "Oracle": (12, -10, "start"),
        "DistilBERT @ 95%": (12, 24, "start"),
        "TF-IDF @ 95%": (12, -14, "start"),
        "Always GPT-4": (-12, -12, "end"),
    }
    dots = []
    for label, point, color in points:
        x, y = xy(point["mean_cost_usd"], point["quality"])
        dx, dy, anchor = labels[label]
        dots.append(
            f'<circle cx="{x:.1f}" cy="{y:.1f}" r="9" fill="{color}"/>'
            f'<text x="{x + dx:.1f}" y="{y + dy:.1f}" font-size="16" font-weight="700" fill="{color}" text-anchor="{anchor}" font-family="system-ui, sans-serif">{label}</text>'
        )
    return f"""<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}" role="img" aria-labelledby="t d">
  <title id="t">Measured cost versus quality for five routing policies</title>
  <desc id="d">Always Mixtral is cheapest at 88.3 percent quality. DistilBERT and TF-IDF sit on the 95 percent line. Always GPT-4 is most expensive.</desc>
  <rect width="100%" height="100%" fill="{BG}"/>
  <text x="48" y="44" font-size="28" font-weight="700" fill="{INK}" font-family="system-ui, sans-serif">Cost vs quality</text>
  <text x="48" y="74" font-size="16" fill="{MUTED}" font-family="system-ui, sans-serif">10,000 held-out prompts · points are measured, not interpolated</text>
  {''.join(grid)}
  <line x1="{floor_x1:.1f}" y1="{floor_y:.1f}" x2="{floor_x2:.1f}" y2="{floor_y:.1f}" stroke="{ACCENT}" stroke-dasharray="6 5" stroke-width="2"/>
  <text x="{width - pad}" y="{floor_y - 12:.1f}" font-size="14" fill="{ACCENT}" text-anchor="end" font-family="system-ui, sans-serif">95% of always-GPT-4 quality</text>
  {''.join(dots)}
  <text x="48" y="{height - 18}" font-size="13" fill="{MUTED}" font-family="system-ui, sans-serif">x = mean list-price USD / prompt · y = mean quality (Mixtral score / 5 if cheap, else 1.0)</text>
</svg>
"""


def split_bar(title: str, desc: str, left_label: str, left: float, left_color: str, right_label: str, right: float, right_color: str, footnote: str) -> str:
    width, height = 1200, 280
    pad = 48
    bar_y, bar_h = 130, 64
    bar_w = width - 2 * pad
    left_w = bar_w * left
    return f"""<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}" role="img" aria-labelledby="t d">
  <title id="t">{title}</title>
  <desc id="d">{desc}</desc>
  <rect width="100%" height="100%" fill="{BG}"/>
  <text x="{pad}" y="48" font-size="26" font-weight="700" fill="{INK}" font-family="system-ui, sans-serif">{title}</text>
  <rect x="{pad}" y="{bar_y}" width="{left_w:.1f}" height="{bar_h}" fill="{left_color}"/>
  <rect x="{pad + left_w:.1f}" y="{bar_y}" width="{bar_w - left_w:.1f}" height="{bar_h}" fill="{right_color}"/>
  <text x="{pad + 20}" y="{bar_y + 42}" font-size="20" font-weight="700" fill="#fff" font-family="system-ui, sans-serif">{left_label} {left:.1%}</text>
  <text x="{pad + bar_w - 20}" y="{bar_y + 42}" font-size="20" font-weight="700" fill="#fff" text-anchor="end" font-family="system-ui, sans-serif">{right_label} {right:.1%}</text>
  <text x="{pad}" y="244" font-size="14" fill="{MUTED}" font-family="system-ui, sans-serif">{footnote}</text>
</svg>
"""


def training_curves() -> str:
    width, height, pad = 1200, 520, 80
    losses = [row[k] for row in history for k in ("train_loss", "eval_loss")]
    accs = [row["eval_accuracy"] for row in history]
    l_lo, l_hi = min(losses) - 0.04, max(losses) + 0.04
    a_lo, a_hi = 0.70, 0.80

    def x_at(epoch: int) -> float:
        return pad + (width - 2 * pad) * ((epoch - 1) / 1)

    def y_loss(value: float) -> float:
        return pad + (height - 2 * pad) * (1 - (value - l_lo) / (l_hi - l_lo))

    def y_acc(value: float) -> float:
        return pad + (height - 2 * pad) * (1 - (value - a_lo) / (a_hi - a_lo))

    def poly(key: str, yfn) -> str:
        return " ".join(f"{x_at(row['epoch']):.1f},{yfn(row[key]):.1f}" for row in history)

    dots = []
    for row in history:
        x = x_at(row["epoch"])
        dots.append(f'<circle cx="{x:.1f}" cy="{y_loss(row["train_loss"]):.1f}" r="6" fill="{STEEL}"/>')
        dots.append(f'<circle cx="{x:.1f}" cy="{y_loss(row["eval_loss"]):.1f}" r="6" fill="{ACCENT}"/>')
        dots.append(f'<circle cx="{x:.1f}" cy="{y_acc(row["eval_accuracy"]):.1f}" r="6" fill="{GREEN}"/>')
    return f"""<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}" role="img" aria-labelledby="t d">
  <title id="t">DistilBERT train loss, eval loss, and eval accuracy over two epochs</title>
  <desc id="d">Train loss fell from 0.594 to 0.522. Eval loss rose slightly from 0.562 to 0.578. Eval accuracy rose from 73.0 percent to 76.9 percent.</desc>
  <rect width="100%" height="100%" fill="{BG}"/>
  <text x="48" y="44" font-size="28" font-weight="700" fill="{INK}" font-family="system-ui, sans-serif">Two epochs on 109,101 prompts</text>
  <text x="48" y="74" font-size="16" fill="{MUTED}" font-family="system-ui, sans-serif">Accuracy is the side metric. The cost-quality point is chosen after training.</text>
  <polyline fill="none" stroke="{STEEL}" stroke-width="3" points="{poly("train_loss", y_loss)}"/>
  <polyline fill="none" stroke="{ACCENT}" stroke-width="3" points="{poly("eval_loss", y_loss)}"/>
  <polyline fill="none" stroke="{GREEN}" stroke-width="3" stroke-dasharray="7 6" points="{poly("eval_accuracy", y_acc)}"/>
  {''.join(dots)}
  <text x="{x_at(1):.1f}" y="{height - 36}" font-size="14" fill="{MUTED}" text-anchor="middle" font-family="system-ui, sans-serif">epoch 1</text>
  <text x="{x_at(2):.1f}" y="{height - 36}" font-size="14" fill="{MUTED}" text-anchor="middle" font-family="system-ui, sans-serif">epoch 2</text>
  <rect x="48" y="470" width="12" height="12" fill="{STEEL}"/><text x="66" y="481" font-size="14" fill="{MUTED}" font-family="system-ui, sans-serif">train loss 0.594 → 0.522</text>
  <rect x="320" y="470" width="12" height="12" fill="{ACCENT}"/><text x="338" y="481" font-size="14" fill="{MUTED}" font-family="system-ui, sans-serif">eval loss 0.562 → 0.578</text>
  <rect x="600" y="470" width="12" height="12" fill="{GREEN}"/><text x="618" y="481" font-size="14" fill="{MUTED}" font-family="system-ui, sans-serif">eval accuracy 73.0% → 76.9%</text>
</svg>
"""


def social_card() -> str:
    """1200×630 share card: the three bills that matter."""
    rows = [BILLS[0], BILLS[2], BILLS[4]]
    max_bill = rows[0][1]
    parts = []
    for i, (label, bill, color, _note) in enumerate(rows):
        y = 250 + i * 100
        w = 80 + 720 * (bill / max_bill)
        if w > 280:
            value_x, value_anchor, value_fill = 64 + w - 16, "end", "#fff"
        else:
            value_x, value_anchor, value_fill = 64 + w + 16, "start", INK
        parts.append(f"""
  <text x="64" y="{y + 18}" font-size="22" font-weight="700" fill="{INK}" font-family="system-ui, sans-serif">{label}</text>
  <rect x="64" y="{y + 30}" width="{w:.1f}" height="36" rx="8" fill="{color}"/>
  <text x="{value_x:.1f}" y="{y + 56}" font-size="22" font-weight="700" fill="{value_fill}" text-anchor="{value_anchor}" font-family="system-ui, sans-serif">{money(bill)}</text>
""")
    return f"""<svg xmlns="http://www.w3.org/2000/svg" width="1200" height="630" viewBox="0 0 1200 630" role="img">
  <rect width="100%" height="100%" fill="{BG}"/>
  <text x="64" y="88" font-size="44" font-weight="700" fill="{INK}" font-family="system-ui, sans-serif">A $0.08 prompt router</text>
  <text x="64" y="140" font-size="24" fill="{MUTED}" font-family="system-ui, sans-serif">Keep 95% of GPT-4 quality. Cut list-price cost 72%.</text>
  <text x="64" y="186" font-size="20" fill="{ACCENT}" font-family="system-ui, sans-serif">One million similar prompts, at public list prices</text>
  {''.join(parts)}
  <text x="64" y="590" font-size="18" fill="{MUTED}" font-family="system-ui, sans-serif">letsusecompute.com · trained on compute.cx</text>
</svg>
"""


def main() -> None:
    write(ROOT / "router_flow.svg", router_flow())
    write(ROOT / "million_bill.svg", million_bill())
    write(ROOT / "cost_quality_curve.svg", cost_quality())
    write(
        ROOT / "label_split.svg",
        split_bar(
            "Most prompts already look easy",
            "On the held-out set, Mixtral scores 4 or 5 on 86.5 percent of prompts.",
            "Mixtral ≥ 4",
            RESULT["eval_cheap_ok_frac"],
            GREEN,
            "needs GPT-4",
            1 - RESULT["eval_cheap_ok_frac"],
            ACCENT,
            "10,000 validation prompts from routellm/gpt4_dataset. This is the label, not the router.",
        ),
    )
    write(
        ROOT / "traffic_split.svg",
        split_bar(
            "What DistilBERT actually sent",
            "At the 95 percent quality target the router sent 67.8 percent of prompts to Mixtral.",
            "Mixtral",
            bert["cheap_frac"],
            GREEN,
            "GPT-4",
            1 - bert["cheap_frac"],
            INK,
            "Threshold 0.575 on P(cheap suffices). Quality held at 95.2% of always-GPT-4.",
        ),
    )
    write(ROOT / "training_curves.svg", training_curves())
    write(ROOT / "social_card.svg", social_card())


if __name__ == "__main__":
    main()
