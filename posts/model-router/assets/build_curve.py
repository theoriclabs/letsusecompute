"""Render the measured cost-vs-quality points from the headline run result."""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
RESULT = json.loads((ROOT / "results" / "run-result.json").read_text())["result"]

always_cheap = RESULT["always_cheap"]
always_strong = RESULT["always_strong"]
oracle = RESULT["oracle"]
bert = RESULT["distilbert_operating_point"]
tfidf = RESULT["tfidf_operating_point"]
quality_floor = RESULT["quality_floor"]

points = [
    ("Always Mixtral", always_cheap, "#3d6b4f"),
    ("TF-IDF @ 95%", tfidf, "#7a8b99"),
    ("DistilBERT @ 95%", bert, "#c45c26"),
    ("Oracle", oracle, "#8a5a2b"),
    ("Always GPT-4", always_strong, "#1a1a1a"),
]

width, height, pad = 760, 440, 58
xs = [p[1]["mean_cost_usd"] for p in points]
ys = [p[1]["quality"] for p in points]
x_lo, x_hi = 0.0, max(xs) * 1.08
y_lo, y_hi = 0.86, 1.005


def xy(cost: float, quality: float) -> tuple[float, float]:
    x = pad + (width - 2 * pad) * (cost - x_lo) / (x_hi - x_lo)
    y = pad + (height - 2 * pad) * (1 - (quality - y_lo) / (y_hi - y_lo))
    return x, y


floor_x1, floor_y = xy(x_lo, quality_floor)
floor_x2, _ = xy(x_hi, quality_floor)

# Light axis ticks
x_ticks = [0.0, 0.002, 0.004, 0.006, 0.008]
y_ticks = [0.88, 0.90, 0.92, 0.94, 0.96, 0.98, 1.00]

grid = []
for tick in x_ticks:
    x, _ = xy(tick, y_lo)
    grid.append(
        f'<line x1="{x:.1f}" y1="{pad}" x2="{x:.1f}" y2="{height - pad}" stroke="#eee6da" />'
        f'<text x="{x:.1f}" y="{height - pad + 16}" font-size="11" fill="#5c5c5c" text-anchor="middle">${tick:.3f}</text>'
    )
for tick in y_ticks:
    _, y = xy(x_lo, tick)
    grid.append(
        f'<line x1="{pad}" y1="{y:.1f}" x2="{width - pad}" y2="{y:.1f}" stroke="#eee6da" />'
        f'<text x="{pad - 8:.1f}" y="{y + 4:.1f}" font-size="11" fill="#5c5c5c" text-anchor="end">{tick:.0%}</text>'
    )

dots = []
for i, (label, point, color) in enumerate(points):
    x, y = xy(point["mean_cost_usd"], point["quality"])
    # Stagger labels so they don't collide.
    if label == "Always Mixtral":
        tx, ty, anchor = x + 10, y + 18, "start"
    elif label == "TF-IDF @ 95%":
        tx, ty, anchor = x + 10, y - 14, "start"
    elif label == "DistilBERT @ 95%":
        tx, ty, anchor = x + 10, y + 18, "start"
    elif label == "Oracle":
        tx, ty, anchor = x + 10, y + 4, "start"
    else:
        tx, ty, anchor = x - 10, y - 10, "end"
    dots.append(
        f'<circle cx="{x:.1f}" cy="{y:.1f}" r="6" fill="{color}" />'
        f'<text x="{tx:.1f}" y="{ty:.1f}" font-size="12" fill="{color}" text-anchor="{anchor}">{label}</text>'
    )

svg = f"""<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}" role="img" aria-labelledby="title desc">
  <title id="title">Measured cost versus quality for prompt routing policies</title>
  <desc id="desc">Always Mixtral is cheapest at 88.3% quality. DistilBERT at the 95% target is cheaper than TF-IDF and much cheaper than always GPT-4. Oracle is the best possible router using the held-out labels.</desc>
  <rect width="100%" height="100%" fill="#fff"/>
  <text x="{pad}" y="24" font-size="15" fill="#1a1a1a">Cost vs quality on 10,000 held-out prompts</text>
  {''.join(grid)}
  <line x1="{floor_x1:.1f}" y1="{floor_y:.1f}" x2="{floor_x2:.1f}" y2="{floor_y:.1f}" stroke="#c45c26" stroke-dasharray="5 4" stroke-width="1.2"/>
  <text x="{width - pad}" y="{floor_y - 8:.1f}" font-size="11" fill="#c45c26" text-anchor="end">95% of always-GPT-4 quality</text>
  {''.join(dots)}
  <text x="{pad}" y="{height - 12}" font-size="11" fill="#5c5c5c">x = mean list-price USD / prompt · y = mean quality · points are measured, not interpolated</text>
</svg>
"""

out = ROOT / "cost_quality_curve.svg"
out.write_text(svg, encoding="utf-8")
print(f"wrote {out}")
