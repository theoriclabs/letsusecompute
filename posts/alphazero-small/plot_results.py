"""Regenerate the strength curve and loss figure from assets/results/history.json (matplotlib)."""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).parent
hist = json.loads((ROOT / "assets/results/history.json").read_text())["history"]
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11, "svg.fonttype": "none"})

series = [
    ("uct1000", "vs rollout MCTS, 1,000 playouts", "#c45c26", "-"),
    ("policy_vs_one_ply", "raw policy, no search, vs one-ply", "#176b8b", "-"),
    ("iteration_0", "vs untrained net + same search", "#6b8e23", "--"),
    ("one_ply", "vs one-ply win/block", "#9a9185", "--"),
    ("random", "vs random mover", "#c9c2b6", ":"),
]
rows = [r for r in hist if r.get("eval")]
fig, ax = plt.subplots(figsize=(8.2, 4.4), layout="constrained")
for key, label, color, style in series:
    xs = [r["iteration"] for r in rows]
    ys = [100 * r["eval"][key]["score"] for r in rows]
    ax.plot(xs, ys, style, color=color, linewidth=2 if style == "-" else 1.4, marker="o", markersize=3, label=label)
runs = [r.get("run_id") for r in hist if r["iteration"] > 0]
boundary = next((hist[i]["iteration"] for i in range(2, len(hist))
                 if hist[i].get("run_id") != hist[i - 1].get("run_id")), None)
if boundary:
    ax.axvline(boundary - 0.5, color="#1a1a1a", linewidth=1, alpha=0.5)
    ax.text(boundary - 0.3, 3, "second run resumes\nfrom the checkpoint", fontsize=9, va="bottom")
ax.set(xlabel="Training iteration (768 self-play games each)", ylabel="Score, % (win 1, draw ½)",
       ylim=(0, 102), xlim=(-0.5, max(r["iteration"] for r in hist) + 0.5))
ax.spines[["top", "right"]].set_visible(False)
ax.grid(alpha=0.16)
ax.legend(frameon=False, fontsize=9, loc="lower right")
fig.savefig(ROOT / "assets/strength-curve.svg", facecolor="white")
plt.close(fig)

train_rows = [r for r in hist if r.get("train")]
fig, ax = plt.subplots(figsize=(8.2, 3.0), layout="constrained")
ax.plot([r["iteration"] for r in train_rows], [r["train"]["policy_loss"] for r in train_rows], color="#c45c26",
        label="policy loss (cross-entropy to MCTS visits)")
ax.plot([r["iteration"] for r in train_rows], [r["train"]["value_loss"] for r in train_rows], color="#176b8b",
        label="value loss (MSE to game result)")
ax.set(xlabel="Training iteration", ylim=(0, None))
ax.spines[["top", "right"]].set_visible(False)
ax.grid(alpha=0.16)
ax.legend(frameon=False, fontsize=9)
fig.savefig(ROOT / "assets/training-loss.svg", facecolor="white")
plt.close(fig)

for path in (ROOT / "assets").glob("*.svg"):
    path.write_text("\n".join(line.rstrip() for line in path.read_text().splitlines()) + "\n")
print("Wrote strength-curve.svg and training-loss.svg")
