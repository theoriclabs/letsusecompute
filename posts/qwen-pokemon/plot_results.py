"""Regenerate the guide's figures from the retrieved full-run results.json (matplotlib)."""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).parent
results = json.loads((ROOT / "assets/results/full/results.json").read_text())
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11, "svg.fonttype": "none"})

names = ["random", "qwen_untuned", "qwen_sft", "expert_no_tera"]
labels = ["Random moves", "Qwen3-0.6B, untuned", "Qwen3-0.6B, fine-tuned", "Scripted expert (teacher)"]
colors = ["#9a9185", "#9a9185", "#c45c26", "#176b8b"]
fig, ax = plt.subplots(figsize=(8.2, 3.6), layout="constrained")
for i, name in enumerate(names):
    r = results["eval"][name]
    lo, hi = r["win_rate_95ci"]
    ax.barh(i, 100 * r["win_rate"], height=0.55, color=colors[i])
    ax.errorbar(100 * r["win_rate"], i, xerr=[[100 * (r["win_rate"] - lo)], [100 * (hi - r["win_rate"])]],
                color="#1a1a1a", capsize=3, linewidth=1)
    ax.text(100 * hi + 1.5, i, f"{r['won']}/{r['battles']}", va="center", fontsize=10)
ax.set(yticks=range(len(names)), yticklabels=labels, xlim=(0, 108), xticks=range(0, 101, 20),
       xlabel="Win rate vs max-damage bot, % (bars: 95% interval)")
ax.invert_yaxis()
ax.spines[["top", "right", "left"]].set_visible(False)
ax.tick_params(axis="y", length=0)
ax.grid(axis="x", alpha=0.16)
ax.set_axisbelow(True)
fig.savefig(ROOT / "assets/win-rates.svg", facecolor="white")
plt.close(fig)

losses = results["training"]["losses"]
fig, ax = plt.subplots(figsize=(8.2, 3.2), layout="constrained")
ax.plot(range(1, len(losses) + 1), losses, color="#c45c26", alpha=0.2, linewidth=0.8)
w = 50
smooth = [sum(losses[max(0, i - w + 1):i + 1]) / len(losses[max(0, i - w + 1):i + 1]) for i in range(len(losses))]
ax.plot(range(1, len(losses) + 1), smooth, color="#c45c26", linewidth=1.8)
ax.set(xlabel="Optimizer step (batch 32)", ylabel="Answer-digit cross-entropy", ylim=(0, None))
ax.spines[["top", "right"]].set_visible(False)
ax.grid(alpha=0.16)
fig.savefig(ROOT / "assets/training-loss.svg", facecolor="white")
plt.close(fig)

for path in (ROOT / "assets").glob("*.svg"):
    path.write_text("\n".join(line.rstrip() for line in path.read_text().splitlines()) + "\n")
print("Wrote win-rates.svg and training-loss.svg")
