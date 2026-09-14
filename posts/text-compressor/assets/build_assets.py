"""Render measured figures for the text-compressor guide from the headline result."""

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


def bpc(value: float) -> str:
    return f"{value:.3f}"


def mb(nbytes: int) -> str:
    return f"{nbytes / 1_000_000:.2f} MB"


def bpc_bars() -> str:
    rows = [
        ("char LM + rANS", RESULT["model_bpc"], STEEL, mb(RESULT["model_bytes"])),
        ("xz -9", RESULT["xz_bpc"], TAN, mb(RESULT["xz_bytes"])),
        ("gzip -9", RESULT["gzip_bpc"], GRAY, mb(RESULT["gzip_bytes"])),
    ]
    width, pad_l, pad_r, pad_t = 1200, 280, 80, 118
    inner = width - pad_l - pad_r
    ceiling = max(row[1] for row in rows)
    bar_h, gap = 64, 36
    blocks = []
    for i, (label, value, color, note) in enumerate(rows):
        y = pad_t + i * (bar_h + gap)
        w = max(16.0, inner * (value / ceiling))
        blocks.append(
            f"""
  <text x="{pad_l - 16}" y="{y + 28}" font-size="18" font-weight="700" fill="{INK}" text-anchor="end" font-family="system-ui, sans-serif">{label}</text>
  <text x="{pad_l - 16}" y="{y + 50}" font-size="13" fill="{MUTED}" text-anchor="end" font-family="system-ui, sans-serif">{note}</text>
  <rect x="{pad_l}" y="{y}" width="{w:.1f}" height="{bar_h}" rx="8" fill="{color}"/>
  <text x="{pad_l + w - 16:.1f}" y="{y + 40}" font-size="22" font-weight="700" fill="#fff" text-anchor="end" font-family="system-ui, sans-serif">{bpc(value)}</text>
"""
        )
    return f"""<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="480" viewBox="0 0 {width} 480" role="img" aria-labelledby="t d">
  <title id="t">Bits per byte on 10 MB of held-out enwik8</title>
  <desc id="d">The character model plus rANS uses 1.817 bits per byte. xz uses 2.144. gzip uses 2.871.</desc>
  <rect width="100%" height="100%" fill="{BG}"/>
  <text x="48" y="48" font-size="28" font-weight="700" fill="{INK}" font-family="system-ui, sans-serif">Bits per byte on the last 10 MB</text>
  <text x="48" y="80" font-size="16" fill="{MUTED}" font-family="system-ui, sans-serif">Lower is better. Same 10,000,000 bytes for every bar.</text>
  {"".join(blocks)}
  <text x="48" y="456" font-size="14" fill="{MUTED}" font-family="system-ui, sans-serif">gzip.compress(level=9) and lzma.compress(preset=9) in the Python standard library.</text>
</svg>
"""


def size_bars() -> str:
    rows = [
        ("char LM + rANS", RESULT["model_bytes"], STEEL),
        ("xz -9", RESULT["xz_bytes"], TAN),
        ("gzip -9", RESULT["gzip_bytes"], GRAY),
        ("original", RESULT["test_bytes"], LINE),
    ]
    width, pad_l, pad_r, pad_t = 1200, 280, 80, 118
    inner = width - pad_l - pad_r
    ceiling = float(RESULT["test_bytes"])
    bar_h, gap = 52, 28
    blocks = []
    for i, (label, value, color) in enumerate(rows):
        y = pad_t + i * (bar_h + gap)
        w = max(16.0, inner * (value / ceiling))
        ink = INK if color == LINE else "#fff"
        blocks.append(
            f"""
  <text x="{pad_l - 16}" y="{y + 34}" font-size="17" font-weight="700" fill="{INK}" text-anchor="end" font-family="system-ui, sans-serif">{label}</text>
  <rect x="{pad_l}" y="{y}" width="{w:.1f}" height="{bar_h}" rx="8" fill="{color}"/>
  <text x="{pad_l + w - 14:.1f}" y="{y + 34}" font-size="18" font-weight="700" fill="{ink}" text-anchor="end" font-family="system-ui, sans-serif">{mb(value)}</text>
"""
        )
    return f"""<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="500" viewBox="0 0 {width} 500" role="img" aria-labelledby="t d">
  <title id="t">Compressed size of the 10 MB holdout</title>
  <desc id="d">The model bitstream is 2.27 MB. xz is 2.68 MB. gzip is 3.59 MB. The original is 10 MB.</desc>
  <rect width="100%" height="100%" fill="{BG}"/>
  <text x="48" y="48" font-size="28" font-weight="700" fill="{INK}" font-family="system-ui, sans-serif">10 MB in, four sizes out</text>
  <text x="48" y="80" font-size="16" fill="{MUTED}" font-family="system-ui, sans-serif">Bitstream only. The 6.7 MB of fp16 weights are not in these bars.</text>
  {"".join(blocks)}
  <text x="48" y="476" font-size="14" fill="{MUTED}" font-family="system-ui, sans-serif">Add the weights and the model is 9.0 MB — worse than gzip for a one-off file.</text>
</svg>
"""


def dataset_split() -> str:
    return f"""<svg xmlns="http://www.w3.org/2000/svg" width="1200" height="280" viewBox="0 0 1200 280" role="img" aria-labelledby="t d">
  <title id="t">enwik8 is split 90 MB train and 10 MB test</title>
  <desc id="d">First 90 million bytes train. Last 10 million bytes are the holdout.</desc>
  <rect width="100%" height="100%" fill="{BG}"/>
  <text x="48" y="48" font-size="26" font-weight="700" fill="{INK}" font-family="system-ui, sans-serif">First 90 MB train. Last 10 MB test.</text>
  <rect x="48" y="100" width="993.6" height="72" fill="{STEEL}"/>
  <rect x="1041.6" y="100" width="110.4" height="72" fill="{ACCENT}"/>
  <text x="68" y="146" font-size="20" font-weight="700" fill="#fff" font-family="system-ui, sans-serif">train 90 MB</text>
  <rect x="48" y="192" width="14" height="14" rx="3" fill="{STEEL}"/>
  <text x="70" y="204" font-size="15" fill="{INK}" font-family="system-ui, sans-serif">train 90,000,000 bytes</text>
  <rect x="340" y="192" width="14" height="14" rx="3" fill="{ACCENT}"/>
  <text x="362" y="204" font-size="15" fill="{INK}" font-family="system-ui, sans-serif">holdout 10,000,000 bytes</text>
  <text x="48" y="248" font-size="14" fill="{MUTED}" font-family="system-ui, sans-serif">enwik8: first 100 MB of the 2006 English Wikipedia XML dump. Split by byte offset, not by article.</text>
</svg>
"""


def compressor_flow() -> str:
    return f"""<svg xmlns="http://www.w3.org/2000/svg" width="1200" height="400" viewBox="0 0 1200 400" role="img" aria-labelledby="t d">
  <title id="t">Bytes go into a character GPT, then a range coder writes a bitstream</title>
  <desc id="d">The model emits a 256-way next-byte distribution. rANS turns that into bits. Decode rebuilt the first 32,768 holdout bytes exactly.</desc>
  <rect width="100%" height="100%" fill="{BG}"/>
  <text x="48" y="48" font-size="28" font-weight="700" fill="{INK}" font-family="system-ui, sans-serif">A language model is a compressor</text>
  <text x="48" y="80" font-size="16" fill="{MUTED}" font-family="system-ui, sans-serif">Next-byte probabilities plus a range coder. Decode must invert that.</text>
  <rect x="48" y="130" width="220" height="150" rx="16" fill="{CARD}" stroke="{LINE}" stroke-width="2"/>
  <text x="68" y="188" font-size="18" font-weight="700" fill="{INK}" font-family="system-ui, sans-serif">Holdout bytes</text>
  <text x="68" y="216" font-size="14" fill="{MUTED}" font-family="system-ui, sans-serif">10 MB of enwik8</text>
  <text x="68" y="240" font-size="14" fill="{MUTED}" font-family="system-ui, sans-serif">never seen in training</text>
  <path d="M278 205 H318" stroke="{GRAY}" stroke-width="3" fill="none" marker-end="url(#a)"/>
  <rect x="328" y="118" width="280" height="174" rx="16" fill="#f4dccb" stroke="{ACCENT}" stroke-width="3"/>
  <text x="348" y="168" font-size="18" font-weight="700" fill="{INK}" font-family="system-ui, sans-serif">Char GPT</text>
  <text x="348" y="196" font-size="14" fill="{MUTED}" font-family="system-ui, sans-serif">3,356,160 params</text>
  <text x="348" y="220" font-size="14" fill="{MUTED}" font-family="system-ui, sans-serif">256-byte context</text>
  <text x="348" y="256" font-size="14" font-weight="700" fill="{ACCENT}" font-family="system-ui, sans-serif">P(next byte)</text>
  <path d="M618 205 H658" stroke="{STEEL}" stroke-width="3" fill="none" marker-end="url(#as)"/>
  <rect x="668" y="130" width="220" height="150" rx="16" fill="{CARD}" stroke="{STEEL}" stroke-width="3"/>
  <text x="688" y="188" font-size="18" font-weight="700" fill="{INK}" font-family="system-ui, sans-serif">rANS</text>
  <text x="688" y="216" font-size="14" fill="{MUTED}" font-family="system-ui, sans-serif">12-bit frequencies</text>
  <text x="688" y="240" font-size="14" fill="{MUTED}" font-family="system-ui, sans-serif">decode checked 32 KB</text>
  <path d="M898 205 H938" stroke="{GREEN}" stroke-width="3" fill="none" marker-end="url(#ag)"/>
  <rect x="948" y="130" width="204" height="150" rx="16" fill="#d5e8dd" stroke="{GREEN}" stroke-width="3"/>
  <text x="968" y="188" font-size="18" font-weight="700" fill="{INK}" font-family="system-ui, sans-serif">2.27 MB</text>
  <text x="968" y="216" font-size="14" fill="{MUTED}" font-family="system-ui, sans-serif">1.817 bits/byte</text>
  <text x="968" y="240" font-size="14" fill="{MUTED}" font-family="system-ui, sans-serif">beats gzip and xz</text>
  <text x="48" y="370" font-size="14" fill="{MUTED}" font-family="system-ui, sans-serif">Measured on run_ea0ca7f6a97ca6bf787f075bb8dc1af1. Cross-entropy is 1.738 bits/byte; the coder spends 0.079 extra on quantized frequencies.</text>
  <defs>
    <marker id="a" markerWidth="10" markerHeight="10" refX="8" refY="3" orient="auto"><path d="M0,0 L0,6 L9,3 z" fill="{GRAY}"/></marker>
    <marker id="as" markerWidth="10" markerHeight="10" refX="8" refY="3" orient="auto"><path d="M0,0 L0,6 L9,3 z" fill="{STEEL}"/></marker>
    <marker id="ag" markerWidth="10" markerHeight="10" refX="8" refY="3" orient="auto"><path d="M0,0 L0,6 L9,3 z" fill="{GREEN}"/></marker>
  </defs>
</svg>
"""


def training_curves() -> str:
    history = RESULT["history"]
    width, height, pad = 1200, 520, 80
    losses = [row["train_loss"] for row in history]
    lo, hi = 0.0, max(losses) * 1.05
    sampled = history[::4] + [history[-1]]

    def x_at(step: int) -> float:
        return pad + (width - 2 * pad) * ((step - 1) / max(history[-1]["step"] - 1, 1))

    def y_at(value: float) -> float:
        return pad + (height - 2 * pad) * (1 - (value - lo) / (hi - lo))

    points = " ".join(f"{x_at(row['step']):.1f},{y_at(row['train_loss']):.1f}" for row in history)
    grid = []
    for tick in (0.0, 1.5, 3.0, 4.5, 6.0):
        y = y_at(tick)
        grid.append(
            f'<line x1="{pad}" y1="{y:.1f}" x2="{width - pad}" y2="{y:.1f}" stroke="{LINE}"/>'
            f'<text x="{pad - 12}" y="{y + 5:.1f}" font-size="13" fill="{MUTED}" text-anchor="end" font-family="system-ui, sans-serif">{tick:.1f}</text>'
        )
    dots = "".join(
        f'<circle cx="{x_at(row["step"]):.1f}" cy="{y_at(row["train_loss"]):.1f}" r="4" fill="{STEEL}"/>'
        for row in sampled
    )
    return f"""<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}" role="img" aria-labelledby="t d">
  <title id="t">Train loss over 20,000 steps</title>
  <desc id="d">Cross-entropy fell from 5.70 nats to 1.20 nats. That is 1.73 bits per byte on the training windows.</desc>
  <rect width="100%" height="100%" fill="{BG}"/>
  <text x="48" y="44" font-size="28" font-weight="700" fill="{INK}" font-family="system-ui, sans-serif">20,000 steps on 90 MB</text>
  <text x="48" y="74" font-size="16" fill="{MUTED}" font-family="system-ui, sans-serif">Train loss in nats. Uniform bytes start at ln(256) = 5.55.</text>
  {"".join(grid)}
  <polyline fill="none" stroke="{STEEL}" stroke-width="2.5" points="{points}"/>
  {dots}
  <text x="{x_at(1):.1f}" y="{height - 36}" font-size="14" fill="{MUTED}" text-anchor="middle" font-family="system-ui, sans-serif">step 1</text>
  <text x="{x_at(20000):.1f}" y="{height - 36}" font-size="14" fill="{MUTED}" text-anchor="end" font-family="system-ui, sans-serif">step 20,000</text>
</svg>
"""


def social_card() -> str:
    return f"""<svg xmlns="http://www.w3.org/2000/svg" width="1200" height="630" viewBox="0 0 1200 630" role="img" aria-labelledby="t">
  <title id="t">A $1.34 language-model compressor</title>
  <rect width="100%" height="100%" fill="{BG}"/>
  <text x="72" y="88" font-size="20" fill="{MUTED}" font-family="system-ui, sans-serif">Let’s use compute · post #7</text>
  <text x="72" y="168" font-size="48" font-weight="700" fill="{INK}" font-family="system-ui, sans-serif">A $1.34 LM compressor</text>
  <text x="72" y="220" font-size="22" fill="{MUTED}" font-family="system-ui, sans-serif">1.82 bits/byte on enwik8. gzip needs 2.87. xz needs 2.14.</text>
  <text x="72" y="300" font-size="16" font-weight="700" fill="{INK}" font-family="system-ui, sans-serif">char LM + rANS</text>
  <rect x="72" y="312" width="668.4" height="48" fill="{STEEL}"/>
  <text x="88" y="344" font-size="18" font-weight="700" fill="#fff" font-family="system-ui, sans-serif">1.817</text>
  <text x="72" y="410" font-size="16" font-weight="700" fill="{INK}" font-family="system-ui, sans-serif">gzip -9</text>
  <rect x="72" y="422" width="1055.8" height="48" fill="{GRAY}"/>
  <text x="88" y="454" font-size="18" font-weight="700" fill="#fff" font-family="system-ui, sans-serif">2.871</text>
  <text x="72" y="560" font-size="18" fill="{ACCENT}" font-family="system-ui, sans-serif">letsusecompute.com/posts/text-compressor</text>
</svg>
"""


def main() -> None:
    write(ROOT / "bpc_bars.svg", bpc_bars())
    write(ROOT / "size_bars.svg", size_bars())
    write(ROOT / "dataset_split.svg", dataset_split())
    write(ROOT / "compressor_flow.svg", compressor_flow())
    write(ROOT / "training_curves.svg", training_curves())
    write(ROOT / "social_card.svg", social_card())


if __name__ == "__main__":
    main()
