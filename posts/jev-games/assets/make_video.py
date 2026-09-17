"""Build a silent captioned replay video from saved jev-games JSON frames."""

from __future__ import annotations

import json
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parent
REPLAYS = ROOT / "results" / "weights" / "replays"
OUT = ROOT / "social"
SIZE = 1080
FPS = 8


def grid_image(lines: list[str], title: str, subtitle: str) -> Image.Image:
    img = Image.new("RGB", (SIZE, SIZE), (247, 244, 239))
    draw = ImageDraw.Draw(img)
    draw.text((48, 40), title, fill=(26, 26, 26))
    draw.text((48, 88), subtitle, fill=(92, 92, 92))
    h, w = len(lines), max(len(row) for row in lines)
    cell = min(72, (SIZE - 160) // max(w, 1), (SIZE - 200) // max(h, 1))
    x0 = (SIZE - w * cell) // 2
    y0 = 160
    colors = {
        ".": (236, 230, 220),
        "W": (90, 84, 74),
        "H": (196, 92, 38),
        "o": (61, 107, 138),
        "*": (196, 92, 38),
        "#": (122, 139, 153),
        "=": (61, 107, 79),
        "K": (196, 164, 56),
        "D": (140, 80, 60),
        "d": (180, 150, 110),
        "G": (61, 138, 90),
        "R": (196, 92, 38),
        "L": (196, 92, 38),
        "U": (196, 92, 38),
        "N": (196, 92, 38),
        "E": (196, 92, 38),
        "S": (196, 92, 38),
    }
    for y, row in enumerate(lines):
        for x, ch in enumerate(row):
            draw.rectangle(
                [x0 + x * cell, y0 + y * cell, x0 + (x + 1) * cell - 2, y0 + (y + 1) * cell - 2],
                fill=colors.get(ch, (40, 40, 40)),
            )
    draw.text((48, SIZE - 64), "Structured observation · recorded replay, not live inference", fill=(92, 92, 92))
    return img


def load(name: str) -> dict:
    return json.loads((REPLAYS / f"{name}.json").read_text())


def write_mp4(frames: list[Image.Image], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    arrays = [np.asarray(frame.convert("RGB")) for frame in frames]
    imageio.mimsave(path, arrays, fps=FPS, codec="libx264", quality=8)
    gif = path.with_suffix(".gif")
    imageio.mimsave(gif, arrays[::2], fps=max(FPS // 2, 4), loop=0)


def main() -> None:
    pairs = [
        ("snake", "Snake 8×8"),
        ("doorkey", "DoorKey-5x5 fully observed"),
        ("breakout", "MinAtar-style Breakout"),
    ]
    frames: list[Image.Image] = []
    title = Image.new("RGB", (SIZE, SIZE), (247, 244, 239))
    draw = ImageDraw.Draw(title)
    draw.text((48, 360), "One decision model, three games", fill=(26, 26, 26))
    draw.text((48, 430), "Random policy vs the trained scorer. Same seeds.", fill=(92, 92, 92))
    frames.extend([title] * 10)
    for game, label in pairs:
        for policy, tag in (("random", "random"), ("model", "trained model")):
            rec = load(f"{game}_{policy}")
            for i, frame in enumerate(rec["frames"][:24]):
                frames.append(
                    grid_image(
                        frame["grid"],
                        f"{label} · {tag}",
                        f"seed {rec['seed']}  score {rec['score']}  frame {i + 1}/{min(24, len(rec['frames']))}  replay",
                    )
                )
    write_mp4(frames, OUT / "jev-games.mp4")
    frames[0].save(OUT / "poster.jpg", quality=90)
    print("wrote", OUT / "jev-games.mp4", "frames", len(frames))


if __name__ == "__main__":
    main()
