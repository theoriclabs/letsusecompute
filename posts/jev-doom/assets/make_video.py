"""Compose a square social video from recorded ViZDoom replays."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parent
REPLAYS = ROOT / "replays"
OUT = ROOT / "social"
SIZE = 1088
FPS = 20


def _font(size: int):
    for path in (
        "/System/Library/Fonts/Supplemental/Arial.ttf",
        "/System/Library/Fonts/Geneva.ttf",
        "/Library/Fonts/Arial.ttf",
    ):
        try:
            return ImageFont.truetype(path, size)
        except OSError:
            continue
    return ImageFont.load_default()


def frame(rgb: np.ndarray, border: tuple[int, int, int], title: str) -> Image.Image:
    play_h = 900
    play_w = int(320 / 240 * play_h)
    game = Image.fromarray(rgb).resize((play_w, play_h), Image.Resampling.NEAREST)
    canvas = Image.new("RGB", (SIZE, SIZE), border)
    canvas.paste(game, ((SIZE - play_w) // 2, 70))
    draw = ImageDraw.Draw(canvas)
    draw.text((32, 22), title, fill=(236, 236, 236), font=_font(28))
    return canvas


def write_mp4(frames: list[Image.Image], path: Path) -> None:
    import imageio.v2 as imageio

    path.parent.mkdir(parents=True, exist_ok=True)
    arrays = [np.asarray(f.convert("RGB")) for f in frames]
    imageio.mimsave(path, arrays, fps=FPS, codec="libx264", quality=6, pixelformat="yuv420p", macro_block_size=16)
    thumbs = [f.resize((544, 544), Image.Resampling.NEAREST) for f in frames[::8]]
    imageio.mimsave(path.with_suffix(".gif"), [np.asarray(f.convert("RGB")) for f in thumbs], fps=6, loop=0)
    Image.fromarray(arrays[len(arrays) // 3]).save(path.with_name("poster.jpg"), quality=88)


def load_mp4(name: str) -> list[np.ndarray]:
    import imageio.v2 as imageio

    return list(imageio.mimread(REPLAYS / f"{name}.mp4", memtest=False))


def main() -> None:
    frames: list[Image.Image] = []
    for name, border, title in (
        ("random", (90, 40, 32), "Random  ·  recorded replay"),
        ("model", (28, 72, 48), "Trained  ·  recorded replay"),
    ):
        for rgb in load_mp4(name):
            frames.append(frame(rgb, border, title))
    write_mp4(frames, OUT / "jev-doom.mp4")
    print("wrote", OUT / "jev-doom.mp4", "frames", len(frames))


if __name__ == "__main__":
    main()
