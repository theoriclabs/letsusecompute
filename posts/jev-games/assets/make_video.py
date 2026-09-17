"""Compose a 1088p ALE Pong social video from recorded RGB replays."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parent
REPLAYS = ROOT / "replays"
OUT = ROOT / "social"
SIZE = 1088
FPS = 24


def frame(rgb: np.ndarray, border: tuple[int, int, int]) -> Image.Image:
    play_h = 960
    play_w = int(160 / 210 * play_h)
    game = Image.fromarray(rgb).resize((play_w, play_h), Image.Resampling.NEAREST)
    canvas = Image.new("RGB", (SIZE, SIZE), border)
    canvas.paste(game, ((SIZE - play_w) // 2, (SIZE - play_h) // 2))
    return canvas


def write_mp4(frames: list[Image.Image], path: Path) -> None:
    import imageio.v2 as imageio

    path.parent.mkdir(parents=True, exist_ok=True)
    arrays = [np.asarray(f.convert("RGB")) for f in frames]
    imageio.mimsave(path, arrays, fps=FPS, codec="libx264", quality=5, pixelformat="yuv420p", macro_block_size=16)
    imageio.mimsave(path.with_suffix(".gif"), arrays[::6], fps=8, loop=0)
    Image.fromarray(arrays[len(arrays) // 2]).save(path.with_name("poster.jpg"), quality=92)


def load_mp4(name: str) -> list[np.ndarray]:
    import imageio.v2 as imageio

    return list(imageio.mimread(REPLAYS / f"{name}.mp4", memtest=False))


def main() -> None:
    frames: list[Image.Image] = []
    for name, border in (("random", (90, 40, 32)), ("model", (28, 72, 48))):
        for rgb in load_mp4(name):
            frames.append(frame(rgb, border))
    write_mp4(frames, OUT / "jev-games.mp4")
    print("wrote", OUT / "jev-games.mp4", "frames", len(frames))


if __name__ == "__main__":
    main()
