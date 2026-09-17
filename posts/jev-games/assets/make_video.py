"""Record Pong + Doom with the fitted controllers' actual per-frame decisions.

Local CPU rendering only; no cloud jobs or Qwen inference. Install
requirements-video.txt, then run this file from any directory.
"""
from __future__ import annotations

import ast
import hashlib
import json
import os
from pathlib import Path
from types import SimpleNamespace

import imageio.v2 as imageio
import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parent
POSTS = ROOT.parents[1]
OUT = ROOT / "social"
SIZE, FPS, HOLD = 1088, 20, 4  # Five recorded decisions per playback second.
BG, PANEL, INK, MUTED, ACCENT = "#101820", "#1c2a35", "#f4f6f8", "#b5c4cf", "#8ce1bc"
FONT_PATHS = (
    "/System/Library/Fonts/Supplemental/Arial.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
)


def font(size):
    for path in FONT_PATHS:
        if Path(path).exists():
            return ImageFont.truetype(path, size)
    raise RuntimeError("Install Arial or DejaVu Sans to render readable video labels")


FONTS = {size: font(size) for size in (28, 30, 32, 34, 44)}


def load_game(slug):
    """Load only local gameplay definitions, without executing Compute setup.

    AST selection keeps the recorder on the exact train.py gameplay code while
    avoiding its cloud decorators, secrets and heavyweight training imports.
    """
    source = POSTS / slug / "train.py"
    tree = ast.parse(source.read_text())
    constants = {"GAME", "OBJECTIVE", "MENU", "MENU_IDS", "ENV_ID", "PLAY_TOP",
                 "PLAY_BOTTOM", "PADDLE_X_MIN", "CPU_X_MAX", "DEADZONE", "FAR",
                 "SCENARIO", "SCREEN_W", "SCREEN_H", "FOV", "SKIP", "CONE", "SELF_NAMES", "ACTS"}
    functions = {"_soft", "make_env", "objects", "parse_state", "reset_env", "threshold_action",
                 "teacher_action", "teacher_target", "wrap_deg", "make_game", "screen_rgb",
                 "_game_var", "record_episode"}
    selected = ast.parse("from __future__ import annotations\nimport math, random, os\n").body
    for node in tree.body:
        if isinstance(node, ast.Assign) and all(isinstance(t, ast.Name) and t.id in constants for t in node.targets):
            selected.append(node)
        elif isinstance(node, ast.FunctionDef) and node.name in functions:
            selected.append(node)
    namespace = {}
    exec(compile(ast.Module(body=selected, type_ignores=[]), str(source), "exec"), namespace)
    return SimpleNamespace(**namespace)


def text(draw, xy, value, size=30, fill=INK):
    draw.text(xy, value, font=FONTS[size], fill=fill)


def render(rgb, decision, game, sequence):
    canvas = Image.new("RGB", (SIZE, SIZE), BG)
    draw = ImageDraw.Draw(canvas)
    title = "01 / PONG" if game == "pong" else "02 / DOOM"
    text(draw, (48, 28), title, 44)
    text(draw, (48, 82), "Fitted controller  /  recorded decisions", 28, MUTED)
    text(draw, (828, 39), f"STEP {sequence:03d}", 28, ACCENT)
    image = Image.fromarray(rgb)
    scale = min(992 / image.width, 612 / image.height)
    image = image.resize((round(image.width * scale), round(image.height * scale)), Image.Resampling.NEAREST)
    x, y = (SIZE - image.width) // 2, 132 + (612 - image.height) // 2
    draw.rounded_rectangle((48, 126, 1040, 752), radius=12, fill="#080d12")
    canvas.paste(image, (x, y))
    state = decision["state"]
    if game == "pong":
        idx = (1, 2, 3).index(decision["action"])
        choices = ("HOLD / SERVE", "MOVE UP", "MOVE DOWN")
        if state["in_play"]:
            lead = 1.6 * state["dy"] if state["approach"] == "toward" else 0.0
            target = state["ball_y"] + lead
            error = target - state["paddle_y"]
            observation = f"Target y {target:.1f}    Paddle y {state['paddle_y']:.1f}    Gap {error:+.1f} px"
            reason = "Gap < -6 px: move up" if idx == 1 else "Gap > +6 px: move down" if idx == 2 else "Within 6 px: hold position"
            # Mark the rule's estimated target on the paddle side of the frame.
            marker_y = y + (34 + min(159, max(0, target))) * scale
            marker_x = x + 140 * scale
            draw.line((marker_x - 18, marker_y, marker_x + 18, marker_y), fill=ACCENT, width=4)
        else:
            observation, reason = "Ball is hidden    Paddle is ready", "No visible ball: serve / hold"
    else:
        idx = decision["action"]
        choices = ("TURN LEFT", "TURN RIGHT", "FIRE")
        bearing = state["bearing"]
        direction = "left" if bearing > 0 else "right"
        observation = f"Target {abs(bearing):.1f}° {direction}    Health {state['health']:.0f}    Kills {state['kills']:.0f}"
        reason = "No target: scan left" if state["n"] <= 0 else "Within 8° of aim: fire" if idx == 2 else f"Outside 8°: turn {direction}"
    text(draw, (48, 777), observation, 30)
    for i, label in enumerate(choices):
        left = 48 + i * 336
        selected = i == idx
        draw.rounded_rectangle((left, 833, left + 320, 919), radius=12, fill=ACCENT if selected else PANEL)
        bbox = draw.textbbox((0, 0), label, font=FONTS[32])
        text(draw, (left + (320 - (bbox[2] - bbox[0])) / 2, 858), label, 32, BG if selected else MUTED)
    text(draw, (48, 950), reason, 34)
    text(draw, (48, 1015), "Recorded actions  ·  playback: 5 decisions / second", 28, MUTED)
    return canvas


def record(game):
    slug = "jev-games" if game == "pong" else "jev-doom"
    module = load_game(slug)
    params = {"t_lo": -6.0, "t_hi": 6.0, "lead": 1.6} if game == "pong" else {"cone": 8.0}
    chooser = lambda state: module.threshold_action(state, params)
    args = dict(kind="fitted_rule", seed=77, max_steps=160, chooser=chooser)
    if game == "pong":
        args["stop_points"] = 5
    rec = module.record_episode(**args)
    return rec, params, hashlib.sha256((POSTS / slug / "train.py").read_bytes()).hexdigest()


def main():
    os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / "jev-games.mp4"
    gif_frames, clips = [], []
    frame_count = 0
    with imageio.get_writer(path, fps=FPS, codec="libx264", quality=7, pixelformat="yuv420p", macro_block_size=16) as writer:
        for game in ("pong", "doom"):
            rec, params, digest = record(game)
            # Fixed, contiguous window, selected without searching for a success.
            decisions = rec["decisions"][20:100]
            if not decisions:
                raise RuntimeError(f"No decisions recorded for {game}")
            # A representative firing frame makes the poster's action easy to read.
            firing = [d for d in decisions if d["action"] == 2] if game == "doom" else []
            poster_step = min(firing, key=lambda d: abs(d["frame"] - 60))["frame"] if firing else None
            clip = {"game": game, "seed": 77, "policy": "fitted_rule", "parameters": params,
                    "source_sha256": digest, "starts_at_seconds": frame_count / FPS, "decisions": []}
            for step in decisions:
                rgb = rec["frames"][step["frame"]]
                image = render(rgb, step, game, step["frame"])
                clip["decisions"].append({**step, "video_frame": frame_count,
                                          "at_seconds": frame_count / FPS, "duration_seconds": HOLD / FPS})
                for _ in range(HOLD):
                    writer.append_data(np.asarray(image))
                gif_frames.append(image.resize((544, 544), Image.Resampling.LANCZOS))
                if step["frame"] == poster_step:
                    image.save(OUT / "poster.jpg", quality=92)
                frame_count += HOLD
            clip["ends_at_seconds"] = frame_count / FPS
            clips.append(clip)
    imageio.mimsave(path.with_suffix(".gif"), [np.asarray(f) for f in gif_frames], duration=HOLD / FPS * 1000, loop=0)
    (OUT / "decisions.json").write_text(json.dumps({"fps": FPS, "frames_per_decision": HOLD,
        "description": "Actual pre-action state and action; fitted rules, no neural probabilities. Contiguous decisions 20–99 from fresh seed-77 recordings.",
        "clips": clips}, indent=2) + "\n")
    def timestamp(seconds):
        milliseconds = round(seconds * 1000)
        return f"{milliseconds // 3600000:02d}:{milliseconds // 60000 % 60:02d}:{milliseconds // 1000 % 60:02d}.{milliseconds % 1000:03d}"
    captions = {
        "pong": "Pong: the fitted controller predicts a target height and selects hold, up or down. The highlighted button is its recorded choice.",
        "doom": "Doom: the fitted controller reads the target bearing, turns toward it, and fires within eight degrees. These are actions, not confidence scores.",
    }
    (OUT / "captions.vtt").write_text("WEBVTT\n\n" + "\n\n".join(
        f"{timestamp(c['starts_at_seconds'])} --> {timestamp(c['ends_at_seconds'])}\n{captions[c['game']]}"
        for c in clips) + "\n")
    print(f"Wrote {path}: {frame_count} frames, {frame_count / FPS:.1f}s")


if __name__ == "__main__":
    main()
