"""Render separate Doom / Pong videos: real Qwen probabilities, then fitted rules.

Requires saved Compute head + adapter checkpoints. No cloud jobs or training.
See ../README.md for reproduction commands and checkpoint provenance.
"""
from __future__ import annotations

import argparse
import ast
import gc
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
                 "SCENARIO", "SCREEN_W", "SCREEN_H", "FOV", "SKIP", "CONE", "SELF_NAMES", "ACTS", "MAX_LEN"}
    functions = {"_soft", "make_env", "objects", "parse_state", "reset_env", "threshold_action",
                 "teacher_action", "teacher_target", "wrap_deg", "make_game", "screen_rgb",
                 "_game_var", "record_episode", "state_obs", "pair_text", "encode_pairs",
                 "last_hidden", "score_rows", "model_prediction"}
    selected = ast.parse("from __future__ import annotations\nimport math, random, os\nfrom collections.abc import Mapping\n").body
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
    title = "PONG" if game == "pong" else "DOOM"
    probabilities = decision.get("probabilities")
    text(draw, (48, 28), title, 44)
    text(draw, (48, 82), "Qwen model  /  action probabilities" if probabilities is not None else "Fitted controller  /  rule-based actions", 28, MUTED)
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
            if probabilities is None:
                marker_y = y + (34 + min(159, max(0, target))) * scale
                marker_x = x + 140 * scale
                draw.line((marker_x - 18, marker_y, marker_x + 18, marker_y), fill=ACCENT, width=4)
            else:
                observation = f"Ball y {state['ball_y']:.1f}    Paddle y {state['paddle_y']:.1f}    {state['approach'].capitalize()}"
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
        draw.rounded_rectangle((left, 826, left + 320, 948), radius=12, fill=PANEL)
        draw.rounded_rectangle((left, 826, left + 320, 948), radius=12,
                               outline=ACCENT if selected else PANEL, width=3)
        bbox = draw.textbbox((0, 0), label, font=FONTS[32])
        text(draw, (left + (320 - (bbox[2] - bbox[0])) / 2, 837), label, 32, ACCENT if selected else INK)
        if probabilities is not None:
            value = probabilities[i]
            text(draw, (left + 18, 879), f"{value * 100:.1f}%", 32, ACCENT if selected else INK)
            draw.rounded_rectangle((left + 140, 893, left + 298, 907), radius=6, fill=BG)
            if value > 0:
                draw.rounded_rectangle((left + 140, 893, left + 140 + max(2, 158 * value), 907),
                                       radius=6, fill=ACCENT if selected else MUTED)
        elif selected:
            text(draw, (left + 82, 886), "CHOSEN", 28, ACCENT)
    text(draw, (48, 970), f"Chosen: {choices[idx]}" if probabilities is not None else reason, 34)
    text(draw, (48, 1026), "Softmax probabilities  ·  calibration not measured" if probabilities is not None
         else "Fitted rule  ·  no probability estimate", 28, MUTED)
    return canvas


BASE_ID = "Qwen/Qwen2.5-0.5B-Instruct"
BASE_REVISION = "7ae557604adf67be50417f59c2c2f167def9a775"
RUNS = {
    "doom": {"run_id": "run_577746f2c18b58372a8c404c2b8f2a1c", "artifact_id": "artifact_9160b4ca29eb1f71514b07432a961087", "version": 1},
    "pong": {"run_id": "run_84d48446e804cd4772e33b240d882643", "artifact_id": "artifact_838cf65caaa78db4a7ad5ea18e697ba4", "version": 1},
}


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_base(args):
    import torch
    from transformers import AutoModel, AutoTokenizer
    settings = dict(revision=BASE_REVISION, cache_dir=args.cache_dir, trust_remote_code=False)
    tokenizer = AutoTokenizer.from_pretrained(BASE_ID, **settings)
    tokenizer.padding_side = "right"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    base = AutoModel.from_pretrained(BASE_ID, torch_dtype=torch.float32, low_cpu_mem_usage=True, **settings)
    return base, tokenizer


def record(game, predictor=None):
    slug = "jev-games" if game == "pong" else "jev-doom"
    module = load_game(slug)
    params = {"t_lo": -6.0, "t_hi": 6.0, "lead": 1.6} if game == "pong" else {"cone": 8.0}
    chooser = predictor or (lambda state: module.threshold_action(state, params))
    args = dict(kind="qwen" if predictor else "fitted_rule", seed=77, max_steps=100, chooser=chooser)
    if game == "pong":
        args["stop_points"] = 5
    return module.record_episode(**args), params, sha256(POSTS / slug / "train.py")


def timestamp(seconds):
    milliseconds = round(seconds * 1000)
    return f"{milliseconds // 3600000:02d}:{milliseconds // 60000 % 60:02d}:{milliseconds // 1000 % 60:02d}.{milliseconds % 1000:03d}"


def write_video(game, recordings, provenance):
    path = OUT / f"{game}.mp4"
    clips, frame_count = [], 0
    with imageio.get_writer(path, fps=FPS, codec="libx264", quality=7, pixelformat="yuv420p", macro_block_size=16) as writer:
        for rec, params, digest in recordings:
            # Contiguous first 80 decisions; no search for a successful window.
            decisions = rec["decisions"][:80]
            if not decisions:
                raise RuntimeError(f"No decisions recorded for {game}")
            clip = {"game": game, "seed": 77, "policy": rec["policy"],
                    "parameters": params if rec["policy"] == "fitted_rule" else None,
                    "source_sha256": digest, "starts_at_seconds": frame_count / FPS, "decisions": []}
            for step in decisions:
                rgb = rec["frames"][step["frame"]]
                image = render(rgb, step, game, step["frame"])
                clip["decisions"].append({**step, "video_frame": frame_count,
                                          "at_seconds": frame_count / FPS, "duration_seconds": HOLD / FPS,
                                          "rgb_sha256": hashlib.sha256(rgb.tobytes()).hexdigest()})
                for _ in range(HOLD):
                    writer.append_data(np.asarray(image))
                if rec["policy"] == "qwen" and step["frame"] == decisions[len(decisions) // 2]["frame"]:
                    image.save(OUT / f"{game}-poster.jpg", quality=92)
                    if game == "doom":
                        image.save(OUT / "poster.jpg", quality=92)
                frame_count += HOLD
            clip["ends_at_seconds"] = frame_count / FPS
            clips.append(clip)
    payload = {"fps": FPS, "frames_per_decision": HOLD, "provenance": provenance,
               "description": "Fresh seed-77 replays. Qwen controls the first segment; the fitted rule controls the second. Qwen probabilities are uncalibrated softmax outputs, not win probabilities.",
               "clips": clips}
    (OUT / f"{game}-decisions.json").write_text(json.dumps(payload, indent=2) + "\n")
    (OUT / f"{game}.vtt").write_text("WEBVTT\n\n" + "\n\n".join(
        f"{timestamp(c['starts_at_seconds'])} --> {timestamp(c['ends_at_seconds'])}\n" +
        (f"{game.title()}: Qwen model probabilities. The highest-probability action controls the game. Calibration has not been measured."
         if c["policy"] == "qwen" else f"{game.title()}: fitted controller. The highlighted action controls the game. This rule has no probability estimate.")
        for c in clips) + "\n")
    print(f"Wrote {path}: {frame_count} frames, {frame_count / FPS:.1f}s", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoints", type=Path, required=True, help="Directory containing doom/ and pong/ checkpoint artifacts")
    parser.add_argument("--cache-dir", default=None)
    args = parser.parse_args()
    import torch
    from peft import PeftModel
    os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
    OUT.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(0)
    torch.set_num_threads(4)
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Inference device: {device}; float32; recorded playback", flush=True)
    base, tokenizer = load_base(args)
    backbone = None
    for game in ("doom", "pong"):
        directory = args.checkpoints / game
        summary = json.loads((directory / "summary.json").read_text())
        checkpoint = torch.load(directory / "head.pt", map_location="cpu", weights_only=True)
        module = load_game("jev-doom" if game == "doom" else "jev-games")
        if list(checkpoint["menu"]) != list(module.MENU):
            raise ValueError("Checkpoint action menu does not match the game")
        if backbone is None:
            backbone = PeftModel.from_pretrained(base, directory / "adapter", adapter_name=game).to(device)
        else:
            backbone.load_adapter(directory / "adapter", adapter_name=game)
        backbone.set_adapter(game)
        if summary["eval_stage"].startswith("head"):
            backbone.disable_adapter_layers()
        else:
            backbone.enable_adapter_layers()
        backbone.eval()
        head = torch.nn.Linear(checkpoint["hidden"], 1).to(device)
        head.load_state_dict(checkpoint["head"])
        head.eval()
        calls = 0
        def predictor(state):
            nonlocal calls
            with torch.inference_mode():
                prediction = module.model_prediction(state, backbone, head, tokenizer, device)
            calls += 1
            if calls % 20 == 0:
                print(f"{game}: {calls} model decisions recorded", flush=True)
            return prediction
        qwen = record(game, predictor)
        rule = record(game)
        provenance = {**RUNS[game], "model_id": BASE_ID, "base_revision": BASE_REVISION,
                      "eval_stage": summary["eval_stage"], "inference_device": str(device), "dtype": "float32",
                      "checkpoint_sha256": {p: sha256(directory / p) for p in ("head.pt", "adapter/adapter_model.safetensors", "adapter/adapter_config.json", "summary.json")}}
        write_video(game, (qwen, rule), provenance)
        del qwen, rule, head
        gc.collect()


if __name__ == "__main__":
    main()
