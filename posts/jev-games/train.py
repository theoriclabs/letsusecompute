"""Train a Jev-inspired decision model on real Atari Pong.

The policy sees structured paddle/ball state extracted from ALE pixels, not
the RGB tensor. Qwen2.5-0.5B-Instruct plus one scalar candidate-scoring head:
Stage A freezes the backbone, Stage B adds LoRA rank 16, then one DAgger
relabel round. Supervised cloning, not RLCD.

    compute run train.py::train --gpu cheap --dry-run
    compute run train.py::train --gpu cheap --timeout 5400 --wait
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import random
import tempfile
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import compute

app = compute.App("jev-games")
image = compute.Image.cuda_pytorch().pip_install(
    "--extra-index-url",
    "https://download.pytorch.org/whl/cu124",
    "torch==2.5.1+cu124",
    "numpy",
    "transformers==4.46.3",
    "peft==0.13.2",
    "accelerate==1.2.1",
    "huggingface_hub>=0.33,<1",
    "pillow",
    "gymnasium[atari]==1.1.1",
    "ale-py==0.11.2",
    "opencv-python-headless",
)
hf_secret = compute.Secret.from_name("hf")

DEFAULT_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
DEFAULT_HUB_REPO = "theoriclabs/jev-games-0.5b"
WORKLOAD_SUBDIR = "jev-games"
ARTIFACT_NAME = "jev-games-decision"
ARTIFACT_MARKER = ".compute-artifact.json"
DEFAULT_ARTIFACT_FALLBACK = Path("/tmp/compute-jev-games")
MAX_LEN = 256

GAME = "pong"
OBJECTIVE = "score against the CPU paddle on Atari Pong"
# ALE/Pong-v5 meanings: NOOP, FIRE, RIGHT, LEFT, RIGHTFIRE, LEFTFIRE.
# RIGHT moves the right paddle up; LEFT moves it down.
MENU = ("serve or hold", "move paddle up", "move paddle down")
MENU_IDS = (1, 2, 3)  # FIRE, RIGHT, LEFT
ENV_ID = "ALE/Pong-v5"
PLAY_TOP = 34
PLAY_BOTTOM = 194
PADDLE_X_MIN = 136
CPU_X_MAX = 24
DEADZONE = 6
FAR = 18


def _bridge_hf_token() -> None:
    token = (
        os.environ.get("HF_TOKEN")
        or os.environ.get("HUGGING_FACE_HUB_TOKEN")
        or os.environ.get("hf")
    )
    if token:
        os.environ.setdefault("HF_TOKEN", token)
        os.environ.setdefault("HUGGING_FACE_HUB_TOKEN", token)


def _py(value: Any) -> Any:
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            pass
    if isinstance(value, float):
        return float(value)
    if isinstance(value, int):
        return int(value)
    return value


def write_artifact_marker(
    directory: Path | str,
    *,
    name: str,
    kind: str,
    compatibility_key: str,
    metadata: dict[str, Any],
) -> Path:
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    marker_path = directory / ARTIFACT_MARKER
    payload = {
        "compatibility_key": compatibility_key,
        "kind": kind,
        "metadata": metadata,
        "name": name,
        "version": 1,
    }
    data = json.dumps(payload, sort_keys=True, indent=2, ensure_ascii=False) + "\n"
    fd, tmp_name = tempfile.mkstemp(prefix=".compute-artifact.", suffix=".tmp", dir=str(directory))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp_name, marker_path)
        dir_fd = os.open(str(directory), os.O_RDONLY)
        try:
            os.fsync(dir_fd)
        finally:
            os.close(dir_fd)
    except Exception:
        try:
            os.unlink(tmp_name)
        except OSError:
            pass
        raise
    for leftover in directory.glob(".compute-artifact.*.tmp"):
        try:
            leftover.unlink()
        except OSError:
            pass
    return marker_path


def resolve_artifact_dirs(env: Mapping[str, str] | None = None) -> Path:
    environ = os.environ if env is None else env
    base = environ.get("COMPUTE_ARTIFACT_DIR")
    root = Path(base) / WORKLOAD_SUBDIR if base else DEFAULT_ARTIFACT_FALLBACK
    root.mkdir(parents=True, exist_ok=True)
    return root


def _soft(ids: list[int], n: int) -> list[float]:
    mass = [0.0] * n
    if not ids:
        return [1.0 / n] * n
    w = 1.0 / len(ids)
    for i in ids:
        mass[i] += w
    return mass


def make_env():
    import ale_py
    import gymnasium as gym

    gym.register_envs(ale_py)
    return gym.make(
        ENV_ID,
        obs_type="rgb",
        render_mode="rgb_array",
        frameskip=4,
        repeat_action_probability=0.0,
        full_action_space=False,
    )


def objects(rgb) -> dict[str, dict[str, float] | None]:
    import numpy as np

    play = rgb[PLAY_TOP:PLAY_BOTTOM]
    h, w = play.shape[:2]
    xs = np.arange(w)[None, :]
    white = (play[:, :, 0] > 200) & (play[:, :, 1] > 200) & (play[:, :, 2] > 200)
    green = (play[:, :, 1] > 150) & (play[:, :, 0] < 140) & (play[:, :, 2] < 140)
    tan = (play[:, :, 0] > 180) & (play[:, :, 1] > 100) & (play[:, :, 1] < 170) & (play[:, :, 2] < 120)

    def bbox(mask):
        yy, xx = np.where(mask)
        if len(xx) == 0:
            return None
        return {
            "x": float(xx.mean()),
            "y": float(yy.mean()),
            "ymin": float(yy.min()),
            "ymax": float(yy.max()),
            "n": float(len(xx)),
        }

    return {
        "ball": bbox(white & (xs > 8) & (xs < 150)),
        "player": bbox(green & (xs >= PADDLE_X_MIN)),
        "cpu": bbox(tan & (xs <= CPU_X_MAX)),
    }


def parse_state(rgb, prev: tuple[float, float] | None) -> dict[str, Any]:
    obj = objects(rgb)
    ball, player = obj["ball"], obj["player"]
    in_play = ball is not None and player is not None
    if not in_play:
        return {
            "in_play": False,
            "delta": 0.0,
            "dx": 0.0,
            "dy": 0.0,
            "rel": "none",
            "approach": "none",
            "ball_x": 0.0,
            "ball_y": 0.0,
            "paddle_y": 0.0 if player is None else player["y"],
            "obj": obj,
        }
    dx = 0.0 if prev is None else ball["x"] - prev[0]
    dy = 0.0 if prev is None else ball["y"] - prev[1]
    delta = ball["y"] - player["y"]
    if abs(delta) <= DEADZONE:
        rel = "aligned"
    elif delta < -FAR:
        rel = "above_far"
    elif delta < 0:
        rel = "above"
    elif delta > FAR:
        rel = "below_far"
    else:
        rel = "below"
    if dx > 1:
        approach = "toward"
    elif dx < -1:
        approach = "away"
    else:
        approach = "none"
    return {
        "in_play": True,
        "delta": float(delta),
        "dx": float(dx),
        "dy": float(dy),
        "rel": rel,
        "approach": approach,
        "ball_x": float(ball["x"]),
        "ball_y": float(ball["y"]),
        "paddle_y": float(player["y"]),
        "obj": obj,
    }


def state_feat(state: dict[str, Any]) -> list[float]:
    return [
        float(state["delta"]) / 80.0,
        float(state["dx"]) / 12.0,
        float(state["dy"]) / 12.0,
        1.0 if state["in_play"] else 0.0,
    ]


def state_obs(state: dict[str, Any]) -> str:
    if not state["in_play"]:
        return "in_play=no ball=hidden paddle_ready=yes serve=yes"
    return (
        f"in_play=yes ball_rel={state['rel']} delta={state['delta']:.0f} "
        f"approach={state['approach']} dx={state['dx']:.0f} dy={state['dy']:.0f} "
        f"ball=({state['ball_x']:.0f},{state['ball_y']:.0f}) paddle_y={state['paddle_y']:.0f}"
    )


def teacher_target(state: dict[str, Any]) -> list[float]:
    if not state["in_play"]:
        return _soft([0], 3)
    target_y = state["ball_y"] + (1.6 * state["dy"] if state["approach"] == "toward" else 0.0)
    err = target_y - state["paddle_y"]
    if err < -DEADZONE:
        return _soft([1], 3)
    if err > DEADZONE:
        return _soft([2], 3)
    return _soft([0], 3)


def teacher_action(state: dict[str, Any]) -> int:
    target = teacher_target(state)
    return MENU_IDS[max(range(3), key=lambda i: target[i])]


def reset_env(env, seed: int):
    rgb, _info = env.reset(seed=int(seed) % (2**31))
    rng = random.Random(seed)
    for _ in range(rng.randint(0, 8)):
        rgb, *_ = env.step(0)
    for _ in range(24):
        if objects(rgb)["ball"] is not None:
            break
        rgb, *_ = env.step(1)
    extra = 2 + (seed % 9)
    for _ in range(extra):
        rgb, *_ = env.step(rng.choice(MENU_IDS))
    return rgb


def pair_text(row: dict[str, Any], cand: str) -> str:
    return (
        f"Game: {row['game']}\nObjective: {row['objective']}\n"
        f"Observation: {row['obs']}\nCandidate action: {cand}"
    )


def collect_states(
    n_states: int,
    seed0: int,
    perturb: float,
    policy=None,
) -> list[dict[str, Any]]:
    env = make_env()
    rows: list[dict[str, Any]] = []
    seed = seed0
    max_eps = max(40, n_states // 8)
    try:
        while len(rows) < n_states and seed < seed0 + max_eps:
            rng = random.Random(seed)
            rgb = reset_env(env, seed)
            prev = None
            for _ in range(80):
                if len(rows) >= n_states:
                    break
                state = parse_state(rgb, prev)
                target = teacher_target(state)
                is_hold = target[0] >= 0.99
                if is_hold and rng.random() > 0.35:
                    action = teacher_action(state) if policy is None else policy(state, rng)
                    prev = (state["ball_x"], state["ball_y"]) if state["in_play"] else None
                    rgb, _r, term, trunc, _ = env.step(action)
                    if term or trunc:
                        break
                    continue
                order = list(range(3))
                rng.shuffle(order)
                rows.append(
                    {
                        "game": GAME,
                        "objective": OBJECTIVE,
                        "obs": state_obs(state),
                        "menu": [MENU[i] for i in order],
                        "target": [target[i] for i in order],
                        "order": order,
                        "episode_seed": seed,
                        "fingerprint": hashlib.sha256(state_obs(state).encode()).hexdigest()[:16],
                        "feat": state_feat(state),
                    }
                )
                if policy is None:
                    action = rng.choice(MENU_IDS) if rng.random() < perturb else teacher_action(state)
                else:
                    action = policy(state, rng)
                prev = (state["ball_x"], state["ball_y"]) if state["in_play"] else None
                rgb, _r, term, trunc, _ = env.step(action)
                if term or trunc:
                    break
            seed += 1
            if (seed - seed0) % 10 == 0:
                print(f"collect episodes={seed - seed0} states={len(rows)}", flush=True)
    finally:
        env.close()
    return rows


def split_rows(rows: list[dict[str, Any]], seed: int) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[int, list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(row["episode_seed"], []).append(row)
    seeds = sorted(grouped)
    rng = random.Random(seed)
    rng.shuffle(seeds)
    n = len(seeds)
    if n < 4:
        parts = {
            "train": seeds[: max(1, n - 2)] or seeds,
            "val": seeds[-2:-1] or seeds[:1],
            "cal": seeds[-2:-1] or seeds[:1],
            "test": seeds[-1:] or seeds[:1],
        }
    else:
        n_train = max(1, int(0.7 * n))
        n_val = max(1, int(0.15 * n))
        n_cal = max(1, int(0.05 * n))
        parts = {
            "train": seeds[:n_train],
            "val": seeds[n_train : n_train + n_val],
            "cal": seeds[n_train + n_val : n_train + n_val + n_cal],
            "test": seeds[n_train + n_val + n_cal :],
        }
        if not parts["test"]:
            parts["test"] = parts["val"][-1:]
        if not parts["val"]:
            parts["val"] = parts["train"][-1:]
        if not parts["cal"]:
            parts["cal"] = parts["val"][:1]
    out = {name: [] for name in parts}
    for name, chosen in parts.items():
        for s in chosen:
            for row in grouped[s]:
                row = dict(row)
                row["split"] = name
                out[name].append(row)
    return out


def encode_pairs(tokenizer, texts: list[str], device):
    enc = tokenizer(
        texts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=MAX_LEN,
    )
    lengths = [len(tokenizer(t, add_special_tokens=True)["input_ids"]) for t in texts]
    if any(n > MAX_LEN for n in lengths):
        raise RuntimeError(f"input truncated: max raw tokens {max(lengths)} > {MAX_LEN}")
    return {k: v.to(device) for k, v in enc.items()}


def last_hidden(backbone, batch):
    out = backbone(**batch, output_hidden_states=True)
    hidden = out.last_hidden_state
    idx = batch["attention_mask"].sum(dim=1) - 1
    return hidden[range(hidden.size(0)), idx]


def score_rows(backbone, head, tokenizer, rows: list[dict[str, Any]], device):
    import torch.nn.functional as F

    texts = []
    groups = []
    for row in rows:
        groups.append(len(row["menu"]))
        texts.extend(pair_text(row, cand) for cand in row["menu"])
    if not texts:
        raise RuntimeError("score_rows called with no candidate texts")
    batch = encode_pairs(tokenizer, texts, device)
    pooled = last_hidden(backbone, batch)
    scores = head(pooled).squeeze(-1)
    outs = []
    cursor = 0
    for n in groups:
        chunk = scores[cursor : cursor + n]
        outs.append(F.softmax(chunk, dim=0))
        cursor += n
    return outs


def train_epoch(backbone, head, tokenizer, rows, device, opt, batch_size: int) -> float:
    import torch
    import torch.nn.functional as F

    order = list(range(len(rows)))
    random.shuffle(order)
    total = 0.0
    n = 0
    backbone.train()
    head.train()
    for i in range(0, len(order), batch_size):
        batch_rows = [rows[j] for j in order[i : i + batch_size]]
        probs = score_rows(backbone, head, tokenizer, batch_rows, device)
        loss = 0.0
        for row, p in zip(batch_rows, probs):
            target = torch.tensor(row["target"], device=device, dtype=p.dtype)
            move_w = 1.0 + 3.0 * float(sum(t for t, name in zip(row["target"], row["menu"]) if name != MENU[0]))
            loss = loss + move_w * F.kl_div(p.clamp_min(1e-8).log(), target, reduction="sum")
        loss = loss / max(len(batch_rows), 1)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            list(head.parameters()) + [p for p in backbone.parameters() if p.requires_grad],
            1.0,
        )
        opt.step()
        total += float(loss.detach())
        n += 1
    return total / max(n, 1)


def agreement(backbone, head, tokenizer, rows, device) -> float:
    if not rows:
        return 0.0
    backbone.eval()
    head.eval()
    hits = 0.0
    import torch

    with torch.no_grad():
        for i in range(0, len(rows), 8):
            chunk = rows[i : i + 8]
            probs = score_rows(backbone, head, tokenizer, chunk, device)
            for row, p in zip(chunk, probs):
                pred = int(p.argmax())
                hits += float(row["target"][pred] > 0)
    return hits / len(rows)


def _row_label(row: dict[str, Any]) -> int:
    target = [0.0, 0.0, 0.0]
    for p, idx in zip(row["target"], row["order"]):
        target[idx] += float(p)
    return max(range(3), key=lambda i: target[i])


def fit_thresholds(rows: list[dict[str, Any]]) -> dict[str, float]:
    usable = [r for r in rows if r.get("feat") and r.get("order")]
    best = {"t_lo": -float(DEADZONE), "t_hi": float(DEADZONE), "lead": 1.6, "agree": -1.0}
    for lead in (0.0, 1.0, 1.6, 2.2):
        for dead in (4, 6, 8):
            hits = 0
            n = 0
            for row in usable:
                feat = row["feat"]
                in_play = feat[3] > 0.5
                delta = feat[0] * 80.0
                dy = feat[2] * 12.0
                err = delta + (lead * dy if feat[1] * 12.0 > 1 else 0.0)
                if not in_play:
                    pred = 0
                elif err < -dead:
                    pred = 1
                elif err > dead:
                    pred = 2
                else:
                    pred = 0
                hits += int(pred == _row_label(row))
                n += 1
            agree = hits / max(n, 1)
            if agree > best["agree"]:
                best = {"t_lo": -float(dead), "t_hi": float(dead), "lead": float(lead), "agree": agree}
    print(
        f"learned intercept lead={best['lead']} dead={best['t_hi']} agree={best['agree']:.3f}",
        flush=True,
    )
    return best


def threshold_action(state: dict[str, Any], thr: dict[str, float]) -> int:
    if not state["in_play"]:
        return MENU_IDS[0]
    lead = float(thr.get("lead", 1.6))
    err = state["delta"] + (lead * state["dy"] if state["approach"] == "toward" else 0.0)
    if err < thr["t_lo"]:
        return MENU_IDS[1]
    if err > thr["t_hi"]:
        return MENU_IDS[2]
    return MENU_IDS[0]


def fit_linear(rows: list[dict[str, Any]], device, steps: int = 400):
    import torch
    from torch import nn

    net = nn.Linear(4, 3).to(device)
    opt = torch.optim.Adam(net.parameters(), lr=0.08)
    usable = [r for r in rows if r.get("feat") and r.get("order")]
    if not usable:
        raise RuntimeError("no feature rows for linear readout")
    for _ in range(steps):
        batch = random.sample(usable, min(64, len(usable)))
        x = torch.tensor([r["feat"] for r in batch], device=device, dtype=torch.float32)
        y = []
        for row in batch:
            t = [0.0, 0.0, 0.0]
            for p, idx in zip(row["target"], row["order"]):
                t[idx] += float(p)
            y.append(t)
        y = torch.tensor(y, device=device, dtype=torch.float32)
        loss = torch.nn.functional.kl_div(net(x).log_softmax(dim=-1), y, reduction="batchmean")
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
    net.eval()
    return net


def linear_action(state, net, device) -> int:
    import torch

    with torch.no_grad():
        x = torch.tensor([state_feat(state)], device=device, dtype=torch.float32)
        return MENU_IDS[int(net(x).argmax(dim=-1).item())]


def model_action(state, backbone, head, tokenizer, device) -> int:
    import torch

    row = {
        "game": GAME,
        "objective": OBJECTIVE,
        "obs": state_obs(state),
        "menu": list(MENU),
        "target": [0.0, 0.0, 0.0],
    }
    with torch.no_grad():
        p = score_rows(backbone, head, tokenizer, [row], device)[0]
        return MENU_IDS[int(p.argmax())]


def closed_loop(kind: str, n_ep: int, seed0: int, stop_points: int, max_steps: int, chooser) -> dict[str, Any]:
    env = make_env()
    scores = []
    lengths = []
    wins = 0
    acts = {aid: 0 for aid in MENU_IDS}
    try:
        for i in range(n_ep):
            rgb = reset_env(env, seed0 + 17 * i)
            prev = None
            you = opp = 0
            ret = 0.0
            steps = 0
            for _ in range(max_steps):
                state = parse_state(rgb, prev)
                action = chooser(state)
                acts[action] = acts.get(action, 0) + 1
                prev = (state["ball_x"], state["ball_y"]) if state["in_play"] else None
                rgb, reward, term, trunc, _ = env.step(action)
                ret += float(reward)
                if reward > 0:
                    you += 1
                elif reward < 0:
                    opp += 1
                steps += 1
                if term or trunc or you >= stop_points or opp >= stop_points:
                    break
            scores.append(ret)
            lengths.append(steps)
            wins += int(you > opp)
            print(f"loop {kind} ep={i} {you}-{opp} ret={ret} steps={steps}", flush=True)
    finally:
        env.close()
    total_a = sum(acts.values()) or 1
    return {
        "episodes": n_ep,
        "mean_score": sum(scores) / max(n_ep, 1),
        "mean_len": sum(lengths) / max(n_ep, 1),
        "win_rate": wins / max(n_ep, 1),
        "scores": scores,
        "stop_points": stop_points,
        "action_hist": {str(k): v / total_a for k, v in acts.items()},
    }


def scale_frame(rgb, scale: int = 5):
    import numpy as np

    return np.repeat(np.repeat(rgb, scale, axis=0), scale, axis=1)


def write_still(rgb, path: Path, caption: str) -> None:
    from PIL import Image, ImageDraw

    img = Image.fromarray(scale_frame(rgb, 4))
    draw = ImageDraw.Draw(img)
    draw.rectangle([0, 0, img.width, 28], fill=(20, 12, 8))
    draw.text((8, 6), caption[:64], fill=(236, 236, 236))
    path.parent.mkdir(parents=True, exist_ok=True)
    img.save(path)


def write_mp4(frames, path: Path, fps: int = 30) -> Path | None:
    if not frames:
        return None
    import numpy as np

    path.parent.mkdir(parents=True, exist_ok=True)
    scaled = [scale_frame(f, 5) for f in frames]
    h, w = scaled[0].shape[:2]
    try:
        import cv2

        writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
        if writer.isOpened():
            for frame in scaled:
                writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
            writer.release()
            if path.exists() and path.stat().st_size > 0:
                return path
    except Exception as err:  # noqa: BLE001
        print(f"cv2 mp4 failed: {err}", flush=True)
    gif = path.with_suffix(".gif")
    from PIL import Image

    images = [Image.fromarray(f) for f in scaled[::2]]
    images[0].save(gif, save_all=True, append_images=images[1:], duration=int(1000 / max(fps // 2, 8)), loop=0)
    return gif


def record_episode(kind: str, seed: int, stop_points: int, max_steps: int, chooser) -> dict[str, Any]:
    env = make_env()
    try:
        rgb = reset_env(env, seed)
        prev = None
        frames = []
        you = opp = 0
        ret = 0.0
        for _ in range(max_steps):
            frames.append(rgb.copy())
            state = parse_state(rgb, prev)
            action = chooser(state)
            prev = (state["ball_x"], state["ball_y"]) if state["in_play"] else None
            rgb, reward, term, trunc, _ = env.step(action)
            ret += float(reward)
            if reward > 0:
                you += 1
            elif reward < 0:
                opp += 1
            if term or trunc or you >= stop_points or opp >= stop_points:
                frames.append(rgb.copy())
                break
        return {
            "policy": kind,
            "seed": seed,
            "score": ret,
            "you": you,
            "opp": opp,
            "steps": len(frames),
            "frames": frames,
            "live": False,
            "observation_mode": "structured-from-ale-rgb",
        }
    finally:
        env.close()


def write_curves(directory: Path, history: list[dict[str, float]]) -> Path:
    width, height, pad = 1200, 420, 72
    path = directory / "training_curves.svg"
    if len(history) < 2:
        path.write_text(f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}"></svg>\n')
        return path
    xs = list(range(len(history)))
    ys = [row["loss"] for row in history]
    min_y, max_y = min(ys), max(ys)
    span_y = max(max_y - min_y, 1e-6)
    inner_w, inner_h = width - 2 * pad, height - 2 * pad
    pts = " ".join(
        f"{pad + inner_w * i / max(len(xs) - 1, 1):.1f},{pad + inner_h * (1 - (y - min_y) / span_y):.1f}"
        for i, y in zip(xs, ys)
    )
    path.write_text(
        f"""<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">
  <rect width="100%" height="100%" fill="#f7f4ef"/>
  <text x="48" y="48" font-size="26" font-weight="700" fill="#1a1a1a" font-family="system-ui, sans-serif">Decision-head training loss</text>
  <polyline fill="none" stroke="#3d6b8a" stroke-width="3" points="{pts}"/>
</svg>
""",
        encoding="utf-8",
    )
    return path


def write_score_bars(directory: Path, loop: dict[str, Any]) -> Path:
    path = directory / "score_bars.svg"
    rows = [("Random", loop["random"]["mean_score"]), ("Trained model", loop["model"]["mean_score"]), ("Teacher", loop["teacher"]["mean_score"])]
    width, height = 1200, 420
    lo = min(-21.0, min(v for _, v in rows) - 1)
    hi = max(21.0, max(v for _, v in rows) + 1)
    span = hi - lo
    bars = []
    colors = ["#8a6a4a", "#3d6b8a", "#3d6b4f"]
    for i, ((name, val), color) in enumerate(zip(rows, colors)):
        y = 90 + i * 90
        x0 = 280
        mid = x0 + (0 - lo) / span * 820
        x1 = x0 + (val - lo) / span * 820
        left, right = (x1, mid) if val < 0 else (mid, x1)
        bars.append(
            f'<text x="48" y="{y + 28}" font-size="22" fill="#1a1a1a" font-family="system-ui, sans-serif">{name}</text>'
            f'<rect x="{left:.1f}" y="{y}" width="{max(right - left, 2):.1f}" height="44" fill="{color}"/>'
            f'<text x="{right + 12:.1f}" y="{y + 30}" font-size="22" font-weight="700" fill="#1a1a1a" font-family="system-ui, sans-serif">{val:.2f}</text>'
        )
    path.write_text(
        f"""<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">
  <rect width="100%" height="100%" fill="#f7f4ef"/>
  <text x="48" y="48" font-size="26" font-weight="700" fill="#1a1a1a" font-family="system-ui, sans-serif">Atari Pong return, first to 5</text>
  {''.join(bars)}
</svg>
""",
        encoding="utf-8",
    )
    return path


def run_unit_tests() -> dict[str, bool]:
    env = make_env()
    rgb, _ = env.reset(seed=0)
    meanings = env.unwrapped.get_action_meanings()
    assert meanings[:4] == ["NOOP", "FIRE", "RIGHT", "LEFT"], meanings
    for _ in range(20):
        rgb, *_ = env.step(1)
    obj = objects(rgb)
    assert obj["player"] is not None, "player paddle not found"
    before = obj["player"]["y"]
    for _ in range(6):
        rgb, *_ = env.step(2)
    up = objects(rgb)["player"]["y"]
    for _ in range(12):
        rgb, *_ = env.step(3)
    down = objects(rgb)["player"]["y"]
    env.close()
    assert up < before, (before, up, down)
    assert down > up, (up, down)
    env = make_env()
    rgb = reset_env(env, 3)
    prev = None
    you = opp = 0
    for _ in range(1600):
        state = parse_state(rgb, prev)
        prev = (state["ball_x"], state["ball_y"]) if state["in_play"] else None
        rgb, reward, term, trunc, _ = env.step(teacher_action(state))
        if reward > 0:
            you += 1
        elif reward < 0:
            opp += 1
        if term or trunc or you >= 3 or opp >= 3:
            break
    env.close()
    assert you > opp, (you, opp)
    hidden = teacher_target({"in_play": False, "delta": 0, "dy": 0, "approach": "none", "ball_y": 0, "paddle_y": 0})
    assert hidden[0] == 1.0
    return {"rom": True, "paddle_polarity": True, "teacher_leads": True}


def _train_impl(
    *,
    sample: bool,
    states_per_game: int,
    episodes: int,
    batch_size: int,
    epochs_head: int,
    epochs_lora: int,
    dagger_states: int,
    dagger_epochs: int,
    lora_rank: int,
    seed: int,
    model_id: str,
    push_to_hub: bool,
    hub_repo: str,
    stop_points: int,
    max_steps: int,
) -> dict:
    import time

    import torch
    from peft import LoraConfig, get_peft_model
    from torch import nn
    from transformers import AutoModel, AutoTokenizer

    tests = run_unit_tests()
    if sample:
        states_per_game = states_per_game or 80
        episodes = episodes or 3
        batch_size = min(batch_size, 2)
        epochs_head = epochs_head or 2
        epochs_lora = epochs_lora or 1
        dagger_states = 0 if dagger_states < 0 else min(dagger_states, 40)
        dagger_epochs = 0 if dagger_states == 0 else max(dagger_epochs, 1)
        stop_points = min(stop_points, 3)
        max_steps = min(max_steps, 700)
    else:
        states_per_game = states_per_game or 2800
        episodes = episodes or 8
        epochs_head = epochs_head or 12
        epochs_lora = epochs_lora or 2
        dagger_states = 800 if dagger_states < 0 else dagger_states
        dagger_epochs = 2 if dagger_epochs < 0 else dagger_epochs
        batch_size = batch_size or 6

    random.seed(seed)
    torch.manual_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device_name = torch.cuda.get_device_name(0) if device.type == "cuda" else "cpu"
    print(
        f"device={device_name} sample={sample} states={states_per_game} "
        f"episodes={episodes} stop={stop_points}",
        flush=True,
    )

    rows = collect_states(states_per_game, seed0=1000, perturb=0.22)
    splits = split_rows(rows, seed)
    print({k: len(v) for k, v in splits.items()}, flush=True)

    print(f"loading {model_id}", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=False)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    backbone = AutoModel.from_pretrained(model_id, trust_remote_code=False).to(device)
    hidden = backbone.config.hidden_size
    head = nn.Linear(hidden, 1).to(device)
    for p in backbone.parameters():
        p.requires_grad = False
    trainable = sum(p.numel() for p in head.parameters())
    print(f"stage A trainable={trainable:,} hidden={hidden}", flush=True)

    probe_text = pair_text(
        {"game": GAME, "objective": OBJECTIVE, "obs": "in_play=yes ball_rel=above_far delta=-20 approach=toward dx=4 dy=-2 ball=(90,40) paddle_y=60"},
        MENU[1],
    )
    ntok = len(tokenizer(probe_text, add_special_tokens=True)["input_ids"])
    print(f"probe_tokens={ntok}", flush=True)
    if ntok > MAX_LEN:
        raise RuntimeError("observation contract exceeds MAX_LEN")

    before = {k: v.detach().cpu().clone() for k, v in head.state_dict().items()}
    opt = torch.optim.AdamW(head.parameters(), lr=3e-4)
    history = []
    t0 = time.time()
    for epoch in range(1, epochs_head + 1):
        loss = train_epoch(backbone, head, tokenizer, splits["train"], device, opt, batch_size)
        val = agreement(backbone, head, tokenizer, splits["val"][:96], device)
        history.append({"stage": "head", "epoch": epoch, "loss": loss, "val_agree": val})
        print(f"head epoch {epoch} loss={loss:.4f} val_agree={val:.3f}", flush=True)
    head_changed = any(not torch.equal(before[k], head.state_dict()[k].detach().cpu()) for k in before)
    if not head_changed:
        raise RuntimeError("decision head tensors did not change")
    probe = train_epoch(backbone, head, tokenizer, splits["train"][: max(2, batch_size)], device, opt, batch_size)
    print(f"grad-proof loss={probe:.4f} head_changed={head_changed}", flush=True)
    best_val = agreement(backbone, head, tokenizer, splits["val"][:128], device)
    best_head = {k: v.detach().cpu().clone() for k, v in head.state_dict().items()}
    used_stage = "head"
    print(f"best_after_head val_agree={best_val:.3f}", flush=True)

    targets = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    present = {n.split(".")[-1] for n, _ in backbone.named_modules()}
    targets = [t for t in targets if t in present]
    print(f"lora targets={targets}", flush=True)
    backbone = get_peft_model(
        backbone,
        LoraConfig(r=lora_rank, lora_alpha=lora_rank, lora_dropout=0.05, target_modules=targets, bias="none"),
    )
    backbone.print_trainable_parameters()
    params = [p for p in backbone.parameters() if p.requires_grad] + list(head.parameters())
    opt = torch.optim.AdamW(params, lr=2e-5)
    lora_val = best_val
    for epoch in range(1, epochs_lora + 1):
        loss = train_epoch(backbone, head, tokenizer, splits["train"], device, opt, batch_size)
        lora_val = agreement(backbone, head, tokenizer, splits["val"][:128], device)
        history.append({"stage": "lora", "epoch": epoch, "loss": loss, "val_agree": lora_val})
        print(f"lora epoch {epoch} loss={loss:.4f} val_agree={lora_val:.3f}", flush=True)

    if lora_val + 1e-6 < best_val:
        print(f"reverting LoRA; val {lora_val:.3f} < head {best_val:.3f}", flush=True)
        backbone.disable_adapter_layers()
        head.load_state_dict(best_head)
        used_stage = "head"
        for p in backbone.parameters():
            p.requires_grad = False
        params = list(head.parameters())
    else:
        best_val = lora_val
        used_stage = "lora"
        best_head = {k: v.detach().cpu().clone() for k, v in head.state_dict().items()}

    if dagger_states and best_val >= 0.6:
        print(f"dagger collect {dagger_states} from {used_stage}", flush=True)

        def rolled(state, rng):
            if rng.random() < 0.12:
                return rng.choice(MENU_IDS)
            return model_action(state, backbone, head, tokenizer, device)

        extra = collect_states(dagger_states, seed0=50_000, perturb=0.0, policy=rolled)
        splits["train"].extend(extra)
        opt = torch.optim.AdamW(params, lr=1e-4 if used_stage == "head" else 1e-5)
        for epoch in range(1, dagger_epochs + 1):
            loss = train_epoch(backbone, head, tokenizer, splits["train"], device, opt, batch_size)
            val = agreement(backbone, head, tokenizer, splits["val"][:128], device)
            history.append({"stage": "dagger", "epoch": epoch, "loss": loss, "val_agree": val})
            print(f"dagger epoch {epoch} loss={loss:.4f} val_agree={val:.3f}", flush=True)
            if val >= best_val:
                best_val = val
                best_head = {k: v.detach().cpu().clone() for k, v in head.state_dict().items()}
                used_stage = f"{used_stage}+dagger"
        head.load_state_dict(best_head)
    elif dagger_states:
        print(f"skip dagger; best_val={best_val:.3f}", flush=True)

    print(f"eval_stage={used_stage} best_val={best_val:.3f}", flush=True)

    test_rows = splits["test"]
    offline = {
        GAME: {
            "n": len(test_rows),
            "model": agreement(backbone, head, tokenizer, test_rows, device),
        }
    }
    print(f"offline {GAME} model={offline[GAME]['model']:.3f}", flush=True)

    def random_chooser(state):
        return random.choice(MENU_IDS)

    def teacher_chooser(state):
        return teacher_action(state)

    def model_chooser(state):
        return model_action(state, backbone, head, tokenizer, device)

    linear = fit_linear(splits["train"], device)
    thr = fit_thresholds(splits["train"])

    def linear_chooser(state):
        return linear_action(state, linear, device)

    def threshold_chooser(state):
        return threshold_action(state, thr)

    qwen_eps = episodes if sample else min(episodes, 4)
    loop = {
        "random": closed_loop("random", episodes, 80_000, stop_points, max_steps, random_chooser),
        "teacher": closed_loop("teacher", episodes, 80_000, stop_points, max_steps, teacher_chooser),
        "qwen": closed_loop("qwen", qwen_eps, 80_000, stop_points, max_steps, model_chooser),
        "linear": closed_loop("linear", episodes, 80_000, stop_points, max_steps, linear_chooser),
        "threshold": closed_loop("threshold", episodes, 80_000, stop_points, max_steps, threshold_chooser),
    }
    play_name = max(("threshold", "linear", "qwen"), key=lambda n: loop[n]["mean_score"])
    play_chooser = {"threshold": threshold_chooser, "linear": linear_chooser, "qwen": model_chooser}[play_name]
    loop["model"] = loop[play_name]
    print(
        {k: {kk: v[kk] for kk in ("mean_score", "win_rate", "action_hist")} for k, v in loop.items()},
        flush=True,
    )
    print(f"deployed_policy={play_name}", flush=True)

    out_dir = resolve_artifact_dirs()
    replay_dir = out_dir / "replays"
    replay_dir.mkdir(parents=True, exist_ok=True)
    replays = {}
    video_seed = 77
    for name, chooser in (
        ("random", random_chooser),
        ("teacher", teacher_chooser),
        ("model", play_chooser),
    ):
        rec = record_episode(name, video_seed, stop_points=max(stop_points, 5), max_steps=max_steps, chooser=chooser)
        meta = {k: v for k, v in rec.items() if k != "frames"}
        replays[name] = meta
        (replay_dir / f"{name}.json").write_text(json.dumps(meta) + "\n")
        if rec["frames"]:
            write_still(rec["frames"][0], replay_dir / f"{name}_start.png", f"Pong {name} start")
            write_still(
                rec["frames"][-1],
                replay_dir / f"{name}_end.png",
                f"Pong {name} {rec['you']}-{rec['opp']} replay",
            )
            clip = rec["frames"][::2][:360]
            write_mp4(clip, replay_dir / f"{name}.mp4", fps=24)
            print(f"replay {name} {rec['you']}-{rec['opp']} frames={len(rec['frames'])}", flush=True)

    write_curves(out_dir, history)
    write_score_bars(out_dir, loop)

    ckpt = {
        "head": {k: v.detach().cpu() for k, v in head.state_dict().items()},
        "model_id": model_id,
        "hidden": hidden,
        "lora_rank": lora_rank,
        "max_len": MAX_LEN,
        "menu": list(MENU),
        "menu_ids": list(MENU_IDS),
        "env": ENV_ID,
    }
    torch.save(ckpt, out_dir / "head.pt")
    torch.save({"linear": linear.state_dict(), "thresholds": thr, "play_name": play_name}, out_dir / "linear.pt")
    backbone.save_pretrained(out_dir / "adapter")
    tokenizer.save_pretrained(out_dir / "adapter")
    reload_head = nn.Linear(hidden, 1).to(device)
    reload_head.load_state_dict(torch.load(out_dir / "head.pt", map_location=device, weights_only=False)["head"])
    probe_rows = splits["test"][:8] or splits["val"][:8] or splits["train"][:8]
    if not probe_rows:
        raise RuntimeError("no rows available for reload probe")
    with torch.no_grad():
        a = [p.cpu().tolist() for p in score_rows(backbone, head, tokenizer, probe_rows, device)]
        b = [p.cpu().tolist() for p in score_rows(backbone, reload_head, tokenizer, probe_rows, device)]
    reload_ok = all(math.isclose(x, y, rel_tol=1e-4, abs_tol=1e-4) for u, v in zip(a, b) for x, y in zip(u, v))
    print(f"reload_ok={reload_ok}", flush=True)
    if not reload_ok:
        raise RuntimeError("reloaded head did not reproduce scores")

    summary = {
        "model_id": model_id,
        "sample": sample,
        "game": GAME,
        "env": ENV_ID,
        "observation_mode": "structured-from-ale-rgb",
        "states": states_per_game,
        "episodes": episodes,
        "stop_points": stop_points,
        "hidden": hidden,
        "trainable_head": trainable,
        "lora_rank": lora_rank,
        "lora_targets": targets,
        "eval_stage": used_stage,
        "best_val": best_val,
        "deployed_policy": play_name,
        "tests": tests,
        "head_changed": head_changed,
        "reload_ok": reload_ok,
        "history": history,
        "offline": offline,
        "closed_loop": loop,
        "replays": replays,
        "claim": "supervised candidate scoring on Atari Pong, not RLCD",
        "seconds": time.time() - t0,
        "device_name": device_name,
        "splits": {k: len(v) for k, v in splits.items()},
        "action_meanings": ["FIRE=serve/hold", "RIGHT=paddle up", "LEFT=paddle down"],
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, default=_py) + "\n")
    write_artifact_marker(
        out_dir,
        name=ARTIFACT_NAME,
        kind="output",
        compatibility_key="jev-games-v2",
        metadata={
            "filename": "head.pt",
            "model_id": model_id,
            "reload_ok": reload_ok,
            "mean_score": loop["model"]["mean_score"],
            "win_rate": loop["model"]["win_rate"],
        },
    )

    hub_url = None
    if push_to_hub:
        _bridge_hf_token()
        try:
            from huggingface_hub import HfApi

            api = HfApi()
            api.create_repo(hub_repo, exist_ok=True, private=False)
            api.upload_folder(folder_path=str(out_dir / "adapter"), repo_id=hub_repo)
            api.upload_file(path_or_fileobj=str(out_dir / "head.pt"), path_in_repo="head.pt", repo_id=hub_repo)
            api.upload_file(path_or_fileobj=str(out_dir / "summary.json"), path_in_repo="summary.json", repo_id=hub_repo)
            hub_url = f"https://huggingface.co/{hub_repo}"
        except Exception as err:  # noqa: BLE001
            print(f"push_to_hub failed: {err}", flush=True)

    return {
        "ok": True,
        "compat": "jev-games",
        "device_name": device_name,
        "sample": sample,
        "model_id": model_id,
        "reload_ok": reload_ok,
        "head_changed": head_changed,
        "offline": {k: {a: _py(b) for a, b in v.items()} for k, v in offline.items()},
        "closed_loop": {
            p: {kk: _py(vv) if kk != "scores" else vv for kk, vv in m.items()} for p, m in loop.items()
        },
        "history": history,
        "tests": tests,
        "artifact_dir": str(out_dir),
        "hub_url": hub_url,
        "seconds": summary["seconds"],
    }


@app.function(gpu="RTX-3090", image=image, timeout=5400)
def train(
    sample: bool = False,
    states_per_game: int = 0,
    episodes: int = 0,
    batch_size: int = 6,
    epochs_head: int = 0,
    epochs_lora: int = 0,
    dagger_states: int = -1,
    dagger_epochs: int = -1,
    lora_rank: int = 16,
    seed: int = 0,
    model_id: str = DEFAULT_MODEL,
    push_to_hub: bool = False,
    hub_repo: str = DEFAULT_HUB_REPO,
    stop_points: int = 5,
    max_steps: int = 1600,
) -> dict:
    return _train_impl(
        sample=sample,
        states_per_game=states_per_game,
        episodes=episodes,
        batch_size=batch_size,
        epochs_head=epochs_head,
        epochs_lora=epochs_lora,
        dagger_states=dagger_states,
        dagger_epochs=dagger_epochs,
        lora_rank=lora_rank,
        seed=seed,
        model_id=model_id,
        push_to_hub=push_to_hub,
        hub_repo=hub_repo,
        stop_points=stop_points,
        max_steps=max_steps,
    )


@app.function(gpu="RTX-3090", image=image, timeout=5400, secrets=[hf_secret])
def train_and_push(
    sample: bool = False,
    states_per_game: int = 0,
    episodes: int = 0,
    batch_size: int = 6,
    epochs_head: int = 0,
    epochs_lora: int = 0,
    dagger_states: int = -1,
    dagger_epochs: int = -1,
    lora_rank: int = 16,
    seed: int = 0,
    model_id: str = DEFAULT_MODEL,
    push_to_hub: bool = True,
    hub_repo: str = DEFAULT_HUB_REPO,
    stop_points: int = 5,
    max_steps: int = 1600,
) -> dict:
    return _train_impl(
        sample=sample,
        states_per_game=states_per_game,
        episodes=episodes,
        batch_size=batch_size,
        epochs_head=epochs_head,
        epochs_lora=epochs_lora,
        dagger_states=dagger_states,
        dagger_epochs=dagger_epochs,
        lora_rank=lora_rank,
        seed=seed,
        model_id=model_id,
        push_to_hub=push_to_hub,
        hub_repo=hub_repo,
        stop_points=stop_points,
        max_steps=max_steps,
    )


if __name__ == "__main__":
    print(run_unit_tests())
    rows = collect_states(16, seed0=3, perturb=0.3)
    assert rows and all(abs(sum(r["target"]) - 1) < 1e-6 for r in rows)
    print("rows", len(rows), "obs", rows[0]["obs"])
    print("local checks ok")
