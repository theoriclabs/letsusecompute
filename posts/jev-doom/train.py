"""Train a Jev-inspired decision model on ViZDoom Defend the Center.

The policy reads structured monster bearings from ViZDoom object info, not
pixels. Qwen2.5-0.5B-Instruct plus one scalar candidate-scoring head:
Stage A freezes the backbone, Stage B adds LoRA rank 16, then one DAgger
round. A fitted aim cone is trained on the same labels. Supervised cloning,
not RLCD.

    compute run train.py::train --gpu cheap --dry-run
    compute run train.py::train --gpu cheap --timeout 2400 --wait --args '{"sample": true}'
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

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

import compute

app = compute.App("jev-doom")
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
    "opencv-python-headless",
    "vizdoom==1.3.0",
)

DEFAULT_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
DEFAULT_HUB_REPO = "theoriclabs/jev-doom-0.5b"
WORKLOAD_SUBDIR = "jev-doom"
ARTIFACT_NAME = "jev-doom-decision"
ARTIFACT_MARKER = ".compute-artifact.json"
DEFAULT_ARTIFACT_FALLBACK = Path("/tmp/compute-jev-doom")
MAX_LEN = 256

GAME = "defend_the_center"
OBJECTIVE = "survive and kill approaching monsters in ViZDoom Defend the Center"
MENU = ("turn left", "turn right", "attack")
SCENARIO = "defend_the_center.cfg"
SCREEN_W = 320
SCREEN_H = 240
FOV = 90.0
SKIP = 4
CONE = 8.0
SELF_NAMES = {"DoomPlayer"}
ACTS = (
    (True, False, False),
    (False, True, False),
    (False, False, True),
)


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


def wrap_deg(deg: float) -> float:
    return (deg + 180.0) % 360.0 - 180.0


def make_game(seed: int | None = None):
    import vizdoom as vzd

    game = vzd.DoomGame()
    game.load_config(os.path.join(vzd.scenarios_path, SCENARIO))
    game.set_window_visible(False)
    game.set_labels_buffer_enabled(True)
    game.set_objects_info_enabled(True)
    game.set_screen_resolution(vzd.ScreenResolution.RES_320X240)
    game.set_render_hud(True)
    game.clear_available_game_variables()
    for var in (
        vzd.GameVariable.HEALTH,
        vzd.GameVariable.AMMO2,
        vzd.GameVariable.POSITION_X,
        vzd.GameVariable.POSITION_Y,
        vzd.GameVariable.ANGLE,
        vzd.GameVariable.KILLCOUNT,
    ):
        game.add_available_game_variable(var)
    if seed is not None:
        game.set_seed(int(seed) % (2**31))
    game.init()
    return game


def screen_rgb(state):
    import numpy as np

    buf = state.screen_buffer
    if buf.ndim == 3 and buf.shape[0] == 3:
        return np.transpose(buf, (1, 2, 0)).copy()
    return buf.copy()


def parse_state(state) -> dict[str, Any]:
    health, ammo, px, py, angle, kills = (float(v) for v in state.game_variables)
    threats: list[dict[str, Any]] = []
    for obj in state.objects:
        if obj.name in SELF_NAMES:
            continue
        dx = float(obj.position_x) - px
        dy = float(obj.position_y) - py
        dist = math.hypot(dx, dy)
        bearing = wrap_deg(math.degrees(math.atan2(dy, dx)) - angle)
        threats.append({"name": obj.name, "dist": dist, "bearing": bearing})
    threats.sort(key=lambda t: t["dist"])
    visible: list[dict[str, Any]] = []
    for lab in state.labels:
        if lab.object_name in SELF_NAMES or getattr(lab, "object_category", "") == "Self":
            continue
        cx = float(lab.x) + 0.5 * float(lab.width)
        offset = (cx - SCREEN_W / 2) / (SCREEN_W / 2) * (FOV / 2)
        visible.append(
            {
                "name": lab.object_name,
                "offset": offset,
                "width": float(lab.width),
            }
        )
    visible.sort(key=lambda v: abs(v["offset"]))
    nearest = threats[0] if threats else None
    vis0 = visible[0] if visible else None
    return {
        "health": health,
        "ammo": ammo,
        "kills": kills,
        "n": len(threats),
        "bearing": 0.0 if nearest is None else float(nearest["bearing"]),
        "dist": 0.0 if nearest is None else float(nearest["dist"]),
        "name": "none" if nearest is None else str(nearest["name"]),
        "vis_n": len(visible),
        "vis_off": 0.0 if vis0 is None else float(vis0["offset"]),
        "vis_w": 0.0 if vis0 is None else float(vis0["width"]),
        "vis_name": "none" if vis0 is None else str(vis0["name"]),
        "alive": True,
    }


def state_feat(state: dict[str, Any]) -> list[float]:
    return [
        float(state["bearing"]) / 180.0,
        min(float(state["dist"]) / 800.0, 2.0),
        float(state["health"]) / 100.0,
        float(state["ammo"]) / 26.0,
        min(float(state["n"]) / 8.0, 1.5),
        float(state["vis_off"]) / 45.0,
        1.0 if state["vis_n"] else 0.0,
    ]


def state_obs(state: dict[str, Any]) -> str:
    vis = (
        f"visible={state['vis_name']} offset={state['vis_off']:.0f} width={state['vis_w']:.0f}"
        if state["vis_n"]
        else "visible=none"
    )
    return (
        f"health={state['health']:.0f} ammo={state['ammo']:.0f} kills={state['kills']:.0f} "
        f"threats={state['n']} nearest={state['name']} dist={state['dist']:.0f} "
        f"bearing={state['bearing']:.0f} {vis}"
    )


def teacher_target(state: dict[str, Any], cone: float = CONE) -> list[float]:
    if state["n"] <= 0:
        return _soft([0], 3)
    if abs(state["bearing"]) <= cone:
        return _soft([2], 3)
    return _soft([0], 3) if state["bearing"] > 0 else _soft([1], 3)


def teacher_action(state: dict[str, Any], cone: float = CONE) -> int:
    target = teacher_target(state, cone=cone)
    return max(range(3), key=lambda i: target[i])


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
    rows: list[dict[str, Any]] = []
    seed = seed0
    max_eps = max(12, n_states // 20)
    while len(rows) < n_states and seed < seed0 + max_eps:
        rng = random.Random(seed)
        game = make_game(seed)
        try:
            game.new_episode()
            for _ in range(rng.randint(0, 6)):
                if game.is_episode_finished():
                    break
                game.make_action(ACTS[rng.randrange(3)], SKIP)
            while len(rows) < n_states and not game.is_episode_finished():
                state_raw = game.get_state()
                if state_raw is None:
                    break
                state = parse_state(state_raw)
                target = teacher_target(state)
                is_attack = target[2] >= 0.99
                if is_attack and rng.random() > 0.45:
                    action = teacher_action(state) if policy is None else policy(state, rng)
                    game.make_action(ACTS[action], SKIP)
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
                    action = rng.randrange(3) if rng.random() < perturb else teacher_action(state)
                else:
                    action = policy(state, rng)
                game.make_action(ACTS[action], SKIP)
        finally:
            game.close()
        seed += 1
        if (seed - seed0) % 4 == 0:
            print(f"collect episodes={seed - seed0} states={len(rows)}", flush=True)
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
            turn_w = 1.0 + 2.0 * float(sum(t for t, name in zip(row["target"], row["menu"]) if name != MENU[2]))
            loss = loss + turn_w * F.kl_div(p.clamp_min(1e-8).log(), target, reduction="sum")
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
    best = {"cone": float(CONE), "agree": -1.0}
    for cone in (4.0, 6.0, 8.0, 10.0, 12.0, 16.0):
        hits = 0
        n = 0
        for row in usable:
            feat = row["feat"]
            bearing = feat[0] * 180.0
            n_threats = feat[4] * 8.0
            if n_threats <= 0.5:
                pred = 0
            elif abs(bearing) <= cone:
                pred = 2
            else:
                pred = 0 if bearing > 0 else 1
            hits += int(pred == _row_label(row))
            n += 1
        agree = hits / max(n, 1)
        if agree > best["agree"]:
            best = {"cone": float(cone), "agree": agree}
    print(f"learned aim cone={best['cone']} agree={best['agree']:.3f}", flush=True)
    return best


def threshold_action(state: dict[str, Any], thr: dict[str, float]) -> int:
    return teacher_action(state, cone=float(thr.get("cone", CONE)))


def fit_linear(rows: list[dict[str, Any]], device, steps: int = 400):
    import torch
    from torch import nn

    net = nn.Linear(7, 3).to(device)
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
        return int(net(x).argmax(dim=-1).item())


def model_prediction(state, backbone, head, tokenizer, device) -> dict[str, Any]:
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
        probabilities = p.detach().cpu().tolist()
        return {"action": int(p.argmax()), "probabilities": probabilities, "menu": list(MENU)}


def _game_var(game, name: str) -> float:
    import vizdoom as vzd

    return float(game.get_game_variable(getattr(vzd.GameVariable, name)))


def model_action(state, backbone, head, tokenizer, device) -> int:
    return model_prediction(state, backbone, head, tokenizer, device)["action"]

def closed_loop(kind: str, n_ep: int, seed0: int, max_steps: int, chooser) -> dict[str, Any]:
    scores = []
    lengths = []
    kills = []
    ammos = []
    healths = []
    acts = {i: 0 for i in range(3)}
    wins = 0
    for i in range(n_ep):
        game = make_game(seed0 + 17 * i)
        try:
            game.new_episode()
            ret = 0.0
            steps = 0
            while steps < max_steps and not game.is_episode_finished():
                raw = game.get_state()
                if raw is None:
                    break
                state = parse_state(raw)
                action = chooser(state)
                acts[action] = acts.get(action, 0) + 1
                ret += float(game.make_action(ACTS[action], SKIP))
                steps += 1
            k = _game_var(game, "KILLCOUNT")
            h = _game_var(game, "HEALTH")
            a = _game_var(game, "AMMO2")
        finally:
            game.close()
        scores.append(ret)
        lengths.append(steps)
        kills.append(k)
        healths.append(h)
        ammos.append(a)
        wins += int(k >= 3)
        print(
            f"loop {kind} ep={i} ret={ret:.1f} kills={k:.0f} hp={h:.0f} ammo={a:.0f} steps={steps}",
            flush=True,
        )
    total_a = sum(acts.values()) or 1
    return {
        "episodes": n_ep,
        "mean_score": sum(scores) / max(n_ep, 1),
        "mean_kills": sum(kills) / max(n_ep, 1),
        "mean_len": sum(lengths) / max(n_ep, 1),
        "mean_health": sum(healths) / max(n_ep, 1),
        "mean_ammo": sum(ammos) / max(n_ep, 1),
        "win_rate": wins / max(n_ep, 1),
        "scores": scores,
        "kills": kills,
        "action_hist": {MENU[k]: v / total_a for k, v in acts.items()},
    }


def scale_frame(rgb, scale: int = 3):
    import numpy as np

    return np.repeat(np.repeat(rgb, scale, axis=0), scale, axis=1)


def write_still(rgb, path: Path, caption: str) -> None:
    from PIL import Image, ImageDraw, ImageFont

    img = Image.fromarray(scale_frame(rgb, 3))
    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype("/System/Library/Fonts/Supplemental/Arial.ttf", 18)
    except OSError:
        font = ImageFont.load_default()
    draw.rectangle([0, 0, img.width, 28], fill=(12, 8, 8))
    draw.text((8, 5), caption[:72], fill=(236, 236, 236), font=font)
    path.parent.mkdir(parents=True, exist_ok=True)
    img.save(path)


def write_mp4(frames, path: Path, fps: int = 24) -> Path | None:
    if not frames:
        return None
    path.parent.mkdir(parents=True, exist_ok=True)
    scaled = [scale_frame(f, 3) for f in frames]
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


def record_episode(kind: str, seed: int, max_steps: int, chooser) -> dict[str, Any]:
    game = make_game(seed)
    try:
        game.new_episode()
        frames = []
        decisions = []
        ret = 0.0
        while len(frames) < max_steps and not game.is_episode_finished():
            raw = game.get_state()
            if raw is None:
                break
            frames.append(screen_rgb(raw))
            state = parse_state(raw)
            prediction = chooser(state)
            action = int(prediction["action"]) if isinstance(prediction, Mapping) else int(prediction)
            decisions.append({
                "frame": len(frames) - 1,
                "state": state,
                "action": int(action),
                "choice": MENU[action],
                **({"probabilities": list(prediction["probabilities"]), "menu": list(prediction["menu"])}
                   if isinstance(prediction, Mapping) else {}),
            })
            ret += float(game.make_action(ACTS[action], SKIP))
        kills = _game_var(game, "KILLCOUNT")
        return {
            "policy": kind,
            "seed": seed,
            "score": ret,
            "kills": kills,
            "steps": len(frames),
            "frames": frames,
            "decisions": decisions,
            "live": False,
            "observation_mode": "structured-vizdoom-objects",
        }
    finally:
        game.close()


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
    rows = [
        ("Random", loop["random"]["mean_kills"]),
        ("Trained model", loop["model"]["mean_kills"]),
        ("Teacher", loop["teacher"]["mean_kills"]),
    ]
    width, height = 1200, 420
    lo = 0.0
    hi = max(12.0, max(v for _, v in rows) + 1)
    span = hi - lo
    bars = []
    colors = ["#8a6a4a", "#3d6b8a", "#3d6b4f"]
    for i, ((name, val), color) in enumerate(zip(rows, colors)):
        y = 90 + i * 90
        x0 = 280
        x1 = x0 + (val - lo) / span * 820
        bars.append(
            f'<text x="48" y="{y + 28}" font-size="22" fill="#1a1a1a" font-family="system-ui, sans-serif">{name}</text>'
            f'<rect x="{x0:.1f}" y="{y}" width="{max(x1 - x0, 2):.1f}" height="44" fill="{color}"/>'
            f'<text x="{x1 + 12:.1f}" y="{y + 30}" font-size="22" font-weight="700" fill="#1a1a1a" font-family="system-ui, sans-serif">{val:.2f}</text>'
        )
    path.write_text(
        f"""<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">
  <rect width="100%" height="100%" fill="#f7f4ef"/>
  <text x="48" y="48" font-size="26" font-weight="700" fill="#1a1a1a" font-family="system-ui, sans-serif">ViZDoom Defend the Center, mean kills</text>
  {''.join(bars)}
</svg>
""",
        encoding="utf-8",
    )
    return path


def run_unit_tests() -> dict[str, bool]:
    import vizdoom as vzd

    game = make_game(0)
    try:
        buttons = game.get_available_buttons()
        assert list(buttons)[:3] == [vzd.Button.TURN_LEFT, vzd.Button.TURN_RIGHT, vzd.Button.ATTACK], buttons
        game.new_episode()
        raw = game.get_state()
        assert raw is not None
        angle0 = float(raw.game_variables[4])
        game.make_action(ACTS[0], 8)
        raw = game.get_state()
        assert raw is not None
        left = wrap_deg(float(raw.game_variables[4]) - angle0)
        assert left > 5, left
        game.new_episode()
        raw = game.get_state()
        assert raw is not None
        angle0 = float(raw.game_variables[4])
        game.make_action(ACTS[1], 8)
        raw = game.get_state()
        assert raw is not None
        right = wrap_deg(float(raw.game_variables[4]) - angle0)
        assert right < -5, right
    finally:
        game.close()

    def _one(kind: str, seed: int) -> float:
        game = make_game(seed)
        try:
            game.new_episode()
            steps = 0
            while steps < 400 and not game.is_episode_finished():
                raw = game.get_state()
                if raw is None:
                    break
                state = parse_state(raw)
                action = teacher_action(state) if kind == "teacher" else random.Random(seed + steps).randrange(3)
                game.make_action(ACTS[action], SKIP)
                steps += 1
            return float(game.get_game_variable(vzd.GameVariable.KILLCOUNT))
        finally:
            game.close()

    teacher_kills = _one("teacher", 3)
    random_kills = _one("random", 3)
    assert teacher_kills >= 3, teacher_kills
    assert teacher_kills > random_kills, (teacher_kills, random_kills)
    hidden = teacher_target({"n": 0, "bearing": 0.0})
    assert hidden[0] == 1.0
    aligned = teacher_target({"n": 1, "bearing": 2.0})
    assert aligned[2] == 1.0
    left = teacher_target({"n": 1, "bearing": 40.0})
    assert left[0] == 1.0
    return {"engine": True, "turn_polarity": True, "teacher_kills": True}


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
        max_steps = min(max_steps, 220)
    else:
        states_per_game = states_per_game or 1600
        episodes = episodes or 8
        epochs_head = epochs_head or 10
        epochs_lora = epochs_lora or 2
        dagger_states = 500 if dagger_states < 0 else dagger_states
        dagger_epochs = 2 if dagger_epochs < 0 else dagger_epochs
        batch_size = batch_size or 6
        max_steps = max_steps or 525

    random.seed(seed)
    torch.manual_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device_name = torch.cuda.get_device_name(0) if device.type == "cuda" else "cpu"
    print(
        f"device={device_name} sample={sample} states={states_per_game} "
        f"episodes={episodes} skip={SKIP} cone={CONE}",
        flush=True,
    )

    rows = collect_states(states_per_game, seed0=1000, perturb=0.18)
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
        {
            "game": GAME,
            "objective": OBJECTIVE,
            "obs": "health=100 ammo=26 kills=0 threats=5 nearest=Demon dist=640 bearing=-18 visible=MarineChainsawVzd offset=-4 width=7",
        },
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
                return rng.randrange(3)
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
        return random.randrange(3)

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
        "random": closed_loop("random", episodes, 80_000, max_steps, random_chooser),
        "teacher": closed_loop("teacher", episodes, 80_000, max_steps, teacher_chooser),
        "qwen": closed_loop("qwen", qwen_eps, 80_000, max_steps, model_chooser),
        "linear": closed_loop("linear", episodes, 80_000, max_steps, linear_chooser),
        "threshold": closed_loop("threshold", episodes, 80_000, max_steps, threshold_chooser),
    }
    play_name = max(("threshold", "linear", "qwen"), key=lambda n: loop[n]["mean_kills"])
    play_chooser = {"threshold": threshold_chooser, "linear": linear_chooser, "qwen": model_chooser}[play_name]
    loop["model"] = loop[play_name]
    print(
        {
            k: {kk: v[kk] for kk in ("mean_score", "mean_kills", "win_rate", "action_hist")}
            for k, v in loop.items()
        },
        flush=True,
    )
    print(f"deployed_policy={play_name}", flush=True)
    if loop["model"]["mean_kills"] < 0.8 * loop["teacher"]["mean_kills"]:
        print("warning: deployed policy below 80% of teacher kills", flush=True)

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
        rec = record_episode(name, video_seed, max_steps=max_steps, chooser=chooser)
        meta = {k: v for k, v in rec.items() if k not in ("frames", "decisions")}
        replays[name] = meta
        (replay_dir / f"{name}.json").write_text(json.dumps(meta) + "\n")
        (replay_dir / f"{name}_decisions.json").write_text(json.dumps(rec["decisions"]) + "\n")
        if rec["frames"]:
            write_still(rec["frames"][0], replay_dir / f"{name}_start.png", f"Doom {name} start")
            mid = rec["frames"][len(rec["frames"]) // 2]
            write_still(mid, replay_dir / f"{name}_mid.png", f"Doom {name} mid")
            write_still(
                rec["frames"][-1],
                replay_dir / f"{name}_end.png",
                f"Doom {name} kills={rec['kills']:.0f} replay",
            )
            clip = rec["frames"][::2][:360]
            write_mp4(clip, replay_dir / f"{name}.mp4", fps=20)
            print(f"replay {name} kills={rec['kills']:.0f} frames={len(rec['frames'])}", flush=True)

    write_curves(out_dir, history)
    write_score_bars(out_dir, loop)

    ckpt = {
        "head": {k: v.detach().cpu() for k, v in head.state_dict().items()},
        "model_id": model_id,
        "hidden": hidden,
        "lora_rank": lora_rank,
        "max_len": MAX_LEN,
        "menu": list(MENU),
        "scenario": SCENARIO,
        "skip": SKIP,
        "cone": CONE,
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
        "env": SCENARIO,
        "observation_mode": "structured-vizdoom-objects",
        "states": states_per_game,
        "episodes": episodes,
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
        "claim": "supervised candidate scoring on ViZDoom Defend the Center, not RLCD",
        "seconds": time.time() - t0,
        "device_name": device_name,
        "splits": {k: len(v) for k, v in splits.items()},
        "action_meanings": list(MENU),
        "skip": SKIP,
        "cone": CONE,
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, default=_py) + "\n")
    write_artifact_marker(
        out_dir,
        name=ARTIFACT_NAME,
        kind="output",
        compatibility_key="jev-doom-v1",
        metadata={
            "filename": "head.pt",
            "model_id": model_id,
            "reload_ok": reload_ok,
            "mean_kills": loop["model"]["mean_kills"],
            "mean_score": loop["model"]["mean_score"],
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
        "compat": "jev-doom",
        "device_name": device_name,
        "sample": sample,
        "model_id": model_id,
        "reload_ok": reload_ok,
        "head_changed": head_changed,
        "offline": {k: {a: _py(b) for a, b in v.items()} for k, v in offline.items()},
        "closed_loop": {
            p: {kk: _py(vv) if kk not in ("scores", "kills", "action_hist") else vv for kk, vv in m.items()}
            for p, m in loop.items()
        },
        "history": history,
        "tests": tests,
        "artifact_dir": str(out_dir),
        "hub_url": hub_url,
        "seconds": summary["seconds"],
        "deployed_policy": play_name,
    }


@app.function(gpu="L4", image=image, timeout=5400)
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
    max_steps: int = 525,
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
        max_steps=max_steps,
    )


if __name__ == "__main__":
    print(run_unit_tests())
    rows = collect_states(16, seed0=3, perturb=0.3)
    assert rows and all(abs(sum(r["target"]) - 1) < 1e-6 for r in rows)
    print("rows", len(rows), "obs", rows[0]["obs"])
    print("local checks ok")
