"""Train one shared instruction-conditioned decision model on three games.

Games: 8x8 Snake, fully observed MiniGrid-style DoorKey-5x5, and a
MinAtar-faithful Breakout (public minimal action IDs 0/1/3). Teachers are
documented heuristics, not oracles. Qwen2.5-0.5B-Instruct plus a shared
scalar candidate-scoring head: Stage A freezes the backbone, Stage B adds
LoRA rank 16. This is supervised decision cloning, not RLCD.

    compute run train.py::train --gpu cheap --dry-run
    compute run train.py::train --gpu cheap --timeout 1800 --wait
"""

from __future__ import annotations

import hashlib
import io
import json
import math
import os
import random
import tempfile
from collections import deque
from collections.abc import Mapping
from dataclasses import dataclass
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
)
hf_secret = compute.Secret.from_name("hf")

DEFAULT_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
DEFAULT_HUB_REPO = "theoriclabs/jev-games-0.5b"
WORKLOAD_SUBDIR = "jev-games"
ARTIFACT_NAME = "jev-games-decision"
ARTIFACT_MARKER = ".compute-artifact.json"
DEFAULT_ARTIFACT_FALLBACK = Path("/tmp/compute-jev-games")
MAX_LEN = 256

SNAKE_ACTIONS = ("turn left", "continue straight", "turn right")
DOOR_ACTIONS = (
    "turn left",
    "turn right",
    "move forward",
    "pick up object",
    "drop object",
    "toggle door or object",
    "done",
)
# MinAtar full map is n,l,u,r,d,f. Breakout minimal set is n,l,r → IDs 0,1,3.
BREAK_ACTIONS = {0: "no-op", 1: "move paddle left", 3: "move paddle right"}
BREAK_MENU = (0, 1, 3)
DIRS = ((0, -1), (1, 0), (0, 1), (-1, 0))  # N E S W
DIR_NAME = "NESW"
MG_DIRS = ((1, 0), (0, 1), (-1, 0), (0, -1))  # MiniGrid: right, down, left, up
MG_DIR_NAME = "RDLU"


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


# ---------------------------------------------------------------------------
# Snake
# ---------------------------------------------------------------------------


@dataclass
class SnakeState:
    body: list[tuple[int, int]]
    heading: int
    food: tuple[int, int]
    steps: int
    since_food: int
    dead: bool
    eaten: int


def snake_reset(rng: random.Random, size: int = 8) -> SnakeState:
    hx, hy = rng.randint(2, size - 3), rng.randint(2, size - 3)
    heading = rng.randint(0, 3)
    dx, dy = DIRS[heading]
    body = [(hx, hy), (hx - dx, hy - dy)]
    food = _snake_food(rng, body, size)
    return SnakeState(body, heading, food, 0, 0, False, 0)


def _snake_food(rng: random.Random, body: list[tuple[int, int]], size: int) -> tuple[int, int]:
    free = [(x, y) for x in range(size) for y in range(size) if (x, y) not in body]
    return rng.choice(free) if free else body[0]


def snake_step(state: SnakeState, action: int, rng: random.Random, size: int = 8) -> SnakeState:
    if state.dead:
        return state
    heading = (state.heading + {0: -1, 1: 0, 2: 1}[action]) % 4
    dx, dy = DIRS[heading]
    hx, hy = state.body[0]
    nxt = (hx + dx, hy + dy)
    grow = nxt == state.food
    body = list(state.body)
    hit = (
        not (0 <= nxt[0] < size and 0 <= nxt[1] < size)
        or nxt in body[:-1]
        or (nxt in body[-1:] and grow)
    )
    if hit or state.since_food >= 32:
        return SnakeState(body, heading, state.food, state.steps + 1, state.since_food + 1, True, state.eaten)
    body = [nxt] + (body if grow else body[:-1])
    food = _snake_food(rng, body, size) if grow else state.food
    return SnakeState(body, heading, food, state.steps + 1, 0 if grow else state.since_food + 1, False, state.eaten + int(grow))


def _snake_legal(state: SnakeState, size: int = 8) -> list[int]:
    legal = []
    for action in range(3):
        heading = (state.heading + {0: -1, 1: 0, 2: 1}[action]) % 4
        dx, dy = DIRS[heading]
        nxt = (state.body[0][0] + dx, state.body[0][1] + dy)
        grow = nxt == state.food
        occ = state.body[:-1] if not grow else state.body
        if 0 <= nxt[0] < size and 0 <= nxt[1] < size and nxt not in occ:
            legal.append(action)
    return legal


def snake_teacher(state: SnakeState, size: int = 8) -> list[float]:
    """Safety/food heuristic: legal moves, then nearest food. Ties kept."""
    legal = _snake_legal(state, size)
    if not legal:
        return _soft([], 3)
    fx, fy = state.food

    def dist(action: int) -> int:
        heading = (state.heading + {0: -1, 1: 0, 2: 1}[action]) % 4
        dx, dy = DIRS[heading]
        nx, ny = state.body[0][0] + dx, state.body[0][1] + dy
        return abs(nx - fx) + abs(ny - fy)

    best = min(dist(a) for a in legal)
    return _soft([a for a in legal if dist(a) == best], 3)


def snake_obs(state: SnakeState) -> str:
    body = " ".join(f"{x},{y}" for x, y in state.body)
    return (
        f"8x8 snake head={state.body[0][0]},{state.body[0][1]} "
        f"dir={DIR_NAME[state.heading]} body={body} "
        f"food={state.food[0]},{state.food[1]} len={len(state.body)}"
    )


def snake_frame(state: SnakeState, size: int = 8) -> list[str]:
    grid = [["."] * size for _ in range(size)]
    for i, (x, y) in enumerate(state.body):
        grid[y][x] = "H" if i == 0 else "o"
    fx, fy = state.food
    if grid[fy][fx] == ".":
        grid[fy][fx] = "*"
    return ["".join(row) for row in grid]


# ---------------------------------------------------------------------------
# DoorKey-5x5, fully observed
# ---------------------------------------------------------------------------


@dataclass
class DoorState:
    x: int
    y: int
    direction: int
    has_key: bool
    door_open: bool
    key: tuple[int, int]
    door: tuple[int, int]
    goal: tuple[int, int]
    steps: int
    done: bool
    success: bool


def door_reset(rng: random.Random) -> DoorState:
    door_y = rng.randint(1, 3)
    key_y = rng.randint(1, 3)
    agent_y = rng.randint(1, 3)
    goal_y = rng.randint(1, 3)
    direction = rng.randint(0, 3)
    return DoorState(1, agent_y, direction, False, False, (1, key_y), (2, door_y), (3, goal_y), 0, False, False)


def _door_front(state: DoorState) -> tuple[int, int]:
    dx, dy = MG_DIRS[state.direction]
    return state.x + dx, state.y + dy


def _door_blocked(state: DoorState, x: int, y: int) -> bool:
    if not (1 <= x <= 3 and 1 <= y <= 3):
        return True
    if x == 2 and (x, y) != state.door:
        return True
    if (x, y) == state.door and not state.door_open:
        return True
    return False


def door_step(state: DoorState, action: int) -> DoorState:
    if state.done:
        return state
    x, y, direction = state.x, state.y, state.direction
    has_key, door_open = state.has_key, state.door_open
    key = state.key
    fx, fy = _door_front(state)
    if action == 0:
        direction = (direction - 1) % 4
    elif action == 1:
        direction = (direction + 1) % 4
    elif action == 2:
        nx, ny = fx, fy
        if not _door_blocked(state, nx, ny):
            x, y = nx, ny
    elif action == 3 and (fx, fy) == key and not has_key:
        has_key = True
        key = (-1, -1)
    elif action == 5 and (fx, fy) == state.door:
        if door_open:
            door_open = False
        elif has_key:
            door_open = True
    success = (x, y) == state.goal
    done = success or state.steps + 1 >= 64
    return DoorState(x, y, direction, has_key, door_open, key, state.door, state.goal, state.steps + 1, done, success)


def door_teacher(state: DoorState) -> list[float]:
    """BFS over (x,y,dir,key,door). All equally short first actions kept."""
    start = (state.x, state.y, state.direction, state.has_key, state.door_open)
    queue = deque([(start, [])])
    seen = {start}
    best: list[int] | None = None
    best_len = None
    while queue:
        cur, path = queue.popleft()
        if best_len is not None and len(path) > best_len:
            break
        key_pos = (-1, -1) if cur[3] else state.key
        probe = DoorState(cur[0], cur[1], cur[2], cur[3], cur[4], key_pos, state.door, state.goal, 0, False, False)
        if (probe.x, probe.y) == state.goal:
            if not path:
                return _soft([6], 7)
            if best_len is None:
                best_len = len(path)
                best = []
            if len(path) == best_len:
                best.append(path[0])
            continue
        for action in range(7):
            nxt_s = door_step(probe, action)
            key = (nxt_s.x, nxt_s.y, nxt_s.direction, nxt_s.has_key, nxt_s.door_open)
            if key in seen:
                continue
            seen.add(key)
            queue.append((key, path + [action]))
    return _soft(sorted(set(best or [])), 7)


def door_obs(state: DoorState) -> str:
    key = "held" if state.has_key else f"{state.key[0]},{state.key[1]}"
    door = "open" if state.door_open else "locked"
    return (
        f"doorkey-5x5 fully-observed pos={state.x},{state.y} "
        f"dir={MG_DIR_NAME[state.direction]} key={key} "
        f"door={state.door[0]},{state.door[1]}:{door} "
        f"goal={state.goal[0]},{state.goal[1]}"
    )


def door_frame(state: DoorState) -> list[str]:
    grid = [["W"] * 5 for _ in range(5)]
    for x in range(1, 4):
        for y in range(1, 4):
            grid[y][x] = "W" if x == 2 and (x, y) != state.door else "."
    grid[state.door[1]][state.door[0]] = "d" if state.door_open else "D"
    if not state.has_key and state.key[0] >= 0:
        grid[state.key[1]][state.key[0]] = "K"
    grid[state.goal[1]][state.goal[0]] = "G"
    grid[state.y][state.x] = MG_DIR_NAME[state.direction]
    return ["".join(row) for row in grid]


# ---------------------------------------------------------------------------
# Breakout — MinAtar rules, original code, public minimal IDs
# ---------------------------------------------------------------------------


@dataclass
class BreakState:
    paddle: int
    ball_x: int
    ball_y: int
    last_x: int
    last_y: int
    ball_dir: int
    bricks: list[list[int]]
    strike: bool
    terminal: bool
    reward: int
    steps: int


def break_reset(rng: random.Random) -> BreakState:
    paddle = rng.randint(0, 9)
    ball_x = rng.randint(0, 9)
    bricks = [[0] * 10 for _ in range(10)]
    for y in range(1, 4):
        for x in range(10):
            bricks[y][x] = 1
    return BreakState(paddle, ball_x, 4, ball_x, 5, rng.randint(0, 1), bricks, False, False, 0, 0)


def break_step(state: BreakState, action: int) -> BreakState:
    if state.terminal:
        return state
    paddle = state.paddle
    if action == 1:
        paddle = max(0, paddle - 1)
    elif action == 3:
        paddle = min(9, paddle + 1)
    last_x, last_y = state.ball_x, state.ball_y
    if state.ball_dir == 0:
        new_x, new_y = state.ball_x - 1, state.ball_y - 1
    elif state.ball_dir == 1:
        new_x, new_y = state.ball_x + 1, state.ball_y - 1
    elif state.ball_dir == 2:
        new_x, new_y = state.ball_x + 1, state.ball_y + 1
    else:
        new_x, new_y = state.ball_x - 1, state.ball_y + 1
    ball_dir = state.ball_dir
    bricks = [row[:] for row in state.bricks]
    strike = state.strike
    reward = 0
    terminal = False
    strike_toggle = False
    if new_x < 0 or new_x > 9:
        new_x = 0 if new_x < 0 else 9
        ball_dir = [1, 0, 3, 2][ball_dir]
    if new_y < 0:
        new_y = 0
        ball_dir = [3, 2, 1, 0][ball_dir]
    elif 0 <= new_y < 10 and bricks[new_y][new_x] == 1:
        strike_toggle = True
        if not strike:
            reward += 1
            strike = True
        bricks[new_y][new_x] = 0
        new_y = last_y
        ball_dir = [3, 2, 1, 0][ball_dir]
    elif new_y == 9:
        if sum(sum(row) for row in bricks) == 0:
            for y in range(1, 4):
                bricks[y] = [1] * 10
        if last_x == paddle:
            ball_dir = [3, 2, 1, 0][ball_dir]
            new_y = last_y
        elif new_x == paddle:
            ball_dir = [2, 3, 0, 1][ball_dir]
            new_y = last_y
        else:
            terminal = True
    if not strike_toggle:
        strike = False
    return BreakState(paddle, new_x, new_y, last_x, last_y, ball_dir, bricks, strike, terminal, state.reward + reward, state.steps + 1)


def _break_land(state: BreakState) -> int:
    probe = state
    for _ in range(20):
        if probe.terminal or probe.ball_y >= 8:
            return probe.ball_x
        probe = break_step(probe, 0)
    return probe.ball_x


def break_teacher(state: BreakState) -> list[float]:
    """Short-rollout intercept. Ties if already aligned."""
    target = _break_land(state)
    if state.paddle < target:
        return _soft([2], 3)  # menu index 2 is action 3 (right)
    if state.paddle > target:
        return _soft([1], 3)
    return _soft([0], 3)


def break_follow(state: BreakState) -> list[float]:
    if state.paddle < state.ball_x:
        return _soft([2], 3)
    if state.paddle > state.ball_x:
        return _soft([1], 3)
    return _soft([0], 3)


def break_obs(state: BreakState) -> str:
    bricks = "".join("1" if state.bricks[y][x] else "0" for y in range(1, 4) for x in range(10))
    return (
        f"breakout-minatar 10x10 paddle={state.paddle} "
        f"ball={state.ball_x},{state.ball_y} last={state.last_x},{state.last_y} "
        f"dir={state.ball_dir} bricks={bricks}"
    )


def break_frame(state: BreakState) -> list[str]:
    grid = [["."] * 10 for _ in range(10)]
    for y in range(10):
        for x in range(10):
            if state.bricks[y][x]:
                grid[y][x] = "#"
    grid[9][state.paddle] = "="
    if 0 <= state.ball_y < 10 and 0 <= state.ball_x < 10:
        grid[state.ball_y][state.ball_x] = "o"
    return ["".join(row) for row in grid]


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------


GAMES = {
    "snake": {
        "objective": "eat food and survive; do not hit walls or the body",
        "menu": list(SNAKE_ACTIONS),
        "reset": snake_reset,
        "step": lambda s, a, rng: snake_step(s, a, rng),
        "teacher": snake_teacher,
        "obs": snake_obs,
        "frame": snake_frame,
        "action_from_menu": lambda i: i,
        "success": lambda s: s.eaten,
        "done": lambda s: s.dead or s.steps >= 80,
    },
    "doorkey": {
        "objective": "use the key to open the door and reach the goal",
        "menu": list(DOOR_ACTIONS),
        "reset": door_reset,
        "step": lambda s, a, rng: door_step(s, a),
        "teacher": door_teacher,
        "obs": door_obs,
        "frame": door_frame,
        "action_from_menu": lambda i: i,
        "success": lambda s: int(s.success),
        "done": lambda s: s.done,
    },
    "breakout": {
        "objective": "keep the ball in play and break bricks",
        "menu": [BREAK_ACTIONS[i] for i in BREAK_MENU],
        "reset": break_reset,
        "step": lambda s, a, rng: break_step(s, BREAK_MENU[a]),
        "teacher": break_teacher,
        "obs": break_obs,
        "frame": break_frame,
        "action_from_menu": lambda i: BREAK_MENU[i],
        "success": lambda s: s.reward,
        "done": lambda s: s.terminal or s.steps >= 200,
    },
}


def fingerprint_state(game: str, obs: str) -> str:
    return hashlib.sha256(f"{game}|{obs}".encode()).hexdigest()[:16]


def collect_game(game: str, n_states: int, seed0: int, perturb: float) -> list[dict[str, Any]]:
    spec = GAMES[game]
    rows = []
    seed = seed0
    max_eps = max(80, n_states)
    while len(rows) < n_states and seed < seed0 + max_eps:
        rng = random.Random(seed)
        state = spec["reset"](rng)
        for _ in range(96):
            if spec["done"](state) or len(rows) >= n_states:
                break
            target = spec["teacher"](state)
            obs = spec["obs"](state)
            fp = fingerprint_state(game, obs)
            order = list(range(len(spec["menu"])))
            rng.shuffle(order)
            rows.append(
                {
                    "game": game,
                    "objective": spec["objective"],
                    "obs": obs,
                    "menu": [spec["menu"][i] for i in order],
                    "target": [target[i] for i in order],
                    "order": order,
                    "episode_seed": seed,
                    "fingerprint": fp,
                }
            )
            if rng.random() < perturb:
                action = rng.choice(list(range(len(target))))
            else:
                action = max(range(len(target)), key=lambda i: target[i])
            state = spec["step"](state, action, rng)
        seed += 1
        if (seed - seed0) % 25 == 0:
            print(f"collect {game} episodes={seed - seed0} states={len(rows)}", flush=True)
    return rows


def split_rows(rows: list[dict[str, Any]], seed: int) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[int, list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(row["episode_seed"], []).append(row)
    seeds = sorted(grouped)
    rng = random.Random(seed)
    rng.shuffle(seeds)
    n = len(seeds)
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
    out = {name: [] for name in parts}
    for name, chosen in parts.items():
        for s in chosen:
            for row in grouped[s]:
                row = dict(row)
                row["split"] = name
                out[name].append(row)
    return out


def pair_text(row: dict[str, Any], cand: str) -> str:
    return (
        f"Game: {row['game']}\nObjective: {row['objective']}\n"
        f"Observation: {row['obs']}\nCandidate action: {cand}"
    )


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------


def render_grid_png(lines: list[str], path: Path, title: str) -> None:
    from PIL import Image, ImageDraw

    cell = 18
    h, w = len(lines), max(len(row) for row in lines)
    img = Image.new("RGB", (w * cell + 16, h * cell + 40), (247, 244, 239))
    draw = ImageDraw.Draw(img)
    draw.text((8, 8), title[:48], fill=(26, 26, 26))
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
    }
    for y, row in enumerate(lines):
        for x, ch in enumerate(row):
            draw.rectangle(
                [8 + x * cell, 28 + y * cell, 8 + (x + 1) * cell - 1, 28 + (y + 1) * cell - 1],
                fill=colors.get(ch, (40, 40, 40)),
            )
    path.parent.mkdir(parents=True, exist_ok=True)
    img.save(path)


# ---------------------------------------------------------------------------
# Unit tests
# ---------------------------------------------------------------------------


def run_unit_tests() -> dict[str, bool]:
    rng = random.Random(0)
    s = snake_reset(rng)
    assert s.body[0] != s.food
    moved = snake_step(s, 1, random.Random(1))
    assert moved.steps == 1
    grow_state = SnakeState([(2, 2)], 1, (3, 2), 0, 0, False, 0)
    grown = snake_step(grow_state, 1, random.Random(2))
    assert len(grown.body) == 2 and grown.eaten == 1
    d = door_reset(random.Random(3))
    assert d.x == 1 and d.goal[0] == 3
    opened = DoorState(1, d.door[1], 0, True, False, (-1, -1), d.door, d.goal, 0, False, False)
    toggled = door_step(opened, 5)
    assert toggled.door_open
    t = door_teacher(door_reset(random.Random(4)))
    assert abs(sum(t) - 1) < 1e-6
    b = break_reset(random.Random(5))
    assert sum(sum(row) for row in b.bricks) == 30
    b2 = break_step(b, 1)
    assert b2.paddle in {b.paddle, max(0, b.paddle - 1)}
    return {"snake": True, "doorkey": True, "breakout": True}


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------


def encode_pairs(tokenizer, texts: list[str], device):
    import torch

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
    import torch
    import torch.nn.functional as F

    texts = []
    groups = []
    for row in rows:
        groups.append(len(row["menu"]))
        texts.extend(pair_text(row, cand) for cand in row["menu"])
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
            loss = loss + F.kl_div(p.clamp_min(1e-8).log(), target, reduction="sum")
        loss = loss / max(len(batch_rows), 1)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(list(head.parameters()) + [p for p in backbone.parameters() if p.requires_grad], 1.0)
        opt.step()
        total += float(loss.detach())
        n += 1
    return total / max(n, 1)


def agreement(backbone, head, tokenizer, rows, device) -> float:
    import torch

    if not rows:
        return 0.0
    backbone.eval()
    head.eval()
    hits = 0.0
    with torch.no_grad():
        probs = score_rows(backbone, head, tokenizer, rows, device)
        for row, p in zip(rows, probs):
            pred = int(p.argmax())
            hits += float(row["target"][pred] > 0)
    return hits / len(rows)


def closed_loop(game: str, policy, n_ep: int, seed0: int) -> dict[str, Any]:
    spec = GAMES[game]
    scores = []
    lengths = []
    fails = 0
    for i in range(n_ep):
        rng = random.Random(seed0 + i)
        state = spec["reset"](rng)
        while not spec["done"](state):
            action = policy(state, rng)
            state = spec["step"](state, action, rng)
        scores.append(spec["success"](state))
        lengths.append(getattr(state, "steps", 0))
        if game == "snake" and state.eaten == 0:
            fails += 1
        if game == "doorkey" and not state.success:
            fails += 1
        if game == "breakout" and state.reward == 0:
            fails += 1
    return {
        "episodes": n_ep,
        "mean_score": sum(scores) / max(n_ep, 1),
        "mean_len": sum(lengths) / max(n_ep, 1),
        "fail_rate": fails / max(n_ep, 1),
        "scores": scores,
    }


def teacher_policy(game: str):
    spec = GAMES[game]

    def _fn(state, rng):
        target = spec["teacher"](state)
        tied = [i for i, p in enumerate(target) if p == max(target)]
        return rng.choice(tied)

    return _fn


def random_policy(game: str):
    n = len(GAMES[game]["menu"])

    def _fn(state, rng):
        return rng.randint(0, n - 1)

    return _fn


def model_policy(game: str, backbone, head, tokenizer, device):
    import torch

    spec = GAMES[game]

    def _fn(state, rng):
        row = {
            "game": game,
            "objective": spec["objective"],
            "obs": spec["obs"](state),
            "menu": spec["menu"],
            "target": [0.0] * len(spec["menu"]),
        }
        with torch.no_grad():
            p = score_rows(backbone, head, tokenizer, [row], device)[0]
            return int(p.argmax())

    return _fn


def record_replay(game: str, policy, seed: int, max_steps: int = 48) -> dict[str, Any]:
    spec = GAMES[game]
    rng = random.Random(seed)
    state = spec["reset"](rng)
    frames = []
    for _ in range(max_steps):
        frames.append({"grid": spec["frame"](state), "obs": spec["obs"](state)})
        if spec["done"](state):
            break
        state = spec["step"](state, policy(state, rng), rng)
    return {
        "game": game,
        "seed": seed,
        "score": spec["success"](state),
        "steps": getattr(state, "steps", len(frames)),
        "frames": frames,
    }


def write_curves(directory: Path, history: list[dict[str, float]]) -> Path:
    width, height, pad = 1200, 420, 72
    if len(history) < 2:
        path = directory / "training_curves.svg"
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
    path = directory / "training_curves.svg"
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


def _train_impl(
    *,
    sample: bool,
    states_per_game: int,
    episodes: int,
    batch_size: int,
    epochs_head: int,
    epochs_lora: int,
    lora_rank: int,
    seed: int,
    model_id: str,
    push_to_hub: bool,
    hub_repo: str,
) -> dict:
    import time

    import torch
    from peft import LoraConfig, get_peft_model
    from torch import nn
    from transformers import AutoModel, AutoTokenizer

    tests = run_unit_tests()
    if sample:
        states_per_game = states_per_game or 80
        episodes = episodes or 6
        batch_size = min(batch_size, 2)
        epochs_head = epochs_head or 1
        epochs_lora = epochs_lora or 1
    else:
        states_per_game = states_per_game or 600
        episodes = episodes or 16
        epochs_head = epochs_head or 1
        epochs_lora = epochs_lora or 1

    random.seed(seed)
    torch.manual_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device_name = torch.cuda.get_device_name(0) if device.type == "cuda" else "cpu"
    print(f"device={device_name} sample={sample} states={states_per_game} episodes={episodes}", flush=True)

    rows = []
    for i, game in enumerate(("snake", "doorkey", "breakout")):
        chunk = collect_game(game, states_per_game, seed0=1000 * (i + 1), perturb=0.25)
        rows.extend(chunk)
        print(f"collected {game} {len(chunk)} states", flush=True)
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

    before = {k: v.detach().cpu().clone() for k, v in head.state_dict().items()}
    opt = torch.optim.AdamW(head.parameters(), lr=3e-4)
    history = []
    t0 = time.time()
    for epoch in range(1, epochs_head + 1):
        loss = train_epoch(backbone, head, tokenizer, splits["train"], device, opt, batch_size)
        val = agreement(backbone, head, tokenizer, splits["val"][:64], device)
        history.append({"stage": "head", "epoch": epoch, "loss": loss, "val_agree": val})
        print(f"head epoch {epoch} loss={loss:.4f} val_agree={val:.3f}", flush=True)
    head_changed = any(not torch.equal(before[k], head.state_dict()[k].detach().cpu()) for k in before)
    if not head_changed:
        raise RuntimeError("decision head tensors did not change")
    # one grad proof batch
    head.train()
    probe = train_epoch(backbone, head, tokenizer, splits["train"][: max(2, batch_size)], device, opt, batch_size)
    print(f"grad-proof loss={probe:.4f} head_changed={head_changed}", flush=True)

    targets = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    present = {n.split(".")[-1] for n, _ in backbone.named_modules()}
    targets = [t for t in targets if t in present]
    print(f"lora targets={targets}", flush=True)
    backbone = get_peft_model(
        backbone,
        LoraConfig(r=lora_rank, lora_alpha=2 * lora_rank, lora_dropout=0.05, target_modules=targets, bias="none"),
    )
    backbone.print_trainable_parameters()
    params = [p for p in backbone.parameters() if p.requires_grad] + list(head.parameters())
    opt = torch.optim.AdamW(params, lr=1e-4)
    for epoch in range(1, epochs_lora + 1):
        loss = train_epoch(backbone, head, tokenizer, splits["train"], device, opt, batch_size)
        val = agreement(backbone, head, tokenizer, splits["val"][:64], device)
        history.append({"stage": "lora", "epoch": epoch, "loss": loss, "val_agree": val})
        print(f"lora epoch {epoch} loss={loss:.4f} val_agree={val:.3f}", flush=True)

    # tiny MLP baseline on hashed obs bags
    mlp_agree: dict[str, float] = {}
    for game in ("snake", "doorkey", "breakout"):
        train_g = [r for r in splits["train"] if r["game"] == game]
        test_g = [r for r in splits["test"] if r["game"] == game][:80]
        if not train_g or not test_g:
            mlp_agree[game] = 0.0
            continue
        dim = 64
        w = torch.randn(dim, max(len(r["menu"]) for r in train_g), device=device) * 0.05
        w.requires_grad_(True)
        opt_m = torch.optim.Adam([w], lr=0.05)

        def bag(obs: str):
            vec = torch.zeros(dim, device=device)
            for tok in obs.replace(",", " ").split():
                vec[int(hashlib.md5(tok.encode()).hexdigest(), 16) % dim] += 1
            return vec

        for _ in range(80 if sample else 200):
            row = random.choice(train_g)
            logits = bag(row["obs"]) @ w[:, : len(row["menu"])]
            target = torch.tensor(row["target"], device=device)
            loss = torch.nn.functional.kl_div(logits.softmax(0).clamp_min(1e-8).log(), target, reduction="sum")
            opt_m.zero_grad()
            loss.backward()
            opt_m.step()
        hits = 0
        with torch.no_grad():
            for row in test_g:
                pred = int((bag(row["obs"]) @ w[:, : len(row["menu"])]).argmax())
                hits += int(row["target"][pred] > 0)
        mlp_agree[game] = hits / len(test_g)
        print(f"mlp {game} agree={mlp_agree[game]:.3f}", flush=True)

    offline = {}
    for game in ("snake", "doorkey", "breakout"):
        test_g = [r for r in splits["test"] if r["game"] == game]
        offline[game] = {
            "n": len(test_g),
            "model": agreement(backbone, head, tokenizer, test_g, device),
            "mlp": mlp_agree[game],
        }
        print(f"offline {game} model={offline[game]['model']:.3f} mlp={mlp_agree[game]:.3f}", flush=True)

    loop = {}
    for game in ("snake", "doorkey", "breakout"):
        loop[game] = {
            "random": closed_loop(game, random_policy(game), episodes, 50_000),
            "teacher": closed_loop(game, teacher_policy(game), episodes, 50_000),
            "model": closed_loop(game, model_policy(game, backbone, head, tokenizer, device), episodes, 50_000),
        }
        if game == "breakout":
            loop[game]["follow_ball"] = closed_loop(game, lambda s, rng: int(break_follow(s).index(max(break_follow(s)))), episodes, 50_000)
        print(f"loop {game} { {k: v['mean_score'] for k, v in loop[game].items()} }", flush=True)

    out_dir = resolve_artifact_dirs()
    replay_dir = out_dir / "replays"
    replay_dir.mkdir(parents=True, exist_ok=True)
    replays = {}
    for game in ("snake", "doorkey", "breakout"):
        for name, pol in (
            ("random", random_policy(game)),
            ("teacher", teacher_policy(game)),
            ("model", model_policy(game, backbone, head, tokenizer, device)),
        ):
            rec = record_replay(game, pol, seed=77 + len(game))
            replays[f"{game}_{name}"] = {
                "game": game,
                "policy": name,
                "seed": rec["seed"],
                "score": rec["score"],
                "steps": rec["steps"],
                "n_frames": len(rec["frames"]),
                "live": False,
                "observation_mode": "structured",
            }
            (replay_dir / f"{game}_{name}.json").write_text(json.dumps(rec) + "\n")
            if rec["frames"]:
                render_grid_png(
                    rec["frames"][0]["grid"],
                    replay_dir / f"{game}_{name}_start.png",
                    f"{game} {name} start",
                )
                render_grid_png(
                    rec["frames"][-1]["grid"],
                    replay_dir / f"{game}_{name}_end.png",
                    f"{game} {name} end score={rec['score']}",
                )

    # reload check
    ckpt = {
        "head": {k: v.detach().cpu() for k, v in head.state_dict().items()},
        "model_id": model_id,
        "hidden": hidden,
        "lora_rank": lora_rank,
        "max_len": MAX_LEN,
    }
    torch.save(ckpt, out_dir / "head.pt")
    backbone.save_pretrained(out_dir / "adapter")
    tokenizer.save_pretrained(out_dir / "adapter")
    reload_head = nn.Linear(hidden, 1).to(device)
    reload_head.load_state_dict(torch.load(out_dir / "head.pt", map_location=device, weights_only=False)["head"])
    probe_rows = splits["test"][:8] or splits["val"][:8]
    with torch.no_grad():
        a = [p.cpu().tolist() for p in score_rows(backbone, head, tokenizer, probe_rows, device)]
        b = [p.cpu().tolist() for p in score_rows(backbone, reload_head, tokenizer, probe_rows, device)]
    reload_ok = all(math.isclose(x, y, rel_tol=1e-4, abs_tol=1e-4) for u, v in zip(a, b) for x, y in zip(u, v))
    print(f"reload_ok={reload_ok}", flush=True)
    if not reload_ok:
        raise RuntimeError("reloaded head did not reproduce scores")

    write_curves(out_dir, history)
    summary = {
        "model_id": model_id,
        "sample": sample,
        "states_per_game": states_per_game,
        "episodes": episodes,
        "hidden": hidden,
        "trainable_head": trainable,
        "lora_rank": lora_rank,
        "lora_targets": targets,
        "tests": tests,
        "head_changed": head_changed,
        "reload_ok": reload_ok,
        "history": history,
        "offline": offline,
        "closed_loop": loop,
        "replays": replays,
        "claim": "supervised candidate scoring, not RLCD",
        "observation_mode": "structured",
        "seconds": time.time() - t0,
        "device_name": device_name,
        "splits": {k: len(v) for k, v in splits.items()},
        "breakout_action_ids": {"no-op": 0, "left": 1, "right": 3, "note": "MinAtar public minimal set"},
        "doorkey_variant": "fully observed DoorKey-5x5, not the default partial-obs MiniGrid benchmark",
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    write_artifact_marker(
        out_dir,
        name=ARTIFACT_NAME,
        kind="output",
        compatibility_key="jev-games-v1",
        metadata={
            "filename": "head.pt",
            "model_id": model_id,
            "reload_ok": reload_ok,
            "offline": {g: v["model"] for g, v in offline.items()},
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
            g: {p: {kk: _py(vv) if kk != "scores" else vv for kk, vv in m.items()} for p, m in games.items()}
            for g, games in loop.items()
        },
        "history": history,
        "tests": tests,
        "artifact_dir": str(out_dir),
        "hub_url": hub_url,
        "seconds": summary["seconds"],
    }


@app.function(gpu="RTX-3090", image=image, timeout=1800)
def train(
    sample: bool = False,
    states_per_game: int = 0,
    episodes: int = 0,
    batch_size: int = 4,
    epochs_head: int = 0,
    epochs_lora: int = 0,
    lora_rank: int = 16,
    seed: int = 0,
    model_id: str = DEFAULT_MODEL,
    push_to_hub: bool = False,
    hub_repo: str = DEFAULT_HUB_REPO,
) -> dict:
    return _train_impl(
        sample=sample,
        states_per_game=states_per_game,
        episodes=episodes,
        batch_size=batch_size,
        epochs_head=epochs_head,
        epochs_lora=epochs_lora,
        lora_rank=lora_rank,
        seed=seed,
        model_id=model_id,
        push_to_hub=push_to_hub,
        hub_repo=hub_repo,
    )


@app.function(gpu="RTX-3090", image=image, timeout=1800, secrets=[hf_secret])
def train_and_push(
    sample: bool = False,
    states_per_game: int = 0,
    episodes: int = 0,
    batch_size: int = 4,
    epochs_head: int = 0,
    epochs_lora: int = 0,
    lora_rank: int = 16,
    seed: int = 0,
    model_id: str = DEFAULT_MODEL,
    push_to_hub: bool = True,
    hub_repo: str = DEFAULT_HUB_REPO,
) -> dict:
    return _train_impl(
        sample=sample,
        states_per_game=states_per_game,
        episodes=episodes,
        batch_size=batch_size,
        epochs_head=epochs_head,
        epochs_lora=epochs_lora,
        lora_rank=lora_rank,
        seed=seed,
        model_id=model_id,
        push_to_hub=push_to_hub,
        hub_repo=hub_repo,
    )


if __name__ == "__main__":
    print(run_unit_tests())
    for game in ("snake", "doorkey", "breakout"):
        rows = collect_game(game, 12, seed0=1, perturb=0.3)
        assert rows and all(abs(sum(r["target"]) - 1) < 1e-6 for r in rows)
        print(game, "rows", len(rows), "obs", rows[0]["obs"][:80])
    print("local checks ok")
