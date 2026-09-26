"""AlphaZero from scratch on Connect Four, with checkpoint artifacts and resume.

One file holds the game, a batched PUCT Monte Carlo tree search, a small
residual policy-value network, self-play in parallel worker processes,
training, and evaluation. The recipe follows AlphaGo Zero / AlphaZero; the
code layout borrows from https://github.com/suragnair/alpha-zero-general.

Each iteration: self-play games with MCTS -> (position, visit distribution,
final result) examples -> a few hundred gradient steps on a sliding replay
window -> evaluation against three fixed opponents -> checkpoint written to
$COMPUTE_ARTIFACT_DIR. A later run resumes from a previous run's artifact by
installing the Compute CLI on the machine and running `compute artifacts get`
with a revocable API key injected as the COMPUTE_API_KEY secret.
"""
from __future__ import annotations

import json
import math
import os
import random
import shutil
import subprocess
import time
from pathlib import Path

import compute

app = compute.App("alphazero-small")
image = compute.Image.cuda_pytorch()
api_key = compute.Secret.from_name("COMPUTE_API_KEY")
hf_secret = compute.Secret.from_name("hf")

ROWS, COLS = 6, 7
ARTIFACT = "alphazero-c4"
# Evaluation, each from the network's side, alternating who moves first:
#   random, one_ply    : network + MCTS vs a random mover / a win-or-block heuristic (the issue's yardsticks)
#   uct1000            : network + MCTS vs classic rollout MCTS with 1,000 random playouts per move
#   iteration_0        : network + MCTS vs the untrained network with the same search
#   policy_vs_one_ply  : the network's raw policy, no search, vs the win-or-block heuristic
OPPONENTS = ("random", "one_ply", "uct1000", "iteration_0", "policy_vs_one_ply")
OPP_SEED = {"random": 11, "one_ply": 23, "uct1000": 29, "iteration_0": 37, "policy_vs_one_ply": 41}
HUB_REPO = "theoriclabs/alphazero-connect4"


# --------------------------------------------------------------------------- #
# Connect Four
# --------------------------------------------------------------------------- #

class State:
    """Board cells are 0 empty, 1 first player, 2 second player; row 0 is the bottom."""

    __slots__ = ("board", "heights", "to_move", "plies", "winner")

    def __init__(self):
        self.board = bytearray(ROWS * COLS)
        self.heights = [0] * COLS
        self.to_move = 1
        self.plies = 0
        self.winner = 0  # 0 ongoing, 1 or 2 winner, 3 draw

    def copy(self) -> "State":
        s = State.__new__(State)
        s.board = bytearray(self.board)
        s.heights = self.heights[:]
        s.to_move, s.plies, s.winner = self.to_move, self.plies, self.winner
        return s

    def legal(self) -> list[int]:
        return [c for c in range(COLS) if self.heights[c] < ROWS]

    def wins_with(self, c: int, p: int) -> bool:
        r = self.heights[c]
        b = self.board
        for dr, dc in ((0, 1), (1, 0), (1, 1), (1, -1)):
            n = 1
            rr, cc = r + dr, c + dc
            while 0 <= rr < ROWS and 0 <= cc < COLS and b[rr * COLS + cc] == p:
                n += 1
                rr += dr
                cc += dc
            rr, cc = r - dr, c - dc
            while 0 <= rr < ROWS and 0 <= cc < COLS and b[rr * COLS + cc] == p:
                n += 1
                rr -= dr
                cc -= dc
            if n >= 4:
                return True
        return False

    def play(self, c: int) -> None:
        p = self.to_move
        won = self.wins_with(c, p)
        self.board[self.heights[c] * COLS + c] = p
        self.heights[c] += 1
        self.plies += 1
        self.to_move = 3 - p
        if won:
            self.winner = p
        elif self.plies == ROWS * COLS:
            self.winner = 3


def random_player(s: State, rng: random.Random) -> int:
    return rng.choice(s.legal())


def one_ply_player(s: State, rng: random.Random) -> int:
    """Win if possible, else block an immediate loss, else random."""
    legal = s.legal()
    for c in legal:
        if s.wins_with(c, s.to_move):
            return c
    for c in legal:
        if s.wins_with(c, 3 - s.to_move):
            return c
    return rng.choice(legal)


class _UNode:
    __slots__ = ("N", "W", "children", "untried", "move", "mover", "parent")

    def __init__(self, s: State, move, mover, parent):
        self.N, self.W, self.children = 0, 0.0, []
        self.untried = [] if s.winner else s.legal()
        self.move, self.mover, self.parent = move, mover, parent


def uct_player(iterations: int):
    """Classic MCTS with uniformly random rollouts and no network: a fixed, stronger yardstick."""

    def choose(state: State, rng: random.Random) -> int:
        root = _UNode(state, None, 3 - state.to_move, None)
        for _ in range(iterations):
            node, s = root, state.copy()
            while not node.untried and node.children:
                log_n = math.log(node.N)
                node = max(node.children, key=lambda c: c.W / c.N + 1.4 * math.sqrt(log_n / c.N))
                s.play(node.move)
            if node.untried:
                a = node.untried.pop(rng.randrange(len(node.untried)))
                mover = s.to_move
                s.play(a)
                child = _UNode(s, a, mover, node)
                node.children.append(child)
                node = child
            while not s.winner:
                s.play(rng.choice(s.legal()))
            while node is not None:
                node.N += 1
                node.W += 0.5 if s.winner == 3 else (1.0 if s.winner == node.mover else 0.0)
                node = node.parent
        return max(root.children, key=lambda c: c.N).move

    return choose


# --------------------------------------------------------------------------- #
# Network
# --------------------------------------------------------------------------- #

def build_net(channels: int = 64, blocks: int = 5):
    import torch.nn as nn

    class Block(nn.Module):
        def __init__(self):
            super().__init__()
            self.c1 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
            self.b1 = nn.BatchNorm2d(channels)
            self.c2 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
            self.b2 = nn.BatchNorm2d(channels)

        def forward(self, x):
            y = self.b1(self.c1(x)).relu()
            return (x + self.b2(self.c2(y))).relu()

    class Net(nn.Module):
        def __init__(self):
            super().__init__()
            self.stem = nn.Sequential(nn.Conv2d(2, channels, 3, padding=1, bias=False),
                                      nn.BatchNorm2d(channels), nn.ReLU())
            self.body = nn.Sequential(*[Block() for _ in range(blocks)])
            self.pol = nn.Sequential(nn.Conv2d(channels, 2, 1, bias=False), nn.BatchNorm2d(2), nn.ReLU(),
                                     nn.Flatten(), nn.Linear(2 * ROWS * COLS, COLS))
            self.val = nn.Sequential(nn.Conv2d(channels, 1, 1, bias=False), nn.BatchNorm2d(1), nn.ReLU(),
                                     nn.Flatten(), nn.Linear(ROWS * COLS, 64), nn.ReLU(), nn.Linear(64, 1), nn.Tanh())

        def forward(self, x):
            h = self.body(self.stem(x))
            return self.pol(h), self.val(h).squeeze(-1)

    return Net()


def encode(boards, to_move):
    """boards: uint8 (B, 42) absolute; to_move: uint8 (B,). Returns float32 (B, 2, 6, 7): mine, theirs."""
    import numpy as np
    me = boards == to_move[:, None]
    opp = (boards != 0) & ~me
    return np.stack([me, opp], 1).reshape(-1, 2, ROWS, COLS).astype(np.float32)


def make_evaluator(net, device):
    import numpy as np
    import torch

    @torch.inference_mode()
    def evaluate(states):
        boards = np.frombuffer(b"".join(bytes(s.board) for s in states), np.uint8).reshape(-1, ROWS * COLS)
        tm = np.fromiter((s.to_move for s in states), np.uint8, len(states))
        x = torch.from_numpy(encode(boards, tm)).to(device)
        logits, v = net(x)
        return logits.float().cpu().numpy(), v.float().cpu().numpy()

    return evaluate


# --------------------------------------------------------------------------- #
# Batched MCTS: one leaf per game per round, all leaves in one forward pass
# --------------------------------------------------------------------------- #

class Node:
    __slots__ = ("P", "N", "W", "children", "legal")

    def __init__(self):
        self.P = None
        self.N = [0] * COLS
        self.W = [0.0] * COLS
        self.children = [None] * COLS
        self.legal = None


def _expand(node: Node, state: State, logits) -> None:
    legal = state.legal()
    m = max(logits[c] for c in legal)
    e = [math.exp(logits[c] - m) if c in legal else 0.0 for c in range(COLS)]
    z = sum(e)
    node.P = [x / z for x in e]
    node.legal = legal


def _select(node: Node, c_puct: float) -> int:
    sq = math.sqrt(sum(node.N) + 1)
    best, best_u = -1, -1e9
    N, W, P = node.N, node.W, node.P
    for a in node.legal:
        q = W[a] / N[a] if N[a] else 0.0
        u = q + c_puct * P[a] * sq / (1 + N[a])
        if u > best_u:
            best, best_u = a, u
    return best


def mcts(states, evaluate, sims, rng, noise=True, c_puct=1.5, alpha=1.0, eps=0.25):
    """Run `sims` simulations from each state in parallel. Returns visit counts per state."""
    roots = [Node() for _ in states]
    logits, _ = evaluate(states)
    for root, s, lg in zip(roots, states, logits):
        _expand(root, s, lg)
        if noise:
            d = [rng.gammavariate(alpha, 1) for _ in root.legal]
            t = sum(d)
            for a, x in zip(root.legal, d):
                root.P[a] = (1 - eps) * root.P[a] + eps * x / t
    for _ in range(sims):
        pending = []
        for root, s0 in zip(roots, states):
            node, s, path = root, s0.copy(), []
            while True:
                a = _select(node, c_puct)
                path.append((node, a))
                s.play(a)
                child = node.children[a]
                if child is None:
                    child = node.children[a] = Node()
                node = child
                if s.winner:
                    v = 0.0 if s.winner == 3 else -1.0  # the player to move has just lost
                    _backup(path, v)
                    break
                if child.P is None:
                    pending.append((child, s, path))
                    break
        if pending:
            lg, vals = evaluate([s for _, s, _ in pending])
            for (node, s, path), l, v in zip(pending, lg, vals):
                _expand(node, s, l)
                _backup(path, float(v))
    return [r.N for r in roots]


def _backup(path, v):
    # v is from the view of the player to move at the leaf; flip once per edge going up.
    for node, a in reversed(path):
        v = -v
        node.N[a] += 1
        node.W[a] += v


def play_games(n_games, evaluate, sims, rng, *, selfplay, opponent=None, temp_plies=8, record=False,
               policy_only=False):
    """Self-play (both sides MCTS, with noise and temperature) or eval (MCTS vs opponent, greedy).

    In eval, the network moves first in even-numbered games.
    Returns (examples, results). results[i] is +1/0/-1 from the network's point of view in eval,
    or the winner (1, 2, 3=draw) in self-play.
    """
    games = [State() for _ in range(n_games)]
    history = [[] for _ in range(n_games)]  # self-play: (board bytes, to_move, pi)
    moves = [[] for _ in range(n_games)]
    net_side = [1 if i % 2 == 0 else 2 for i in range(n_games)]
    while True:
        active = [i for i, g in enumerate(games) if not g.winner]
        if not active:
            break
        if not selfplay:
            for i in active:
                g = games[i]
                if g.to_move != net_side[i]:
                    a = opponent(g, rng)
                    moves[i].append(a)
                    g.play(a)
            active = [i for i in active if not games[i].winner and games[i].to_move == net_side[i]]
            if not active:
                continue
        if policy_only:  # the network's raw move choice, no search
            logits, _ = evaluate([games[i] for i in active])
            counts = [[(1 + lg[c]) if games[i].heights[c] < ROWS else -1e9 for c in range(COLS)]
                      for i, lg in zip(active, logits)]
            counts = [[x - min(row) for x in row] for row in counts]
        else:
            counts = mcts([games[i] for i in active], evaluate, sims, rng, noise=selfplay)
        for i, N in zip(active, counts):
            g = games[i]
            total = sum(N)
            if selfplay:
                pi = [n / total for n in N]
                history[i].append((bytes(g.board), g.to_move, pi))
                if g.plies < temp_plies:
                    a = rng.choices(range(COLS), weights=N)[0]
                else:
                    a = max(range(COLS), key=lambda c: (N[c], rng.random()))
            else:
                a = max(range(COLS), key=lambda c: (N[c], rng.random()))
            moves[i].append(a)
            g.play(a)
    examples = []
    if selfplay:
        for g, h in zip(games, history):
            for board, tm, pi in h:
                z = 0 if g.winner == 3 else (1 if g.winner == tm else -1)
                examples.append((board, tm, pi, z))
        results = [g.winner for g in games]
    else:
        results = [0 if g.winner == 3 else (1 if g.winner == side else -1) for g, side in zip(games, net_side)]
    return examples, results, (moves if record else None)


# --------------------------------------------------------------------------- #
# Worker processes
# --------------------------------------------------------------------------- #

def _pack(sd) -> bytes:
    """State dict -> bytes. Tensors pickled by multiprocessing go through fd sharing, which needs the
    parent's process authkey; plain bytes do not."""
    import io
    import torch
    buf = io.BytesIO()
    torch.save(sd, buf)
    return buf.getvalue()


def _unpack(b: bytes):
    import io
    import torch
    return torch.load(io.BytesIO(b), map_location="cpu")


def _worker(wid, conn, device, net_kwargs):
    import torch
    torch.set_num_threads(1)
    net = build_net(**net_kwargs).to(device).eval()
    ev = make_evaluator(net, device)
    opponents = {"random": random_player, "one_ply": one_ply_player, "uct1000": uct_player(1000),
                 "policy_vs_one_ply": one_ply_player}
    base_net = None
    conn.send(("ready", wid))
    while True:
        msg = conn.recv()
        if msg is None:
            return
        kind, sd, p = msg
        net.load_state_dict(_unpack(sd))
        rng = random.Random(p["seed"])
        t = time.time()
        if kind == "selfplay":
            ex, res, _ = play_games(p["games"], ev, p["sims"], rng, selfplay=True)
            conn.send((wid, kind, None, ex, res, None, time.time() - t))
        else:
            if p["opponent"] == "iteration_0":
                if base_net is None:
                    base_net = build_net(**net_kwargs).to(device).eval()
                    base_net.load_state_dict(_unpack(p["base_sd"]))
                    base_ev = make_evaluator(base_net, device)
                    base_rng = random.Random(p["seed"] + 1)

                def opp(s, _rng):
                    N = mcts([s], base_ev, p["sims"], base_rng, noise=False)[0]
                    return max(range(COLS), key=lambda c: (N[c], base_rng.random()))
            else:
                opp = opponents[p["opponent"]]
            _, res, mv = play_games(p["games"], ev, p["sims"], rng, selfplay=False, opponent=opp, record=True,
                                    policy_only=p["opponent"] == "policy_vs_one_ply")
            conn.send((wid, kind, p["opponent"], None, res, mv, time.time() - t))


# Runs train.py as __main__ in a worker. If the Compute SDK is not importable in a plain
# subprocess, a stub stands in: workers only need the game and network code.
_WORKER_BOOT = """
import runpy, sys, types
try:
    import compute
except ImportError:
    class _Any:
        def __getattr__(self, name): return self
        def __call__(self, *a, **k): return a[0] if len(a) == 1 and callable(a[0]) and not k else self
    stub = types.ModuleType("compute"); stub.__getattr__ = lambda name: _Any()
    sys.modules["compute"] = stub
script = sys.argv[1]
sys.argv = sys.argv[1:]
runpy.run_path(script, run_name="__main__")
"""


class Pool:
    """Worker processes started as `python train.py --az-worker ...`, talking over a local socket.

    Neither fork (the parent may already hold a CUDA context) nor multiprocessing's spawn (the Compute
    runner imports this file under a generated module name that children cannot re-import) works here.
    """

    def __init__(self, n, device, net_kwargs):
        import sys
        import threading
        from multiprocessing.connection import Listener
        key = os.urandom(16)
        self.listener = Listener(("127.0.0.1", 0), authkey=key)
        port = self.listener.address[1]
        script = os.path.abspath(__file__)
        self.logs = [Path(f"/tmp/az-worker-{i}.log") for i in range(n)]
        t = time.time()
        self.procs = [subprocess.Popen([sys.executable, "-c", _WORKER_BOOT, script, "--az-worker", str(port),
                                        key.hex(), str(i), device, json.dumps(net_kwargs)],
                                       stdout=open(self.logs[i], "w"),
                                       stderr=subprocess.STDOUT, cwd=os.path.dirname(script),
                                       env=dict(os.environ, PYTHONPATH=os.pathsep.join(p for p in sys.path if p)))
                      for i in range(n)]
        self.conns = [None] * n
        accepted = []

        def accept_all():
            for _ in range(n):
                accepted.append(self.listener.accept())

        threading.Thread(target=accept_all, daemon=True).start()
        while len(accepted) < n:
            time.sleep(0.5)
            self._check_alive()
            if time.time() - t > 300:
                raise TimeoutError(f"only {len(accepted)}/{n} workers connected in 300 s")
        for conn in accepted:
            tag, wid = conn.recv()
            assert tag == "ready"
            self.conns[wid] = conn
        self.startup_seconds = round(time.time() - t, 1)

    def _check_alive(self):
        dead = [i for i, p in enumerate(self.procs) if p.poll() is not None]
        if dead:
            for i in dead[:2]:
                print(f"--- worker {i} exited with {self.procs[i].returncode}:\n"
                      + self.logs[i].read_text()[-3000:], flush=True)
            raise RuntimeError(f"worker(s) {dead} exited")

    def run(self, jobs):
        """jobs: list of (kind, state_dict, params). Returns results in completion order."""
        from multiprocessing.connection import wait
        for i, job in enumerate(jobs):
            self.conns[i % len(self.conns)].send(job)
        out = []
        while len(out) < len(jobs):
            ready = wait(self.conns, timeout=10)
            for c in ready:
                out.append(c.recv())
            self._check_alive()
        return out

    def close(self):
        for c in self.conns:
            try:
                c.send(None)
            except OSError:
                pass
        for p in self.procs:
            try:
                p.wait(timeout=10)
            except subprocess.TimeoutExpired:
                p.kill()
        self.listener.close()


# --------------------------------------------------------------------------- #
# Checkpoints and resume
# --------------------------------------------------------------------------- #

def fetch_checkpoint(run_id: str, dest: Path) -> tuple[Path, dict]:
    """Install the Compute CLI on this machine and download `run_id`'s checkpoint artifact."""
    timings = {}
    key = os.environ.get("COMPUTE_API_KEY")
    if not key:
        raise RuntimeError("Resuming needs a Compute API key: `compute secrets set COMPUTE_API_KEY`")
    t = time.time()
    cli = shutil.which("compute")
    if cli is None or not os.environ.get("ALPHAZERO_LOCAL_CLI"):
        subprocess.run("curl -fsSL https://compute.cx/install.sh | sh", shell=True, check=True,
                       stdout=subprocess.DEVNULL)
        cli = str(Path.home() / ".local" / "bin" / "compute")
    timings["cli_install_seconds"] = round(time.time() - t, 1)
    t = time.time()
    listing = json.loads(subprocess.run([cli, "artifacts", "--json", "list", run_id], check=True,
                                        capture_output=True, text=True).stdout)
    items = [a for a in listing["items"] if a["name"] == ARTIFACT and a["status"] == "completed"]
    if not items:
        raise RuntimeError(f"{run_id} has no completed {ARTIFACT} artifact: {listing}")
    art = max(items, key=lambda a: a["version"])
    subprocess.run([cli, "artifacts", "get", run_id, art["artifact_id"], str(art["version"]), "--out", str(dest)],
                   check=True, stdout=subprocess.DEVNULL)
    timings["download_seconds"] = round(time.time() - t, 1)
    timings.update(run_id=run_id, artifact_id=art["artifact_id"], version=art["version"],
                   bytes=sum(f["size_bytes"] for f in art["files"]))
    return dest / "checkpoint.pt", timings


def _hf_token() -> str:
    token = os.environ.get("HF_TOKEN") or os.environ.get("hf") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    assert token, "No Hugging Face token; attach the hf secret"
    return token


# --------------------------------------------------------------------------- #
# Main loop
# --------------------------------------------------------------------------- #

def _train(iterations=40, games_per_iter=768, sims=64, eval_games=64, eval_sims=64, train_steps=300,
           batch_size=512, window=8, max_seconds=None, resume_run=None, resume_path=None, resume_hub=False,
           sample=False, workers=None, push=False, device=None, eval_every=1):
    import numpy as np
    import torch

    t0 = time.time()
    if sample:
        iterations, games_per_iter, eval_games, train_steps = min(iterations, 3), 64, 20, 50
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    # More processes than this mostly queue on the one GPU; each worker batches its own games.
    workers = workers or max(1, min(12, (os.cpu_count() or 2) - 2))
    import sys
    print("BOOT", json.dumps({"file": os.path.abspath(__file__), "exists": os.path.exists(__file__),
                              "python": sys.executable, "cwd": os.getcwd(), "workers": workers,
                              "device": device}), flush=True)
    net_kwargs = {"channels": 64, "blocks": 5}
    pool = Pool(workers, device, net_kwargs)

    out = Path(os.environ.get("COMPUTE_ARTIFACT_DIR", "/tmp/compute-artifacts")) / ARTIFACT
    out.mkdir(parents=True, exist_ok=True)
    (out / ".compute-artifact.json").write_text(json.dumps({
        "name": ARTIFACT, "version": 1, "kind": "checkpoint", "compatibility_key": "alphazero-c4-r5c64",
        "metadata": {"game": "connect4", "net": net_kwargs}}))

    torch.manual_seed(0)
    net = build_net(**net_kwargs).to(device)
    opt = torch.optim.AdamW(net.parameters(), lr=1e-3, weight_decay=1e-4)
    buffer: list[dict] = []
    history: list[dict] = []
    start_iter = 1
    resume_info = None
    base_sd = {k: v.detach().cpu().clone() for k, v in net.state_dict().items()}

    if resume_run or resume_path or resume_hub:
        if resume_run:
            path, resume_info = fetch_checkpoint(resume_run, Path("/tmp/alphazero-resume"))
        elif resume_hub:
            from huggingface_hub import hf_hub_download
            t = time.time()
            path = Path(hf_hub_download(HUB_REPO, "checkpoint/checkpoint.pt", token=_hf_token()))
            resume_info = {"hub_repo": HUB_REPO, "download_seconds": round(time.time() - t, 1)}
        else:
            path, resume_info = Path(resume_path), {"path": str(resume_path)}
        ck = torch.load(path, map_location="cpu", weights_only=False)
        net.load_state_dict(ck["model"])
        opt.load_state_dict(ck["optimizer"])
        base_sd = ck["iteration_0_model"]
        buffer = ck["buffer"]
        history = ck["history"]
        start_iter = ck["iteration"] + 1
        resume_info["resumed_after_iteration"] = ck["iteration"]
        print("RESUME", json.dumps(resume_info), flush=True)

    base_bytes = _pack(base_sd)
    params_count = sum(p.numel() for p in net.parameters())
    run_meta = {"device": torch.cuda.get_device_name() if device == "cuda" else "cpu", "workers": workers,
                "cpu_count": os.cpu_count(), "worker_startup_seconds": pool.startup_seconds, "parameters": params_count, "sims": sims, "eval_sims": eval_sims,
                "games_per_iter": games_per_iter, "train_steps": train_steps, "batch_size": batch_size,
                "window": window, "eval_games": eval_games, "sample": sample,
                "run_id": os.environ.get("COMPUTE_RUN_ID"), "resume": resume_info,
                "start_iteration": start_iter}
    print("START", json.dumps(run_meta), flush=True)

    def cpu_sd():
        return {k: v.detach().cpu() for k, v in net.state_dict().items()}

    def evaluate_all(it):
        sd = _pack(cpu_sd())
        row = {}
        per = max(2, eval_games // workers // 2 * 2)
        for name in OPPONENTS:
            jobs, left, seed = [], eval_games, 0
            while left > 0:
                n = min(per, left)
                p = {"games": n, "sims": eval_sims, "seed": 10_000 * it + 100 * seed + OPP_SEED[name],
                     "opponent": name}
                if name == "iteration_0":
                    p["base_sd"] = base_bytes
                jobs.append(("eval", sd, p))
                left -= n
                seed += 1
            t = time.time()
            res = pool.run(jobs)
            flat = [r for (_, _, _, _, rs, _, _) in res for r in rs]
            games = [m for (_, _, _, _, _, mv, _) in res for m in mv]
            w, d, l = flat.count(1), flat.count(0), flat.count(-1)
            row[name] = {"won": w, "drawn": d, "lost": l, "games": len(flat),
                         "score": round((w + 0.5 * d) / len(flat), 4), "seconds": round(time.time() - t, 1),
                         "sample_games": games[:4]}
        return row

    def save(it):
        ck = {"model": cpu_sd(), "optimizer": opt.state_dict(), "iteration_0_model": base_sd,
              "buffer": buffer, "history": history, "iteration": it, "net": net_kwargs}
        tmp = out / "checkpoint.pt.tmp"
        torch.save(ck, tmp)
        tmp.replace(out / "checkpoint.pt")
        (out / "history.json").write_text(json.dumps({"meta": run_meta, "history": history}, indent=1))
        if push:  # a second copy off the machine, so resuming does not depend on artifacts alone
            from huggingface_hub import HfApi
            api = HfApi(token=_hf_token())
            api.create_repo(HUB_REPO, exist_ok=True)
            api.upload_folder(repo_id=HUB_REPO, folder_path=str(out), path_in_repo="checkpoint",
                              allow_patterns=["checkpoint.pt", "history.json"],
                              commit_message=f"Checkpoint after iteration {it}")

    if start_iter == 1:
        row = {"iteration": 0, "eval": evaluate_all(0), "elapsed": round(time.time() - t0, 1)}
        history.append(row)
        print("ITER", json.dumps({k: (v if k != "eval" else {n: e["score"] for n, e in v.items()})
                                   for k, v in row.items()}), flush=True)
        save(0)

    stop_reason = "completed"
    for it in range(start_iter, iterations + 1):
        if max_seconds and time.time() - t0 > max_seconds:
            stop_reason = "max_seconds"
            break
        ti = time.time()
        # 1. self-play
        sd = _pack(cpu_sd())
        per = max(1, games_per_iter // workers)
        jobs = [("selfplay", sd, {"games": per, "sims": sims, "seed": 1_000_000 + 1000 * it + w})
                for w in range(workers)]
        res = pool.run(jobs)
        examples = [e for r in res for e in r[3]]
        winners = [x for r in res for x in r[4]]
        sp_seconds = time.time() - ti
        boards = np.frombuffer(b"".join(e[0] for e in examples), np.uint8).reshape(-1, ROWS * COLS)
        tm = np.array([e[1] for e in examples], np.uint8)
        pis = np.array([e[2] for e in examples], np.float32)
        zs = np.array([e[3] for e in examples], np.int8)
        # relative encoding: 1 = player to move, 2 = opponent
        rel = np.where(boards == 0, 0, np.where(boards == tm[:, None], 1, 2)).astype(np.uint8)
        buffer.append({"boards": rel, "pis": pis.astype(np.float16), "z": zs})
        buffer = buffer[-window:]

        # 2. train
        tt = time.time()
        B = np.concatenate([b["boards"] for b in buffer])
        P = np.concatenate([b["pis"] for b in buffer]).astype(np.float32)
        Z = np.concatenate([b["z"] for b in buffer]).astype(np.float32)
        net.train()
        losses = []
        g = torch.Generator().manual_seed(it)
        for _ in range(train_steps):
            idx = torch.randint(0, len(B), (batch_size,), generator=g).numpy()
            b = B[idx].reshape(-1, ROWS, COLS)
            p = P[idx]
            flip = np.random.default_rng(it * 7919 + len(losses)).random(batch_size) < 0.5
            b = np.where(flip[:, None, None], b[:, :, ::-1], b)
            p = np.where(flip[:, None], p[:, ::-1], p)
            x = np.stack([b == 1, b == 2], 1).astype(np.float32)
            x = torch.from_numpy(x).to(device)
            pt = torch.from_numpy(np.ascontiguousarray(p)).to(device)
            zt = torch.from_numpy(Z[idx]).to(device)
            logits, v = net(x)
            lp = -(pt * torch.log_softmax(logits, -1)).sum(-1).mean()
            lv = ((v - zt) ** 2).mean()
            loss = lp + lv
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            losses.append((float(lp.detach()), float(lv.detach())))
        net.eval()
        train_seconds = time.time() - tt

        # 3. evaluate (every eval_every iterations, and always on the last one)
        te = time.time()
        ev = evaluate_all(it) if (it % eval_every == 0 or it == iterations) else None
        row = {"iteration": it,
               "selfplay": {"games": len(winners), "positions": len(examples), "seconds": round(sp_seconds, 1),
                            "first_player_wins": winners.count(1), "second_player_wins": winners.count(2),
                            "draws": winners.count(3), "mean_plies": round(len(examples) / len(winners), 1)},
               "train": {"steps": train_steps, "buffer_positions": int(len(B)), "seconds": round(train_seconds, 1),
                         "policy_loss": round(sum(a for a, _ in losses[-50:]) / len(losses[-50:]), 4),
                         "value_loss": round(sum(b for _, b in losses[-50:]) / len(losses[-50:]), 4)},
               "eval": ev, "eval_seconds": round(time.time() - te, 1),
               "iteration_seconds": round(time.time() - ti, 1), "elapsed": round(time.time() - t0, 1),
               "run_id": run_meta["run_id"]}
        history.append(row)
        # 4. checkpoint every iteration so a timeout loses at most one iteration
        save(it)
        print("ITER", json.dumps({"iteration": it, "score": {n: e["score"] for n, e in ev.items()} if ev else None,
                                   "selfplay_s": row["selfplay"]["seconds"], "train_s": row["train"]["seconds"],
                                   "eval_s": row["eval_seconds"], "loss": [row["train"]["policy_loss"],
                                   row["train"]["value_loss"]], "elapsed": row["elapsed"]}), flush=True)

    pool.close()
    last = next(r for r in reversed(history) if r["eval"])
    summary = {"stop_reason": stop_reason, "last_iteration": history[-1]["iteration"],
               "last_evaluated_iteration": last["iteration"],
               "last_scores": {n: e["score"] for n, e in last["eval"].items()},
               "elapsed_seconds": round(time.time() - t0, 1), "meta": run_meta, "hub_repo": None}
    if push:
        from huggingface_hub import HfApi
        token = _hf_token()
        pub = Path("/tmp/alphazero-publish")
        pub.mkdir(exist_ok=True)
        torch.save({"model": cpu_sd(), "net": net_kwargs, "iteration": history[-1]["iteration"]}, pub / "model.pt")
        shutil.copy2(out / "history.json", pub / "history.json")
        shutil.copy2(Path(__file__), pub / "train.py")
        (pub / "README.md").write_text(
            "---\nlicense: mit\ntags:\n- alphazero\n- connect-four\n- reinforcement-learning\n---\n"
            "# AlphaZero Connect Four, trained from scratch\n\n"
            f"Residual policy-value network ({params_count:,} parameters) after {history[-1]['iteration']} "
            "self-play iterations. Load with `build_net()` from `train.py`. Scores against a random mover, "
            "a one-ply win/block heuristic and the untrained network are in `history.json`.\n\n"
            f"Final-iteration scores (win = 1, draw = 0.5): `{json.dumps(summary['last_scores'])}`\n\n"
            "Guide: https://letsusecompute.com/posts/alphazero-small\n")
        api = HfApi(token=token)
        api.create_repo(HUB_REPO, exist_ok=True)
        api.upload_folder(repo_id=HUB_REPO, folder_path=str(pub), commit_message=f"Iteration {history[-1]['iteration']}")
        summary["hub_repo"] = HUB_REPO
    print("RESULT", json.dumps(summary), flush=True)
    return summary


hub_image = image.pip_install("huggingface_hub==0.30.2")


@app.function(gpu="runpod/H100-SXM", image=image, timeout=7200)
def train(iterations: int = 40, max_seconds: int = 0, sample: bool = False) -> dict:
    """Fresh start. Saves the checkpoint artifact after every iteration."""
    return _train(iterations=iterations, max_seconds=max_seconds or None, sample=sample)


@app.function(gpu="runpod/H100-SXM", image=hub_image, timeout=7200, secrets=[hf_secret])
def train_and_push(iterations: int = 40, max_seconds: int = 0, sample: bool = False) -> dict:
    """Fresh start; also pushes each checkpoint and the final network to Hugging Face."""
    return _train(iterations=iterations, max_seconds=max_seconds or None, sample=sample, push=True)


@app.function(gpu="runpod/H100-SXM", image=image, timeout=7200, secrets=[api_key])
def resume(run_id: str, iterations: int = 40, max_seconds: int = 0, eval_every: int = 1) -> dict:
    """Continue from run_id's latest checkpoint artifact."""
    return _train(iterations=iterations, max_seconds=max_seconds or None, resume_run=run_id, eval_every=eval_every)


@app.function(gpu="runpod/H100-SXM", image=hub_image, timeout=7200, secrets=[api_key, hf_secret])
def resume_and_push(run_id: str, iterations: int = 40, max_seconds: int = 0, eval_every: int = 1) -> dict:
    """Continue from run_id's checkpoint artifact; push checkpoints and the final network."""
    return _train(iterations=iterations, max_seconds=max_seconds or None, resume_run=run_id, push=True,
                  eval_every=eval_every)


@app.function(gpu="runpod/H100-SXM", image=hub_image, timeout=7200, secrets=[hf_secret])
def resume_from_hub(iterations: int = 40, max_seconds: int = 0, eval_every: int = 1) -> dict:
    """Continue from the checkpoint a *_and_push run left on Hugging Face (no Compute API key needed)."""
    return _train(iterations=iterations, max_seconds=max_seconds or None, resume_hub=True, push=True,
                  eval_every=eval_every)


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "--az-worker":
        from multiprocessing.connection import Client
        _port, _key, _wid, _device, _kw = sys.argv[2:7]
        _worker(int(_wid), Client(("127.0.0.1", int(_port)), authkey=bytes.fromhex(_key)), _device, json.loads(_kw))
        sys.exit(0)
    # Local CPU smoke test: python train.py [resume_path]
    rp = sys.argv[1] if len(sys.argv) > 1 else None
    print(_train(iterations=2 if rp is None else 3, games_per_iter=24, sims=16, eval_games=8, eval_sims=16,
                 train_steps=20, batch_size=64, workers=3, device="cpu", resume_path=rp))
