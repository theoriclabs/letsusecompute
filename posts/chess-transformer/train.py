"""Train a small GPT on chess SAN moves from public Lichess games.

Preferred dataset: ``Lichess/standard-chess-games`` (one recent month, streamed).
Fallback: ``adamkarvonen/chess_games`` ``lichess_100mb.zip``.

The model never sees the rules — only games. After training it is scored on
next-move top-1, unmasked legal-move rate, and matches vs a random mover and
Stockfish skill 1.

    compute run train.py::train --gpu cheap --dry-run
    compute run train.py::train --gpu cheap --timeout 1800 --wait --args '{"sample": true}'
"""

from __future__ import annotations

import json
import math
import os
import random
import re
import shutil
import sys
import tarfile
import tempfile
import time
import urllib.request
from collections.abc import Iterator, Mapping
from pathlib import Path
from typing import Any

import compute

app = compute.App("chess-transformer")
# Pin cu124 wheels: many Vast consumer hosts still expose driver 12.4
# (CUDA_VERSION 12040). Unpinned torch can refuse to initialize on that driver.
image = compute.Image.cuda_pytorch().pip_install(
    "--extra-index-url",
    "https://download.pytorch.org/whl/cu124",
    "torch==2.5.1+cu124",
    "datasets",
    "huggingface_hub",
    "chess",
)

# Stored with `compute secrets set hf`. Injected only when a function lists
# `secrets=[hf_secret]`. Training the public dataset does not need it.
hf_secret = compute.Secret.from_name("hf")

PRIMARY_DATASET = "Lichess/standard-chess-games"
FALLBACK_DATASET = "adamkarvonen/chess_games"
FALLBACK_FILE = "lichess_100mb.zip"
DEFAULT_HUB_REPO = "theoriclabs/chess-move-transformer"
WORKLOAD_SUBDIR = "chess-transformer"
ARTIFACT_NAME = "chess-move-transformer"
ARTIFACT_MARKER = ".compute-artifact.json"
DEFAULT_ARTIFACT_FALLBACK = Path("/tmp/compute-chess-transformer")

PAD_TOK, BOS_TOK, EOS_TOK, UNK_TOK = "<pad>", "<bos>", "<eos>", "<unk>"
SPECIAL_TOKENS = (PAD_TOK, BOS_TOK, EOS_TOK, UNK_TOK)
DROP_TERMINATIONS = frozenset({"abandoned", "rules infraction"})
STOCKFISH_LIMIT_S = 0.02
STOCKFISH_AVX2_URL = (
    "https://github.com/official-stockfish/Stockfish/releases/download/"
    "sf_17.1/stockfish-ubuntu-x86-64-avx2.tar"
)
STOCKFISH_X86_URL = (
    "https://github.com/official-stockfish/Stockfish/releases/download/"
    "sf_17.1/stockfish-ubuntu-x86-64.tar"
)
MAX_MATCH_PLIES = 300

_COMMENT_RE = re.compile(r"\{[^{}]*\}")
_LINE_COMMENT_RE = re.compile(r";[^\n]*")
_VARIATION_RE = re.compile(r"\([^()]*\)")
_NAG_DOLLAR_RE = re.compile(r"\$\d+")
_MOVE_NUM_RE = re.compile(r"\b\d+\.+")
_RESULT_RE = re.compile(r"\b(?:1-0|0-1|1/2-1/2|\*)\b")
_NAG_SUFFIX_RE = re.compile(r"[!?]+")
_YEAR_MONTH_RE = re.compile(r"data/year=(\d{4})/month=(\d{2})/")


def _bridge_hf_token() -> None:
    token = (
        os.environ.get("HF_TOKEN")
        or os.environ.get("HUGGING_FACE_HUB_TOKEN")
        or os.environ.get("hf")
    )
    if token:
        os.environ.setdefault("HF_TOKEN", token)
        os.environ.setdefault("HUGGING_FACE_HUB_TOKEN", token)


def write_artifact_marker(
    directory: Path | str,
    *,
    name: str,
    kind: str,
    compatibility_key: str,
    metadata: dict[str, Any],
) -> Path:
    """Atomically write ``.compute-artifact.json`` (temp + flush + fsync + replace)."""
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
    fd, tmp_name = tempfile.mkstemp(
        prefix=".compute-artifact.",
        suffix=".tmp",
        dir=str(directory),
    )
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


def write_loss_curve(directory: Path, history: list[dict[str, float]]) -> Path:
    """Write JSON + a tiny SVG of train/test loss. No extra deps."""
    directory.mkdir(parents=True, exist_ok=True)
    json_path = directory / "loss_curve.json"
    json_path.write_text(json.dumps(history, indent=2) + "\n", encoding="utf-8")

    width, height, pad = 720, 280, 36
    losses = [row["train_loss"] for row in history] + [row["test_loss"] for row in history]
    lo = min(losses) if losses else 0.0
    hi = max(losses) if losses else 1.0
    if hi <= lo:
        hi = lo + 1e-6

    def xy(index: int, value: float, count: int) -> str:
        x = pad + (width - 2 * pad) * (index / max(count - 1, 1))
        y = pad + (height - 2 * pad) * (1 - (value - lo) / (hi - lo))
        return f"{x:.1f},{y:.1f}"

    def polyline(key: str) -> str:
        count = len(history)
        pts = " ".join(xy(i, row[key], count) for i, row in enumerate(history))
        return pts

    svg = f"""<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">
  <rect width="100%" height="100%" fill="#fff"/>
  <text x="{pad}" y="20" font-size="13" fill="#333">train loss (steel) vs test loss (orange)</text>
  <polyline fill="none" stroke="#3d6b8a" stroke-width="2" points="{polyline("train_loss")}"/>
  <polyline fill="none" stroke="#c45c26" stroke-width="2" points="{polyline("test_loss")}"/>
  <text x="{pad}" y="{height - 8}" font-size="11" fill="#666">min {lo:.3f} · max {hi:.3f} · {len(history)} epochs</text>
</svg>
"""
    (directory / "loss_curve.svg").write_text(svg, encoding="utf-8")
    return json_path


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


def extract_san(movetext: str) -> list[str]:
    """Strip clocks, evals, NAGs, move numbers, and the result; keep SAN tokens."""
    if not movetext or not str(movetext).strip():
        return []
    text = str(movetext)
    while True:
        new = _COMMENT_RE.sub(" ", text)
        if new == text:
            break
        text = new
    text = _LINE_COMMENT_RE.sub(" ", text)
    while True:
        new = _VARIATION_RE.sub(" ", text)
        if new == text:
            break
        text = new
    text = _NAG_DOLLAR_RE.sub(" ", text)
    text = _RESULT_RE.sub(" ", text)
    text = _MOVE_NUM_RE.sub(" ", text)
    tokens: list[str] = []
    for tok in text.split():
        tok = _NAG_SUFFIX_RE.sub("", tok).strip()
        if tok:
            tokens.append(tok)
    return tokens


def _as_int(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _termination_ok(value: Any) -> bool:
    if value is None or value == "":
        return True
    return str(value).strip().lower() not in DROP_TERMINATIONS


def _row_movetext(row: Mapping[str, Any]) -> str:
    for key in ("movetext", "transcript", "pgn", "moves"):
        val = row.get(key)
        if val:
            return str(val)
    return ""


def _keep_game(row: Mapping[str, Any], min_elo: int, min_plies: int, block_size: int) -> dict[str, Any] | None:
    white = _as_int(row.get("WhiteElo") if "WhiteElo" in row else row.get("white_elo"))
    black = _as_int(row.get("BlackElo") if "BlackElo" in row else row.get("black_elo"))
    if white is None or black is None or white < min_elo or black < min_elo:
        return None
    if not _termination_ok(row.get("Termination") if "Termination" in row else row.get("termination")):
        return None
    movetext = _row_movetext(row)
    if not movetext.strip():
        return None
    moves = extract_san(movetext)
    if len(moves) < min_plies:
        return None
    truncated = len(moves) > block_size
    if truncated:
        moves = moves[:block_size]
    return {
        "moves": moves,
        "truncated": truncated,
        "white_elo": white,
        "black_elo": black,
        "result": str(row.get("Result") or row.get("result") or "*"),
    }


def _latest_month_files(repo_id: str) -> tuple[str, str, list[str]]:
    from huggingface_hub import list_repo_files

    files = list_repo_files(repo_id, repo_type="dataset")
    months: dict[tuple[str, str], list[str]] = {}
    for path in files:
        match = _YEAR_MONTH_RE.search(path)
        if match and path.endswith(".parquet"):
            months.setdefault((match.group(1), match.group(2)), []).append(path)
    if not months:
        raise FileNotFoundError(f"no hive-partitioned parquet files in {repo_id}")
    year, month = max(months)
    return year, month, sorted(months[(year, month)])


def _iter_primary() -> tuple[str, Iterator[dict[str, Any]]]:
    from datasets import load_dataset

    year, month, files = _latest_month_files(PRIMARY_DATASET)
    pattern = f"data/year={year}/month={month}/*.parquet"
    print(
        f"primary dataset={PRIMARY_DATASET} year={year} month={month} shards={len(files)}",
        flush=True,
    )
    ds = load_dataset(
        PRIMARY_DATASET,
        data_files=pattern,
        split="train",
        streaming=True,
    )
    label = f"{PRIMARY_DATASET} year={year} month={month}"
    return label, ds


def _iter_fallback() -> tuple[str, Iterator[dict[str, Any]]]:
    from datasets import load_dataset

    print(f"fallback dataset={FALLBACK_DATASET} file={FALLBACK_FILE}", flush=True)
    ds = load_dataset(
        FALLBACK_DATASET,
        data_files=FALLBACK_FILE,
        split="train",
        streaming=True,
    )
    return f"{FALLBACK_DATASET}/{FALLBACK_FILE}", ds


def collect_games(
    *,
    max_positions: int,
    min_elo: int,
    min_plies: int,
    block_size: int,
) -> tuple[list[dict[str, Any]], str, dict[str, Any]]:
    """Stream filtered games until train+holdout plies cover ``max_positions``."""
    t0 = time.time()
    target = int(max_positions / 0.98) + min_plies
    used = ""
    source = None
    primary_error = None
    try:
        used, source = _iter_primary()
    except Exception as err:  # noqa: BLE001 — fallback is the product test
        primary_error = f"{type(err).__name__}: {err}"
        print(f"primary dataset unavailable ({primary_error}); falling back", flush=True)
        used, source = _iter_fallback()

    games: list[dict[str, Any]] = []
    scanned = 0
    kept = 0
    positions = 0
    t_filter = time.time()
    try:
        for row in source:
            scanned += 1
            game = _keep_game(row, min_elo=min_elo, min_plies=min_plies, block_size=block_size)
            if game is None:
                continue
            games.append(game)
            kept += 1
            positions += len(game["moves"])
            if positions >= target:
                break
            if kept % 500 == 0:
                print(
                    f"collected games={kept} positions={positions} scanned={scanned}",
                    flush=True,
                )
    except Exception as err:  # noqa: BLE001
        if not games:
            if used.startswith(PRIMARY_DATASET):
                print(f"primary stream failed ({err}); falling back", flush=True)
                used, source = _iter_fallback()
                scanned = kept = positions = 0
                games = []
                for row in source:
                    scanned += 1
                    game = _keep_game(row, min_elo=min_elo, min_plies=min_plies, block_size=block_size)
                    if game is None:
                        continue
                    games.append(game)
                    kept += 1
                    positions += len(game["moves"])
                    if positions >= target:
                        break
            else:
                raise

    if not games:
        raise RuntimeError("no games passed the Elo/length/termination filter")

    timings = {
        "download_and_scan_s": round(time.time() - t0, 3),
        "filter_s": round(time.time() - t_filter, 3),
        "scanned_rows": scanned,
        "kept_games": kept,
        "positions_before_split": positions,
        "primary_error": primary_error,
    }
    print(
        f"dataset={used} games={kept} positions={positions} "
        f"scanned={scanned} seconds={timings['download_and_scan_s']:.1f}",
        flush=True,
    )
    return games, used, timings


def build_vocab(games: list[dict[str, Any]]) -> dict[str, Any]:
    tokens = set()
    for game in games:
        tokens.update(game["moves"])
    itos = list(SPECIAL_TOKENS) + sorted(tokens)
    stoi = {tok: i for i, tok in enumerate(itos)}
    return {"itos": itos, "stoi": stoi}


def encode_game(moves: list[str], stoi: dict[str, int], block_size: int) -> tuple[list[int], list[int]]:
    unk = stoi[UNK_TOK]
    bos = stoi[BOS_TOK]
    ids = [bos] + [stoi.get(move, unk) for move in moves[:block_size]]
    x = ids[:-1][:block_size]
    y = ids[1:][:block_size]
    pad = stoi[PAD_TOK]
    x = x + [pad] * (block_size - len(x))
    y = y + [pad] * (block_size - len(y))
    return x, y


def build_model(vocab_size: int, n_embd: int, n_head: int, n_layer: int, block_size: int, dropout: float = 0.1):
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    if n_embd % n_head != 0:
        raise ValueError(f"n_embd={n_embd} must be divisible by n_head={n_head}")

    class CausalSelfAttention(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.n_head = n_head
            self.n_embd = n_embd
            self.c_attn = nn.Linear(n_embd, 3 * n_embd)
            self.c_proj = nn.Linear(n_embd, n_embd)
            self.resid_dropout = nn.Dropout(dropout)

        def forward(self, x):
            batch, steps, width = x.size()
            q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
            head = width // self.n_head
            q = q.view(batch, steps, self.n_head, head).transpose(1, 2)
            k = k.view(batch, steps, self.n_head, head).transpose(1, 2)
            v = v.view(batch, steps, self.n_head, head).transpose(1, 2)
            out = F.scaled_dot_product_attention(q, k, v, is_causal=True)
            out = out.transpose(1, 2).contiguous().view(batch, steps, width)
            return self.resid_dropout(self.c_proj(out))

    class Block(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.ln1 = nn.LayerNorm(n_embd)
            self.attn = CausalSelfAttention()
            self.ln2 = nn.LayerNorm(n_embd)
            self.mlp = nn.Sequential(
                nn.Linear(n_embd, 4 * n_embd),
                nn.GELU(),
                nn.Linear(4 * n_embd, n_embd),
                nn.Dropout(dropout),
            )

        def forward(self, x):
            x = x + self.attn(self.ln1(x))
            x = x + self.mlp(self.ln2(x))
            return x

    class ChessGPT(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.block_size = block_size
            self.tok = nn.Embedding(vocab_size, n_embd)
            self.pos = nn.Embedding(block_size, n_embd)
            self.drop = nn.Dropout(dropout)
            self.blocks = nn.ModuleList([Block() for _ in range(n_layer)])
            self.ln_f = nn.LayerNorm(n_embd)
            self.lm_head = nn.Linear(n_embd, vocab_size, bias=False)
            self.lm_head.weight = self.tok.weight
            self.apply(self._init_weights)
            for name, param in self.named_parameters():
                if name.endswith("c_proj.weight") or name.endswith("mlp.2.weight"):
                    nn.init.normal_(param, mean=0.0, std=0.02 / math.sqrt(2 * n_layer))

        @staticmethod
        def _init_weights(module) -> None:
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)

        def forward(self, idx):
            _batch, steps = idx.shape
            if steps > self.block_size:
                idx = idx[:, -self.block_size :]
                steps = idx.shape[1]
            pos = torch.arange(0, steps, device=idx.device)
            x = self.drop(self.tok(idx) + self.pos(pos))
            for block in self.blocks:
                x = block(x)
            return self.lm_head(self.ln_f(x))

    return ChessGPT()


def _http_download(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    request = urllib.request.Request(url, headers={"User-Agent": "letsusecompute-chess-transformer/1.0"})
    with urllib.request.urlopen(request, timeout=180) as response, dest.open("wb") as handle:
        shutil.copyfileobj(response, handle)


def _extract_stockfish_binary(archive: Path, dest_dir: Path) -> Path | None:
    dest_dir.mkdir(parents=True, exist_ok=True)
    with tarfile.open(archive) as tar:
        try:
            tar.extractall(dest_dir, filter="data")
        except TypeError:
            tar.extractall(dest_dir)
    candidates = [p for p in dest_dir.rglob("*") if p.is_file() and os.access(p, os.X_OK)]
    named = [p for p in candidates if "stockfish" in p.name.lower() and p.suffix == ""]
    pick = named[0] if named else (candidates[0] if candidates else None)
    return pick


def resolve_stockfish() -> tuple[Any, str]:
    """Return ``(engine_or_None, status_string)``. Never raises on a missing binary."""
    import chess.engine

    which = shutil.which("stockfish")
    path_candidates = [which] if which else []
    for extra in ("/usr/games/stockfish", "/usr/bin/stockfish", "/usr/local/bin/stockfish"):
        if extra not in path_candidates and Path(extra).is_file():
            path_candidates.append(extra)

    for path in path_candidates:
        try:
            engine = chess.engine.SimpleEngine.popen_uci(path)
            engine.configure({"Skill Level": 1})
            return engine, f"path:{path}"
        except Exception as err:  # noqa: BLE001
            print(f"stockfish at {path} failed: {err}", flush=True)

    if sys.platform != "linux":
        return None, f"unavailable (platform {sys.platform}; Image has no apt API, skipped linux binary)"

    dest = Path(tempfile.mkdtemp(prefix="stockfish-"))
    for label, url in (("avx2", STOCKFISH_AVX2_URL), ("x86-64", STOCKFISH_X86_URL)):
        try:
            archive = dest / f"stockfish-{label}.tar"
            print(f"downloading Stockfish 17.1 {label}", flush=True)
            _http_download(url, archive)
            binary = _extract_stockfish_binary(archive, dest / label)
            if binary is None:
                continue
            binary.chmod(0o755)
            engine = chess.engine.SimpleEngine.popen_uci(str(binary))
            engine.configure({"Skill Level": 1})
            return engine, f"downloaded:sf_17.1:{label}:{binary.name}"
        except Exception as err:  # noqa: BLE001
            print(f"stockfish {label} install failed: {err}", flush=True)
    return None, "unavailable (no PATH binary; avx2 and x86-64 downloads failed)"


def _special_ids(stoi: dict[str, int]) -> set[int]:
    return {stoi[tok] for tok in SPECIAL_TOKENS}


def _token_of(itos: list[str], idx: int) -> str:
    if 0 <= idx < len(itos):
        return itos[idx]
    return UNK_TOK


def _parse_legal(board, san: str) -> bool:
    if not san or san in SPECIAL_TOKENS:
        return False
    try:
        board.parse_san(san)
        return True
    except Exception:
        return False


def _highest_legal(logits, board, itos: list[str], stoi: dict[str, int]) -> str:
    legal = [board.san(move) for move in board.legal_moves]
    if not legal:
        raise RuntimeError("no legal moves")
    best_san = legal[0]
    best_val = float("-inf")
    for san in legal:
        idx = stoi.get(san)
        if idx is None:
            continue
        val = float(logits[idx])
        if val > best_val:
            best_val = val
            best_san = san
    return best_san


def _top_k(probs, itos: list[str], k: int = 3) -> list[dict[str, Any]]:
    import torch

    values, indices = torch.topk(probs, k=min(k, probs.numel()))
    out = []
    for val, idx in zip(values.tolist(), indices.tolist()):
        out.append({"move": _token_of(itos, int(idx)), "p": float(val)})
    return out


def _model_logits(model, prefix: list[int], device):
    import torch

    if not prefix:
        raise ValueError("empty prefix")
    idx = torch.tensor([prefix[-model.block_size :]], dtype=torch.long, device=device)
    with torch.no_grad():
        logits = model(idx)[0, -1]
    return logits


def eval_top1_and_legal(
    model,
    games: list[dict[str, Any]],
    stoi: dict[str, int],
    itos: list[str],
    device,
    legal_eval_positions: int,
) -> dict[str, Any]:
    import chess
    import torch
    import torch.nn.functional as F

    model.eval()
    pad = stoi[PAD_TOK]
    block_size = model.block_size
    correct = 0
    total = 0
    sampled_legal = 0
    sampled_n = 0
    argmax_legal = 0
    argmax_n = 0
    special = _special_ids(stoi)

    with torch.no_grad():
        for game in games:
            moves = game["moves"]
            if not moves:
                continue
            x, y = encode_game(moves, stoi, block_size)
            length = min(len(moves), block_size)
            idx = torch.tensor([x[:length]], dtype=torch.long, device=device)
            logits = model(idx)[0]
            targets = torch.tensor(y[:length], dtype=torch.long, device=device)
            pred = logits.argmax(dim=-1)
            mask = targets != pad
            correct += int((pred[mask] == targets[mask]).sum().item())
            total += int(mask.sum().item())

            if sampled_n >= legal_eval_positions:
                continue
            board = chess.Board()
            prefix = [stoi[BOS_TOK]]
            for t, move in enumerate(moves[:length]):
                if sampled_n >= legal_eval_positions:
                    break
                step_logits = logits[t]
                probs = F.softmax(step_logits.float(), dim=-1)
                sampled = int(torch.multinomial(probs, num_samples=1).item())
                argmax = int(step_logits.argmax().item())
                sampled_n += 1
                argmax_n += 1
                if sampled not in special and _parse_legal(board, _token_of(itos, sampled)):
                    sampled_legal += 1
                if argmax not in special and _parse_legal(board, _token_of(itos, argmax)):
                    argmax_legal += 1
                try:
                    board.push_san(move)
                except Exception:
                    break
                prefix.append(stoi.get(move, stoi[UNK_TOK]))

    return {
        "top1_acc": (correct / total) if total else 0.0,
        "top1_n": total,
        "legal_rate_sampled": (sampled_legal / sampled_n) if sampled_n else 0.0,
        "legal_rate_argmax": (argmax_legal / argmax_n) if argmax_n else 0.0,
        "legal_eval_n": sampled_n,
    }


def play_matches(
    model,
    opponent: str,
    n_games: int,
    stoi: dict[str, int],
    itos: list[str],
    device,
    engine,
    seed: int,
    annotate: bool,
) -> dict[str, Any]:
    import chess
    import chess.pgn
    import torch
    import torch.nn.functional as F

    rng = random.Random(seed)
    model.eval()
    wins = draws = losses = 0
    illegal_attempts = 0
    games_pgn: list[str] = []
    annotated: dict[str, Any] | None = None
    stockfish_limit = chess.engine.Limit(time=STOCKFISH_LIMIT_S)

    def opponent_move(board: chess.Board):
        if opponent == "random":
            return rng.choice(list(board.legal_moves))
        if engine is None:
            raise RuntimeError("stockfish engine missing")
        result = engine.play(board, stockfish_limit)
        return result.move

    for game_i in range(n_games):
        board = chess.Board()
        model_is_white = game_i % 2 == 0
        prefix = [stoi[BOS_TOK]]
        record_plies: list[dict[str, Any]] = []
        pgn_game = chess.pgn.Game()
        pgn_game.headers["Event"] = f"chess-transformer vs {opponent}"
        pgn_game.headers["White"] = "model" if model_is_white else opponent
        pgn_game.headers["Black"] = opponent if model_is_white else "model"
        node = pgn_game
        ply = 0
        while not board.is_game_over(claim_draw=True) and ply < MAX_MATCH_PLIES:
            ply += 1
            model_turn = board.turn == chess.WHITE if model_is_white else board.turn == chess.BLACK
            if model_turn:
                logits = _model_logits(model, prefix, device)
                probs = F.softmax(logits.float(), dim=-1)
                argmax = int(logits.argmax().item())
                san = _token_of(itos, argmax)
                legal = _parse_legal(board, san)
                played = san
                if not legal:
                    illegal_attempts += 1
                    played = _highest_legal(logits, board, itos, stoi)
                move = board.parse_san(played)
                if annotate and annotated is None:
                    record_plies.append(
                        {
                            "ply": ply,
                            "mover": "model",
                            "played": played,
                            "argmax": san,
                            "legal": legal,
                            "prob": float(probs[argmax].item()) if legal else float(probs[stoi.get(played, 0)].item()) if played in stoi else None,
                            "top3": _top_k(probs, itos, 3),
                        }
                    )
            else:
                move = opponent_move(board)
                played = board.san(move)
                if annotate and annotated is None:
                    record_plies.append({"ply": ply, "mover": opponent, "played": played})
            board.push(move)
            node = node.add_variation(move)
            prefix.append(stoi.get(played, stoi[UNK_TOK]))

        if not board.is_game_over(claim_draw=True):
            outcome_result = "1/2-1/2"
        else:
            outcome = board.outcome(claim_draw=True)
            outcome_result = outcome.result() if outcome else "1/2-1/2"
        pgn_game.headers["Result"] = outcome_result
        games_pgn.append(str(pgn_game) + "\n")

        if outcome_result == "1/2-1/2":
            draws += 1
            model_result = "draw"
        elif (outcome_result == "1-0" and model_is_white) or (outcome_result == "0-1" and not model_is_white):
            wins += 1
            model_result = "win"
        else:
            losses += 1
            model_result = "loss"

        if annotate and annotated is None:
            annotated = {
                "opponent": opponent,
                "model_color": "white" if model_is_white else "black",
                "result": outcome_result,
                "model_result": model_result,
                "plies": record_plies,
            }

    n = max(n_games, 1)
    return {
        "games": n_games,
        "wins": wins,
        "draws": draws,
        "losses": losses,
        "win_rate": wins / n if n_games else 0.0,
        "illegal_attempts": illegal_attempts,
        "pgn": "".join(games_pgn),
        "annotated": annotated,
    }


def _cosine_lr(step: int, total_steps: int, lr: float, warmup: int) -> float:
    if total_steps <= 0:
        return lr
    if step < warmup:
        return lr * (step + 1) / max(warmup, 1)
    progress = (step - warmup) / max(total_steps - warmup, 1)
    return lr * 0.5 * (1.0 + math.cos(math.pi * min(1.0, max(0.0, progress))))


def _apply_sample_preset(kwargs: dict[str, Any]) -> dict[str, Any]:
    if not kwargs.get("sample"):
        return kwargs
    kwargs["n_layer"] = 4
    kwargs["n_embd"] = 256
    kwargs["n_head"] = 4
    kwargs["max_positions"] = 100_000
    kwargs["games_vs_random"] = 20
    kwargs["games_vs_stockfish"] = 20
    if kwargs.get("max_train_minutes", 25) > 8:
        kwargs["max_train_minutes"] = 8
    return kwargs


def _train_impl(
    *,
    sample: bool,
    allow_cpu: bool,
    max_train_minutes: float,
    max_positions: int,
    max_steps: int,
    epochs: int,
    batch_size: int,
    block_size: int,
    n_embd: int,
    n_head: int,
    n_layer: int,
    lr: float,
    weight_decay: float,
    min_elo: int,
    min_plies: int,
    games_vs_random: int,
    games_vs_stockfish: int,
    legal_eval_positions: int,
    seed: int,
    push_to_hub: bool,
    hub_repo: str,
) -> dict:
    import torch
    import torch.nn.functional as F
    from contextlib import nullcontext

    cfg = _apply_sample_preset(
        {
            "sample": sample,
            "n_layer": n_layer,
            "n_embd": n_embd,
            "n_head": n_head,
            "max_positions": max_positions,
            "games_vs_random": games_vs_random,
            "games_vs_stockfish": games_vs_stockfish,
            "max_train_minutes": max_train_minutes,
        }
    )
    n_layer = cfg["n_layer"]
    n_embd = cfg["n_embd"]
    n_head = cfg["n_head"]
    max_positions = cfg["max_positions"]
    games_vs_random = cfg["games_vs_random"]
    games_vs_stockfish = cfg["games_vs_stockfish"]
    max_train_minutes = cfg["max_train_minutes"]

    torch.manual_seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        device = torch.device("cuda")
        device_name = torch.cuda.get_device_name(0)
        autocast = torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    elif allow_cpu:
        device = torch.device("cpu")
        device_name = "cpu"
        autocast = nullcontext()
    else:
        raise RuntimeError("CUDA required (torch.cuda.is_available() is False); pass allow_cpu=True for a local smoke test")

    print(
        f"device={device_name} sample={sample} layers={n_layer} d_model={n_embd} "
        f"heads={n_head} max_positions={max_positions}",
        flush=True,
    )

    games, dataset_id, data_timings = collect_games(
        max_positions=max_positions,
        min_elo=min_elo,
        min_plies=min_plies,
        block_size=block_size,
    )
    n_hold = max(1, int(round(len(games) * 0.02)))
    if n_hold >= len(games):
        n_hold = 1 if len(games) > 1 else 0
    holdout = games[-n_hold:] if n_hold else []
    train_games = games[:-n_hold] if n_hold else games
    if not train_games:
        train_games, holdout = games, games[:1]

    vocab = build_vocab(train_games)
    stoi, itos = vocab["stoi"], vocab["itos"]
    truncations = sum(1 for g in games if g["truncated"])
    train_positions = sum(len(g["moves"]) for g in train_games)
    hold_positions = sum(len(g["moves"]) for g in holdout)
    print(
        f"split train_games={len(train_games)} holdout_games={len(holdout)} "
        f"train_positions={train_positions} vocab={len(itos)} truncations={truncations}",
        flush=True,
    )

    model = build_model(len(itos), n_embd, n_head, n_layer, block_size).to(device)
    param_count = sum(p.numel() for p in model.parameters())
    print(f"params={param_count:,}", flush=True)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay, betas=(0.9, 0.95))

    xs, ys = [], []
    for game in train_games:
        x, y = encode_game(game["moves"], stoi, block_size)
        xs.append(x)
        ys.append(y)
    x_train = torch.tensor(xs, dtype=torch.long)
    y_train = torch.tensor(ys, dtype=torch.long)
    n_train = x_train.size(0)
    steps_per_epoch = max(1, math.ceil(n_train / batch_size))
    n_epochs = 3 if sample else epochs
    planned_steps = steps_per_epoch * n_epochs
    if max_steps > 0:
        planned_steps = min(planned_steps, max_steps)
    warmup = min(100, max(1, planned_steps // 10))
    pad = stoi[PAD_TOK]
    deadline = time.time() + max_train_minutes * 60
    log_every = max(20, planned_steps // 100)
    hold_pairs = [encode_game(g["moves"], stoi, block_size) for g in holdout[:256]]
    x_hold = torch.tensor([p[0] for p in hold_pairs], dtype=torch.long) if hold_pairs else None
    y_hold = torch.tensor([p[1] for p in hold_pairs], dtype=torch.long) if hold_pairs else None

    def holdout_loss() -> float:
        if x_hold is None:
            return float("nan")
        model.eval()
        total, count = 0.0, 0
        with torch.no_grad(), autocast:
            for start in range(0, x_hold.size(0), batch_size):
                xb = x_hold[start : start + batch_size].to(device)
                yb = y_hold[start : start + batch_size].to(device)
                logits = model(xb)
                total += float(
                    F.cross_entropy(
                        logits.reshape(-1, logits.size(-1)),
                        yb.reshape(-1),
                        ignore_index=pad,
                        reduction="sum",
                    )
                )
                count += int((yb != pad).sum().item())
        model.train()
        return total / max(count, 1)

    history: list[dict[str, float]] = []
    t_train = time.time()
    step = 0
    last_loss = 0.0
    model.train()
    stop = False
    for epoch in range(1, n_epochs + 1):
        perm = torch.randperm(n_train)
        for start in range(0, n_train, batch_size):
            if time.time() >= deadline or step >= planned_steps:
                stop = True
                break
            step += 1
            batch = perm[start : start + batch_size]
            xb = x_train[batch].to(device)
            yb = y_train[batch].to(device)
            lengths = (yb != pad).sum(dim=1)
            tmax = int(lengths.max().item()) if lengths.numel() else 1
            tmax = max(1, min(block_size, tmax))
            xb = xb[:, :tmax]
            yb = yb[:, :tmax]
            lr_now = _cosine_lr(step - 1, planned_steps, lr, warmup)
            for group in opt.param_groups:
                group["lr"] = lr_now
            opt.zero_grad(set_to_none=True)
            with autocast:
                logits = model(xb)
                loss = F.cross_entropy(
                    logits.reshape(-1, logits.size(-1)),
                    yb.reshape(-1),
                    ignore_index=pad,
                )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            last_loss = float(loss.detach())
            if step == 50:
                # Shrink the cosine schedule to what fits the time budget so the LR still decays to zero.
                rate = 50 / max(time.time() - t_train, 1e-6)
                budget_steps = 50 + int(rate * (deadline - time.time()) * 0.9)
                if budget_steps < planned_steps:
                    print(f"time budget caps steps at {budget_steps} ({rate:.1f} steps/s)", flush=True)
                    planned_steps = budget_steps
                    log_every = max(20, planned_steps // 100)
            if step == 1 or step % log_every == 0 or step == planned_steps:
                hold = holdout_loss()
                history.append({"step": float(step), "train_loss": last_loss, "test_loss": hold})
                print(
                    f"step {step}/{planned_steps} epoch={epoch} loss={last_loss:.4f} "
                    f"holdout={hold:.4f} lr={lr_now:.2e}",
                    flush=True,
                )
        if stop:
            break

    train_s = time.time() - t_train
    print(f"training done steps={step} seconds={train_s:.1f}", flush=True)

    t_eval = time.time()
    metrics = eval_top1_and_legal(
        model,
        holdout or train_games[:1],
        stoi,
        itos,
        device,
        legal_eval_positions=legal_eval_positions,
    )
    if not history or history[-1]["step"] != float(step):
        history.append({"step": float(step), "train_loss": last_loss, "test_loss": holdout_loss()})
    val_loss = history[-1]["test_loss"]

    engine, stockfish_status = resolve_stockfish()
    print(f"stockfish: {stockfish_status}", flush=True)
    illegal_attempts = 0
    random_stats = play_matches(
        model,
        "random",
        games_vs_random,
        stoi,
        itos,
        device,
        engine=None,
        seed=seed + 1,
        annotate=True,
    )
    illegal_attempts += random_stats["illegal_attempts"]
    stockfish_stats: dict[str, Any] | None = None
    if engine is not None:
        try:
            stockfish_stats = play_matches(
                model,
                "stockfish",
                games_vs_stockfish,
                stoi,
                itos,
                device,
                engine=engine,
                seed=seed + 2,
                annotate=True,
            )
            illegal_attempts += stockfish_stats["illegal_attempts"]
        except Exception as err:  # noqa: BLE001
            stockfish_status = f"unavailable ({err})"
            stockfish_stats = None
        finally:
            try:
                engine.quit()
            except Exception:
                pass

    eval_s = time.time() - t_eval
    out_dir = resolve_artifact_dirs()
    config = {
        "sample": sample,
        "n_layer": n_layer,
        "n_embd": n_embd,
        "n_head": n_head,
        "block_size": block_size,
        "batch_size": batch_size,
        "lr": lr,
        "max_positions": max_positions,
        "max_train_minutes": max_train_minutes,
        "epochs": n_epochs,
        "min_elo": min_elo,
        "min_plies": min_plies,
        "games_vs_random": games_vs_random,
        "games_vs_stockfish": games_vs_stockfish,
        "stockfish_limit_s": STOCKFISH_LIMIT_S,
        "seed": seed,
    }
    results = {
        "config": config,
        "dataset_id": dataset_id,
        "device": str(device),
        "device_name": device_name,
        "param_count": param_count,
        "vocab_size": len(itos),
        "n_games": len(games),
        "n_train_games": len(train_games),
        "n_holdout_games": len(holdout),
        "train_positions": train_positions,
        "holdout_positions": hold_positions,
        "truncations": truncations,
        "steps": step,
        "timings": {
            **data_timings,
            "train_s": round(train_s, 3),
            "eval_s": round(eval_s, 3),
        },
        "top1_acc": metrics["top1_acc"],
        "top1_n": metrics["top1_n"],
        "legal_rate_sampled": metrics["legal_rate_sampled"],
        "legal_rate_argmax": metrics["legal_rate_argmax"],
        "legal_eval_n": metrics["legal_eval_n"],
        "games_vs_random": {
            "W": random_stats["wins"],
            "D": random_stats["draws"],
            "L": random_stats["losses"],
            "win_rate": random_stats["win_rate"],
        },
        "games_vs_stockfish": (
            {
                "W": stockfish_stats["wins"],
                "D": stockfish_stats["draws"],
                "L": stockfish_stats["losses"],
                "win_rate": stockfish_stats["win_rate"],
                "limit": f"chess.engine.Limit(time={STOCKFISH_LIMIT_S})",
            }
            if stockfish_stats
            else stockfish_status
        ),
        "illegal_attempts_in_matches": illegal_attempts,
        "stockfish": stockfish_status,
    }

    torch.save(
        {
            "state_dict": model.state_dict(),
            "config": {
                "vocab_size": len(itos),
                "n_embd": n_embd,
                "n_head": n_head,
                "n_layer": n_layer,
                "block_size": block_size,
                "param_count": param_count,
            },
        },
        out_dir / "model.pt",
    )
    (out_dir / "vocab.json").write_text(json.dumps(vocab, indent=2) + "\n", encoding="utf-8")
    (out_dir / "results.json").write_text(json.dumps(results, indent=2, default=str) + "\n", encoding="utf-8")
    annotated = {
        "random": random_stats.get("annotated"),
        "stockfish": stockfish_stats.get("annotated") if stockfish_stats else None,
    }
    (out_dir / "annotated_game.json").write_text(json.dumps(annotated, indent=2) + "\n", encoding="utf-8")
    (out_dir / "games_vs_random.pgn").write_text(random_stats["pgn"], encoding="utf-8")
    if stockfish_stats:
        (out_dir / "games_vs_stockfish.pgn").write_text(stockfish_stats["pgn"], encoding="utf-8")
    else:
        (out_dir / "games_vs_stockfish.pgn").write_text(
            f"; stockfish {stockfish_status}\n", encoding="utf-8"
        )
    write_loss_curve(out_dir, history)
    write_artifact_marker(
        out_dir,
        name=ARTIFACT_NAME,
        kind="output",
        compatibility_key=f"chess-move-transformer-v1-{n_layer}x{n_embd}",
        metadata={
            "filename": "model.pt",
            "param_count": param_count,
            "vocab_size": len(itos),
            "dataset_id": dataset_id,
            "top1_acc": metrics["top1_acc"],
            "legal_rate_sampled": metrics["legal_rate_sampled"],
        },
    )

    sf_wdl = (
        f"{stockfish_stats['wins']}/{stockfish_stats['draws']}/{stockfish_stats['losses']}"
        if stockfish_stats
        else stockfish_status
    )
    print("", flush=True)
    print("=" * 64, flush=True)
    print("RESULTS", flush=True)
    print("=" * 64, flush=True)
    print(f"top1_acc                 {metrics['top1_acc']:.4f}  (n={metrics['top1_n']})", flush=True)
    print(f"legal_rate_sampled       {metrics['legal_rate_sampled']:.4f}  (n={metrics['legal_eval_n']})", flush=True)
    print(f"legal_rate_argmax        {metrics['legal_rate_argmax']:.4f}", flush=True)
    print(
        f"vs_random W/D/L          {random_stats['wins']}/{random_stats['draws']}/{random_stats['losses']}  "
        f"win_rate={random_stats['win_rate']:.4f}",
        flush=True,
    )
    print(f"vs_stockfish W/D/L       {sf_wdl}", flush=True)
    print(f"illegal_attempts_in_matches  {illegal_attempts}", flush=True)
    print(f"stockfish                {stockfish_status}", flush=True)
    print(f"stockfish_limit          chess.engine.Limit(time={STOCKFISH_LIMIT_S})", flush=True)
    print(f"dataset                  {dataset_id}", flush=True)
    print(f"games/positions          {len(games)}/{train_positions + hold_positions}", flush=True)
    print(f"truncations              {truncations}", flush=True)
    print(f"vocab                    {len(itos)}", flush=True)
    print(f"params                   {param_count}", flush=True)
    print(f"device                   {device_name}", flush=True)
    print("=" * 64, flush=True)

    hub_url = None
    if push_to_hub:
        _bridge_hf_token()
        try:
            from huggingface_hub import HfApi

            api = HfApi()
            api.create_repo(hub_repo, exist_ok=True, private=False)
            for filename in ("model.pt", "vocab.json", "results.json"):
                api.upload_file(
                    path_or_fileobj=str(out_dir / filename),
                    path_in_repo=filename,
                    repo_id=hub_repo,
                )
            hub_url = f"https://huggingface.co/{hub_repo}"
        except Exception as err:  # noqa: BLE001 — publish is optional; keep weights via artifacts
            print(f"push_to_hub failed: {err}", flush=True)
            hub_url = None

    return {
        "ok": True,
        "compat": "chess-move-transformer",
        "device": str(device),
        "device_name": device_name,
        "dataset_id": dataset_id,
        "param_count": param_count,
        "vocab_size": len(itos),
        "steps": step,
        "sample": sample,
        "top1_acc": _py(metrics["top1_acc"]),
        "legal_rate_sampled": _py(metrics["legal_rate_sampled"]),
        "legal_rate_argmax": _py(metrics["legal_rate_argmax"]),
        "games_vs_random": {
            "W": random_stats["wins"],
            "D": random_stats["draws"],
            "L": random_stats["losses"],
            "win_rate": _py(random_stats["win_rate"]),
        },
        "games_vs_stockfish": results["games_vs_stockfish"],
        "illegal_attempts_in_matches": illegal_attempts,
        "stockfish": stockfish_status,
        "n_games": len(games),
        "train_positions": train_positions,
        "truncations": truncations,
        "artifact_dir": str(out_dir),
        "hub_url": hub_url,
        "torch": torch.__version__,
    }


@app.function(
    gpu="RTX-3090",
    image=image,
    timeout=3600,
)
def train(
    sample: bool = False,
    allow_cpu: bool = False,
    max_train_minutes: float = 25,
    max_positions: int = 4_000_000,
    max_steps: int = 0,
    epochs: int = 8,
    batch_size: int = 32,
    block_size: int = 256,
    n_embd: int = 512,
    n_head: int = 8,
    n_layer: int = 8,
    lr: float = 3e-4,
    weight_decay: float = 0.1,
    min_elo: int = 1800,
    min_plies: int = 20,
    games_vs_random: int = 100,
    games_vs_stockfish: int = 100,
    legal_eval_positions: int = 5000,
    seed: int = 0,
    push_to_hub: bool = False,
    hub_repo: str = DEFAULT_HUB_REPO,
) -> dict:
    """Train only. No Hugging Face token required."""
    return _train_impl(
        sample=sample,
        allow_cpu=allow_cpu,
        max_train_minutes=max_train_minutes,
        max_positions=max_positions,
        max_steps=max_steps,
        epochs=epochs,
        batch_size=batch_size,
        block_size=block_size,
        n_embd=n_embd,
        n_head=n_head,
        n_layer=n_layer,
        lr=lr,
        weight_decay=weight_decay,
        min_elo=min_elo,
        min_plies=min_plies,
        games_vs_random=games_vs_random,
        games_vs_stockfish=games_vs_stockfish,
        legal_eval_positions=legal_eval_positions,
        seed=seed,
        push_to_hub=push_to_hub,
        hub_repo=hub_repo,
    )


@app.function(
    gpu="RTX-3090",
    image=image,
    timeout=3600,
    secrets=[hf_secret],
)
def train_and_push(
    sample: bool = False,
    allow_cpu: bool = False,
    max_train_minutes: float = 25,
    max_positions: int = 4_000_000,
    max_steps: int = 0,
    epochs: int = 8,
    batch_size: int = 32,
    block_size: int = 256,
    n_embd: int = 512,
    n_head: int = 8,
    n_layer: int = 8,
    lr: float = 3e-4,
    weight_decay: float = 0.1,
    min_elo: int = 1800,
    min_plies: int = 20,
    games_vs_random: int = 100,
    games_vs_stockfish: int = 100,
    legal_eval_positions: int = 5000,
    seed: int = 0,
    push_to_hub: bool = True,
    hub_repo: str = DEFAULT_HUB_REPO,
) -> dict:
    """Same train, then upload ``model.pt``, ``vocab.json``, and ``results.json`` using the stored ``hf`` secret."""
    return _train_impl(
        sample=sample,
        allow_cpu=allow_cpu,
        max_train_minutes=max_train_minutes,
        max_positions=max_positions,
        max_steps=max_steps,
        epochs=epochs,
        batch_size=batch_size,
        block_size=block_size,
        n_embd=n_embd,
        n_head=n_head,
        n_layer=n_layer,
        lr=lr,
        weight_decay=weight_decay,
        min_elo=min_elo,
        min_plies=min_plies,
        games_vs_random=games_vs_random,
        games_vs_stockfish=games_vs_stockfish,
        legal_eval_positions=legal_eval_positions,
        seed=seed,
        push_to_hub=push_to_hub,
        hub_repo=hub_repo,
    )


if __name__ == "__main__":
    with app.run():
        print(train.remote(sample=True))
