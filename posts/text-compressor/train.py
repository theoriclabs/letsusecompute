"""Train a tiny character LM and use it as a compressor.

Dataset: enwik8 (first 90 MB train, last 10 MB test), downloaded on the
machine. A GPT-style byte model produces next-byte probabilities; a range
coder turns those into a bitstream. gzip -9 and xz -9 run on the same bytes.

    compute run train.py::train --gpu cheap --dry-run
    compute run train.py::train --gpu cheap --timeout 2400 --wait
"""

from __future__ import annotations

import gzip
import io
import json
import lzma
import math
import os
import tempfile
import urllib.request
import zipfile
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import compute

app = compute.App("text-compressor")
# Pin cu124 wheels: many Vast consumer hosts still expose driver 12.4.
image = compute.Image.cuda_pytorch().pip_install(
    "--extra-index-url",
    "https://download.pytorch.org/whl/cu124",
    "torch==2.5.1+cu124",
    "numpy",
    "datasets==3.6.0",
    "huggingface_hub>=0.33,<1",
)

hf_secret = compute.Secret.from_name("hf")

ENWIK8_URLS = (
    "https://mattmahoney.net/dc/enwik8.zip",
    "http://mattmahoney.net/dc/enwik8.zip",
    "https://web.archive.org/web/20201027130031/http://mattmahoney.net/dc/enwik8.zip",
    "https://cs.fit.edu/~mmahoney/compression/enwik8.zip",
    "http://cs.fit.edu/~mmahoney/compression/enwik8.zip",
)
ENWIK8_BYTES = 100_000_000
DEFAULT_HUB_REPO = "theoriclabs/char-lm-compressor"
WORKLOAD_SUBDIR = "text-compressor"
ARTIFACT_NAME = "char-lm-compressor"
ARTIFACT_MARKER = ".compute-artifact.json"
DEFAULT_ARTIFACT_FALLBACK = Path("/tmp/compute-text-compressor")
VOCAB = 256
BOS = 0
FREQ_TOTAL = 1 << 12
RANS_L = 1 << 23


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


def _http_get(url: str) -> bytes:
    request = urllib.request.Request(
        url,
        headers={"User-Agent": "letsusecompute-text-compressor/1.0"},
    )
    with urllib.request.urlopen(request, timeout=180) as response:
        return response.read()


def _from_zip_bytes(blob: bytes) -> bytes:
    if blob[:2] != b"PK":
        preview = blob[:80]
        raise RuntimeError(f"not a zip ({len(blob)} bytes, start={preview!r})")
    with zipfile.ZipFile(io.BytesIO(blob)) as archive:
        payload = archive.read(archive.namelist()[0])
    if len(payload) < ENWIK8_BYTES:
        raise RuntimeError(f"enwik8 too short: {len(payload)} bytes")
    return payload[:ENWIK8_BYTES]


def _from_hf() -> bytes:
    from datasets import load_dataset

    print("loading LTCB/enwik8 enwik8-raw from Hugging Face", flush=True)
    dataset = load_dataset(
        "LTCB/enwik8",
        "enwik8-raw",
        split="train",
        trust_remote_code=True,
    )
    text = dataset[0]["text"]
    for encoding in ("utf-8", "latin-1"):
        try:
            payload = text.encode(encoding)
        except UnicodeEncodeError:
            continue
        if len(payload) >= ENWIK8_BYTES:
            return payload[:ENWIK8_BYTES]
    raise RuntimeError("Hugging Face enwik8-raw was shorter than 100 MB")


def _from_wikitext() -> bytes:
    from datasets import load_dataset

    print("loading Salesforce/wikitext wikitext-103-raw-v1", flush=True)
    dataset = load_dataset("Salesforce/wikitext", "wikitext-103-raw-v1", split="train")
    chunks: list[bytes] = []
    total = 0
    for row in dataset:
        piece = str(row["text"]).encode("utf-8")
        if not piece:
            continue
        chunks.append(piece)
        total += len(piece)
        if total >= ENWIK8_BYTES:
            break
    payload = b"".join(chunks)
    if len(payload) < ENWIK8_BYTES:
        raise RuntimeError(f"wikitext-103 too short: {len(payload)} bytes")
    return payload[:ENWIK8_BYTES]


def download_corpus(dest: Path) -> tuple[bytes, str]:
    """Return 100 MB of public text and the corpus name."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    raw_path = dest.with_suffix("")
    meta_path = dest.with_name("corpus_name.txt")
    if raw_path.exists() and raw_path.stat().st_size >= ENWIK8_BYTES and meta_path.exists():
        return raw_path.read_bytes()[:ENWIK8_BYTES], meta_path.read_text().strip()
    last_err: Exception | None = None
    for url in ENWIK8_URLS:
        try:
            print(f"downloading {url}", flush=True)
            payload = _from_zip_bytes(_http_get(url))
            raw_path.write_bytes(payload)
            meta_path.write_text("enwik8\n")
            return payload, "enwik8"
        except Exception as err:  # noqa: BLE001 — try the next mirror
            last_err = err
            print(f"download failed: {err}", flush=True)
    try:
        payload = _from_hf()
        raw_path.write_bytes(payload)
        meta_path.write_text("enwik8\n")
        return payload, "enwik8"
    except Exception as err:
        print(f"hf enwik8 failed: {err}", flush=True)
        last_err = err
    payload = _from_wikitext()
    raw_path.write_bytes(payload)
    meta_path.write_text("wikitext-103-raw\n")
    print(f"using wikitext-103-raw after enwik8 failed: {last_err}", flush=True)
    return payload, "wikitext-103-raw"


def quantize_probs(probs) -> tuple[list[int], list[int]]:
    """Turn a 256-way distribution into integer start/freq that sum to FREQ_TOTAL."""
    values = [max(float(p), 1e-12) for p in probs]
    total = sum(values)
    values = [p / total for p in values]
    remaining = FREQ_TOTAL - VOCAB
    extra = [int(p * remaining) for p in values]
    freq = [1 + e for e in extra]
    leftover = FREQ_TOTAL - sum(freq)
    if leftover:
        top = max(range(VOCAB), key=lambda i: values[i])
        freq[top] += leftover
    start = [0] * VOCAB
    running = 0
    for i, f in enumerate(freq):
        start[i] = running
        running += f
    return start, freq


class RansEncoder:
    """32-bit rANS. Symbols are encoded last-to-first."""

    def __init__(self) -> None:
        self.state = RANS_L
        self.tail = bytearray()

    def encode(self, start: int, freq: int) -> None:
        x_max = ((RANS_L >> 12) << 8) * freq
        while self.state >= x_max:
            self.tail.append(self.state & 0xFF)
            self.state >>= 8
        self.state = ((self.state // freq) << 12) + (self.state % freq) + start

    def finish(self) -> bytes:
        for _ in range(4):
            self.tail.append(self.state & 0xFF)
            self.state >>= 8
        return bytes(reversed(self.tail))


class RansDecoder:
    def __init__(self, blob: bytes) -> None:
        self.data = blob
        self.pos = 0
        self.state = 0
        for _ in range(4):
            self.state = (self.state << 8) | self._read()

    def _read(self) -> int:
        if self.pos >= len(self.data):
            return 0
        byte = self.data[self.pos]
        self.pos += 1
        return byte

    def peek_slot(self) -> int:
        return self.state & (FREQ_TOTAL - 1)

    def advance(self, start: int, freq: int) -> None:
        self.state = freq * (self.state >> 12) + (self.state & (FREQ_TOTAL - 1)) - start
        while self.state < RANS_L:
            self.state = (self.state << 8) | self._read()


def rans_roundtrip(symbols: list[int], tables: list[tuple[list[int], list[int]]]) -> bytes:
    enc = RansEncoder()
    for symbol, (start, freq) in zip(reversed(symbols), reversed(tables)):
        enc.encode(start[symbol], freq[symbol])
    return enc.finish()


def rans_decode_symbols(blob: bytes, tables: list[tuple[list[int], list[int]]]) -> list[int]:
    dec = RansDecoder(blob)
    out: list[int] = []
    for start, freq in tables:
        slot = dec.peek_slot()
        symbol = 0
        for i, (s, f) in enumerate(zip(start, freq)):
            if s <= slot < s + f:
                symbol = i
                break
        out.append(symbol)
        dec.advance(start[symbol], freq[symbol])
    return out


def bpc(nbytes: int, n_symbols: int) -> float:
    return 8.0 * nbytes / max(n_symbols, 1)


def write_bpc_svg(directory: Path, rows: list[tuple[str, float, str]]) -> Path:
    width, height, pad_l, pad_r, pad_t = 1200, 420, 220, 80, 100
    inner = width - pad_l - pad_r
    ceiling = max(row[1] for row in rows)
    bar_h, gap = 52, 28
    blocks = []
    for i, (label, value, color) in enumerate(rows):
        y = pad_t + i * (bar_h + gap)
        w = max(12.0, inner * (value / ceiling))
        blocks.append(
            f'<text x="{pad_l - 16}" y="{y + 34}" font-size="18" font-weight="700" '
            f'fill="#1a1a1a" text-anchor="end" font-family="system-ui, sans-serif">{label}</text>'
            f'<rect x="{pad_l}" y="{y}" width="{w:.1f}" height="{bar_h}" rx="8" fill="{color}"/>'
            f'<text x="{pad_l + w - 14:.1f}" y="{y + 34}" font-size="18" font-weight="700" '
            f'fill="#fff" text-anchor="end" font-family="system-ui, sans-serif">{value:.3f}</text>'
        )
    svg = f"""<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}" role="img">
  <rect width="100%" height="100%" fill="#f7f4ef"/>
  <text x="48" y="48" font-size="26" font-weight="700" fill="#1a1a1a" font-family="system-ui, sans-serif">Bits per byte on the held-out slice</text>
  <text x="48" y="76" font-size="15" fill="#5c5c5c" font-family="system-ui, sans-serif">Lower is better. Same bytes for every bar.</text>
  {"".join(blocks)}
</svg>
"""
    path = directory / "bpc_bars.svg"
    path.write_text(svg, encoding="utf-8")
    return path


def write_loss_svg(directory: Path, history: list[dict[str, float]]) -> Path:
    width, height, pad = 1200, 480, 72
    losses = [row["train_loss"] for row in history]
    lo, hi = (min(losses), max(losses)) if losses else (0.0, 1.0)
    if hi <= lo:
        hi = lo + 1e-6

    def xy(index: int, value: float) -> str:
        x = pad + (width - 2 * pad) * (index / max(len(history) - 1, 1))
        y = pad + (height - 2 * pad) * (1 - (value - lo) / (hi - lo))
        return f"{x:.1f},{y:.1f}"

    points = " ".join(xy(i, row["train_loss"]) for i, row in enumerate(history))
    svg = f"""<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}" role="img">
  <rect width="100%" height="100%" fill="#f7f4ef"/>
  <text x="48" y="44" font-size="26" font-weight="700" fill="#1a1a1a" font-family="system-ui, sans-serif">Train loss</text>
  <polyline fill="none" stroke="#3d6b8a" stroke-width="3" points="{points}"/>
</svg>
"""
    path = directory / "training_curves.svg"
    path.write_text(svg, encoding="utf-8")
    return path


def build_model(n_embd: int, n_head: int, n_layer: int, block_size: int):
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    class CausalSelfAttention(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.n_head = n_head
            self.key = nn.Linear(n_embd, n_embd)
            self.query = nn.Linear(n_embd, n_embd)
            self.value = nn.Linear(n_embd, n_embd)
            self.proj = nn.Linear(n_embd, n_embd)
            self.register_buffer("mask", torch.tril(torch.ones(block_size, block_size)))

        def forward(self, x):
            batch, steps, width = x.size()
            head = width // self.n_head
            key = self.key(x).view(batch, steps, self.n_head, head).transpose(1, 2)
            query = self.query(x).view(batch, steps, self.n_head, head).transpose(1, 2)
            value = self.value(x).view(batch, steps, self.n_head, head).transpose(1, 2)
            att = (query @ key.transpose(-2, -1)) * (1.0 / math.sqrt(head))
            att = att.masked_fill(self.mask[:steps, :steps] == 0, float("-inf"))
            att = F.softmax(att, dim=-1)
            out = (att @ value).transpose(1, 2).contiguous().view(batch, steps, width)
            return self.proj(out)

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
            )

        def forward(self, x):
            x = x + self.attn(self.ln1(x))
            x = x + self.mlp(self.ln2(x))
            return x

    class CharGPT(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.block_size = block_size
            self.tok = nn.Embedding(VOCAB, n_embd)
            self.pos = nn.Embedding(block_size, n_embd)
            self.blocks = nn.ModuleList([Block() for _ in range(n_layer)])
            self.ln = nn.LayerNorm(n_embd)
            self.head = nn.Linear(n_embd, VOCAB, bias=False)

        def forward(self, idx):
            _batch, steps = idx.shape
            pos = torch.arange(0, steps, device=idx.device)
            x = self.tok(idx) + self.pos(pos)
            for block in self.blocks:
                x = block(x)
            return self.head(self.ln(x))

    return CharGPT()


def _block_batch(data: bytes, starts: list[int], block_size: int, device):
    import torch

    xs, lengths = [], []
    for start in starts:
        chunk = data[start : start + block_size]
        lengths.append(len(chunk))
        inp = [BOS] + list(chunk[:-1])
        inp.extend([BOS] * (block_size - len(inp)))
        xs.append(inp)
    return torch.tensor(xs, device=device, dtype=torch.long), lengths


def quantize_batch(probs):
    """Vectorized start/freq tables. ``probs`` is (N, 256)."""
    import torch

    p = probs.double().clamp_min(1e-12)
    p = p / p.sum(dim=-1, keepdim=True)
    remaining = FREQ_TOTAL - VOCAB
    freq = (p * remaining).floor().long() + 1
    leftover = FREQ_TOTAL - freq.sum(dim=-1)
    top = p.argmax(dim=-1)
    freq[torch.arange(freq.size(0), device=freq.device), top] += leftover
    start = torch.zeros_like(freq)
    start[:, 1:] = torch.cumsum(freq, dim=-1)[:, :-1]
    return start, freq


def collect_symbol_freqs(model, data: bytes, block_size: int, batch_size: int, device):
    """Teacher-forced (start, freq) of each actual byte, plus nats NLL."""
    import torch
    import torch.nn.functional as F

    model.eval()
    n = len(data)
    starts = [0] * n
    freqs = [0] * n
    nll = 0.0
    bytes_t = torch.tensor(list(data), device=device, dtype=torch.long)
    block_starts = list(range(0, n, block_size))
    with torch.no_grad():
        for i in range(0, len(block_starts), batch_size):
            batch = block_starts[i : i + batch_size]
            idx, lengths = _block_batch(data, batch, block_size, device)
            logits = model(idx)
            probs = F.softmax(logits, dim=-1)
            log_probs = torch.log(probs.clamp_min(1e-12))
            start_tbl, freq_tbl = quantize_batch(probs.reshape(-1, VOCAB))
            start_tbl = start_tbl.view(idx.size(0), idx.size(1), VOCAB)
            freq_tbl = freq_tbl.view(idx.size(0), idx.size(1), VOCAB)
            for bi, start in enumerate(batch):
                length = lengths[bi]
                nll += -float(log_probs[bi, :length].gather(1, bytes_t[start : start + length][:, None]).sum())
                for t in range(length):
                    byte = int(bytes_t[start + t])
                    starts[start + t] = int(start_tbl[bi, t, byte])
                    freqs[start + t] = int(freq_tbl[bi, t, byte])
    return starts, freqs, nll


def encode_symbols(starts: list[int], freqs: list[int]) -> bytes:
    enc = RansEncoder()
    for start, freq in zip(reversed(starts), reversed(freqs)):
        enc.encode(start, freq)
    return enc.finish()


def decode_with_model(model, blob: bytes, n: int, block_size: int, device) -> bytes:
    """Decode ``n`` bytes from the bitstream by re-running the model."""
    import torch
    import torch.nn.functional as F

    model.eval()
    dec = RansDecoder(blob)
    out = bytearray()
    with torch.no_grad():
        for start in range(0, n, block_size):
            length = min(block_size, n - start)
            inp = [BOS] * block_size
            for t in range(length):
                idx = torch.tensor([inp], device=device, dtype=torch.long)
                probs = F.softmax(model(idx)[0, t], dim=-1)
                start_tbl, freq_tbl = quantize_batch(probs.unsqueeze(0))
                start_freq = start_tbl[0].tolist()
                freq = freq_tbl[0].tolist()
                slot = dec.peek_slot()
                symbol = 0
                for i, (s, f) in enumerate(zip(start_freq, freq)):
                    if s <= slot < s + f:
                        symbol = i
                        break
                out.append(symbol)
                dec.advance(start_freq[symbol], freq[symbol])
                if t + 1 < block_size:
                    inp[t + 1] = symbol
    return bytes(out)


def _train_impl(
    *,
    sample: bool,
    steps: int,
    train_bytes: int,
    test_bytes: int,
    batch_size: int,
    block_size: int,
    n_embd: int,
    n_head: int,
    n_layer: int,
    lr: float,
    seed: int,
    push_to_hub: bool,
    hub_repo: str,
) -> dict:
    import torch
    import torch.nn.functional as F

    if sample:
        train_bytes = train_bytes or 900_000
        test_bytes = test_bytes or 100_000
        steps = steps or 1_500
        batch_size = min(batch_size, 32)
    else:
        train_bytes = train_bytes or 90_000_000
        test_bytes = test_bytes or 10_000_000
        steps = steps or 20_000

    torch.manual_seed(seed)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device_name = torch.cuda.get_device_name(0) if device.type == "cuda" else "cpu"
    print(f"device={device_name} sample={sample} steps={steps}", flush=True)

    payload, dataset_name = download_corpus(Path("/tmp/enwik8.zip"))
    train_data = payload[:train_bytes]
    test_data = payload[train_bytes : train_bytes + test_bytes]
    if len(train_data) < block_size + 1 or not test_data:
        raise RuntimeError("not enough corpus bytes for the requested split")
    print(f"corpus={dataset_name}", flush=True)

    model = build_model(n_embd, n_head, n_layer, block_size).to(device)
    param_count = sum(p.numel() for p in model.parameters())
    print(f"params={param_count:,} train={len(train_data):,} test={len(test_data):,}", flush=True)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.1, betas=(0.9, 0.95))

    train_tensor = torch.tensor(list(train_data), device=device, dtype=torch.long)
    history: list[dict[str, float]] = []
    model.train()
    for step in range(1, steps + 1):
        ix = torch.randint(0, len(train_tensor) - block_size, (batch_size,), device=device)
        offsets = torch.arange(block_size, device=device)
        x = train_tensor[ix[:, None] + offsets]
        y = train_tensor[ix[:, None] + offsets + 1]
        logits = model(x)
        loss = F.cross_entropy(logits.reshape(-1, VOCAB), y.reshape(-1))
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if step == 1 or step % 100 == 0 or step == steps:
            row = {"step": step, "train_loss": float(loss.detach())}
            history.append(row)
            print(f"step {step}/{steps} loss={row['train_loss']:.4f}", flush=True)

    print("encoding holdout", flush=True)
    check_n = len(test_data) if sample else min(len(test_data), 32_768)
    starts, freqs, nll = collect_symbol_freqs(model, test_data, block_size, batch_size=1, device=device)
    bitstream = encode_symbols(starts, freqs)
    decoded = decode_with_model(model, bitstream, check_n, block_size, device)
    roundtrip = decoded == test_data[:check_n]
    print(f"roundtrip={roundtrip} checked={check_n} encoded_bytes={len(bitstream)}", flush=True)
    if not roundtrip:
        mismatch = next((i for i, (a, b) in enumerate(zip(decoded, test_data)) if a != b), -1)
        print(f"first mismatch at {mismatch}", flush=True)

    gzip_blob = gzip.compress(test_data, compresslevel=9)
    xz_blob = lzma.compress(test_data, preset=9)
    model_bpc = bpc(len(bitstream), len(test_data))
    nll_bpc = nll / math.log(2) / len(test_data)
    gzip_bpc = bpc(len(gzip_blob), len(test_data))
    xz_bpc = bpc(len(xz_blob), len(test_data))
    weight_bytes = param_count * 2
    with_weights_bpc = bpc(len(bitstream) + weight_bytes, len(test_data))
    beats_gzip = bool(roundtrip and model_bpc < gzip_bpc)
    print(
        f"bpc model={model_bpc:.4f} nll={nll_bpc:.4f} gzip={gzip_bpc:.4f} xz={xz_bpc:.4f} "
        f"with_weights={with_weights_bpc:.4f} beats_gzip={beats_gzip}",
        flush=True,
    )

    out_dir = resolve_artifact_dirs()
    torch.save(
        {
            "state_dict": {k: v.detach().cpu().half() for k, v in model.state_dict().items()},
            "n_embd": n_embd,
            "n_head": n_head,
            "n_layer": n_layer,
            "block_size": block_size,
            "param_count": param_count,
            "vocab": VOCAB,
        },
        out_dir / "model.pt",
    )
    (out_dir / "holdout.rans").write_bytes(bitstream)
    write_loss_svg(out_dir, history)
    write_bpc_svg(
        out_dir,
        [
            ("char LM + rANS", model_bpc, "#3d6b8a"),
            ("gzip -9", gzip_bpc, "#7a8b99"),
            ("xz -9", xz_bpc, "#c4a484"),
        ],
    )
    summary = {
        "dataset": dataset_name,
        "sample": sample,
        "steps": steps,
        "param_count": param_count,
        "train_bytes": len(train_data),
        "test_bytes": len(test_data),
        "n_embd": n_embd,
        "n_head": n_head,
        "n_layer": n_layer,
        "block_size": block_size,
        "model_bytes": len(bitstream),
        "gzip_bytes": len(gzip_blob),
        "xz_bytes": len(xz_blob),
        "model_bpc": model_bpc,
        "nll_bpc": nll_bpc,
        "gzip_bpc": gzip_bpc,
        "xz_bpc": xz_bpc,
        "weight_bytes": weight_bytes,
        "with_weights_bpc": with_weights_bpc,
        "roundtrip": roundtrip,
        "decode_checked_bytes": check_n,
        "beats_gzip": beats_gzip,
        "history": history,
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    write_artifact_marker(
        out_dir,
        name=ARTIFACT_NAME,
        kind="output",
        compatibility_key="char-lm-compressor-v1",
        metadata={
            "filename": "model.pt",
            "param_count": param_count,
            "model_bpc": model_bpc,
            "gzip_bpc": gzip_bpc,
            "roundtrip": roundtrip,
        },
    )

    hub_url = None
    if push_to_hub:
        _bridge_hf_token()
        try:
            from huggingface_hub import HfApi

            hub_dir = Path(tempfile.mkdtemp(prefix="char-lm-compressor-"))
            (hub_dir / "model.pt").write_bytes((out_dir / "model.pt").read_bytes())
            (hub_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
            api = HfApi()
            api.create_repo(hub_repo, exist_ok=True, private=False)
            api.upload_folder(folder_path=str(hub_dir), repo_id=hub_repo)
            hub_url = f"https://huggingface.co/{hub_repo}"
        except Exception as err:  # noqa: BLE001 — publish is optional
            print(f"push_to_hub failed: {err}", flush=True)
            hub_url = None

    return {
        "ok": True,
        "compat": "char-lm-compressor",
        "device": str(device),
        "device_name": device_name,
        "dataset": dataset_name,
        "param_count": param_count,
        "steps": steps,
        "sample": sample,
        "train_bytes": len(train_data),
        "test_bytes": len(test_data),
        "model_bpc": _py(model_bpc),
        "nll_bpc": _py(nll_bpc),
        "gzip_bpc": _py(gzip_bpc),
        "xz_bpc": _py(xz_bpc),
        "with_weights_bpc": _py(with_weights_bpc),
        "model_bytes": len(bitstream),
        "gzip_bytes": len(gzip_blob),
        "xz_bytes": len(xz_blob),
        "roundtrip": roundtrip,
        "decode_checked_bytes": check_n,
        "beats_gzip": beats_gzip,
        "history": history,
        "artifact_dir": str(out_dir),
        "hub_url": hub_url,
        "torch": torch.__version__,
    }


@app.function(
    gpu="RTX-3090",
    image=image,
    timeout=2400,
)
def train(
    sample: bool = False,
    steps: int = 0,
    train_bytes: int = 0,
    test_bytes: int = 0,
    batch_size: int = 64,
    block_size: int = 256,
    n_embd: int = 256,
    n_head: int = 4,
    n_layer: int = 4,
    lr: float = 3e-4,
    seed: int = 0,
    push_to_hub: bool = False,
    hub_repo: str = DEFAULT_HUB_REPO,
) -> dict:
    """Train the compressor. No Hugging Face token required."""
    return _train_impl(
        sample=sample,
        steps=steps,
        train_bytes=train_bytes,
        test_bytes=test_bytes,
        batch_size=batch_size,
        block_size=block_size,
        n_embd=n_embd,
        n_head=n_head,
        n_layer=n_layer,
        lr=lr,
        seed=seed,
        push_to_hub=push_to_hub,
        hub_repo=hub_repo,
    )


@app.function(
    gpu="RTX-3090",
    image=image,
    timeout=2400,
    secrets=[hf_secret],
)
def train_and_push(
    sample: bool = False,
    steps: int = 0,
    train_bytes: int = 0,
    test_bytes: int = 0,
    batch_size: int = 64,
    block_size: int = 256,
    n_embd: int = 256,
    n_head: int = 4,
    n_layer: int = 4,
    lr: float = 3e-4,
    seed: int = 0,
    push_to_hub: bool = True,
    hub_repo: str = DEFAULT_HUB_REPO,
) -> dict:
    """Same train, then upload weights using the stored ``hf`` secret."""
    return _train_impl(
        sample=sample,
        steps=steps,
        train_bytes=train_bytes,
        test_bytes=test_bytes,
        batch_size=batch_size,
        block_size=block_size,
        n_embd=n_embd,
        n_head=n_head,
        n_layer=n_layer,
        lr=lr,
        seed=seed,
        push_to_hub=push_to_hub,
        hub_repo=hub_repo,
    )


if __name__ == "__main__":
    # Local check of the coder, no GPU.
    start, freq = quantize_probs([1.0] * VOCAB)
    tables = [(start, freq)] * 8
    symbols = [3, 1, 4, 1, 5, 9, 2, 6]
    blob = rans_roundtrip(symbols, tables)
    assert rans_decode_symbols(blob, tables) == symbols
    print("coder ok", len(blob), "bytes")
