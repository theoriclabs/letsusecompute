"""Bounded, checksum-verified model export through Compute's JSON result path."""

from __future__ import annotations

import base64
import hashlib
import io
import json
import tarfile
from pathlib import Path

RESULT_LIMIT = 64 * 1024 * 1024


def bundle_result(root: Path, summary: dict) -> dict:
    # This bundle stays inside the bootstrap child. Only a small upload receipt
    # reaches Compute: its current supervisor cannot poll multi-MiB results.
    # Base/tokenizer are pinned on the Hub; checkpoints and teacher .blend files
    # are reproducible intermediates, not needed to load the trained adapter.
    files = []
    for path in sorted(root.rglob("*")):
        relative = path.relative_to(root)
        if not path.is_file() or "checkpoints" in relative.parts:
            continue
        if path.name.startswith(".compute-artifact") or path.suffix == ".blend":
            continue
        if path.name in {"teacher-validation.tar.gz", "artifact-hashes.json", "early-export.json"}:
            continue
        if relative.parts[0] == "adapter" and path.name not in {
            "adapter_model.safetensors",
            "adapter_config.json",
            "README.md",
        }:
            continue
        if relative.parts[0] == "review" and path.suffix == ".png":
            continue  # reconstructed from the original images and review key
        files.append(path)
    hashes = {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
    (root / "artifact-hashes.json").write_text(json.dumps(hashes, indent=2) + "\n")
    files.append(root / "artifact-hashes.json")
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w:gz", compresslevel=6) as archive:
        for path in files:
            archive.add(path, arcname=str(path.relative_to(root)), recursive=False)
    raw = buffer.getvalue()
    result = {
        **summary,
        "export": {
            "format": "tar.gz+base64",
            "sha256": hashlib.sha256(raw).hexdigest(),
            "bytes": len(raw),
            "data": base64.b64encode(raw).decode("ascii"),
        },
    }
    size = len(json.dumps(result, allow_nan=False).encode()) + 1
    if size > RESULT_LIMIT:
        raise RuntimeError(f"Result export {size} bytes exceeds {RESULT_LIMIT}; refusing upload")
    print(
        json.dumps({"phase": "export", "json_bytes": size, "sha256": result["export"]["sha256"]}),
        flush=True,
    )
    return result


def unpack_result(result: dict, output: Path) -> dict:
    export = result["export"]
    if export["format"] != "tar.gz+base64":
        raise ValueError("Unknown export format")
    raw = base64.b64decode(export["data"], validate=True)
    if len(raw) != export["bytes"] or hashlib.sha256(raw).hexdigest() != export["sha256"]:
        raise ValueError("Export checksum mismatch")
    output.mkdir(parents=True, exist_ok=False)
    with tarfile.open(fileobj=io.BytesIO(raw), mode="r:gz") as archive:
        members = archive.getmembers()
        if len(members) > 4096 or sum(m.size for m in members) > 128 * 1024 * 1024:
            raise ValueError("Expanded export exceeds safety bounds")
        for member in members:
            target = (output / member.name).resolve()
            if not target.is_relative_to(output.resolve()) or not member.isfile():
                raise ValueError("Invalid export member")
        archive.extractall(output, filter="data")
    hashes = json.loads((output / "artifact-hashes.json").read_text())
    for name, expected in hashes.items():
        target = (output / name).resolve()
        if not target.is_relative_to(output.resolve()):
            raise ValueError("Invalid manifest path")
        if hashlib.sha256(target.read_bytes()).hexdigest() != expected:
            raise ValueError(f"File checksum mismatch: {name}")
    # Restore neutral render copies for the human review packet.
    key_path = output / "review-key.json"
    if key_path.exists():
        import shutil

        for pair in json.loads(key_path.read_text()):
            for side in ("A", "B"):
                source = output / pair[side] / pair["id"] / "render.png"
                if source.exists():
                    shutil.copy2(source, output / "review" / f"{pair['pair_id']}-{side}.png")
    return {"sha256": export["sha256"], "bytes": len(raw), "verified_files": len(hashes)}


def upload_result(result: dict, url: str, token: str) -> dict:
    """Send only this archive to the scoped receiver; return a tiny receipt."""
    import time
    from urllib.parse import urlparse
    from urllib.request import Request, urlopen

    target = urlparse(url)
    if target.scheme != "https" and not (
        target.scheme == "http" and target.hostname == "127.0.0.1"
    ):
        raise ValueError("Export requires HTTPS or a local test receiver")
    raw = base64.b64decode(result["export"]["data"], validate=True)
    digest = hashlib.sha256(raw).hexdigest()
    if digest != result["export"]["sha256"]:
        raise ValueError("Export checksum mismatch before upload")
    receipt = None
    for attempt in range(3):
        try:
            request = Request(
                url,
                data=raw,
                method="PUT",
                headers={
                    "Authorization": "Bearer " + token,
                    "Content-Type": "application/gzip",
                    "Content-Length": str(len(raw)),
                    "X-Content-SHA256": digest,
                },
            )
            with urlopen(request, timeout=120) as response:
                receipt = json.load(response)
            if receipt["sha256"] != digest or receipt["bytes"] != len(raw):
                raise ValueError("Receiver checksum mismatch")
            break
        except OSError:
            if attempt == 2:
                raise RuntimeError("Scoped archive upload failed after three attempts") from None
            time.sleep(2**attempt)
    summary = {k: v for k, v in result.items() if k != "export"}
    summary["export"] = {"format": "tar.gz", **receipt}
    return summary


def unpack_archive(raw: bytes, digest: str, output: Path) -> dict:
    return unpack_result(
        {
            "export": {
                "format": "tar.gz+base64",
                "bytes": len(raw),
                "sha256": digest,
                "data": base64.b64encode(raw).decode("ascii"),
            }
        },
        output,
    )
