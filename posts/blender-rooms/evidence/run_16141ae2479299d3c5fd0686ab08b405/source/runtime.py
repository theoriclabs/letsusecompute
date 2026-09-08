"""Pin Python 3.11 for bpy independently of Compute's runner interpreter."""

from __future__ import annotations

import ctypes.util
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

PACKAGES = (
    "torch==2.8.0",
    "transformers==5.16.1",
    "peft==0.20.0",
    "accelerate==1.14.0",
    "datasets==5.0.1",
    "bpy==4.5.3",
    "Pillow==11.3.0",
    "fla-core==0.5.2",
)


def run_child(command, *, early_path=None, export_url=None, export_token=None):
    """Watch for an atomic checkpoint export while the GPU child evaluates."""
    from export import upload_result

    attempted = False
    with subprocess.Popen(command) as child:
        while True:
            if early_path is not None and early_path.exists() and not attempted:
                attempted = True
                try:
                    receipt = upload_result(
                        json.loads(early_path.read_text()),
                        export_url.rsplit("/", 1)[0] + "/sft-checkpoint",
                        export_token,
                    )
                    print(
                        json.dumps({"phase": "checkpoint_durable", "export": receipt["export"]}),
                        flush=True,
                    )
                except (RuntimeError, ValueError, OSError):
                    print(
                        json.dumps(
                            {"phase": "checkpoint_transfer_failed", "final_export_will_retry": True}
                        ),
                        flush=True,
                    )
            try:
                code = child.wait(timeout=1)
                break
            except subprocess.TimeoutExpired:
                continue
        if code:
            raise subprocess.CalledProcessError(code, command)


def launch(*, export_url, export_token, **kwargs):
    if sys.platform != "linux":
        raise RuntimeError("This entrypoint bootstraps a disposable Linux GPU worker")
    # The curated CUDA image's python3 may be 3.10. Blender 4.5 wheels are
    # CPython 3.11 only. Keep its environment separate from the Compute runner.
    if not all(
        ctypes.util.find_library(name) for name in ("GL", "EGL", "X11", "Xi", "Xrender", "SM")
    ):
        if os.geteuid() != 0:
            raise RuntimeError("Blender system libraries are missing and this worker is not root")
        subprocess.run(["apt-get", "update", "-qq"], check=True, timeout=180)
        subprocess.run(
            [
                "apt-get",
                "install",
                "-y",
                "--no-install-recommends",
                "libgl1",
                "libegl1",
                "libx11-6",
                "libxi6",
                "libxrender1",
                "libxfixes3",
                "libxkbcommon0",
                "libsm6",
                "libice6",
                "libgomp1",
                "libxxf86vm1",
                "libxrandr2",
                "libxcursor1",
            ],
            check=True,
            timeout=180,
        )
    work = Path(tempfile.mkdtemp(prefix="blender-runtime-"))
    python = work / "venv/bin/python"
    uv = [sys.executable, "-m", "uv"]
    subprocess.run([*uv, "venv", "--python", "3.11", str(work / "venv")], check=True, timeout=180)
    subprocess.run(
        [*uv, "pip", "install", "--python", str(python), *PACKAGES], check=True, timeout=600
    )
    subprocess.run(
        [
            str(python),
            "-c",
            'import bpy, torch; print("Blender", bpy.app.version_string, "Torch", torch.__version__, flush=True)',
        ],
        check=True,
        timeout=60,
    )
    if kwargs.get("action") == "evaluate":
        from urllib.request import Request, urlopen

        from export import RESULT_LIMIT

        source_url = kwargs.pop("training_archive_url")
        request = Request(source_url, headers={"Authorization": "Bearer " + export_token})
        with urlopen(request, timeout=120) as response:
            raw = response.read(RESULT_LIMIT + 1)
        if len(raw) > RESULT_LIMIT:
            raise ValueError("Training archive exceeds transfer bound")
        training_archive = work / "training.tar.gz"
        training_archive.write_bytes(raw)
        kwargs["training_archive_path"] = str(training_archive)
    config = work / "args.json"
    config.write_text(json.dumps(kwargs))
    result = work / "result.json"
    run_child(
        [str(python), str(Path(__file__).resolve()), str(config), str(result)],
        early_path=(
            Path("/tmp/blender-result-export/sft/early-export.json")
            if not kwargs.get("smoke") and kwargs.get("action", "train") == "train"
            else None
        ),
        export_url=export_url,
        export_token=export_token,
    )
    from export import upload_result

    summary = upload_result(json.loads(result.read_text()), export_url, export_token)
    print(json.dumps({"phase": "download_durable", "export": summary["export"]}), flush=True)
    return summary


if __name__ == "__main__":
    from pipeline import train_impl

    args = json.loads(Path(sys.argv[1]).read_text())
    action = args.pop("action", "train")
    if action == "evaluate":
        from evaluate import evaluate_from_archive

        result = evaluate_from_archive(**args)
    elif action == "train":
        result = train_impl(**args)
    else:
        raise ValueError("Unknown runtime action")
    Path(sys.argv[2]).write_text(json.dumps(result, allow_nan=False))
