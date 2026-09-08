"""Render camera motion around frozen actual model scenes on a disposable worker."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

from media_sources import SCENES


def render_media(output):
    output.mkdir(parents=True)
    manifest = []
    for index, scene in enumerate(SCENES):
        destination = output / scene["id"]
        destination.mkdir()
        env = {k: v for k, v in os.environ.items() if k in {"PATH", "LD_LIBRARY_PATH", "TMPDIR"}}
        env.update({"HOME": str(destination), "PYTHONNOUSERSITE": "1", "OMP_NUM_THREADS": "4"})
        with (destination / "video-render.log").open("w") as log:
            try:
                process = subprocess.run(
                    [
                        sys.executable,
                        str(Path(__file__).resolve()),
                        "worker",
                        str(index),
                        str(destination),
                    ],
                    env=env,
                    cwd=destination,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    timeout=420,
                )
                status = "completed" if process.returncode == 0 else "failed"
            except subprocess.TimeoutExpired:
                status = "timeout"
        frames = sorted(destination.glob("frame-*.png"))
        frame_hashes = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in frames}
        video = destination / "orbit.mp4"
        if len(frames) == 36:
            import imageio_ffmpeg

            subprocess.run(
                [
                    imageio_ffmpeg.get_ffmpeg_exe(),
                    "-y",
                    "-loglevel",
                    "error",
                    "-framerate",
                    "12",
                    "-i",
                    str(destination / "frame-%03d.png"),
                    "-c:v",
                    "libx264",
                    "-crf",
                    "20",
                    "-pix_fmt",
                    "yuv420p",
                    "-movflags",
                    "+faststart",
                    str(video),
                ],
                check=True,
                timeout=60,
            )

        manifest.append(
            {k: v for k, v in scene.items() if k != "raw"}
            | {
                "status": status,
                "frames": len(frames),
                "fps": 12,
                "camera_motion": "36-frame gentle orbit, +/-12 degrees; original geometry unchanged",
                "frame_hashes": frame_hashes,
                "video_sha256": hashlib.sha256(video.read_bytes()).hexdigest()
                if video.exists()
                else None,
            }
        )
        for frame_path in frames:
            frame_path.unlink()
        print(
            json.dumps(
                {"phase": "media", "id": scene["id"], "frames": len(frames), "status": status}
            ),
            flush=True,
        )
    return manifest


def worker(index, output):
    import math

    import bpy
    from mathutils import Vector

    from render import blender_worker, extract_code

    scene = SCENES[index]
    assert hashlib.sha256(scene["raw"].encode()).hexdigest() == scene["raw_sha256"]
    source = output / "program.py"
    source.write_text(extract_code(scene["raw"]))
    blender_worker(source, output, resolution=512, samples=12)
    camera = bpy.context.scene.camera
    angle = math.atan2(-10, 8)
    radius = math.hypot(8, -10)
    for frame in range(36):
        theta = angle + math.radians(12) * math.sin(2 * math.pi * frame / 36)
        camera.location = (radius * math.cos(theta), radius * math.sin(theta), 8)
        camera.rotation_euler = (
            (Vector((0, 0, 1)) - camera.location).to_track_quat("-Z", "Y").to_euler()
        )
        bpy.context.scene.render.filepath = str((output / f"frame-{frame:03d}.png").resolve())
        bpy.ops.render.render(write_still=True)


if __name__ == "__main__":
    if sys.platform != "linux":
        raise RuntimeError("Execute generated scenes only on a disposable Linux worker")
    worker(int(sys.argv[2]), Path(sys.argv[3]))
