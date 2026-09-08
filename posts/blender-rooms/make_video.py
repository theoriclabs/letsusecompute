"""Make a captioned MP4 and GIF from verified actual Blender camera-orbit clips."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path

import imageio_ffmpeg
from PIL import Image, ImageDraw, ImageFont

BG = "#f7f4ef"
INK = "#1a1a1a"
MUTED = "#625d55"
ACCENT = "#bc592c"
GREEN = "#326454"


def fonts(size, *, serif=False, bold=False):
    family = "Georgia" if serif else "Arial"
    name = family + (" Bold" if bold else "") + ".ttf"
    mac = Path("/System/Library/Fonts/Supplemental") / name
    fallback = Path("/usr/share/fonts/truetype/dejavu") / (
        ("DejaVuSerif" if serif else "DejaVuSans") + ("-Bold" if bold else "") + ".ttf"
    )
    for path in [mac, fallback]:
        if path.exists():
            return ImageFont.truetype(str(path), size)
    raise RuntimeError("Install DejaVu fonts or use the macOS Arial/Georgia fonts")


def text(draw, xy, value, size=34, *, color=INK, serif=False, bold=False):
    draw.text(xy, value, font=fonts(size, serif=serif, bold=bold), fill=color)


def title_frame(width=1080):
    canvas = Image.new("RGB", (width, width), BG)
    return canvas, ImageDraw.Draw(canvas)


def load_clips(artifact):
    manifest = json.loads((artifact / "media.json").read_text())
    clips = []
    for row in manifest:
        path = artifact / "media" / row["id"] / "orbit.mp4"
        if row["status"] != "completed" or row["frames"] != 36:
            raise RuntimeError("A required predetermined media scene did not render: " + row["id"])
        assert hashlib.sha256(path.read_bytes()).hexdigest() == row["video_sha256"]
        reader = imageio_ffmpeg.read_frames(str(path))
        meta = next(reader)
        frames = [Image.frombytes("RGB", tuple(meta["size"]), data) for data in reader]
        assert len(frames) == 36
        clips.append((row, frames))
    return clips


def grid_frame(clips, frame, *, size=1080):
    canvas, draw = title_frame()
    text(draw, (56, 42), "LET’S USE COMPUTE", 25, color=ACCENT, bold=True)
    text(draw, (56, 89), "Qwen writes tiny rooms", 60, serif=True, bold=True)
    text(draw, (56, 165), "9B model · 192 procedural training examples", 29, color=MUTED)
    edge, gap = 470, 28
    for i, (row, frames) in enumerate(clips):
        x = 56 + (i % 2) * (edge + gap)
        y = 233 + (i // 2) * 375
        # The rendered square is fully visible; no furniture is cropped.
        image = frames[frame % len(frames)].resize((340, 340), Image.Resampling.LANCZOS)
        canvas.paste(image, (x + 65, y))
        label = row["family"].replace("_", " ").title()
        status = "geometry passed" if row["geometry_valid"] else "geometry failed"
        text(
            draw,
            (x + 65, y + 343),
            label + " · " + status,
            20,
            color=GREEN if row["geometry_valid"] else ACCENT,
        )
    text(draw, (56, 1005), "Primary test: 30/32 rendered · 18/32 geometry-valid", 29, bold=True)
    return canvas.resize((size, size), Image.Resampling.LANCZOS) if size != 1080 else canvas


def room_frame(row, source, index):
    canvas, draw = title_frame()
    text(draw, (56, 42), "ACTUAL SAVED MODEL OUTPUT", 25, color=ACCENT, bold=True)
    text(draw, (56, 100), row["family"].replace("_", " ").title(), 70, serif=True, bold=True)
    canvas.paste(source.resize((780, 780), Image.Resampling.LANCZOS), (150, 202))
    draw.rectangle((0, 998, 1080, 1080), fill=BG)
    status = (
        "Geometry check passed" if row["geometry_valid"] else "Geometry failed: bounds and support"
    )
    text(draw, (56, 1020), status, 28, color=GREEN if row["geometry_valid"] else ACCENT, bold=True)
    text(draw, (915, 115), f"{index + 1}/4", 30, color=MUTED)
    return canvas


def result_frame(summary, spent):
    canvas, draw = title_frame()
    text(draw, (56, 48), "THEN WE GAVE IT ONE RETRY", 25, color=ACCENT, bold=True)
    text(draw, (56, 115), "Can errors help?", 75, serif=True, bold=True)
    text(draw, (56, 230), "Same saved adapter. 16 new validation prompts.", 34, color=MUTED)
    sft = summary["sft"]
    rows = [
        ("First attempt", sft["first"]["valid"]),
        ("Generic retry", sft["retry"]["within_two_valid"]),
        ("Measured feedback", sft["feedback"]["within_two_valid"]),
    ]
    for i, (label, value) in enumerate(rows):
        y = 350 + i * 145
        text(draw, (56, y), label, 38)
        text(draw, (825, y), f"{value}/16", 45, bold=True)
        draw.rounded_rectangle((56, y + 62, 1000, y + 82), radius=8, fill="#dfd9ce")
        if value:
            draw.rounded_rectangle(
                (56, y + 62, 56 + 944 * value / 16, y + 82), radius=8, fill=ACCENT
            )
    text(draw, (56, 825), "Geometry-valid outputs within the attempt budget.", 30, color=MUTED)
    text(
        draw, (56, 875), "Retries use extra inference. No aesthetic score claimed.", 29, color=MUTED
    )
    text(draw, (56, 944), "letsusecompute.com", 30, color=ACCENT)
    text(draw, (56, 992), f"Total experiment spend: ${spent:.2f}", 34, bold=True)
    return canvas


def build(artifact, output, budget):
    output.mkdir(parents=True, exist_ok=True)
    clips = load_clips(artifact)
    summary = json.loads((artifact / "summary.json").read_text())
    spent = json.loads(budget.read_text())["gpu_api_spend_usd"]
    fps = 24
    video = output / "qwen-blender-rooms.mp4"
    writer = imageio_ffmpeg.write_frames(
        str(video),
        (1080, 1080),
        fps=fps,
        codec="libx264",
        quality=8,
        macro_block_size=2,
        output_params=["-movflags", "+faststart", "-pix_fmt", "yuv420p"],
    )
    writer.send(None)
    for frame in range(3 * fps):
        writer.send(grid_frame(clips, frame // 2).tobytes())
    for i, (row, frames) in enumerate(clips):
        for frame in range(3 * fps):
            writer.send(room_frame(row, frames[frame // 2], i).tobytes())
    result = result_frame(summary, spent)
    for _ in range(6 * fps):
        writer.send(result.tobytes())
    writer.close()
    loop = output / "rooms-loop.mp4"
    writer = imageio_ffmpeg.write_frames(
        str(loop),
        (720, 720),
        fps=12,
        codec="libx264",
        quality=8,
        macro_block_size=2,
        output_params=["-movflags", "+faststart", "-pix_fmt", "yuv420p"],
    )
    writer.send(None)
    for frame in range(36):
        writer.send(grid_frame(clips, frame, size=720).tobytes())
    writer.close()
    gif = output / "qwen-blender-rooms.gif"
    subprocess.run(
        [
            imageio_ffmpeg.get_ffmpeg_exe(),
            "-y",
            "-loglevel",
            "error",
            "-i",
            str(loop),
            "-filter_complex",
            "[0:v]split[a][b];[a]palettegen=max_colors=128[p];[b][p]paletteuse=dither=bayer:bayer_scale=3",
            "-loop",
            "0",
            str(gif),
        ],
        check=True,
        timeout=120,
    )
    grid_frame(clips, 0).save(output / "poster.jpg", quality=95, subsampling=0)
    (output / "captions.vtt").write_text(
        "WEBVTT\n\n00:00.000 --> 00:03.000\nQwen3.5-9B, fine-tuned on 192 procedural Blender programs.\n\n00:03.000 --> 00:06.000\nBedroom: actual model output, geometry check passed.\n\n00:06.000 --> 00:09.000\nStudio: rendered, but the bookshelf fails the floor-bound check.\n\n00:09.000 --> 00:12.000\nKitchen: actual model output, geometry check passed.\n\n00:12.000 --> 00:15.000\nLiving room: actual model output, geometry check passed.\n\n00:15.000 --> 00:21.000\n"
        + f"Validation geometry: first attempt {summary['sft']['first']['valid']}/16; generic retry {summary['sft']['retry']['within_two_valid']}/16; measured feedback {summary['sft']['feedback']['within_two_valid']}/16.\n"
    )
    evidence = {
        "encoder_version": imageio_ffmpeg.get_ffmpeg_version(),
        "video_seconds": 21,
        "video_size": [1080, 1080],
        "video_fps": 24,
        "gif_seconds": 3,
        "gif_size": [720, 720],
        "gif_fps": 12,
        "audio": "none; captions are burned into the video",
        "total_spend_usd": spent,
        "sources": [row for row, _ in clips],
        "claim": "Original primary-test geometry, camera motion added for presentation. Follow-up metrics are separate validation results.",
        "files": {
            p.name: {
                "bytes": p.stat().st_size,
                "sha256": hashlib.sha256(p.read_bytes()).hexdigest(),
            }
            for p in output.iterdir()
            if p.is_file() and p.name != "provenance.json"
        },
    }
    (output / "provenance.json").write_text(json.dumps(evidence, indent=2) + "\n")
    print(
        json.dumps(
            {
                "video": str(video),
                "gif": str(gif),
                "video_bytes": video.stat().st_size,
                "gif_bytes": gif.stat().st_size,
            }
        )
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--budget", type=Path, required=True)
    args = parser.parse_args()
    build(args.artifact, args.output, args.budget)
