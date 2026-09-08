"""Encode a shareable slideshow from the four predetermined, saved test renders."""

import hashlib
import json
import subprocess
from pathlib import Path

import imageio_ffmpeg
from PIL import Image

from make_video import ACCENT, MUTED, grid_frame, room_frame, text, title_frame

ROOT = Path(__file__).resolve().parent


def build():
    output = ROOT / "assets/social"
    output.mkdir(parents=True, exist_ok=True)
    originals = json.loads((ROOT / "assets/results/provenance.json").read_text())
    clips = []
    sources = []
    for family in ["bedroom", "studio", "kitchen", "living_room"]:
        name = f"test-{family}-l7-p0-sft.png"
        path = ROOT / "assets/results" / name
        original = next(r for r in originals if r["file"] == name)
        assert hashlib.sha256(path.read_bytes()).hexdigest() == original["sha256"]
        row = {"id": f"{family}-l7-p0", "family": family, "geometry_valid": family != "studio"}
        clips.append((row, [Image.open(path).convert("RGB")]))
        sources.append({**row, "file": "../results/" + name, "sha256": original["sha256"]})
    overview = grid_frame(clips, 0)
    overview.save(output / "poster.jpg", quality=95, subsampling=0)
    result, draw = title_frame()
    text(draw, (56, 65), "THE COMPLETED $7.82 PILOT", 28, color=ACCENT, bold=True)
    text(draw, (56, 145), "It learned to render.", 66, serif=True, bold=True)
    text(draw, (56, 230), "Geometry is harder.", 66, serif=True, bold=True)
    text(draw, (56, 405), "30/32", 110, bold=True)
    text(draw, (470, 446), "test prompts rendered", 37)
    text(draw, (56, 585), "18/32", 110, bold=True)
    text(draw, (470, 626), "passed geometry checks", 37)
    text(draw, (56, 820), "4 layout groups with correlated palette variants.", 31, color=MUTED)
    text(draw, (56, 870), "No independent aesthetic score.", 31, color=MUTED)
    text(draw, (56, 984), "letsusecompute.com", 34, color=ACCENT, bold=True)
    cards = (
        [overview]
        + [room_frame(row, frames[0], i) for i, (row, frames) in enumerate(clips)]
        + [result]
    )
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
    # Three seconds per card, with a quarter-second crossfade at transitions.
    for i, card in enumerate(cards):
        for frame in range(3 * fps):
            displayed = (
                Image.blend(cards[i - 1], card, (frame + 1) / 6) if i and frame < 6 else card
            )
            writer.send(displayed.tobytes())
    writer.close()
    gif = output / "qwen-blender-rooms.gif"
    subprocess.run(
        [
            imageio_ffmpeg.get_ffmpeg_exe(),
            "-y",
            "-loglevel",
            "error",
            "-i",
            str(video),
            "-filter_complex",
            "[0:v]fps=6,scale=640:640,split[a][b];[a]palettegen=max_colors=128[p];[b][p]paletteuse=dither=bayer:bayer_scale=3",
            "-loop",
            "0",
            str(gif),
        ],
        check=True,
        timeout=120,
    )
    captions = [
        "Qwen3.5-9B, fine-tuned on 192 procedural Blender programs.",
        "Bedroom: actual saved render, geometry passed.",
        "Studio: actual saved render, geometry failed on bounds and support.",
        "Kitchen: actual saved render, geometry passed.",
        "Living room: actual saved render, geometry passed.",
        "Completed pilot: 30/32 renders, 18/32 geometry passes, $7.82 in GPU charges.",
    ]
    (output / "captions.vtt").write_text(
        "WEBVTT\n\n"
        + "\n\n".join(
            f"00:{i * 3:02}.000 --> 00:{(i + 1) * 3:02}.000\n{s}" for i, s in enumerate(captions)
        )
        + "\n"
    )
    (output / "provenance.json").write_text(
        json.dumps(
            {
                "presentation": "Crossfade slideshow of unchanged saved primary-test still renders; no generated imagery or 3D camera motion.",
                "selection": "First palette in every primary-test family, selected before follow-up.",
                "primary_test_run": "run_f62cfd31c65501be529a975b1eb6a139",
                "pilot_cost_usd": 7.82,
                "scope": "Original completed pilot only; excludes unfinished repair experiment.",
                "video_seconds": 18,
                "video_size": [1080, 1080],
                "video_fps": 24,
                "gif_seconds": 18,
                "gif_size": [640, 640],
                "audio": "none; visible text and VTT captions",
                "encoder_version": imageio_ffmpeg.get_ffmpeg_version(),
                "sources": sources,
                "files": {
                    p.name: {
                        "bytes": p.stat().st_size,
                        "sha256": hashlib.sha256(p.read_bytes()).hexdigest(),
                    }
                    for p in output.iterdir()
                    if p.is_file() and p.name != "provenance.json"
                },
            },
            indent=2,
        )
        + "\n"
    )
    url = "https://letsusecompute.com/posts/blender-rooms/"
    short = (
        "Fine-tuned Qwen3.5-9B on 192 procedural Blender programs.\n\n30/32 test renders. 18/32 geometry passes. $7.82 on Compute.\n\nActual outputs, including the failures:\n"
        + url
    )
    assert len(short) <= 280
    (ROOT / "SOCIAL.md").write_text(
        "# Shareable media and copy\n\n[MP4](assets/social/qwen-blender-rooms.mp4): 18 seconds, 1080×1080, H.264.\n[GIF](assets/social/qwen-blender-rooms.gif): 18-second loop, 640×640.\n[Poster](assets/social/poster.jpg) · [Captions](assets/social/captions.vtt) · [Provenance](assets/social/provenance.json).\n\n## Short caption\n\n"
        + short
        + "\n\n## Longer post\n\nI fine-tuned Qwen3.5-9B to write low-poly rooms in Blender.\n\n192 procedural examples, one H100, and real saved adapter weights. On the primary test, 30/32 prompts rendered and 18/32 passed geometry checks. The completed pilot cost $7.82.\n\nThe slideshow shows the first palette from each test family, including the studio that failed. These are four layout groups with correlated color variants. A separate compatibility diagnostic broke all eight adapter programs; geometry and robustness remain open problems. We have no independent aesthetic score.\n\nCode, actual outputs, and limitations: "
        + url
        + "\n\nInspired by Merve’s Blender challenge: https://huggingface.co/buckets/merve/blender-grpo-gepa\n\n## Alt text\n\nA slideshow of four actual renders from a fine-tuned Qwen3.5-9B model: a bedroom, studio, kitchen, and living room. Labels show that the pictured studio failed geometry checks; the other three pictured examples passed. The final card reports 30/32 test renders, 18/32 geometry passes, and $7.82 in completed-pilot GPU charges.\n\n## Scope\n\nThis media covers the completed pilot. The separate repair experiment is excluded. This file is prepared copy; no social post has been sent.\n"
    )
    print(
        json.dumps(
            {
                "mp4_bytes": video.stat().st_size,
                "gif_bytes": gif.stat().st_size,
                "short_caption_characters": len(short),
            }
        )
    )


if __name__ == "__main__":
    build()
