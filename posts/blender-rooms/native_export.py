"""Compact still-image previews, below the deployed supervisor buffer."""

import hashlib
import json
from pathlib import Path

from export import bundle_result

MAX_RESULT_BYTES = 4 * 1024 * 1024


def assert_result_bound(value):
    size = len(json.dumps(value, allow_nan=False).encode()) + 1
    # The supervisor wraps result JSON in base64. Leave >2 MiB for logs/envelope.
    if size > MAX_RESULT_BYTES:
        raise ValueError(f"Native result {size} exceeds the 4 MiB supervisor-safe bound")
    return size


def compact_result(root: Path, summary):
    from PIL import Image

    transports = []
    for path in root.rglob("*.png"):
        with Image.open(path) as image:
            rgb = image.convert("RGB")
            target = path.with_suffix(".jpg")
            rgb.save(target, "JPEG", quality=90, subsampling=0, optimize=True)
            transports.append(
                {
                    "source": str(path.relative_to(root)),
                    "file": str(target.relative_to(root)),
                    "size": list(rgb.size),
                    "original_pixel_sha256": hashlib.sha256(rgb.tobytes()).hexdigest(),
                    "preview_encoding": "JPEG quality 90, no chroma subsampling; geometry measured before compression",
                }
            )
        path.unlink()
    (root / "image-transports.json").write_text(json.dumps(transports, indent=2) + "\n")
    result = bundle_result(root, summary)
    assert_result_bound(result)
    return result
