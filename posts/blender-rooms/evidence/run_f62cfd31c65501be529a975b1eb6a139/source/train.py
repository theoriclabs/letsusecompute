"""Train Qwen3.5-9B to write Blender rooms on a fresh Compute GPU.

compute run train.py::smoke --gpu runpod/A100-PCIe-80GB --timeout 1800 --wait
compute run train.py::train --gpu runpod/A100-PCIe-80GB --timeout 7200 --wait

Use a current 80 GB NVIDIA SKU from `compute gpu list`; the example SKU must be
checked against the live catalog before spending. No secrets are required.
"""

import compute

from runtime import launch

app = compute.App("blender-rooms")
image = compute.Image.cuda_pytorch().pip_install("uv==0.8.22")


@app.function(gpu="runpod/A100-PCIe-80GB", image=image, timeout=1800)
def smoke(export_url: str, export_token: str, max_length: int = 6144):
    return launch(
        export_url=export_url, export_token=export_token, smoke=True, max_length=max_length
    )


@app.function(gpu="runpod/A100-PCIe-80GB", image=image, timeout=7200)
def train(
    export_url: str,
    export_token: str,
    max_length: int = 6144,
    epochs: float = 3.0,
    eval_count: int = 8,
):
    return launch(
        export_url=export_url,
        export_token=export_token,
        max_length=max_length,
        epochs=epochs,
        eval_count=eval_count,
    )


@app.function(gpu="runpod/A100-PCIe-80GB", image=image, timeout=10800)
def test(
    export_url: str,
    export_token: str,
    training_archive_url: str,
    training_sha256: str,
    training_run_id: str,
    count: int = 32,
):
    return launch(
        export_url=export_url,
        export_token=export_token,
        action="evaluate",
        training_archive_url=training_archive_url,
        training_sha256=training_sha256,
        training_run_id=training_run_id,
        count=count,
    )
