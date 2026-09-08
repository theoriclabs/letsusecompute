"""Train Qwen3.5-9B to write Blender rooms on a fresh Compute GPU.

Use submit.py to provide the scoped transfer credentials and enforce a live
quoted budget; see README.md. Select a current NVIDIA 80 GB offer from the
Compute catalog. No Compute account secrets are injected into workers.
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


@app.function(gpu="runpod/H100-SXM", image=image, timeout=1800)
def control(
    export_url: str,
    export_token: str,
    training_archive_url: str,
    training_sha256: str,
    training_run_id: str,
    count: int = 8,
):
    """Post-hoc validation-only API-hint control; the primary test is unchanged."""
    return launch(
        export_url=export_url,
        export_token=export_token,
        action="evaluate",
        training_archive_url=training_archive_url,
        training_sha256=training_sha256,
        training_run_id=training_run_id,
        count=count,
        api_hint_control=True,
    )
