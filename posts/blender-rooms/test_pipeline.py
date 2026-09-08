"""Offline contracts plus a real, tiny Qwen3.5 adapter round trip on CPU."""

import ast
import json

import pytest

from pipeline import collate, encode_row, lora_targets, select_balanced, training_arguments
from render import check_code, extract_code, geometry_metrics, render_code
from rooms import FAMILIES, build_rows, write_dataset


def test_split_and_source_integrity(tmp_path):
    rows, manifest = write_dataset(tmp_path)
    assert len(rows) == 256
    assert [manifest["splits"][s]["rows"] for s in ("train", "validation", "test")] == [192, 32, 32]
    assert len({r["code_sha256"] for r in rows}) == len(rows)
    groups = {
        split: {r["layout_group"] for r in rows if r["split"] == split}
        for split in ("train", "validation", "test")
    }
    assert not (
        groups["train"] & groups["validation"]
        or groups["train"] & groups["test"]
        or groups["validation"] & groups["test"]
    )
    for row in rows:
        ast.parse(row["messages"][-1]["content"])
        check_code(row["messages"][-1]["content"])
    assert json.loads((tmp_path / "manifest.json").read_text()) == manifest


def test_balanced_evaluation():
    rows = [r for r in build_rows() if r["split"] == "validation"]
    selected = select_balanced(rows, 8)
    assert {f: sum(r["family"] == f for r in selected) for f in FAMILIES} == dict.fromkeys(
        FAMILIES, 2
    )
    with pytest.raises(ValueError):
        select_balanced(rows, 5)


@pytest.mark.parametrize(
    "code",
    [
        'import os\nos.system("echo unsafe")',
        'import bpy\nbpy.ops.wm.open_mainfile(filepath="x")',
        "import bpy\nbpy.app.handlers.load_post.append(print)",
        "x = ().__class__",
        'exec("pass")',
    ],
)
def test_reject_obvious_host_access(code):
    with pytest.raises(ValueError):
        check_code(code)


def test_completion_parsing():
    assert extract_code("<think>analysis</think>\n```python\nimport bpy\n```") == "import bpy"
    with pytest.raises(ValueError):
        extract_code("```python\nimport bpy")


def test_failed_render_cannot_reuse_a_stale_image(tmp_path):
    (tmp_path / "render.png").write_bytes(b"old render")
    (tmp_path / "scene.json").write_text("[]")
    result = render_code("import os", tmp_path, ["bed"])
    assert not result["valid"]
    assert not (tmp_path / "render.png").exists()
    assert not (tmp_path / "scene.json").exists()


def test_geometry_does_not_reward_missing_or_floating_meshes():
    floor = {"name": "floor", "min": [-2.8, -2.8, 0.0], "max": [2.8, 2.8, 0.2], "polygons": 6}
    wall_back = {
        "name": "wall_back",
        "min": [-2.8, 2.625, 0.2],
        "max": [2.8, 2.775, 3.2],
        "polygons": 6,
    }
    wall_side = {
        "name": "wall_side",
        "min": [-2.775, -2.8, 0.2],
        "max": [-2.625, 2.8, 3.2],
        "polygons": 6,
    }
    bed = {"name": "bed_frame", "min": [0, 0, 0.2], "max": [1, 1, 0.5], "polygons": 6}
    scene = [floor, wall_back, wall_side, bed]
    assert geometry_metrics(scene, ["bed"])["valid"]
    bad = [*scene[:-1], {**bed, "min": [0, 0, 1.0], "max": [1, 1, 1.5]}]
    assert "unsupported:bed_frame" in geometry_metrics(bad, ["bed"])["failures"]
    assert not geometry_metrics(scene, ["bed", "chair"])["valid"]
    bad = [*scene[:-1], {**bed, "max": [3.1, 1, 0.5]}]
    assert "bounds:bed_frame" in geometry_metrics(bad, ["bed"])["failures"]
    assert not geometry_metrics([bed], ["bed"])["valid"]


def test_completion_mask_and_padding():
    torch = pytest.importorskip("torch")

    class Tokenizer:
        eos_token_id = 2

        def apply_chat_template(self, *args, **kwargs):
            return "prompt"

        def __call__(self, text, **kwargs):
            return {"input_ids": [3] * len(text)}

    row = {"id": "test", "messages": [{}, {}, {"content": "code"}]}
    encoded = encode_row(row, Tokenizer(), 20)
    assert encoded["labels"][:6] == [-100] * 6
    assert encoded["labels"][6:] == [3, 3, 3, 3, 2]
    with pytest.raises(ValueError):
        encode_row(row, Tokenizer(), 8)
    shorter = {k: v[:-2] for k, v in encoded.items()}
    batch = collate([encoded, shorter], 0)
    assert torch.equal(batch["labels"][1, -2:], torch.tensor([-100, -100]))
    assert batch["attention_mask"][1, -1].item() == 0


def test_real_qwen_adapter_step_and_reload(tmp_path):
    torch = pytest.importorskip("torch")
    transformers = pytest.importorskip("transformers")
    peft = pytest.importorskip("peft")
    torch.manual_seed(42)
    config = transformers.Qwen3_5Config(
        text_config={
            "vocab_size": 256,
            "hidden_size": 32,
            "intermediate_size": 64,
            "num_hidden_layers": 4,
            "num_attention_heads": 2,
            "num_key_value_heads": 1,
            "head_dim": 16,
            "linear_num_key_heads": 2,
            "linear_num_value_heads": 2,
            "linear_key_head_dim": 16,
            "linear_value_head_dim": 16,
            "max_position_embeddings": 128,
            "pad_token_id": 0,
            "eos_token_id": 2,
            "rope_parameters": {
                "rope_type": "default",
                "rope_theta": 10000.0,
                "partial_rotary_factor": 0.5,
                "mrope_section": [2, 1, 1],
            },
        },
        vision_config={
            "depth": 1,
            "hidden_size": 32,
            "intermediate_size": 64,
            "num_heads": 2,
            "out_hidden_size": 32,
            "num_position_embeddings": 16,
        },
        image_token_id=253,
        video_token_id=254,
        vision_start_token_id=251,
        vision_end_token_id=252,
    )
    base = transformers.Qwen3_5ForConditionalGeneration(config)
    targets = lora_targets(base)
    assert any("linear_attn.in_proj_qkv" in name for name in targets)
    assert any("self_attn.q_proj" in name for name in targets)
    assert not any("visual" in name for name in targets)
    original = {k: v.detach().clone() for k, v in base.state_dict().items()}
    model = peft.get_peft_model(
        base, peft.LoraConfig(r=2, lora_alpha=4, target_modules=targets, task_type="CAUSAL_LM")
    )
    before = {k: v.detach().clone() for k, v in model.named_parameters() if v.requires_grad}
    ids = torch.randint(3, 100, (1, 24))
    labels = ids.clone()
    labels[:, :8] = -100
    optimizer = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad), lr=0.001)
    model.train()
    loss = model(input_ids=ids, labels=labels).loss
    assert torch.isfinite(loss)
    loss.backward()
    optimizer.step()
    assert any(not torch.equal(before[k], v) for k, v in model.named_parameters() if k in before)
    assert all(not p.requires_grad for n, p in model.named_parameters() if ".visual." in n)
    model.eval()
    with torch.no_grad():
        expected = model(input_ids=ids).logits
    model.save_pretrained(tmp_path / "adapter")
    restored = transformers.Qwen3_5ForConditionalGeneration(config)
    restored.load_state_dict(original)
    restored = peft.PeftModel.from_pretrained(restored, tmp_path / "adapter").eval()
    with torch.no_grad():
        actual = restored(input_ids=ids).logits
    torch.testing.assert_close(actual, expected)
    # Check the real hybrid-attention generation path with unequal prompt lengths.
    from tokenizers import Tokenizer, models, pre_tokenizers

    from inference import batch_tokens, completion_tokens

    backend = Tokenizer(
        models.WordLevel(
            {"[PAD]": 0, "[UNK]": 1, "[EOS]": 2, "room": 3, "blue": 4, "small": 5},
            unk_token="[UNK]",
        )
    )
    backend.pre_tokenizer = pre_tokenizers.Whitespace()
    tokenizer = transformers.PreTrainedTokenizerFast(
        tokenizer_object=backend, pad_token="[PAD]", eos_token="[EOS]", unk_token="[UNK]"
    )
    prompts = ["room", "small blue room"] * 4
    batched = batch_tokens(model, tokenizer, prompts, 4)
    singles = [batch_tokens(model, tokenizer, [prompt], 4)[0] for prompt in prompts]
    assert batched == singles
    assert tokenizer.padding_side == "right"
    assert completion_tokens([3, 2, 0, 0], 2) == [3, 2]
    assert completion_tokens([3, 4, 5, 6], 2) == [3, 4, 5, 6]
    # Exercise the pinned Trainer API and gradient-checkpointing path too.
    restored = peft.PeftModel.from_pretrained(
        transformers.Qwen3_5ForConditionalGeneration(config),
        tmp_path / "adapter",
        is_trainable=True,
    )
    restored.config.use_cache = False
    restored.enable_input_require_grads()
    data = [
        {"input_ids": ids[0].tolist(), "attention_mask": [1] * 24, "labels": labels[0].tolist()}
    ]
    args = training_arguments(tmp_path / "trainer", smoke=True, train_rows=1, use_cpu=True)
    assert args.warmup_steps == 0
    full_args = training_arguments(tmp_path / "full", smoke=False, train_rows=192, use_cpu=True)
    assert full_args.warmup_steps == 4
    assert full_args.gradient_accumulation_steps == 8

    trainer = transformers.Trainer(
        model=restored,
        args=args,
        train_dataset=data,
        eval_dataset=data,
        data_collator=lambda rows: collate(rows, 0),
    )
    result = trainer.train()
    assert result.global_step == 1
    assert result.training_loss > 0
    assert trainer.state.best_model_checkpoint is not None


def test_result_export_round_trip_and_corruption(tmp_path):
    import copy

    from export import bundle_result, unpack_result

    source = tmp_path / "source"
    (source / "adapter").mkdir(parents=True)
    (source / "adapter/adapter_model.safetensors").write_bytes(b"trained weights")
    (source / "adapter/adapter_config.json").write_text('{"r": 4}')
    (source / "adapter/tokenizer.json").write_text("fetch pinned tokenizer instead")
    (source / "scene.blend").write_bytes(b"not exported")
    result = bundle_result(source, {"mode": "test"})
    verified = unpack_result(result, tmp_path / "download")
    assert verified["verified_files"] == 2
    assert (
        tmp_path / "download/adapter/adapter_model.safetensors"
    ).read_bytes() == b"trained weights"
    assert not (tmp_path / "download/scene.blend").exists()
    corrupt = copy.deepcopy(result)
    corrupt["export"]["sha256"] = "0" * 64
    with pytest.raises(ValueError, match="checksum"):
        unpack_result(corrupt, tmp_path / "bad")


def test_scoped_receiver_large_upload_auth_checksum_and_immutability(tmp_path):
    import base64
    import hashlib
    import os
    import threading
    from urllib.error import HTTPError
    from urllib.request import Request, urlopen

    from export import upload_result
    from receiver import server

    key_file = tmp_path / "worker.key"
    receiver = server(tmp_path / "received", key_file, 0)
    thread = threading.Thread(target=receiver.serve_forever, daemon=True)
    thread.start()
    url = f"http://127.0.0.1:{receiver.server_port}/upload/probe"
    token = key_file.read_text()
    raw = os.urandom(8 * 1024 * 1024)
    digest = hashlib.sha256(raw).hexdigest()
    result = {
        "mode": "transport-test",
        "export": {
            "data": base64.b64encode(raw).decode(),
            "sha256": digest,
        },
    }
    try:
        with pytest.raises(HTTPError) as error:
            urlopen(Request(url, data=b"x", method="PUT"))
        assert error.value.code == 401
        receipt = upload_result(result, url, token)
        assert len(json.dumps(receipt)) < 1024
        assert receipt["export"]["sha256"] == digest
        assert (tmp_path / "received/probe.tar.gz").read_bytes() == raw
        assert upload_result(result, url, token) == receipt
        for body, checksum, status in [
            (b"changed", hashlib.sha256(b"changed").hexdigest(), 409),
            (b"wrong", "0" * 64, 400),
        ]:
            request = Request(
                url,
                data=body,
                method="PUT",
                headers={
                    "Authorization": "Bearer " + token,
                    "X-Content-SHA256": checksum,
                },
            )
            with pytest.raises(HTTPError) as error:
                urlopen(request)
            assert error.value.code == status
        with pytest.raises(HTTPError) as error:
            urlopen(
                Request(
                    url.replace("upload/probe", "download/../../private"),
                    headers={"Authorization": "Bearer " + token},
                )
            )
        assert error.value.code == 404
        with urlopen(
            Request(url.replace("upload", "download"), headers={"Authorization": "Bearer " + token})
        ) as response:
            assert hashlib.sha256(response.read()).hexdigest() == digest
    finally:
        receiver.shutdown()
        receiver.server_close()
        thread.join(timeout=5)


def test_checkpoint_transfers_even_if_evaluation_child_fails(tmp_path, monkeypatch):
    import subprocess
    import sys

    import export
    from runtime import run_child

    early = tmp_path / "early-export.json"
    early.write_text(json.dumps({"mode": "sft_checkpoint", "export": {"data": "ready"}}))
    calls = []

    def upload(result, url, token):
        calls.append((result["mode"], url, token))
        return {"export": {"sha256": "verified"}}

    monkeypatch.setattr(export, "upload_result", upload)
    with pytest.raises(subprocess.CalledProcessError):
        run_child(
            [sys.executable, "-c", "raise SystemExit(1)"],
            early_path=early,
            export_url="https://example.test/upload/sft",
            export_token="scoped-test-token",
        )
    assert calls == [
        ("sft_checkpoint", "https://example.test/upload/sft-checkpoint", "scoped-test-token")
    ]


def test_geometry_modifier_operator_matches_data_api(tmp_path):
    row = next(r for r in build_rows() if r["family"] == "bedroom" and r["split"] == "train")
    original = row["messages"][-1]["content"]
    equivalent = original.replace(
        "bevel = obj.modifiers.new('edge', 'BEVEL')",
        "bpy.ops.object.mode_set(mode='EDIT')\n    bpy.ops.mesh.select_all(action='SELECT')\n    bpy.ops.object.mode_set(mode='OBJECT')\n    bpy.ops.object.parent_clear(type='CLEAR_KEEP_TRANSFORM')\n    bpy.ops.object.modifier_add(type='BEVEL')\n    bevel = obj.modifiers[-1]",
    )
    assert equivalent != original
    check_code(equivalent)
    before = render_code(original, tmp_path / "data-api", row["required"], render=False)
    after = render_code(equivalent, tmp_path / "operator-api", row["required"], render=False)
    assert before["valid"] and after["valid"]
    assert before["mesh_count"] == after["mesh_count"]
    assert before["polygons"] == after["polygons"]


def test_fixed_renderer_ignores_program_lights(tmp_path):
    from PIL import Image

    row = next(r for r in build_rows() if r["family"] == "bedroom" and r["split"] == "train")
    code = row["messages"][-1]["content"]
    extra_light = (
        code
        + '\nbpy.ops.object.light_add(type="POINT", location=(0, 0, 3))\nbpy.context.object.data.energy = 100000\n'
    )
    before = render_code(code, tmp_path / "fixed", row["required"], resolution=64, samples=1)
    after = render_code(extra_light, tmp_path / "extra", row["required"], resolution=64, samples=1)
    assert before["valid"] and after["valid"]
    assert (
        Image.open(tmp_path / "fixed/render.png").tobytes()
        == Image.open(tmp_path / "extra/render.png").tobytes()
    )
