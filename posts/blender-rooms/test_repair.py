"""Contracts for the fixed retry comparison and media provenance."""

import hashlib
import json
from pathlib import Path

from media_sources import SCENES
from repair import repair_messages, select_final, selected_rows
from rooms import build_rows


def test_retry_prompts_exclude_teacher_and_isolate_measured_feedback(tmp_path):
    row = selected_rows([r for r in build_rows() if r["split"] == "validation"])[0]
    metrics = {
        "valid": False,
        "executed": False,
        "token_limit": False,
        "failures": ["RuntimeError: Blender exited 1"],
    }
    (tmp_path / "blender.log").write_text(
        "startup\nTraceback (most recent call last):\nTypeError: RGB list plus tuple"
    )
    retry = repair_messages(row, "previous generated code", metrics, tmp_path, "retry")
    feedback = repair_messages(row, "previous generated code", metrics, tmp_path, "feedback")
    assert retry[:3] == feedback[:3]
    assert feedback[-1]["content"].startswith(retry[-1]["content"])
    assert "RGB list plus tuple" not in retry[-1]["content"]
    assert "RGB list plus tuple" in feedback[-1]["content"]
    assert row["messages"][-1]["content"] not in json.dumps(feedback)
    assert [r["role"] for r in feedback] == ["system", "user", "assistant", "user"]


def test_fixed_selection_and_monotonic_pass_retention():
    rows = selected_rows([r for r in build_rows() if r["split"] == "validation"])
    assert len(rows) == 16
    assert len({r["id"] for r in rows}) == 16
    assert all(
        r["split"] == "validation" and int(r["id"].rsplit("p", 1)[1]) in range(2, 6) for r in rows
    )
    passed = {"valid": True, "executed": True}
    invalid = {"valid": False, "executed": True}
    crashed = {"valid": False, "executed": False}
    assert select_final(passed, crashed) is passed
    assert select_final(invalid, passed) is passed
    assert select_final(invalid, crashed) is invalid
    assert select_final(crashed, invalid) is invalid


def test_video_preserves_original_primary_test_outputs():
    root = Path(__file__).parent
    assert len(SCENES) == 4
    assert {r["family"] for r in SCENES} == {"bedroom", "studio", "kitchen", "living_room"}
    for row in SCENES:
        assert row["id"] == row["family"] + "-l7-p0"
        assert hashlib.sha256(row["raw"].encode()).hexdigest() == row["raw_sha256"]
        original = root / "assets/results" / ("test-" + row["id"] + "-sft.txt")
        assert original.read_text() == row["raw"]
    assert not next(r for r in SCENES if r["family"] == "studio")["geometry_valid"]
