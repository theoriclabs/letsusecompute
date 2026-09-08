"""Blender subprocess renderer and measured geometry checks.

The AST checks reject obvious file/process/network access; they are NOT an OS
sandbox. Render policy outputs only on disposable, credential-free workers.
The local dataset command executes only this project's trusted demonstrations.
"""

from __future__ import annotations

import ast
import hashlib
import json
import math
import os
import subprocess
import sys
from pathlib import Path

EVALUATOR_REVISION = "geometry-v2-object-operators"


def extract_code(text):
    if "</think>" in text:
        text = text.split("</think>", 1)[1]
    text = text.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        if lines[-1].strip() != "```":
            raise ValueError("Unclosed code fence (possibly truncated)")
        text = "\n".join(lines[1:-1])
    if not text:
        raise ValueError("Empty program")
    return text


def check_code(code):
    if len(code) > 100_000:
        raise ValueError("Program exceeds 100 KB")
    tree = ast.parse(code)
    blocked = {
        "open",
        "exec",
        "eval",
        "compile",
        "getattr",
        "setattr",
        "delattr",
        "globals",
        "locals",
        "vars",
        "input",
        "breakpoint",
        "__import__",
        "help",
        "dir",
        "memoryview",
    }
    forbidden_attrs = {
        "handlers",
        "drivers",
        "driver_add",
        "load",
        "save",
        "write",
        "read",
        "filepath",
        "libraries",
        "texts",
        "scripts",
        "preferences",
        "window_manager",
    }
    allowed_ops = {
        "bpy.ops.mesh.primitive_cube_add",
        "bpy.ops.mesh.primitive_cylinder_add",
        "bpy.ops.mesh.primitive_cone_add",
        "bpy.ops.mesh.primitive_uv_sphere_add",
        "bpy.ops.mesh.primitive_ico_sphere_add",
        "bpy.ops.mesh.primitive_torus_add",
        "bpy.ops.object.transform_apply",
        "bpy.ops.object.select_all",
        "bpy.ops.object.delete",
        "bpy.ops.object.shade_flat",
        "bpy.ops.object.shade_smooth",
        "bpy.ops.object.modifier_apply",
        "bpy.ops.object.modifier_add",
        "bpy.ops.object.parent_clear",
        "bpy.ops.object.join",
        "bpy.ops.mesh.primitive_plane_add",
        "bpy.ops.mesh.primitive_circle_add",
        "bpy.ops.object.parent_set",
        "bpy.ops.object.origin_set",
        "bpy.ops.object.duplicate",
        "bpy.ops.object.convert",
        "bpy.ops.object.mode_set",
        "bpy.ops.object.empty_add",
        "bpy.ops.object.light_add",
        "bpy.ops.object.camera_add",
        "bpy.ops.transform.translate",
        "bpy.ops.transform.rotate",
        "bpy.ops.transform.resize",
    }

    # Built-in mesh and transform operations edit scene geometry; the worker
    # installs no custom operators or add-ons. Other namespaces stay explicit.
    geometry_namespaces = {"bpy.ops.mesh", "bpy.ops.transform"}

    def dotted(node):
        if isinstance(node, ast.Name):
            return node.id
        if isinstance(node, ast.Attribute):
            return dotted(node.value) + "." + node.attr
        return ""

    for node in ast.walk(tree):
        if isinstance(
            node, (ast.ImportFrom, ast.ClassDef, ast.With, ast.AsyncWith, ast.Global, ast.Nonlocal)
        ):
            raise ValueError(f"Disallowed syntax: {type(node).__name__}")
        if isinstance(node, ast.Import):
            if any(a.name not in {"bpy", "math"} or a.asname for a in node.names):
                raise ValueError("Only unaliased bpy and math imports are allowed")
        if isinstance(node, ast.Name) and (node.id in blocked or node.id.startswith("__")):
            raise ValueError(f"Disallowed name: {node.id}")
        if isinstance(node, ast.Attribute):
            if node.attr.startswith("_") or node.attr in forbidden_attrs:
                raise ValueError(f"Disallowed attribute: {node.attr}")
            path = dotted(node)
            if path.startswith("bpy.ops.") and not (
                any(op == path or op.startswith(path + ".") for op in allowed_ops)
                or any(path == ns or path.startswith(ns + ".") for ns in geometry_namespaces)
            ):
                raise ValueError(f"Disallowed Blender operator: {path}")
            if path.startswith("bpy.app") or path.startswith("bpy.utils"):
                raise ValueError(f"Disallowed Blender API: {path}")


def geometry_metrics(objects, required):
    """A diagnostic, not a visual judge. Count every failure separately."""
    shells = {"floor", "wall_back", "wall_side"}
    failures = []
    names = {o["name"] for o in objects}
    if not shells <= names:
        failures.append("missing_shell")
    expected = {
        "floor": ((0, 0, 0.1), (5.6, 5.6, 0.2)),
        "wall_back": ((0, 2.7, 1.7), (5.6, 0.15, 3)),
        "wall_side": ((-2.7, 0, 1.7), (0.15, 5.6, 3)),
    }
    for o in objects:
        values = o["min"] + o["max"]
        if not all(math.isfinite(x) for x in values):
            failures.append("nonfinite:" + o["name"])
            continue
        if any(b - a < 0.001 for a, b in zip(o["min"], o["max"], strict=False)):
            failures.append("degenerate:" + o["name"])
        if o["name"] in shells:
            loc, size = expected[o["name"]]
            if any(
                abs((a + b) / 2 - c) > 0.06 or abs((b - a) - s) > 0.06
                for a, b, c, s in zip(o["min"], o["max"], loc, size, strict=False)
            ):
                failures.append("shell_dimensions:" + o["name"])
        elif (
            o["min"][0] < -2.81
            or o["min"][1] < -2.81
            or o["max"][0] > 2.81
            or o["max"][1] > 2.81
            or o["min"][2] < 0.19
            or o["max"][2] > 3.8
        ):
            failures.append("bounds:" + o["name"])
    polygons = sum(o["polygons"] for o in objects)
    if polygons > 20000:
        failures.append("polygon_budget")
    # Support is a connected component of touching/overlapping world-space
    # bounds rooted at the floor/walls. This is approximate for curved meshes.
    supported = {o["name"] for o in objects if o["name"] in shells}
    for _ in range(len(objects)):
        added = set()
        for obj in objects:
            if obj["name"] in supported:
                continue
            for other in objects:
                if other["name"] not in supported:
                    continue
                if all(
                    min(obj["max"][i], other["max"][i]) - max(obj["min"][i], other["min"][i])
                    >= -0.025
                    for i in range(3)
                ):
                    added.add(obj["name"])
                    break
        if not added:
            break
        supported.update(added)
    floating = [o["name"] for o in objects if o["name"] not in supported]
    failures.extend("unsupported:" + n for n in floating)
    counts = {
        r: sum(o["name"] == r or o["name"].startswith(r + "_") for o in objects) for r in required
    }
    missing = [r for r, c in counts.items() if c == 0]
    failures.extend("missing:" + r for r in missing)
    return {
        "valid": not failures,
        "failures": failures,
        "mesh_count": len(objects),
        "polygons": polygons,
        "coverage": sum(c > 0 for c in counts.values()) / max(len(counts), 1),
        "part_counts": counts,
        "supported_fraction": len(supported) / max(len(objects), 1),
        "measurement_note": "AABB support/coverage proxy; no visibility, collision, aesthetic or semantic proof",
    }


def blender_worker(code_path: Path, output: Path, resolution=384, samples=12, render=True):
    import bpy
    from mathutils import Vector

    code = code_path.read_text()
    check_code(code)
    bpy.ops.wm.read_factory_settings(use_empty=True)
    exec(compile(code, str(code_path), "exec"), {"__name__": "__room__"})
    bpy.context.view_layer.update()
    objects = []
    depsgraph = bpy.context.evaluated_depsgraph_get()
    for obj in bpy.context.scene.objects:
        if obj.type != "MESH" or obj.hide_render:
            continue
        evaluated = obj.evaluated_get(depsgraph)
        mesh = evaluated.to_mesh()
        try:
            coords = [evaluated.matrix_world @ v.co for v in mesh.vertices]
            if not coords:
                continue
            objects.append(
                {
                    "name": obj.name,
                    "min": [min(v[i] for v in coords) for i in range(3)],
                    "max": [max(v[i] for v in coords) for i in range(3)],
                    "polygons": len(mesh.polygons),
                }
            )
        finally:
            evaluated.to_mesh_clear()
    output.mkdir(parents=True, exist_ok=True)
    (output / "scene.json").write_text(json.dumps(objects, indent=2) + "\n")
    if not render:
        return
    scene = bpy.context.scene
    # Keep the supplied lighting identical even if a program adds its own.
    for existing in scene.objects:
        if existing.type == "LIGHT":
            existing.hide_render = True
    scene.render.engine = "CYCLES"
    scene.cycles.device = "CPU"
    scene.cycles.samples = samples
    scene.cycles.use_denoising = True
    scene.render.threads_mode = "FIXED"
    scene.render.threads = 4
    scene.render.resolution_x = resolution
    scene.render.resolution_y = resolution
    scene.render.resolution_percentage = 100
    scene.render.image_settings.file_format = "PNG"
    scene.render.film_transparent = False
    scene.view_settings.view_transform = "AgX"
    scene.world = bpy.data.worlds.new("World")
    scene.world.use_nodes = True
    scene.world.node_tree.nodes["Background"].inputs[0].default_value = (0.72, 0.77, 0.82, 1)
    scene.world.node_tree.nodes["Background"].inputs[1].default_value = 0.35
    camera_data = bpy.data.cameras.new("Camera")
    camera = bpy.data.objects.new("Camera", camera_data)
    scene.collection.objects.link(camera)
    camera.location = (8, -10, 8)
    camera.rotation_euler = (
        (Vector((0, 0, 1)) - camera.location).to_track_quat("-Z", "Y").to_euler()
    )
    camera_data.type = "ORTHO"
    camera_data.ortho_scale = 9.2
    scene.camera = camera
    for name, loc, power, size in [("Key", (1, -4, 8), 500, 6), ("Fill", (5, 1, 5), 200, 5)]:
        light_data = bpy.data.lights.new(name, "AREA")
        light_data.energy = power
        light_data.shape = "DISK"
        light_data.size = size
        light = bpy.data.objects.new(name, light_data)
        scene.collection.objects.link(light)
        light.location = loc
        light.rotation_euler = (
            (Vector((0, 0, 0)) - light.location).to_track_quat("-Z", "Y").to_euler()
        )
    scene.render.filepath = str((output / "render.png").resolve())
    bpy.ops.render.render(write_still=True)
    bpy.ops.wm.save_as_mainfile(filepath=str((output / "scene.blend").resolve()))


def render_code(
    code, destination: Path, required, *, resolution=384, samples=12, timeout=90, render=True
):
    destination = destination.resolve()
    destination.mkdir(parents=True, exist_ok=True)
    # A failed retry must never inherit an earlier successful scene or image.
    for name in (
        "program.py",
        "scene.json",
        "render.png",
        "scene.blend",
        "metrics.json",
        "blender.log",
    ):
        (destination / name).unlink(missing_ok=True)
    try:
        code = extract_code(code)
        check_code(code)
        source = destination / "program.py"
        source.write_text(code)
        args = [
            sys.executable,
            str(Path(__file__).resolve()),
            "worker",
            str(source),
            str(destination),
            str(resolution),
            str(samples),
        ]
        if not render:
            args.append("--no-render")
        env = {
            k: v
            for k, v in os.environ.items()
            if k in {"PATH", "SYSTEMROOT", "LD_LIBRARY_PATH", "DYLD_LIBRARY_PATH", "TMPDIR"}
        }
        env.update({"HOME": str(destination), "PYTHONNOUSERSITE": "1", "OMP_NUM_THREADS": "4"})
        with (destination / "blender.log").open("w") as log:
            proc = subprocess.run(
                args,
                env=env,
                cwd=destination,
                stdout=log,
                stderr=subprocess.STDOUT,
                timeout=timeout,
            )
        scene_path = destination / "scene.json"
        if proc.returncode or not scene_path.exists():
            raise RuntimeError(f"Blender exited {proc.returncode}; see blender.log")
        if render and not (destination / "render.png").is_file():
            raise RuntimeError("Blender returned without a render")
        metrics = geometry_metrics(json.loads(scene_path.read_text()), required)
        metrics["executed"] = True
    except (ValueError, SyntaxError, RuntimeError, OSError, subprocess.TimeoutExpired) as error:
        metrics = {
            "valid": False,
            "executed": False,
            "coverage": 0.0,
            "failures": [f"{type(error).__name__}: {error}"],
        }
    metrics["code_sha256"] = hashlib.sha256(code.encode()).hexdigest()
    metrics["evaluator_revision"] = EVALUATOR_REVISION
    (destination / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    return metrics


def prepare(output: Path, limit=0, resolution=256, workers=2):
    from concurrent.futures import ThreadPoolExecutor

    from rooms import write_dataset

    if not 1 <= workers <= 4:
        raise ValueError("workers must be between one and four")
    rows, manifest = write_dataset(output / "data")
    results = []

    def check(row):
        folder = output / "teacher" / row["id"]
        result = render_code(
            row["messages"][-1]["content"],
            folder,
            row["required"],
            resolution=resolution,
            samples=8,
        )
        result.update({"id": row["id"], "split": row["split"]})
        return result

    with ThreadPoolExecutor(max_workers=workers) as pool:
        for result in pool.map(check, rows[: limit or None]):
            results.append(result)
            print(json.dumps({key: result[key] for key in ("id", "valid", "failures")}), flush=True)
    summary = {
        "dataset": manifest,
        "checked": len(results),
        "valid": sum(r["valid"] for r in results),
        "results": results,
    }
    (output / "teacher-validation.json").write_text(json.dumps(summary, indent=2) + "\n")
    if any(not r["valid"] for r in results):
        raise RuntimeError("Teacher validation failed; inspect teacher-validation.json")
    return summary


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "worker":
        blender_worker(
            Path(sys.argv[2]),
            Path(sys.argv[3]),
            int(sys.argv[4]),
            int(sys.argv[5]),
            render="--no-render" not in sys.argv,
        )
    else:
        import argparse

        parser = argparse.ArgumentParser()
        parser.add_argument("--output", type=Path, default=Path("artifacts"))
        parser.add_argument("--limit", type=int, default=0)
        parser.add_argument("--resolution", type=int, default=256)
        parser.add_argument("--workers", type=int, default=2)
        args = parser.parse_args()
        prepare(args.output, args.limit, args.resolution, args.workers)
