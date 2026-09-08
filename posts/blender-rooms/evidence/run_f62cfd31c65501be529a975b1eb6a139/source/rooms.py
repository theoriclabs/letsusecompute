"""Original procedural bpy demonstrations; no external teacher API required."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

MODEL = "Qwen/Qwen3.5-9B"
REVISION = "c202236235762e1c871ad0ccb60c8ee5ba337b9a"
FAMILIES = ("bedroom", "studio", "kitchen", "living_room")
PALETTES = (
    ("sage and warm oak", "91aa8a", "e8dfcd", "9b704d", "e2b96b"),
    ("dusty blue and cream", "859eae", "ece3d0", "9b7359", "cb8d74"),
    ("terracotta and sand", "bd7963", "eddbc0", "8b6650", "7d9a8c"),
    ("lavender and butter", "a799be", "ece1af", "927861", "cf8596"),
    ("forest green and blush", "668b79", "eedacf", "8d6951", "d4a25f"),
    ("slate and mustard", "7a899a", "e5ddca", "8f6b50", "d7ad48"),
    ("rose and ivory", "c28e98", "f0e6d6", "9d7454", "83a29a"),
    ("teal and apricot", "71a4a3", "edddc8", "967255", "dfac83"),
)
SYSTEM = """Create a low-poly isometric room using Blender 4.5 bpy. Return only a complete Python program, without Markdown or explanations. The scene is empty; create a raised floor centered at (0,0,0.1) with size (5.6,5.6,0.2), a back wall at (0,2.7,1.7) with size (5.6,0.15,3.0), and a left wall at (-2.7,0,1.7) with size (0.15,5.6,3.0). Name these floor, wall_back, wall_side. The renderer supplies camera and lights. Use meters, Z up, keep furniture within the floor and above z=0.2, and support every part. Make recognizable multipart furniture, small bevels, flat colors, and at most 20000 evaluated mesh polygons. Name furniture parts with semantic prefixes, for example bed_frame, bed_mattress, chair_seat, chair_back. Use only bpy and math; do not access files, processes, network, scene handlers, or render settings. You may define small primitive helpers. Do not render or save; the evaluator handles that."""

HELPERS = """import bpy
import math

def material(color):
    mat = bpy.data.materials.get(color)
    if mat is None:
        mat = bpy.data.materials.new(color)
        rgb = [int(color[i:i+2], 16) / 255 for i in (0, 2, 4)]
        linear = tuple(v / 12.92 if v < 0.04045 else ((v + 0.055) / 1.055) ** 2.4 for v in rgb)
        mat.diffuse_color = linear + (1,)
    return mat

def box(name, loc, size, color, angle=0):
    bpy.ops.mesh.primitive_cube_add(size=1, location=loc)
    obj = bpy.context.object
    obj.name = name
    obj.dimensions = size
    bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)
    obj.rotation_euler.z = angle
    obj.data.materials.append(material(color))
    bevel = obj.modifiers.new('edge', 'BEVEL')
    bevel.width = min(0.025, min(size) / 5)
    bevel.segments = 1
    return obj

def cylinder(name, loc, radius, depth, color):
    bpy.ops.mesh.primitive_cylinder_add(vertices=10, radius=radius, depth=depth, location=loc)
    obj = bpy.context.object
    obj.name = name
    obj.data.materials.append(material(color))
    return obj

"""


class Program:
    def __init__(self, palette, reflected=False):
        self.lines = [HELPERS]
        self.wall, self.cream, self.wood, self.accent = palette[1:]
        self.reflected = reflected

    @staticmethod
    def numbers(values):
        return tuple(round(v, 4) for v in values)

    def box(self, name, loc, size, color, angle=0, shell=False):
        x, y, z = loc
        if self.reflected and not shell:
            x, y = -y, -x
            angle = -math.pi / 2 - angle
        self.lines.append(
            f"box({name!r}, {self.numbers((x, y, z))!r}, {self.numbers(size)!r}, {color!r}, {round(angle, 4)})"
        )

    def part(self, name, origin, offset, size, color, angle=0):
        x, y, z = origin
        ox, oy, oz = offset
        self.box(
            name,
            (
                x + ox * math.cos(angle) - oy * math.sin(angle),
                y + ox * math.sin(angle) + oy * math.cos(angle),
                z + oz,
            ),
            size,
            color,
            angle,
        )

    def cylinder(self, name, loc, radius, depth, color):
        x, y, z = loc
        if self.reflected:
            x, y = -y, -x
        self.lines.append(
            f"cylinder({name!r}, {self.numbers((x, y, z))!r}, {radius}, {depth}, {color!r})"
        )

    def table(self, name, x, y, width=1.4, depth=0.8, height=0.85):
        self.box(name + "_top", (x, y, 0.2 + height), (width, depth, 0.12), self.wood)
        for i, (ox, oy) in enumerate(((-1, -1), (-1, 1), (1, -1), (1, 1))):
            self.box(
                name + f"_leg_{i}",
                (
                    x + ox * (width / 2 - 0.13),
                    y + oy * (depth / 2 - 0.13),
                    0.2 + (height - 0.06) / 2,
                ),
                (0.1, 0.1, height - 0.06),
                self.wood,
            )

    def chair(self, name, x, y, angle=0, wide=0.64):
        for i, (ox, oy) in enumerate(((-1, -1), (-1, 1), (1, -1), (1, 1))):
            self.part(
                name + f"_leg_{i}",
                (x, y, 0.2),
                (ox * (wide / 2 - 0.1), oy * 0.21, 0.2),
                (0.08, 0.08, 0.4),
                self.wood,
                angle,
            )
        self.part(name + "_seat", (x, y, 0.2), (0, 0, 0.45), (wide, 0.62, 0.14), self.accent, angle)
        self.part(
            name + "_back", (x, y, 0.2), (0, 0.25, 0.77), (wide, 0.12, 0.66), self.accent, angle
        )

    def cabinet(self, name, x, y, width=1, depth=0.65, height=1.1):
        self.box(name + "_body", (x, y, 0.2 + height / 2), (width, depth, height), self.wood)
        for i, side in enumerate((-1, 1)):
            self.box(
                name + f"_door_{i}",
                (x + side * width / 4, y - depth / 2 - 0.015, 0.2 + height / 2),
                (width / 2 - 0.035, 0.035, height - 0.09),
                self.cream,
            )
            self.box(
                name + f"_handle_{i}",
                (x + side * 0.08, y - depth / 2 - 0.05, 0.2 + height * 0.55),
                (0.04, 0.04, 0.18),
                "504945",
            )

    def plant(self, name, x, y, bottom=0.2, size=1):
        self.cylinder(
            name + "_pot", (x, y, bottom + 0.15 * size), 0.19 * size, 0.3 * size, self.accent
        )
        self.cylinder(
            name + "_stem", (x, y, bottom + 0.4 * size), 0.035 * size, 0.35 * size, "6d8051"
        )
        for i, (ox, oy) in enumerate(((-0.11, 0), (0.11, 0.02), (0, 0.1))):
            self.box(
                name + f"_leaf_{i}",
                (x + ox * size, y + oy * size, bottom + 0.55 * size),
                (0.25 * size, 0.17 * size, 0.3 * size),
                "739662",
                angle=i * 0.8,
            )

    def lamp(self, x, y, bottom=0.2, height=1.45):
        self.cylinder("lamp_base", (x, y, bottom + 0.035), 0.23, 0.07, "514b47")
        self.cylinder("lamp_stem", (x, y, bottom + height / 2), 0.035, height, "514b47")
        self.cylinder("lamp_shade", (x, y, bottom + height), 0.26, 0.35, self.cream)

    def books(self, x, y, z, count=4, prefix="books"):
        for i in range(count):
            h = 0.22 + 0.035 * (i % 3)
            self.box(
                prefix + f"_{i}",
                (x + i * 0.105, y, z + h / 2),
                (0.08, 0.22, h),
                (self.wall, self.cream, self.accent)[i % 3],
            )

    def artwork(self):
        self.box("artwork_frame", (0.55, 2.59, 2.2), (0.95, 0.065, 0.7), self.wood)
        self.box("artwork_canvas", (0.55, 2.547, 2.2), (0.83, 0.025, 0.58), self.cream)
        self.box("artwork_shape", (0.62, 2.527, 2.2), (0.37, 0.02, 0.33), self.accent)


def example(family: str, layout: int, palette_index: int) -> dict:
    # Eight composition groups with distinct dimension/offset combinations.
    # Shared furniture primitives remain a limitation of this procedural pilot.
    split = "train" if layout < 6 else "validation" if layout == 6 else "test"
    p = Program(PALETTES[palette_index], reflected=bool(layout % 2))
    variant = layout
    shift = (-0.15, 0.1, -0.05, 0.2, -0.2, 0.0, 0.05, -0.1)[variant]
    p.box("floor", (0, 0, 0.1), (5.6, 5.6, 0.2), p.cream, shell=True)
    p.box("wall_back", (0, 2.7, 1.7), (5.6, 0.15, 3), p.wall, shell=True)
    p.box("wall_side", (-2.7, 0, 1.7), (0.15, 5.6, 3), p.wall, shell=True)
    p.box("trim_back", (0, 2.595, 0.29), (5.2, 0.05, 0.18), p.cream, shell=True)
    p.box("trim_side", (-2.595, 0, 0.29), (0.05, 5.2, 0.18), p.cream, shell=True)
    if family == "bedroom":
        x, y = -1.1 + shift, 0.25
        w = (1.5, 1.7, 1.6, 1.8, 1.45, 1.65, 1.55, 1.75)[variant]
        p.box("bed_frame", (x, y, 0.4), (w, 2.25, 0.4), p.wood)
        p.box("bed_mattress", (x, y, 0.68), (w - 0.05, 2.15, 0.22), p.cream)
        p.box("bed_headboard", (x, y + 1.1, 0.88), (w + 0.08, 0.1, 1.25), p.wood)
        p.box("bed_blanket", (x, y - 0.42, 0.807), (w - 0.03, 1.12, 0.055), p.accent)
        for i, ox in enumerate((-0.36, 0.36)):
            p.box(f"bed_pillow_{i}", (x + ox, y + 0.68, 0.86), (0.59, 0.44, 0.16), p.cream)
        p.cabinet("nightstand", 0.35 + shift, 1.0, 0.58, 0.58, 0.52)
        p.lamp(0.35 + shift, 1.0, bottom=0.72, height=0.46)
        p.cabinet("wardrobe", 1.75, 1.9, 1.15, 0.75, 2.25)
        p.box("rug", (1.3, -1, 0.22), (1.6, 1.4, 0.04), p.wall)
        p.plant("plant", 2, -1.7)
        required = ["bed", "nightstand", "wardrobe", "rug", "lamp", "plant", "artwork"]
        desc = f"A {w:.1f} m wide bed with two pillows and a folded blanket, a bedside cabinet and lamp, a two-door wardrobe, rug, potted plant and framed art."
    elif family == "studio":
        x = shift
        w = (1.7, 2.0, 1.8, 2.1, 1.75, 1.85, 1.95, 2.05)[variant]
        p.table("desk", x, 1.85, w, 0.8, 0.85)
        p.chair("chair", x, 0.7, angle=math.pi)
        p.box("monitor_foot", (x, 1.92, 1.145), (0.38, 0.25, 0.08), "4c515a")
        p.box("monitor_stand", (x, 1.98, 1.33), (0.09, 0.07, 0.32), "4c515a")
        p.box("monitor_frame", (x, 1.99, 1.57), (0.85, 0.09, 0.5), "4c515a")
        p.box("monitor_screen", (x, 1.93, 1.57), (0.77, 0.02, 0.42), "a5c6ce")
        p.box("keyboard", (x, 1.61, 1.13), (0.5, 0.18, 0.04), p.cream)
        p.cabinet("storage", 1.95, 0.45, 0.72, 0.8, 0.8)
        p.plant("plant", 1.95, 0.45, bottom=1.0, size=0.65)
        sx, sy = -1.95, 1.6
        for i, ox in enumerate((-0.4, 0.4)):
            p.box(f"shelf_side_{i}", (sx + ox, sy, 1.29), (0.08, 0.48, 2.18), p.wood)
        for i in range(4):
            z = 0.24 + i * 0.64
            p.box(f"shelf_level_{i}", (sx, sy, z), (0.88, 0.48, 0.08), p.wood)
            p.books(sx - 0.26, sy, z + 0.04, 4, prefix=f"books_{i}")
        p.box("rug", (0, -0.9, 0.22), (2.0, 1.3, 0.04), p.accent)
        p.lamp(-1.9, -1.2)
        required = [
            "desk",
            "chair",
            "monitor",
            "keyboard",
            "storage",
            "shelf",
            "rug",
            "lamp",
            "plant",
            "artwork",
        ]
        desc = f"A {w:.1f} m wide desk with monitor and keyboard, a chair facing the desk, an open four-level bookcase with books, a storage cabinet, rug, standing lamp and plant."
    elif family == "kitchen":
        p.cabinet("refrigerator", -1.8, 1.95, 0.92, 0.95, 2.1)
        p.cabinet("counter", 0.3, 2.06, 2.35, 0.75, 0.84)
        p.box("counter_worktop", (0.3, 2.06, 1.1), (2.43, 0.8, 0.12), p.cream)
        p.box("stove_top", (-0.3, 2.04, 1.177), (0.76, 0.63, 0.035), "4a4a51")
        for i, (ox, oy) in enumerate(((-1, -1), (-1, 1), (1, -1), (1, 1))):
            p.cylinder(
                f"stove_burner_{i}",
                (-0.3 + ox * 0.18, 2.04 + oy * 0.14, 1.201),
                0.105,
                0.02,
                "85838b",
            )
        p.box("sink_basin", (0.85, 2.04, 1.173), (0.57, 0.5, 0.026), "646b72")
        for i, ox in enumerate((-0.32, 0.32)):
            p.box(f"sink_rim_{i}", (0.85 + ox, 2.04, 1.19), (0.06, 0.61, 0.04), "b9c6c9")
        for i, oy in enumerate((-0.28, 0.28)):
            p.box(f"sink_edge_{i}", (0.85, 2.04 + oy, 1.19), (0.66, 0.06, 0.04), "b9c6c9")
        p.box("sink_tap_stem", (0.85, 2.32, 1.35), (0.055, 0.055, 0.32), "b9c6c9")
        p.box("sink_tap_spout", (0.85, 2.23, 1.49), (0.055, 0.22, 0.055), "b9c6c9")
        x = 0.45 + shift
        p.table("table", x, -0.45, 1.35, 0.9, 0.85)
        p.chair("chair_1", x, -1.43, angle=math.pi)
        p.chair("chair_2", 1.8 + shift, -0.45, angle=-math.pi / 2)
        p.cylinder("bowl", (x, -0.45, 1.15), 0.19, 0.08, p.accent)
        p.plant("plant", -1.85, -1.45)
        required = [
            "refrigerator",
            "counter",
            "stove",
            "sink",
            "table",
            "chair_1",
            "chair_2",
            "plant",
            "artwork",
        ]
        desc = "A refrigerator, two-door counter with a four-burner stove and rimmed sink with tap, a dining table with a bowl, two chairs and a plant."
    elif family == "living_room":
        x, y = -1.83, 0.3 + shift
        angle = math.pi / 2
        width = (2, 2.2, 2.1, 2.3, 2.05, 2.15, 2.25, 2.35)[variant]
        for name, off, size in [
            ("base", (0, 0, 0.22), (width, 0.82, 0.44)),
            ("back", (0, 0.35, 0.74), (width, 0.16, 0.85)),
        ]:
            p.part("sofa_" + name, (x, y, 0.2), off, size, p.wall, angle)
        for i, ox in enumerate((-width / 2 + 0.1, width / 2 - 0.1)):
            p.part(f"sofa_arm_{i}", (x, y, 0.2), (ox, 0, 0.52), (0.2, 0.85, 0.64), p.wall, angle)
        for i, ox in enumerate((-0.47, 0.47)):
            p.part(
                f"sofa_cushion_{i}",
                (x, y, 0.2),
                (ox, -0.04, 0.52),
                (0.83, 0.65, 0.18),
                p.accent,
                angle,
            )
        p.box("rug", (0.05, 0.25, 0.22), (2.6, 2.3, 0.04), p.cream)
        p.table("coffee_table", -0.05, 0.15, 1.2, 0.7, 0.5)
        p.books(-0.32, 0.15, 0.76, 3)
        p.chair("chair", 0.25, 1.75, wide=0.78)
        p.cabinet("media_unit", 1.65, 1.92, 1.4, 0.55, 0.55)
        p.box("television_base", (1.65, 1.92, 0.79), (0.5, 0.3, 0.08), "414851")
        p.box("television_stand", (1.65, 1.95, 0.94), (0.1, 0.08, 0.24), "414851")
        p.box("television_frame", (1.65, 1.95, 1.31), (1.18, 0.09, 0.66), "414851")
        p.box("television_screen", (1.65, 1.89, 1.31), (1.08, 0.025, 0.56), "a5bbc3")
        p.lamp(-1.95, 1.94)
        p.plant("plant", 1.95, -1.6)
        required = [
            "sofa",
            "coffee_table",
            "chair",
            "media_unit",
            "television",
            "rug",
            "lamp",
            "plant",
            "artwork",
        ]
        desc = f"A {width:.1f} m wide sofa with cushions and arms, a low coffee table with books, an arm-free accent chair, television on a media cabinet, rug, lamp and plant."
    else:
        raise ValueError(f"Unknown room family {family}")
    p.artwork()
    p.lines.append("bpy.context.view_layer.update()\n")
    code = "\n".join(p.lines)
    orientation = "left wall" if p.reflected else "back wall"
    user = (
        f"Create a cozy low-poly {family.replace('_', ' ')} in {PALETTES[palette_index][0]}. "
        f"{desc} Arrange the main storage or work zone along the {orientation}, "
        f"leaving the front corner open. Layout variation {variant + 1}: "
        f"use accent-colored textiles, warm wood frames, and cream highlights. "
        f"Palette hex colors: {', '.join(PALETTES[palette_index][1:])}."
    )
    return {
        "id": f"{family}-l{layout}-p{palette_index}",
        "family": family,
        "layout_group": f"{family}-l{layout}",
        "split": split,
        "required": required,
        "messages": [
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": user},
            {"role": "assistant", "content": code},
        ],
        "code_sha256": hashlib.sha256(code.encode()).hexdigest(),
        "provenance": "original procedural demonstration; agent-authored generator v1",
    }


def build_rows():
    return [
        example(family, layout, palette)
        for family in FAMILIES
        for layout in range(8)
        for palette in range(8)
    ]


def write_dataset(destination: Path):
    destination.mkdir(parents=True, exist_ok=True)
    rows = build_rows()
    manifest = {
        "schema": 1,
        "generator_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "provenance": "procedural; no remote teacher API",
        "splits": {},
    }
    for split in ("train", "validation", "test"):
        subset = [r for r in rows if r["split"] == split]
        data = "".join(json.dumps(r, sort_keys=True) + "\n" for r in subset).encode()
        (destination / f"{split}.jsonl").write_bytes(data)
        manifest["splits"][split] = {
            "rows": len(subset),
            "sha256": hashlib.sha256(data).hexdigest(),
            "layout_groups": sorted({r["layout_group"] for r in subset}),
        }
    (destination / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return rows, manifest


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("data"))
    args = parser.parse_args()
    _, manifest = write_dataset(args.output)
    print(json.dumps(manifest, indent=2))
