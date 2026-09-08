import bpy
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


box('floor', (0, 0, 0.1), (5.6, 5.6, 0.2), 'eddbc0', 0)
box('wall_back', (0, 2.7, 1.7), (5.6, 0.15, 3), 'bd7963', 0)
box('wall_side', (-2.7, 0, 1.7), (0.15, 5.6, 3), 'bd7963', 0)
box('trim_back', (0, 2.595, 0.29), (5.2, 0.05, 0.18), 'eddbc0', 0)
box('trim_side', (-2.595, 0, 0.29), (0.05, 5.2, 0.18), 'eddbc0', 0)
box('refrigerator_body', (-1.8, 1.95, 1.25), (0.92, 0.95, 2.1), '8b6650', 0)
box('refrigerator_door_0', (-2.03, 1.46, 1.25), (0.425, 0.035, 2.01), 'eddbc0', 0)
box('refrigerator_handle_0', (-1.88, 1.425, 1.355), (0.04, 0.04, 0.18), '504945', 0)
box('refrigerator_door_1', (-1.57, 1.46, 1.25), (0.425, 0.035, 2.01), 'eddbc0', 0)
box('refrigerator_handle_1', (-1.72, 1.425, 1.355), (0.04, 0.04, 0.18), '504945', 0)
box('counter_body', (0.3, 2.06, 0.62), (2.35, 0.75, 0.84), '8b6650', 0)
box('counter_door_0', (-0.2875, 1.67, 0.62), (1.14, 0.035, 0.75), 'eddbc0', 0)
box('counter_handle_0', (0.22, 1.635, 0.662), (0.04, 0.04, 0.18), '504945', 0)
box('counter_door_1', (0.8875, 1.67, 0.62), (1.14, 0.035, 0.75), 'eddbc0', 0)
box('counter_handle_1', (0.38, 1.635, 0.662), (0.04, 0.04, 0.18), '504945', 0)
box('counter_worktop', (0.3, 2.06, 1.1), (2.43, 0.8, 0.12), 'eddbc0', 0)
box('stove_top', (-0.3, 2.04, 1.177), (0.76, 0.63, 0.035), '4a4a51', 0)
cylinder('stove_burner_0', (-0.48, 1.9, 1.201), 0.105, 0.02, '85838b')
cylinder('stove_burner_1', (-0.48, 2.18, 1.201), 0.105, 0.02, '85838b')
cylinder('stove_burner_2', (-0.12, 1.9, 1.201), 0.105, 0.02, '85838b')
cylinder('stove_burner_3', (-0.12, 2.18, 1.201), 0.105, 0.02, '85838b')
box('sink_basin', (0.85, 2.04, 1.173), (0.57, 0.5, 0.026), '646b72', 0)
box('sink_rim_0', (0.53, 2.04, 1.19), (0.06, 0.61, 0.04), 'b9c6c9', 0)
box('sink_rim_1', (1.17, 2.04, 1.19), (0.06, 0.61, 0.04), 'b9c6c9', 0)
box('sink_edge_0', (0.85, 1.76, 1.19), (0.66, 0.06, 0.04), 'b9c6c9', 0)
box('sink_edge_1', (0.85, 2.32, 1.19), (0.66, 0.06, 0.04), 'b9c6c9', 0)
box('sink_tap_stem', (0.85, 2.32, 1.35), (0.055, 0.055, 0.32), 'b9c6c9', 0)
box('sink_tap_spout', (0.85, 2.23, 1.49), (0.055, 0.22, 0.055), 'b9c6c9', 0)
box('table_top', (0.3, -0.45, 1.05), (1.35, 0.9, 0.12), '8b6650', 0)
box('table_leg_0', (-0.245, -0.77, 0.595), (0.1, 0.1, 0.79), '8b6650', 0)
box('table_leg_1', (-0.245, -0.13, 0.595), (0.1, 0.1, 0.79), '8b6650', 0)
box('table_leg_2', (0.845, -0.77, 0.595), (0.1, 0.1, 0.79), '8b6650', 0)
box('table_leg_3', (0.845, -0.13, 0.595), (0.1, 0.1, 0.79), '8b6650', 0)
box('chair_1_leg_0', (0.52, -1.22, 0.4), (0.08, 0.08, 0.4), '8b6650', 3.1416)
box('chair_1_leg_1', (0.52, -1.64, 0.4), (0.08, 0.08, 0.4), '8b6650', 3.1416)
box('chair_1_leg_2', (0.08, -1.22, 0.4), (0.08, 0.08, 0.4), '8b6650', 3.1416)
box('chair_1_leg_3', (0.08, -1.64, 0.4), (0.08, 0.08, 0.4), '8b6650', 3.1416)
box('chair_1_seat', (0.3, -1.43, 0.65), (0.64, 0.62, 0.14), '7d9a8c', 3.1416)
box('chair_1_back', (0.3, -1.68, 0.97), (0.64, 0.12, 0.66), '7d9a8c', 3.1416)
box('chair_2_leg_0', (1.44, -0.23, 0.4), (0.08, 0.08, 0.4), '8b6650', -1.5708)
box('chair_2_leg_1', (1.86, -0.23, 0.4), (0.08, 0.08, 0.4), '8b6650', -1.5708)
box('chair_2_leg_2', (1.44, -0.67, 0.4), (0.08, 0.08, 0.4), '8b6650', -1.5708)
box('chair_2_leg_3', (1.86, -0.67, 0.4), (0.08, 0.08, 0.4), '8b6650', -1.5708)
box('chair_2_seat', (1.65, -0.45, 0.65), (0.64, 0.62, 0.14), '7d9a8c', -1.5708)
box('chair_2_back', (1.9, -0.45, 0.97), (0.64, 0.12, 0.66), '7d9a8c', -1.5708)
cylinder('bowl', (0.3, -0.45, 1.15), 0.19, 0.08, '7d9a8c')
cylinder('plant_pot', (-1.85, -1.45, 0.35), 0.19, 0.3, '7d9a8c')
cylinder('plant_stem', (-1.85, -1.45, 0.6), 0.035, 0.35, '6d8051')
box('plant_leaf_0', (-1.96, -1.45, 0.75), (0.25, 0.17, 0.3), '739662', 0.0)
box('plant_leaf_1', (-1.74, -1.43, 0.75), (0.25, 0.17, 0.3), '739662', 0.8)
box('plant_leaf_2', (-1.85, -1.35, 0.75), (0.25, 0.17, 0.3), '739662', 1.6)
box('artwork_frame', (0.55, 2.59, 2.2), (0.95, 0.065, 0.7), '8b6650', 0)
box('artwork_canvas', (0.55, 2.547, 2.2), (0.83, 0.025, 0.58), 'eddbc0', 0)
box('artwork_shape', (0.62, 2.527, 2.2), (0.37, 0.02, 0.33), '7d9a8c', 0)
bpy.context.view_layer.update()
