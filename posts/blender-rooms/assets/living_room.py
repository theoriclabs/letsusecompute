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


box('floor', (0, 0, 0.1), (5.6, 5.6, 0.2), 'ece1af', 0)
box('wall_back', (0, 2.7, 1.7), (5.6, 0.15, 3), 'a799be', 0)
box('wall_side', (-2.7, 0, 1.7), (0.15, 5.6, 3), 'a799be', 0)
box('trim_back', (0, 2.595, 0.29), (5.2, 0.05, 0.18), 'ece1af', 0)
box('trim_side', (-2.595, 0, 0.29), (0.05, 5.2, 0.18), 'ece1af', 0)
box('sofa_base', (-1.83, 0.15, 0.42), (2, 0.82, 0.44), 'a799be', 1.5708)
box('sofa_back', (-2.18, 0.15, 0.94), (2, 0.16, 0.85), 'a799be', 1.5708)
box('sofa_arm_0', (-1.83, -0.75, 0.72), (0.2, 0.85, 0.64), 'a799be', 1.5708)
box('sofa_arm_1', (-1.83, 1.05, 0.72), (0.2, 0.85, 0.64), 'a799be', 1.5708)
box('sofa_cushion_0', (-1.79, -0.32, 0.72), (0.83, 0.65, 0.18), 'cf8596', 1.5708)
box('sofa_cushion_1', (-1.79, 0.62, 0.72), (0.83, 0.65, 0.18), 'cf8596', 1.5708)
box('rug', (0.05, 0.25, 0.22), (2.6, 2.3, 0.04), 'ece1af', 0)
box('coffee_table_top', (-0.05, 0.15, 0.7), (1.2, 0.7, 0.12), '927861', 0)
box('coffee_table_leg_0', (-0.52, -0.07, 0.42), (0.1, 0.1, 0.44), '927861', 0)
box('coffee_table_leg_1', (-0.52, 0.37, 0.42), (0.1, 0.1, 0.44), '927861', 0)
box('coffee_table_leg_2', (0.42, -0.07, 0.42), (0.1, 0.1, 0.44), '927861', 0)
box('coffee_table_leg_3', (0.42, 0.37, 0.42), (0.1, 0.1, 0.44), '927861', 0)
box('books_0', (-0.32, 0.15, 0.87), (0.08, 0.22, 0.22), 'a799be', 0)
box('books_1', (-0.215, 0.15, 0.8875), (0.08, 0.22, 0.255), 'ece1af', 0)
box('books_2', (-0.11, 0.15, 0.905), (0.08, 0.22, 0.29), 'cf8596', 0)
box('chair_leg_0', (-0.04, 1.54, 0.4), (0.08, 0.08, 0.4), '927861', 0)
box('chair_leg_1', (-0.04, 1.96, 0.4), (0.08, 0.08, 0.4), '927861', 0)
box('chair_leg_2', (0.54, 1.54, 0.4), (0.08, 0.08, 0.4), '927861', 0)
box('chair_leg_3', (0.54, 1.96, 0.4), (0.08, 0.08, 0.4), '927861', 0)
box('chair_seat', (0.25, 1.75, 0.65), (0.78, 0.62, 0.14), 'cf8596', 0)
box('chair_back', (0.25, 2.0, 0.97), (0.78, 0.12, 0.66), 'cf8596', 0)
box('media_unit_body', (1.65, 1.92, 0.475), (1.4, 0.55, 0.55), '927861', 0)
box('media_unit_door_0', (1.3, 1.63, 0.475), (0.665, 0.035, 0.46), 'ece1af', 0)
box('media_unit_handle_0', (1.57, 1.595, 0.5025), (0.04, 0.04, 0.18), '504945', 0)
box('media_unit_door_1', (2.0, 1.63, 0.475), (0.665, 0.035, 0.46), 'ece1af', 0)
box('media_unit_handle_1', (1.73, 1.595, 0.5025), (0.04, 0.04, 0.18), '504945', 0)
box('television_base', (1.65, 1.92, 0.79), (0.5, 0.3, 0.08), '414851', 0)
box('television_stand', (1.65, 1.95, 0.94), (0.1, 0.08, 0.24), '414851', 0)
box('television_frame', (1.65, 1.95, 1.31), (1.18, 0.09, 0.66), '414851', 0)
box('television_screen', (1.65, 1.89, 1.31), (1.08, 0.025, 0.56), 'a5bbc3', 0)
cylinder('lamp_base', (-1.95, 1.94, 0.235), 0.23, 0.07, '514b47')
cylinder('lamp_stem', (-1.95, 1.94, 0.925), 0.035, 1.45, '514b47')
cylinder('lamp_shade', (-1.95, 1.94, 1.65), 0.26, 0.35, 'ece1af')
cylinder('plant_pot', (1.95, -1.6, 0.35), 0.19, 0.3, 'cf8596')
cylinder('plant_stem', (1.95, -1.6, 0.6), 0.035, 0.35, '6d8051')
box('plant_leaf_0', (1.84, -1.6, 0.75), (0.25, 0.17, 0.3), '739662', 0.0)
box('plant_leaf_1', (2.06, -1.58, 0.75), (0.25, 0.17, 0.3), '739662', 0.8)
box('plant_leaf_2', (1.95, -1.5, 0.75), (0.25, 0.17, 0.3), '739662', 1.6)
box('artwork_frame', (0.55, 2.59, 2.2), (0.95, 0.065, 0.7), '927861', 0)
box('artwork_canvas', (0.55, 2.547, 2.2), (0.83, 0.025, 0.58), 'ece1af', 0)
box('artwork_shape', (0.62, 2.527, 2.2), (0.37, 0.02, 0.33), 'cf8596', 0)
bpy.context.view_layer.update()
