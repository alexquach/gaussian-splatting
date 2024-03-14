import bpy
import os
import math
import numpy as np

bpy.ops.preferences.addon_enable(module="io_scene_obj")

def load_obj_mtl(obj_filepath, mtl_filepath):
    # Ensure the .mtl file is in the same directory as the .obj for auto-loading materials
    bpy.ops.import_scene.obj(filepath=obj_filepath, filter_glob="*.obj;*.mtl")

def set_camera_location(angle_x, angle_y, distance=10):
    # Set camera angles and distance from origin
    camera = bpy.data.objects["Camera"]
    camera.location.x = distance * math.sin(angle_x) * math.cos(angle_y) / 2
    camera.location.y = distance * math.sin(angle_x) * math.sin(angle_y) / 2
    camera.location.z = distance * math.cos(angle_x) / 2
    camera.rotation_euler = (angle_x, 0, angle_y + math.pi/2) # The added math.pi/2 is to orient the camera correctly
    # Point camera to the origin
    bpy.ops.object.select_all(action='DESELECT')
    camera.select_set(True)
    track_to = camera.constraints.new('TRACK_TO')
    track_to.target = bpy.context.scene.objects["Empty"]
    track_to.track_axis = 'TRACK_NEGATIVE_Z'
    track_to.up_axis = 'UP_Y'

    return camera.location, camera.rotation_quaternion, camera.name

def render_and_save(output_path):
    bpy.context.scene.render.filepath = output_path
    bpy.ops.render.render(write_still=True)

# Check if the default cube exists
if "Cube" in bpy.data.objects:
    # Select the cube
    bpy.data.objects["Cube"].select_set(True)
    # Delete the cube
    bpy.ops.object.delete()

# Load the .OBJ and .MTL files
obj_filepath = './data/sphere_smooth.obj'
mtl_filepath = './data/sphere_smooth.mtl' # This is just for reference; Blender will automatically search for the .MTL file in the same directory as the .OBJ
save_dir = './data/new_red_ball'
load_obj_mtl(obj_filepath, mtl_filepath)

def set_material(obj):
    # Create a new material
    mat = bpy.data.materials.new(name="Red Color")
    # Set the material's diffuse color
    mat.diffuse_color = (1, 0, 0, 1)
    # Reduce specularity and glossiness
    mat.specular_intensity = 0.5  # Reduce specularity (range 0-1)
    mat.roughness = 1.0  # Increase roughness to reduce glossiness (range 0-1)
    
    # Assign the material to the object
    if obj.data.materials:
        # Object already has material, replace it
        obj.data.materials[0] = mat
    else:
        # No material yet, add one
        obj.data.materials.append(mat)

# Select the loaded object
obj = bpy.context.selected_objects[0]
set_material(obj)

def adjust_sizing(ratio):
    bpy.ops.transform.resize(value=(ratio, ratio, ratio))
    bpy.data.cameras["Camera"].lens /= ratio

# resize ball to be 1/4 the size
# zoom in camera, by zooming in the lens by 4x
adjust_sizing(0.25)

# Create an empty at the origin, which the camera will point to
bpy.ops.object.empty_add(location=(0, 0, 0))

# set background to white
bpy.context.scene.world.use_nodes = True
bpy.context.scene.world.node_tree.nodes["Background"].inputs[0].default_value = (0, 0, 0, 0)


def setup_lights():
    # Duplicate the light source to create 6 lights for octahedron formation
    light = bpy.data.objects["Light"]
    light.select_set(True)
    for i in range(5):
        bpy.ops.object.duplicate_move()

    # Select all the lights
    bpy.ops.object.select_all(action='DESELECT')
    lights = [light for light in bpy.data.objects if light.name.startswith("Light")]

    dist = 5
    # Position the lights in octahedron formation
    for i, light in enumerate(lights):
        if i == 0:
            light.location.x = 0
            light.location.y = 0
            light.location.z = dist
        elif i == 1:
            light.location.x = dist
            light.location.y = 0
            light.location.z = 0
        elif i == 2:
            light.location.x = 0
            light.location.y = dist
            light.location.z = 0
        elif i == 3:
            light.location.x = -dist
            light.location.y = 0
            light.location.z = 0
        elif i == 4:
            light.location.x = 0
            light.location.y = -dist
            light.location.z = 0
        elif i == 5:
            light.location.x = 0
            light.location.y = 0
            light.location.z = -dist

    # make all the lights lower power
    for light in lights:
        light.data.energy = 100
        # make all lights point to the origin
        bpy.ops.object.select_all(action='DESELECT')
        light.select_set(True)
        track_to = light.constraints.new('TRACK_TO')
        track_to.target = bpy.context.scene.objects["Empty"]
setup_lights()

def generate_views():
    # Render views from different angles
    xs = np.linspace(0.1, 2*math.pi, 10)
    ys = np.linspace(0, 2*math.pi, 100)
    angles = [(x, y) for x in xs for y in ys]

    camera = bpy.data.objects["Camera"]

    transforms_train = {}
    transforms_val = {}
    transforms_test = {}
    transforms_train['camera_angle_x'] = camera.data.angle_x
    transforms_train['frames'] = []
    transforms_val['camera_angle_x'] = camera.data.angle_x
    transforms_val['frames'] = []
    transforms_test['camera_angle_x'] = camera.data.angle_x
    transforms_test['frames'] = []

    for i, (angle_x, angle_y) in enumerate(angles):
        loc, rot, camera_name = set_camera_location(angle_x, angle_y)
        
        output_path = f'{save_dir}/input/render_{i}.png'
        render_and_save(output_path)

        # Save the camera transform
        transform = {}
        transform['file_path'] = f'./input/render_{i}'
        transform['rotation'] = rot[1]
        transform['transform_matrix'] = np.array(camera.matrix_world).tolist()

        if i < 50:
            transforms_test['frames'].append(transform)
        elif i < 100:
            transforms_val['frames'].append(transform)
        else:
            transforms_train['frames'].append(transform)

    import json
    with open(f'{save_dir}/transforms_train.json', 'w') as f:
        json.dump(transforms_train, f)
    with open(f'{save_dir}/transforms_val.json', 'w') as f:
        json.dump(transforms_val, f)
    with open(f'{save_dir}/transforms_test.json', 'w') as f:
        json.dump(transforms_test, f)

generate_views()