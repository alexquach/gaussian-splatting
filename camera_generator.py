import json
import numpy as np
import os
import random

from camera_custom_utils import parse_keycamera, process_keycamera_to_W2C, replace_w2c, get_forward_direction, rotate_about_forward_direction, rotate_camera_dict_about_up_direction, dist_from_origin, move_forward
    
camera_json_path = "/home/makramchahine/repos/gaussian-splatting/output/holodeck2/cameras.json"
keycamera_file_path = "./camera_assets/key_cameras_4"
base_dir = "./camera_assets/generated"
debug_dir = "./camera_assets/debug"

def get_reference_camera_structure(camera_json_path=camera_json_path):
    with open(camera_json_path) as f:
        cameras = json.load(f)
    return cameras

def get_keycameras(file_path=keycamera_file_path):
    parsed_data = parse_keycamera(file_path)
    keycameras = [process_keycamera_to_W2C(keycamera) for keycamera in parsed_data]

    return keycameras

INCREMENT_FORWARD = 0.05
INCREMENT_YAW = 2*np.pi / 80
START_TURNING_DIST = 1.5
STOP_TURN_THRESHOLD = np.pi / 2

debug = False

def generate_one_camera_path(save_path, color):
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    
    save_list = []
    ref_camera = get_reference_camera_structure()[185]
    keycamera = get_keycameras()[0]
    deltas = []


    start_dict = replace_w2c(ref_camera, keycamera)
    # ? Rotate camera to correct orientation in the forward axis
    start_dict = rotate_about_forward_direction(start_dict, np.pi/2)
    
    # ? Move forward/backwards for randomization
    random_dist = random.uniform(-1, 1)
    # base_forward_direction = get_forward_direction(start_dict)
    start_dict, _ = move_forward(start_dict, random_dist, np.array([0, 0, 0, 0]))
    
    # ? Rotate camera for randomization about the up axis
    save_list.append(start_dict)

    # ? Generate rest of trajectory
    accumulated_yaw = 0
    has_turned = False
    for i in range(200):
        delta = np.array([0, 0, 0, 0])
        last_dict = save_list[-1].copy()

        dist = dist_from_origin(last_dict)
        last_dict, delta = move_forward(last_dict, INCREMENT_FORWARD, delta)
        if dist < START_TURNING_DIST and abs(accumulated_yaw) <= STOP_TURN_THRESHOLD:
            if color == "R":
                turn_yaw = INCREMENT_YAW
            else:
                turn_yaw = -INCREMENT_YAW
            
            last_dict, delta = rotate_camera_dict_about_up_direction(last_dict, turn_yaw, delta)
            accumulated_yaw += INCREMENT_YAW
            has_turned = True
        elif has_turned:
            break

        save_list.append(last_dict)
        deltas.append(delta)

    # ? Save trajectory and delta velocities
    with open(f"{save_path}/path.json", 'w') as outfile:
        json.dump(save_list, outfile)
    np.savetxt(f"{save_path}/deltas.csv", np.array(deltas), delimiter=",")
    with open(f"{save_path}/colors.txt", 'w') as f:
        f.write(str(color))
    
NUM_SAMPLES = 300

if debug:
    generate_one_camera_path(f"{debug_dir}")
else:
    # Create an array of length NUM_SAMPLES, where half the entries are "B" and half are "R"
    colors = ["B"] * (NUM_SAMPLES // 2) + ["R"] * (NUM_SAMPLES // 2)
    random.shuffle(colors)
    
    for i, color in enumerate(colors):
        generate_one_camera_path(f"{base_dir}/path_{i}", color)
