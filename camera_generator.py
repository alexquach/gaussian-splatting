import json
import numpy as np
import os
import random

from camera_custom_utils import parse_keycamera, process_keycamera_to_W2C, replace_w2c, get_yaw_diff_relative_to_origin, rise_relative_to_camera, rotate_about_forward_direction, rotate_camera_dict_about_up_direction, dist_from_origin, move_forward, move_sideways
    
NUM_SAMPLES = 300
camera_json_path = "/home/makramchahine/repos/gaussian-splatting/output/holodeck2/cameras.json"
keycamera_file_path = "./camera_assets/key_cameras_4"
base_dir = f"./camera_assets/train_g1_xyzs_{NUM_SAMPLES}"
debug_dir = "./camera_assets/debug"

def get_reference_camera_structure(camera_json_path=camera_json_path):
    with open(camera_json_path) as f:
        cameras = json.load(f)
    return cameras

def get_keycameras(file_path=keycamera_file_path):
    parsed_data = parse_keycamera(file_path)
    keycameras = [process_keycamera_to_W2C(keycamera) for keycamera in parsed_data]

    return keycameras

MAX_FRAMES = 200
CRITICAL_DIST = 1.5
CRITICAL_DIST_BUFFER = 0.5
STOP_TURN_THRESHOLD = np.pi / 2

DEFAULT_SPEED = 0.05
CRITICAL_SPEED = 0.03

DEFAULT_YAW_SPEED = 0.0175 * np.pi
CRITICAL_YAW_SPEED = 0.0175 * np.pi
MIN_YAW_SPEED = 0.0 * np.pi
APPROX_CORRECT_YAW = 0.000001
YAW_START_SLOWDOWN = 0.015 * np.pi
STABILIZE_LIFT_SPEED = 0.05 / 8 * 2
LIFT_HEIGHT_BUFFER = 0.1

def get_start_camera():
    ref_camera = get_reference_camera_structure()[185]
    
    new_width = 256
    new_height = 144

    # Calculate the new focal lengths based on the new dimensions
    old_fx = ref_camera['fx']
    old_fy = ref_camera['fy']
    old_width = ref_camera['width']
    old_height = ref_camera['height']

    new_fx = old_fx * new_width / old_width
    new_fy = old_fy * new_height / old_height

    # Update the camera parameters
    ref_camera['width'] = new_width
    ref_camera['height'] = new_height
    ref_camera['fx'] = new_fx
    ref_camera['fy'] = new_fy

    keycamera = get_keycameras()[0]
    start_dict = replace_w2c(ref_camera, keycamera)
    # ? Rotate camera to correct orientation in the forward axis
    start_dict = rotate_about_forward_direction(start_dict, np.pi/2)
    return start_dict

def interpolate_speeds(dist, critical_dist, critical_dist_buffer, speed1, speed2):
    # as dist gets closer to critical_dist, speed gets closer to speed1
    return speed1 + (speed2 - speed1) * np.clip(abs(dist - critical_dist) / critical_dist_buffer, 0, 1)

def scale_yaw_speed(yaw_speed, yaw_dist):
    return yaw_speed * np.sign(yaw_dist)

def generate_one_camera_path(save_path, color):
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    
    save_list = []
    deltas = []
    start_dict = get_start_camera()

    # ? Move forward/backwards for randomization
    random_dist = random.uniform(-1, 1)
    start_dict, _ = move_forward(start_dict, random_dist, np.array([0, 0, 0, 0]))
    Theta_offset = random.choice([0.175 * np.pi, -0.175 * np.pi])
    start_dict, _ = rotate_camera_dict_about_up_direction(start_dict, Theta_offset, np.array([0, 0, 0, 0]))
    height_offset = random.uniform(-0.35, 0.35)
    start_dict, _ = rise_relative_to_camera(start_dict, height_offset, np.array([0, 0, 0, 0]))
    active_height_offset = height_offset
    
    # ? Rotate camera for randomization about the up axis
    save_list.append(start_dict)

    # ? Generate rest of trajectory
    accumulated_yaw = 0
    previously_hit_critical = False
    for i in range(MAX_FRAMES):
        delta = np.array([0, 0, 0, 0]) # LABELS IN ORDER: forward, right, up, yaw
        last_dict = save_list[-1].copy()

        # Get distances
        dist = dist_from_origin(last_dict)
        yaw_dist, yaw_gt = get_yaw_diff_relative_to_origin(last_dict)

        # print(yaw_dist)

        lift_speed = -1 * np.sign(active_height_offset) * interpolate_speeds(abs(active_height_offset), 0, LIFT_HEIGHT_BUFFER, 0, STABILIZE_LIFT_SPEED)
        last_dict, delta = rise_relative_to_camera(last_dict, lift_speed, delta)
        active_height_offset += lift_speed
        print(active_height_offset)

        if dist > CRITICAL_DIST + CRITICAL_DIST_BUFFER and not previously_hit_critical:
            last_dict, delta = move_forward(last_dict, DEFAULT_SPEED, delta)

            if abs(yaw_dist) < APPROX_CORRECT_YAW:
                last_dict, delta = rotate_camera_dict_about_up_direction(last_dict, 0, delta)
            else:
                yaw_speed = interpolate_speeds(yaw_dist, 0, YAW_START_SLOWDOWN, MIN_YAW_SPEED, DEFAULT_YAW_SPEED)
                # yaw_speed = yaw_speed * 0.75 if abs(yaw_dist) < YAW_START_SLOWDOWN else yaw_speed
                last_dict, delta = rotate_camera_dict_about_up_direction(last_dict, scale_yaw_speed(yaw_speed, yaw_dist), delta)
                # last_dict, delta = rotate_camera_dict_about_up_direction(last_dict, scale_yaw_speed(DEFAULT_YAW_SPEED, yaw_dist), delta)

        elif dist > CRITICAL_DIST and not previously_hit_critical:
            speed = interpolate_speeds(dist, CRITICAL_DIST, CRITICAL_DIST_BUFFER, CRITICAL_SPEED, DEFAULT_SPEED)
            last_dict, delta = move_forward(last_dict, speed, delta)

            if abs(yaw_dist) < APPROX_CORRECT_YAW:
                last_dict, delta = rotate_camera_dict_about_up_direction(last_dict, yaw_dist, delta)
            else:
                yaw_speed = interpolate_speeds(yaw_dist, 0, YAW_START_SLOWDOWN, MIN_YAW_SPEED, DEFAULT_YAW_SPEED)
                # yaw_speed = yaw_speed * 0.75 if abs(yaw_dist) < YAW_START_SLOWDOWN else yaw_speed
                last_dict, delta = rotate_camera_dict_about_up_direction(last_dict, scale_yaw_speed(yaw_speed, yaw_dist), delta)

        elif dist < CRITICAL_DIST and abs(accumulated_yaw) <= STOP_TURN_THRESHOLD:
            last_dict, delta = move_forward(last_dict, CRITICAL_SPEED, delta)
            if color == "R":
                turn_yaw = CRITICAL_YAW_SPEED
            else:
                turn_yaw = -CRITICAL_YAW_SPEED
            
            last_dict, delta = rotate_camera_dict_about_up_direction(last_dict, turn_yaw, delta)
            accumulated_yaw += CRITICAL_YAW_SPEED
        elif abs(accumulated_yaw) >= STOP_TURN_THRESHOLD:
            break

        save_list.append(last_dict)
        deltas.append(delta)

    # ? Save trajectory and delta velocities
    with open(f"{save_path}/path.json", 'w') as outfile:
        json.dump(save_list, outfile)
    np.savetxt(f"{save_path}/deltas.csv", np.array(deltas), delimiter=",")
    with open(f"{save_path}/colors.txt", 'w') as f:
        f.write(str(color))

def generate_pybullet_camera_path(run_path, color):
    save_list = []
    start_dict = get_start_camera()

    with open(os.path.join(base_path, run, "start_dist.txt"), "r") as f:
        start_dist = float(f.readline().strip())

    with open(os.path.join(base_path, run, "start_h.txt"), "r") as f:
        start_h = f.readline().strip()
        height_offset = float(start_h) - 0.1

    with open(os.path.join(base_path, run, "theta.txt"), "r") as f:
        theta_key = f.readline().strip()
        theta_map = {
            "P": 0.175 * np.pi,
            "N": -0.175 * np.pi,
        }
        theta_offset = theta_map[theta_key]

    # SCALING_FACTOR = 3.5 / start_dist
    SCALING_FACTOR = 2.5
    # read timestepwise_displacement.csv using numpy
    timestepwise_displacement = np.loadtxt(os.path.join(base_path, run, "timestepwise_displacement.csv"), delimiter=",")
    pos = np.loadtxt(os.path.join(base_path, run, "pos.csv"), delimiter=",")
    
    # ? Move forward/backwards for randomization
    start_dict, _ = move_forward(start_dict, 3.5 - (start_dist * SCALING_FACTOR), np.array([0, 0, 0, 0]))
    start_dict, _ = rotate_camera_dict_about_up_direction(start_dict, theta_offset, np.array([0, 0, 0, 0]))
    start_dict, _ = rise_relative_to_camera(start_dict, height_offset * SCALING_FACTOR, np.array([0, 0, 0, 0]))
    # active_height_offset = height_offset

    save_list.append(start_dict)

    for displacement in timestepwise_displacement:
        start_dict, _ = move_forward(start_dict, displacement[0] * SCALING_FACTOR, np.array([0, 0, 0, 0]))
        start_dict, _ = move_sideways(start_dict, displacement[1] * SCALING_FACTOR, np.array([0, 0, 0, 0]))
        start_dict, _ = rise_relative_to_camera(start_dict, displacement[2] * SCALING_FACTOR, np.array([0, 0, 0, 0]))
        start_dict, _ = rotate_camera_dict_about_up_direction(start_dict, displacement[3], np.array([0, 0, 0, 0]))
        save_list.append(start_dict)

    with open(f"{run_path}/path.json", 'w') as outfile:
        json.dump(save_list, outfile)

mode = "pybullet"

if __name__ == "__main__":
    if mode == "debug":
        color = "R"
        generate_one_camera_path(f"{debug_dir}", color)
    elif mode == "naive":
        # Create an array of length NUM_SAMPLES, where half the entries are "B" and half are "R"
        colors = ["B"] * (NUM_SAMPLES // 2) + ["R"] * (NUM_SAMPLES // 2)
        random.shuffle(colors)
        
        for i, color in enumerate(colors):
            generate_one_camera_path(f"{base_dir}/path_{i}", color)
    elif mode == "pybullet":
        base_path = "/home/makramchahine/repos/gym-pybullet-drones/gym_pybullet_drones/examples/train_d0"

        for run in os.listdir(base_path):
            if not os.path.isdir(os.path.join(base_path, run)):
                continue

            # read color from color.txt
            with open(os.path.join(base_path, run, "colors.txt"), "r") as f:
                color = f.readline().strip()

            generate_pybullet_camera_path(f"{base_path}/{run}", color)
