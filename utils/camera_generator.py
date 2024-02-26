import json
import numpy as np
import os
from copy import deepcopy
import matplotlib.pyplot as plt
import argparse

from utils.camera_custom_utils import (get_start_camera, 
                                 rise_relative_to_camera, 
                                 rotate_camera_dict_about_up_direction, 
                                 move_forward, move_sideways, set_position_to_origin)
from env_configs import ENV_CONFIGS
from gym_pybullet_drones.examples.schemas import parse_init_conditions

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


def generate_one_camera_path(run_path, env_name):
    """
    Generates a path.json that contains the camera positions used for rendering in GS, using the positions in the sim_pos.csv file from pybullet
    
    """
    save_list = []
    start_dict = get_start_camera(ENV_CONFIGS[env_name]["keycamera_path"])
    PYBULLET_TO_GS_SCALING_FACTOR = ENV_CONFIGS[env_name]["PYBULLET_TO_GS_SCALING_FACTOR"]
    init_conditions = parse_init_conditions(run_path)

    start_dist = init_conditions["start_dist"]
    theta_offset = init_conditions["theta_offset"]
    theta_environment = init_conditions["theta_environment"]

    positions = np.loadtxt(os.path.join(run_path, "sim_pos.csv"), delimiter=",")
    init_forward = positions[0][0]
    init_side = positions[0][1]
    init_height_offset = positions[0][2] - 0.1 - 0.5
    init_yaw = positions[0][3] - theta_environment
    
    start_dict = set_position_to_origin(start_dict)
    start_dict, _ = move_forward(start_dict, -(init_forward * PYBULLET_TO_GS_SCALING_FACTOR), np.array([0, 0, 0, 0]))
    start_dict, _ = move_sideways(start_dict, -(init_side * PYBULLET_TO_GS_SCALING_FACTOR), np.array([0, 0, 0, 0]))
    global_start = deepcopy(start_dict)
    start_dict, _ = rise_relative_to_camera(start_dict, init_height_offset * PYBULLET_TO_GS_SCALING_FACTOR, np.array([0, 0, 0, 0]))
    start_dict, _ = rotate_camera_dict_about_up_direction(start_dict, init_yaw, np.array([0, 0, 0, 0]))

    save_list.append(start_dict)

    for position in positions:
        position[0] -= init_forward
        position[2] -= (0.1 + 0.5)
        position[3] -= theta_environment

        # Uses absolute positioning from an origin point
        start_dict = deepcopy(global_start)
        start_dict, _ = move_forward(start_dict, position[0] * PYBULLET_TO_GS_SCALING_FACTOR, np.array([0, 0, 0, 0]))
        start_dict, _ = move_sideways(start_dict, -position[1] * PYBULLET_TO_GS_SCALING_FACTOR, np.array([0, 0, 0, 0]))
        start_dict, _ = rise_relative_to_camera(start_dict, position[2] * PYBULLET_TO_GS_SCALING_FACTOR, np.array([0, 0, 0, 0]))
        start_dict, _ = rotate_camera_dict_about_up_direction(start_dict, position[3], np.array([0, 0, 0, 0]))
        save_list.append(start_dict)

    with open(f"{run_path}/path.json", 'w') as outfile:
        json.dump(save_list, outfile)

    plot_camera_path(save_list, run_path, start_dist)

def plot_camera_path(save_list, run_path, start_dist):
    # Extract x, y positions from the save_list
    x_positions = [camera_dict['position'][1] for camera_dict in save_list]
    y_positions = [camera_dict['position'][0] for camera_dict in save_list]

    targets = [(start_dist, 0.2), (start_dist, -0.2)]

    # Create a plot of x, y positions
    plt.figure(figsize=(10, 6))
    plt.plot(x_positions, y_positions, marker='o')
    for target in targets:
        plt.plot(target[0], target[1], marker='o', color='red')
    plt.title('Camera Path')
    plt.xlabel('X Position')
    plt.ylabel('Y Position')
    plt.grid(True)
    plt.axis('equal')

    # Save the figure
    plt.savefig(f"{run_path}/camera_path.png")
    plt.close()

def generate_camera_paths_for_all_subfolders(base_dir, env_name):
    for run in os.listdir(base_dir):
        if not os.path.isdir(os.path.join(base_dir, run)):
            continue
        generate_one_camera_path(f"{base_dir}/{run}", env_name)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Provide base directory.')
    parser.add_argument('--base_dir', type=str, default="./generated_paths/train_fly_and_turn", help='Base directory for the script')
    parser.add_argument('--env_name', type=str, default="holodeck", help='Environment name')
    args = parser.parse_args()

    base_dir = args.base_dir
    env_name = args.env_name

    generate_camera_paths_for_all_subfolders(base_dir, env_name)
