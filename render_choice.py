#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import torch
from scene import Scene
import os
import sys
from tqdm import tqdm
from os import makedirs
from gaussian_renderer import render
import torchvision
from utils.camera_utils import parse_custom_cameras, camera_from_dict
from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams, get_combined_args
from gaussian_renderer import GaussianModel
import numpy as np
import copy

from utils.camera_generator import get_start_camera
from utils.camera_custom_utils import move_forward, rotate_camera_dict_about_up_direction, rise_relative_to_camera, move_sideways, set_position_to_origin
from render_utils import *
from env_configs import COLOR_MAP, ENV_CONFIGS
from drone_causality.utils.model_utils import get_params_from_json, load_model_from_weights, generate_hidden_list
from gym_pybullet_drones.examples.schemas import parse_init_conditions
from gym_pybullet_drones.examples.simulator_utils import get_x_y_z_yaw_relative_to_base_env

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(SCRIPT_DIR, ".."))
sys.path.append(os.path.join(SCRIPT_DIR, "..", "gym-pybullet-drones"))
sys.path.append(os.path.join(SCRIPT_DIR, "..", "gym-pybullet-drones", "gym_pybullet_drones", "examples"))
from gym_pybullet_drones.examples.simulator_eval import EvalSimulator


pybullet_inference = False

def render_set(model_path, name, iteration, views, gaussians, pipeline, background, custom_camera_path=None):
    if custom_camera_path is None:
        render_path = os.path.join(model_path, name, "ours_{}".format(iteration), "renders")
    else:
        render_path = os.path.dirname(custom_camera_path)
    gts_path = os.path.join(model_path, name, "ours_{}".format(iteration), "gt")

    makedirs(render_path, exist_ok=True)
    makedirs(gts_path, exist_ok=True)
    makedirs(os.path.join(render_path, "pics0"), exist_ok=True)

    for idx, view in enumerate(tqdm(views, desc="Rendering progress")):
        rendering = render(view, gaussians, pipeline, background)["render"]
        # flip across the x axis to match the original image
        rendering = torch.flip(rendering, [1])
        torchvision.utils.save_image(rendering, os.path.join(render_path, "pics0", '{0:05d}'.format(idx) + ".png"))
        if custom_camera_path is None:
            gt = view.original_image[0:3, :, :]
            torchvision.utils.save_image(gt, os.path.join(gts_path, '{0:05d}'.format(idx) + ".png"))

def closed_loop_render_set(gaussians, pipeline, background, init_conditions, inference_configs, record_hz, closed_loop_save_path, params_path=None, checkpoint_path=None, object_colors=None, keycamera_path=None, PYBULLET_TO_GS_SCALING_FACTOR=None):
    assert closed_loop_save_path is not None
    render_path = closed_loop_save_path
    makedirs(render_path, exist_ok=True)
    makedirs(os.path.join(render_path, "pics0"), exist_ok=True)

    model_params = get_params_from_json(params_path, checkpoint_path)
    model_params.no_norm_layer = False
    model_params.single_step = True
    single_step_model = load_model_from_weights(model_params, checkpoint_path)
    is_lstm = not (single_step_model.input_shape[1][1] == 1)
    print(f"\n\n {single_step_model.input_shape}")
    hiddens = generate_hidden_list(model=single_step_model, return_numpy=True, lstm=is_lstm)

    # ! Setup Simulator
    sim = EvalSimulator(render_path, init_conditions, record_hz)
    sim.setup_simulation()
    height_offset = sim.start_height - 0.1 - 0.5
    num_objects = len(sim.objects_color_target)
    
    CLOSED_LOOP_NUM_FRAMES = int(num_objects * 240.0 * record_hz / 8.0) * 4

    # init stabilization
    vel_cmd = np.array([0, 0, 0, 0])
    sim.vel_cmd_world = vel_cmd
    updated_state, pybullet_img, finished = sim.dynamic_step_simulation(vel_cmd)
    updated_position = get_x_y_z_yaw_relative_to_base_env(updated_state, sim.theta_environment)
    pybullet_img = pybullet_img[None,:,:,0:3]

    init_forward = updated_position[0]
    init_side = updated_position[1]
    init_height_offset = updated_position[2] - 0.1 - 0.5
    init_yaw = updated_position[3] - sim.theta_environment

    start_dict = get_start_camera(keycamera_path)
    start_dict = set_position_to_origin(start_dict)
    start_dict, _ = move_forward(start_dict, -(init_forward * PYBULLET_TO_GS_SCALING_FACTOR), np.array([0, 0, 0, 0]))
    start_dict, _ = move_sideways(start_dict, -(init_side * PYBULLET_TO_GS_SCALING_FACTOR), np.array([0, 0, 0, 0]))
    global_start = copy.deepcopy(start_dict)
    start_dict, _ = rise_relative_to_camera(start_dict, init_height_offset * PYBULLET_TO_GS_SCALING_FACTOR, np.array([0, 0, 0, 0]))
    start_dict, _ = rotate_camera_dict_about_up_direction(start_dict, init_yaw, np.array([0, 0, 0, 0]))
    view = camera_from_dict(start_dict)

    unnormalized_vel_cmds = []
    for idx in tqdm(range(CLOSED_LOOP_NUM_FRAMES)):
        if pybullet_inference:
            input = copy.deepcopy(pybullet_img)
        else:
            rendering = render(view, gaussians, pipeline, background)["render"]
            rendering = torch.flip(rendering, [1]) # flip across the x axis to match the original image
            torchvision.utils.save_image(rendering, os.path.join(render_path, "pics0", '{0:05d}'.format(idx) + ".png"))
            gs_img = transform_gs_img_to_network_input(rendering)
            
            input = copy.deepcopy(gs_img)

        if is_lstm:
            inputs = [input, *hiddens]
        else:
            # inputs = [input, np.array(1.0).reshape(-1, 1), *hiddens]
            inputs = [input, np.array(1.0 / record_hz * 5).reshape(-1, 1), *hiddens]
        out = single_step_model.predict(inputs)
        unnormalized_vel_cmds.append(out[0][0])
        vel_cmd = out[0][0]  # shape: 1 x 8
        hiddens = out[1:]

        vel_cmd[0] = vel_cmd[0] * 0.5

        # Put into simulator
        updated_state, pybullet_img, finished = sim.dynamic_step_simulation(vel_cmd)
        if finished:
            break
        pybullet_img = pybullet_img[None,:,:,0:3]
        
        updated_position = get_x_y_z_yaw_relative_to_base_env(updated_state, sim.theta_environment)
        updated_position -= [init_forward, 0, 0.6, sim.theta_environment]
    
        start_dict = copy.deepcopy(global_start)
        start_dict, _ = move_forward(start_dict, updated_position[0] * PYBULLET_TO_GS_SCALING_FACTOR, np.array([0, 0, 0, 0]))
        start_dict, _ = move_sideways(start_dict, -updated_position[1] * PYBULLET_TO_GS_SCALING_FACTOR, np.array([0, 0, 0, 0]))
        start_dict, _ = rise_relative_to_camera(start_dict, updated_position[2] * PYBULLET_TO_GS_SCALING_FACTOR, np.array([0, 0, 0, 0]))
        start_dict, _ = rotate_camera_dict_about_up_direction(start_dict, updated_position[3], np.array([0, 0, 0, 0]))

        view = camera_from_dict(start_dict)

    sim.export_plots()
    np.savetxt(os.path.join(render_path, "vel_cmds_unnorm.csv"), np.array(unnormalized_vel_cmds), delimiter=",")


def render_sets(dataset : ModelParams, iteration : int, pipeline : PipelineParams, inference_configs: dict, scene_config: dict, init_conditions):
    custom_camera_paths = inference_configs["custom_camera_paths"]
    is_closed_loop = inference_configs["is_closed_loop"]
    
    with torch.no_grad():
        gaussians = GaussianModel(dataset.sh_degree)

        # ! Load Scene
        scene = Scene(dataset, gaussians, scene_config=scene_config, load_iteration=iteration, shuffle=False)
        bg_color = [1,1,1] if dataset.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

        # ! Closed Loop inference with dynamic interplay between GS and pybullet simulators
        if is_closed_loop:
            for closed_loop_save_path, params_path, checkpoint_path, record_hz in tqdm(zip(closed_loop_save_paths, params_paths, checkpoint_paths, inference_configs["record_hzs"]), desc="Closed loop rendering progress"):
                closed_loop_render_set(gaussians, pipeline, background, init_conditions, inference_configs, record_hz, closed_loop_save_path, params_path, checkpoint_path, objects_color, keycamera_path, PYBULLET_TO_GS_SCALING_FACTOR)
        # ! Regular Inference with specified camera paths
        elif custom_camera_paths is not None:
            for custom_camera_path in tqdm(custom_camera_paths):
                render_set(dataset.model_path, "custom_train", scene.loaded_iter, parse_custom_cameras(custom_camera_path), gaussians, pipeline, background, custom_camera_path)


if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Testing script parameters")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--record_hzs", nargs='*', default=None, type=str)
    parser.add_argument("--quiet", action="store_true")
    
    parser.add_argument("--checkpoint_paths", nargs='*', default=None, type=str)
    parser.add_argument("--closed_loop_save_paths", nargs='*', default=None, type=str)
    parser.add_argument("--custom_camera_paths", nargs='*', default=None, type=str)
    parser.add_argument("--params_paths", nargs='*', default=None, type=str)
    parser.add_argument("--objects_color", nargs='*', default=None, type=str)
    parser.add_argument("--is_closed_loop", action="store_true", default=False)
    parser.add_argument("--full_path", default=None, type=str)
    parser.add_argument('--env_name', type=str, default="holodeck", help='Environment name')
    args = get_combined_args(parser)

    env_name = args.env_name

    is_closed_loop = getattr(args, "is_closed_loop", False)
    record_hzs = getattr(args, "record_hzs", None)
    custom_camera_paths = getattr(args, "custom_camera_paths", None)

    keycamera_path = ENV_CONFIGS[env_name]["keycamera_path"]
    PYBULLET_TO_GS_SCALING_FACTOR = ENV_CONFIGS[env_name]["PYBULLET_TO_GS_SCALING_FACTOR"]

    if is_closed_loop:
        closed_loop_save_paths = getattr(args, "closed_loop_save_paths", None)
        print(f"closed_loop_save_paths: {closed_loop_save_paths}")
        params_paths = getattr(args, "params_paths", None)
        checkpoint_paths = getattr(args, "checkpoint_paths", None)

        init_conditions = generate_init_conditions_closed_loop_inference(args.objects_color, PYBULLET_TO_GS_SCALING_FACTOR, closed_loop_save_paths)
        objects_color = init_conditions["objects_color"]
        objects_relative = init_conditions["objects_relative"]
        theta_environment = init_conditions["theta_environment"]
    else:

        init_conditions = parse_init_conditions(args.full_path)    

        start_dist = init_conditions["start_dist"]
        theta_environment = init_conditions["theta_environment"]
        theta_offset = init_conditions["theta_offset"]
        objects_relative = init_conditions["objects_relative"]
        objects_color = init_conditions["objects_color"]
        correct_side = init_conditions.get("correct_side", None)

        gs_start_dist = start_dist * PYBULLET_TO_GS_SCALING_FACTOR

    gs_offsets_from_camera = [[0, 0, 0]] # environment offset is 0
    for object_relative in objects_relative:
        # NOTE: y is opposite in pybullet compared to GS coordinates, so we flip it here:
        gs_offsets_from_camera.append([object_relative[0] * PYBULLET_TO_GS_SCALING_FACTOR, -object_relative[1] * PYBULLET_TO_GS_SCALING_FACTOR, 0])

    color_paths = [COLOR_MAP[color] for color in objects_color]
    object_paths = [ENV_CONFIGS[env_name]["ply_path"], *color_paths]


    #TODO: Make smaller depending what is used
    inference_configs = {
        "is_closed_loop": is_closed_loop,
        "custom_camera_paths": custom_camera_paths,

        # objects / positioning
        "object_paths": object_paths,
        "objects_color": objects_color,
        "gs_offsets_from_camera": gs_offsets_from_camera,
        "PYBULLET_TO_GS_SCALING_FACTOR": PYBULLET_TO_GS_SCALING_FACTOR,
    }

    if is_closed_loop:
        inference_configs.update(
            {
                # closed loop params
                "closed_loop_save_paths": closed_loop_save_paths,
                "params_paths": params_paths,
                "checkpoint_paths": checkpoint_paths,
                "record_hzs": [int(x) for x in record_hzs] if record_hzs is not None else None,
            }
        )
    else:
        inference_configs.update(
            {
                "custom_camera_paths": custom_camera_paths,
                "keycamera_path": keycamera_path,
            }
        )

    scene_config = {
        "object_paths": object_paths,
        "theta_environment": theta_environment,
        "gs_offsets_from_camera": gs_offsets_from_camera,
        "keycamera_path": keycamera_path,
        "train_mode": not is_closed_loop
    }

    render_sets(model.extract(args), args.iteration, pipeline.extract(args), inference_configs, scene_config, init_conditions)