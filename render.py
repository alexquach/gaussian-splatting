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
from utils.general_utils import safe_state
from utils.camera_utils import parse_custom_cameras, camera_from_dict
from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams, get_combined_args
from gaussian_renderer import GaussianModel
import random
import numpy as np
import pandas as pd
import math

import copy
import matplotlib.pyplot as plt

from camera_generator import get_start_camera
from camera_custom_utils import move_forward, rotate_camera_dict_about_up_direction, rise_relative_to_camera, move_sideways
from render_utils import *
from env_configs import COLOR_MAP, ENV_CONFIGS

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(SCRIPT_DIR, ".."))
sys.path.append(os.path.join(SCRIPT_DIR, "..", "gym-pybullet-drones"))
sys.path.append(os.path.join(SCRIPT_DIR, "..", "gym-pybullet-drones", "gym_pybullet_drones", "examples"))
from gym_pybullet_drones.examples.simulator_eval import EvalSimulator


pybullet_inference = False
vanish_mode = False
NUM_BALLS = 2

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


def dynamic_closed_loop_render_set(gaussians, pipeline, background, sim_dict, closed_loop_save_path, normalize_path=None, params_path=None, checkpoint_path=None, object_colors=None, keycamera_path=None, PYBULLET_TO_GS_SCALING_FACTOR=None):
    assert closed_loop_save_path is not None
    render_path = closed_loop_save_path
    makedirs(render_path, exist_ok=True)
    makedirs(os.path.join(render_path, "pics0"), exist_ok=True)

    start_H = sim_dict['start_H']
    target_Hs = sim_dict['target_Hs']
    Theta = sim_dict['Theta']
    Theta_offset = sim_dict['Theta_offset']
    rel_obj = sim_dict['rel_obj']
    record_hz = sim_dict['record_hz']

    model_params = get_params_from_json(params_path, checkpoint_path)
    model_params.no_norm_layer = False
    model_params.single_step = True
    single_step_model = load_model_from_weights(model_params, checkpoint_path)
    is_lstm = not (single_step_model.input_shape[1][1] == 1)
    print(f"\n\n {single_step_model.input_shape}")
    hiddens = generate_hidden_list(model=single_step_model, return_numpy=True, lstm=is_lstm)
    assert(len(hiddens) == 3, f"Expected 3 hidden layers, got {len(hiddens)}")
    
    if normalize_path is not None:
        df_norm = pd.read_csv(normalize_path, index_col=0)
        np_mean = df_norm.iloc[0].to_numpy()
        np_std = df_norm.iloc[1].to_numpy()

    # ! Setup Simulator
    forward_dist = rel_obj[0][0]
    sim = EvalSimulator(object_colors, rel_obj, render_path, start_H, target_Hs, Theta, Theta_offset, record_hz)
    sim.setup_simulation()
    height_offset = sim.start_H - 0.1 - 0.5
    num_objects = len(sim.ordered_objs)
    CLOSED_LOOP_NUM_FRAMES = int(num_objects * 240.0 * record_hz / 8.0) * 2 

    # init stabilization
    vel_cmd = np.array([0, 0, 0, 0])
    sim.vel_cmd_world = vel_cmd
    _, pybullet_img, finished = sim.dynamic_step_simulation(vel_cmd)
    pybullet_img = pybullet_img[None,:,:,0:3]

    camera_dict = get_start_camera(keycamera_path)
    # old_offset = random.uniform(-1.7, 0.9)
    # camera_dict, _ = move_forward(camera_dict, old_offset, np.array([0, 0, 0, 0]))
    # old_offset =  #random.uniform(-0.75, 0.75) # TODO: FIX
    camera_dict, _ = move_forward(camera_dict, 3.5 - (forward_dist * PYBULLET_TO_GS_SCALING_FACTOR), np.array([0, 0, 0, 0]))
    # camera_dict, _ = move_forward(camera_dict, 3, np.array([0, 0, 0, 0]))
    camera_dict, _ = rotate_camera_dict_about_up_direction(camera_dict, -0.05, np.array([0, 0, 0, 0]))
    camera_dict, _ = rise_relative_to_camera(camera_dict, height_offset * PYBULLET_TO_GS_SCALING_FACTOR, np.array([0, 0, 0, 0]))
    camera_dict, _ = rotate_camera_dict_about_up_direction(camera_dict, Theta_offset, np.array([0, 0, 0, 0]))
    view = camera_from_dict(camera_dict)

    unnormalized_vel_cmds = []

    # if is_lstm:
    # double all
    CLOSED_LOOP_NUM_FRAMES *= 2
    print(f"CLOSED_LOOP_NUM_FRAMES: {CLOSED_LOOP_NUM_FRAMES}")
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
            # TODO: FIX THIS FIXED TIMESTEP
            inputs = [input, np.array(1.0 / 9 * 5).reshape(-1, 1), *hiddens]
            # inputs = [input, np.array(1.0 / record_hz * 5).reshape(-1, 1), *hiddens]
        out = single_step_model.predict(inputs)
        unnormalized_vel_cmds.append(out[0][0])
        # if normalize_path is not None:
        #     out[0][0] = out[0][0] * np_std + np_mean
        vel_cmd = out[0][0]  # shape: 1 x 8
        hiddens = out[1:]

        vel_cmd[0] = vel_cmd[0] * 0.5

        # Put into simulator
        _, pybullet_img, finished = sim.dynamic_step_simulation(vel_cmd)
        if finished:
            break
        pybullet_img = pybullet_img[None,:,:,0:3]

        # move camera according to velocity commands
        displacement = sim.get_latest_displacement()
        camera_dict, _ = move_forward(camera_dict, displacement[0] * PYBULLET_TO_GS_SCALING_FACTOR, np.array([0, 0, 0, 0]))
        camera_dict, _ = move_sideways(camera_dict, displacement[1] * PYBULLET_TO_GS_SCALING_FACTOR, np.array([0, 0, 0, 0]))
        camera_dict, _ = rise_relative_to_camera(camera_dict, displacement[2] * PYBULLET_TO_GS_SCALING_FACTOR, np.array([0, 0, 0, 0]))
        camera_dict, _ = rotate_camera_dict_about_up_direction(camera_dict, displacement[3], np.array([0, 0, 0, 0]))
        view = camera_from_dict(camera_dict)

    sim.export_plots()
    np.savetxt(os.path.join(render_path, "vel_cmds_unnorm.csv"), np.array(unnormalized_vel_cmds), delimiter=",")


def render_sets(dataset : ModelParams, iteration : int, pipeline : PipelineParams, skip_train : bool, skip_test : bool, inference_configs: dict):
    custom_camera_paths = inference_configs["custom_camera_paths"]
    object_paths = inference_configs["object_paths"]
    object_colors = inference_configs["object_colors"]
    gs_offsets = inference_configs["gs_offsets"]
    keycamera_path = inference_configs["keycamera_path"]
    rotation_theta = inference_configs["rotation_theta"]
    closed_loop_save_paths = inference_configs["closed_loop_save_paths"]
    normalize_path = inference_configs["normalize_path"]
    params_paths = inference_configs["params_paths"]
    checkpoint_paths = inference_configs["checkpoint_paths"]
    use_dynamic = inference_configs["use_dynamic"]
    PYBULLET_TO_GS_SCALING_FACTOR = inference_configs["PYBULLET_TO_GS_SCALING_FACTOR"]
    
    with torch.no_grad():
        gaussians = GaussianModel(dataset.sh_degree)

        # ! Load Scene
        scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False, object_paths=object_paths, rotation_theta=rotation_theta, gs_offsets=gs_offsets, keycamera_path=keycamera_path, train_mode=not use_dynamic)
        bg_color = [1,1,1] if dataset.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

        # ! Closed Loop inference with dynamic interplay between GS and pybullet simulators
        if use_dynamic:
            sim_dict = generate_init_conditions(object_colors, gs_offsets[1:], PYBULLET_TO_GS_SCALING_FACTOR)
            for closed_loop_save_path, params_path, checkpoint_path, record_hz in tqdm(zip(closed_loop_save_paths, params_paths, checkpoint_paths, inference_configs["record_hzs"]), desc="Closed loop rendering progress"):
                sim_dict['record_hz'] = record_hz
                dynamic_closed_loop_render_set(gaussians, pipeline, background, sim_dict, closed_loop_save_path, normalize_path, params_path, checkpoint_path, object_colors, keycamera_path, PYBULLET_TO_GS_SCALING_FACTOR)
        # Regular Inference with specified camera paths
        elif custom_camera_paths is not None:
            for custom_camera_path in tqdm(custom_camera_paths):
                render_set(dataset.model_path, "custom_train", scene.loaded_iter, parse_custom_cameras(custom_camera_path), gaussians, pipeline, background, custom_camera_path)
        # Original Inference w/ train / test camera paths
        else:
            if not skip_train:
                render_set(dataset.model_path, "train", scene.loaded_iter, scene.getTrainCameras(), gaussians, pipeline, background, custom_camera_paths)

            if not skip_test:
                render_set(dataset.model_path, "test", scene.loaded_iter, scene.getTestCameras(), gaussians, pipeline, background)


if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Testing script parameters")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--record_hzs", nargs='*', default=None, type=str)
    parser.add_argument("--skip_train", action="store_true")
    parser.add_argument("--skip_test", action="store_true")
    parser.add_argument("--object_colors", nargs='*', default=None, type=str)
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--custom_camera_paths", nargs='*', default=None, type=str)
    parser.add_argument("--closed_loop_save_paths", nargs='*', default=None, type=str)
    parser.add_argument("--normalize_path", default=None, type=str)
    parser.add_argument("--params_paths", nargs='*', default=None, type=str)
    parser.add_argument("--checkpoint_paths", nargs='*', default=None, type=str)
    parser.add_argument("--use_dynamic", action="store_true", default=False)
    parser.add_argument("--rand_theta", type=float)
    args = get_combined_args(parser)
    print("Rendering " + args.model_path)

    if not "custom_camera_paths" in args:
        args.custom_camera_paths = None
    if not "closed_loop_save_paths" in args:
        args.closed_loop_save_paths = None
    if not "normalize_path" in args:
        args.normalize_path = None
    if not "params_paths" in args:
        args.params_paths = None
    if not "checkpoint_paths" in args:
        args.checkpoint_paths = None
    if not "record_hzs" in args:
        args.record_hzs = None
    if not "object_colors" in args:
        args.object_colors = None

    env_name = "holodeck" # colosseum, holodeck
    rand_theta = args.rand_theta
    # if env_name == "holodeck":
    #     rand_theta = random.uniform(0, 2 * np.pi)
    # else: 
    #     rand_theta = 0
    
    # Initialize system state (RNG)
    # safe_state(args.quiet)

    pybullet_rand_forward = random.uniform(1.5, 2)
    keycamera_path = ENV_CONFIGS[env_name]["keycamera_path"]
    PYBULLET_TO_GS_SCALING_FACTOR = ENV_CONFIGS[env_name]["PYBULLET_TO_GS_SCALING_FACTOR"]
    if args.use_dynamic:
        object_colors = args.object_colors
        # object_colors = [random.choice(["R", "B"]) for _ in range(NUM_BALLS)]
        pybullet_rand_forward = random.uniform(1.5, 2)
        gs_rand_forward = pybullet_rand_forward * PYBULLET_TO_GS_SCALING_FACTOR
        # gs_offsets = [[0, 0, -1], [2, 0, 0], [1.9, -3 if object_colors[0] == "R" else 3, 0]] # forward, right, up
        gs_offsets_from_camera = [[0, 0, 0], [gs_rand_forward, 0, 0], [gs_rand_forward * 0.9, -3 if object_colors[0] == "R" else 3, 0]] # forward, right, up
        
        # gs_offsets_from_camera = [[0, 0, 0]] # forward, right, up
        # cur_point = (0, 0)
        # cur_direction = 0 

        # for object_color in object_colors:
        #     pybullet_rand_forward = random.uniform(1.5, 2)
        #     gs_rand_forward = pybullet_rand_forward * PYBULLET_TO_GS_SCALING_FACTOR
            
        #     target_loc = (cur_point[0] + (gs_rand_forward + 0.5) * math.cos(cur_direction), cur_point[1] + (gs_rand_forward + 0.5) * math.sin(cur_direction), 0)
        #     cur_point = (cur_point[0] + gs_rand_forward * math.cos(cur_direction), cur_point[1] + gs_rand_forward * math.sin(cur_direction), 0)
        #     gs_offsets_from_camera.append(target_loc)

        #     if object_color == 'R':
        #         cur_direction += math.pi / 2
        #     elif object_color == 'B':
        #         cur_direction += -math.pi / 2
        
        
        # gs_offsets_from_camera = [[0, 0, 0], [gs_rand_forward, 0, 0], [gs_rand_forward * 0.9, -3 if object_colors[0] == "R" else 3, 0]] # forward, right, up
        # for _ in range(10):
        #     gs_sideways_offset = random.uniform(*GS_SIDEWAYS_OFFSET_RAND_VALUES)
        #     pybullet_sideways_offset = gs_sideways_offset / SECOND_BALL_SCALING_FACTOR
        color_paths = [COLOR_MAP[color] for color in object_colors]
        object_paths = [ENV_CONFIGS[env_name]["ply_path"], *color_paths]
    else:
        object_colors = [args.object_color]
        gs_offsets_from_camera = [[0, 0, 0], [0, 0, 0]]
        color_paths = [COLOR_MAP[color] for color in object_colors]
        object_paths = [ENV_CONFIGS[env_name]["ply_path"], *color_paths]

    print(f"use_dynamic: {args.use_dynamic}")
    print(f"custom_camera_paths: {args.custom_camera_paths}")
    print(f"rand_theta: {rand_theta}")

    # print(f"args.record_hzs: {args.record_hzs}")
    inference_configs = {
        "custom_camera_paths": args.custom_camera_paths,
        "rotation_theta": rand_theta,
        "closed_loop_save_paths": args.closed_loop_save_paths,
        "normalize_path": args.normalize_path,
        "params_paths": args.params_paths,
        "checkpoint_paths": args.checkpoint_paths,
        "use_dynamic": args.use_dynamic,
        "record_hzs": [int(x) for x in args.record_hzs],
        "object_paths": object_paths,
        "object_colors": object_colors,
        "keycamera_path": keycamera_path,
        "gs_offsets": gs_offsets_from_camera,
        "pybullet_rand_forward": pybullet_rand_forward,
        "PYBULLET_TO_GS_SCALING_FACTOR": PYBULLET_TO_GS_SCALING_FACTOR,
    }

    render_sets(model.extract(args), args.iteration, pipeline.extract(args), args.skip_train, args.skip_test, inference_configs)