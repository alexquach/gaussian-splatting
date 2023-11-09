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
from PIL import Image
import copy
import matplotlib.pyplot as plt

from camera_generator import get_start_camera
from camera_custom_utils import move_forward, rotate_camera_dict_about_up_direction, rise_relative_to_camera, move_sideways

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(SCRIPT_DIR, ".."))
sys.path.append(os.path.join(SCRIPT_DIR, "..", "gym-pybullet-drones"))
sys.path.append(os.path.join(SCRIPT_DIR, "..", "gym-pybullet-drones", "gym_pybullet_drones", "examples"))
from drone_multimodal.utils.model_utils import load_model_from_weights, generate_hidden_list, get_readable_name, \
    get_params_from_json
from drone_multimodal.keras_models import IMAGE_SHAPE
from drone_multimodal.preprocess.process_data_util import resize_and_crop
import importlib  
from gym_pybullet_drones.examples.simulator_eval import EvalSimulator as Simulator

COLOR_MAP = {
    "R": "/home/makramchahine/repos/gaussian-splatting/output/solid_red_ball",
    "B": "/home/makramchahine/repos/gaussian-splatting/output/solid_blue_ball",
}
PYBULLET_TO_GS_SCALING_FACTOR = 2.5

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
        # put a white dot in the center of the image for calibration
        # rendering[:, 376, 500] = 1
        torchvision.utils.save_image(rendering, os.path.join(render_path, "pics0", '{0:05d}'.format(idx) + ".png"))
        if custom_camera_path is None:
            gt = view.original_image[0:3, :, :]
            torchvision.utils.save_image(gt, os.path.join(gts_path, '{0:05d}'.format(idx) + ".png"))

def closed_loop_render_set(model_path, name, iteration, views, gaussians, pipeline, background, custom_camera_path=None, closed_loop_save_path=None, normalize_path=None, params_path=None, checkpoint_path=None):
    assert closed_loop_save_path is not None
    render_path = closed_loop_save_path
    makedirs(render_path, exist_ok=True)
    makedirs(os.path.join(render_path, "pics0"), exist_ok=True)

    print(f"params_path: {params_path}")
    print(f"checkpoint_path: {checkpoint_path}")
    model_params = get_params_from_json(params_path, checkpoint_path)
    model_params.no_norm_layer = False
    model_params.single_step = True
    single_step_model = load_model_from_weights(model_params, checkpoint_path)
    hiddens = generate_hidden_list(model=single_step_model, return_numpy=True)
    
    if normalize_path is not None:
        df_norm = pd.read_csv(normalize_path, index_col=0)
        np_mean = df_norm.iloc[0].to_numpy()
        np_std = df_norm.iloc[1].to_numpy()
    print('Loaded Model')

    camera_dict = get_start_camera()
    random_dist = random.uniform(-1, 1)
    random_yaw = 0 #random.uniform(0.175 * np.pi, -0.175 * np.pi)
    camera_dict, _ = move_forward(camera_dict, random_dist, np.array([0, 0, 0, 0]))
    camera_dict, _ = rotate_camera_dict_about_up_direction(camera_dict, random_yaw, np.array([0, 0, 0, 0]))
    view = camera_from_dict(camera_dict)
    # optinoally do some translation
    for idx in tqdm(range(CLOSED_LOOP_NUM_FRAMES)):
        rendering = render(view, gaussians, pipeline, background)["render"]
        rendering = torch.flip(rendering, [1]) # flip across the x axis to match the original image
        torchvision.utils.save_image(rendering, os.path.join(render_path, "pics0", '{0:05d}'.format(idx) + ".png"))

        # calculate next camera position with model velocity updates
        # transform rendering to (h, w, 4)-shaped array of uint8's containing the RBG(A) image
        img = rendering.cpu().numpy().transpose(1, 2, 0) * 255 

        # center crop img to IMAGE_SHAPE
        img = Image.fromarray(img.astype(np.uint8))
        # resize to keep aspect ratio
        img = resize_and_crop(img, (IMAGE_SHAPE[1], IMAGE_SHAPE[0]))
        # img.save(os.path.join(render_path, "pics0", '{0:05d}r'.format(idx) + ".png"))
        img = np.array(img).astype(np.float32) / 255.0
        
        # save the resulting img as a png

        # save img to disk
        # img = img[None,:,:,0:3]
        # torchvision.utils.save_image(torch.from_numpy(img), os.path.join(render_path, "pics0", '{0:05d}r'.format(idx) + ".png"))
        
        # save img to disk as png using numpy functions
        # np.save(os.path.join(render_path, "pics0", '{0:05d}r'.format(idx) + ".npy"), img)

        img = img[None,:,:,0:3]
        inputs = [img, *hiddens]
        out = single_step_model.predict(inputs)
        if normalize_path is not None:
            out[0][0] = out[0][0] * np_std + np_mean
        vel_cmd = out[0][0]  # shape: 1 x 8
        hiddens = out[1:]

        # move camera according to velocity commands
        camera_dict, _ = move_forward(camera_dict, vel_cmd[0], np.array([0, 0, 0, 0]))
        camera_dict, _ = rotate_camera_dict_about_up_direction(camera_dict, vel_cmd[3], np.array([0, 0, 0, 0]))
        view = camera_from_dict(camera_dict)


def generate_init_conditions(object_color, pybullet_sideways_offset=0):
    """Shared between training and CL inference"""
    start_H = 0.1 + random.choice([0, 1])
    # start_H = 0.1 + random.uniform(0, 1)
    target_Hs = [0.1 + 0.5]
    Theta = 0 #random.random() * 2 * np.pi
    Theta_offset = random.uniform(0.175 * np.pi, -0.175 * np.pi) #random.choice([0.175 * np.pi, -0.175 * np.pi])
    rel_obj = [(random.uniform(1, 2), 0)]

    pybullet_sideways_offset = pybullet_sideways_offset if object_color == "R" else -pybullet_sideways_offset
    rel_obj.append((rel_obj[0][0] - 0.2 / PYBULLET_TO_GS_SCALING_FACTOR, pybullet_sideways_offset))

    return start_H, target_Hs, Theta, Theta_offset, rel_obj


def dynamic_closed_loop_render_set(gaussians, pipeline, background, start_H, target_Hs, Theta, Theta_offset, rel_obj, closed_loop_save_path=None, normalize_path=None, params_path=None, checkpoint_path=None, object_colors=None, use_dynamic=False, sideways_offset=0):
    assert closed_loop_save_path is not None
    render_path = closed_loop_save_path
    makedirs(render_path, exist_ok=True)
    makedirs(os.path.join(render_path, "pics0"), exist_ok=True)
    pybullet_path = os.path.join(render_path, "pybullet_pics0")

    print(f"params_path: {params_path}")
    print(f"checkpoint_path: {checkpoint_path}")
    model_params = get_params_from_json(params_path, checkpoint_path)
    model_params.no_norm_layer = False
    model_params.single_step = True
    single_step_model = load_model_from_weights(model_params, checkpoint_path)
    hiddens = generate_hidden_list(model=single_step_model, return_numpy=True)
    
    if normalize_path is not None:
        df_norm = pd.read_csv(normalize_path, index_col=0)
        np_mean = df_norm.iloc[0].to_numpy()
        np_std = df_norm.iloc[1].to_numpy()
    print('Loaded Model')

    # ! Generate random dimensions
    forward_dist = rel_obj[0][0]

    record_hz = 4
    CLOSED_LOOP_NUM_FRAMES = int(2 * 240.0 * record_hz / 8.0)
    sim = Simulator(object_colors, rel_obj, render_path, start_H, target_Hs, Theta, Theta_offset, record_hz)
    sim.setup_simulation()
    height_offset = sim.start_H - 0.1 - 0.5

    # init stabilization
    vel_cmd = np.array([0, 0, 0, 0])
    sim.vel_cmd_world = vel_cmd
    _, rgb, finished = sim.dynamic_step_simulation(vel_cmd)
    rgb = rgb[None,:,:,0:3]

    camera_dict = get_start_camera()
    camera_dict, _ = move_forward(camera_dict, 3.5 - (forward_dist * PYBULLET_TO_GS_SCALING_FACTOR), np.array([0, 0, 0, 0]))
    camera_dict, _ = rise_relative_to_camera(camera_dict, height_offset * PYBULLET_TO_GS_SCALING_FACTOR, np.array([0, 0, 0, 0]))
    camera_dict, _ = rotate_camera_dict_about_up_direction(camera_dict, Theta_offset, np.array([0, 0, 0, 0]))
    view = camera_from_dict(camera_dict)

    unnormalized_vel_cmds = []

    # optinoally do some translation
    for idx in tqdm(range(CLOSED_LOOP_NUM_FRAMES)):
        rendering = render(view, gaussians, pipeline, background)["render"]
        rendering = torch.flip(rendering, [1]) # flip across the x axis to match the original image
        torchvision.utils.save_image(rendering, os.path.join(render_path, "pics0", '{0:05d}'.format(idx) + ".png"))

        # calculate next camera position with model velocity updates
        # transform rendering to (h, w, 4)-shaped array of uint8's containing the RBG(A) image
        img = rendering.cpu().numpy().transpose(1, 2, 0) * 255 

        img = Image.fromarray(img.astype(np.uint8))
        img = resize_and_crop(img, (IMAGE_SHAPE[1], IMAGE_SHAPE[0]))
        img = np.array(img).astype(np.uint8)

        # Get velocity labels from network inference
        img = img[None,:,:,0:3]
        input = copy.deepcopy(rgb if pybullet_inference else img)
        # plt.imsave(os.path.join(render_path, "network_inputs", f"img_{idx}.png"), input[0])
        # inputs = [input, *hiddens]
        # inputs = [input, np.array(0.375 * 5).reshape(-1, 1), *hiddens]
        inputs = [input, np.array(0.25 * 5).reshape(-1, 1), *hiddens]
        out = single_step_model.predict(inputs)
        unnormalized_vel_cmds.append(out[0][0])
        # if normalize_path is not None:
        #     out[0][0] = out[0][0] * np_std + np_mean
        vel_cmd = out[0][0]  # shape: 1 x 8
        hiddens = out[1:]

        # Put into simulator
        _, rgb, finished = sim.dynamic_step_simulation(vel_cmd)
        if finished:
            break
        rgb = rgb[None,:,:,0:3]

        # move camera according to velocity commands
        displacement = sim.get_latest_displacement()
        camera_dict, _ = move_forward(camera_dict, displacement[0] * PYBULLET_TO_GS_SCALING_FACTOR, np.array([0, 0, 0, 0]))
        camera_dict, _ = move_sideways(camera_dict, displacement[1] * PYBULLET_TO_GS_SCALING_FACTOR, np.array([0, 0, 0, 0]))
        camera_dict, _ = rise_relative_to_camera(camera_dict, displacement[2] * PYBULLET_TO_GS_SCALING_FACTOR, np.array([0, 0, 0, 0]))
        camera_dict, _ = rotate_camera_dict_about_up_direction(camera_dict, displacement[3], np.array([0, 0, 0, 0]))
        
        view = camera_from_dict(camera_dict)

    print(f"sim.window_outcomes: {sim.window_outcomes}")
    sim.export_plots()
    np.savetxt(os.path.join(render_path, "vel_cmds_unnorm.csv"), np.array(unnormalized_vel_cmds), delimiter=",")


def render_sets(dataset : ModelParams, iteration : int, pipeline : PipelineParams, skip_train : bool, skip_test : bool, custom_camera_paths : str, object_color : str, rotation_theta: float = 0.0, closed_loop_save_paths: str = None, normalize_path: str = None, params_paths: str = None, checkpoint_paths: str = None, use_dynamic: bool = False):
    object_path = COLOR_MAP[object_color]
    
    print("Before torch grad")
    with torch.no_grad():
        gaussians = GaussianModel(dataset.sh_degree)
        add_second_ball = use_dynamic
        second_color = random.choice(["R", "B"])
        sideways_offset = random.uniform(2.5, 3)
        scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False, object_path=object_path, rotation_theta=rotation_theta, add_second_ball=add_second_ball, second_color=second_color, sideways_offset=sideways_offset)

        bg_color = [1,1,1] if dataset.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

        if use_dynamic:
            start_H, target_Hs, Theta, Theta_offset, rel_obj = generate_init_conditions(object_color, sideways_offset / PYBULLET_TO_GS_SCALING_FACTOR)
            for closed_loop_save_path, params_path, checkpoint_path in tqdm(zip(closed_loop_save_paths, params_paths, checkpoint_paths), desc="Closed loop rendering progress"):
                dynamic_closed_loop_render_set(gaussians, pipeline, background, start_H, target_Hs, Theta, Theta_offset, rel_obj, closed_loop_save_path, normalize_path, params_path, checkpoint_path, [object_color, second_color], use_dynamic, sideways_offset)
        elif closed_loop_save_paths is not None:
            closed_loop_render_set(dataset.model_path, "closed_loop", scene.loaded_iter, None, gaussians, pipeline, background, custom_camera_paths, closed_loop_save_path, normalize_path, params_path, checkpoint_paths)
        elif custom_camera_paths is not None:
            for custom_camera_path in tqdm(custom_camera_paths):
                render_set(dataset.model_path, "custom_train", scene.loaded_iter, parse_custom_cameras(custom_camera_path), gaussians, pipeline, background, custom_camera_path)
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
    parser.add_argument("--skip_train", action="store_true")
    parser.add_argument("--skip_test", action="store_true")
    parser.add_argument("--object_color", default="something", type=str)
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--custom_camera_paths", nargs='*', default=None, type=str)
    parser.add_argument("--closed_loop_save_paths", nargs='*', default=None, type=str)
    parser.add_argument("--normalize_path", default=None, type=str)
    parser.add_argument("--params_paths", nargs='*', default=None, type=str)
    parser.add_argument("--checkpoint_paths", nargs='*', default=None, type=str)
    parser.add_argument("--use_dynamic", action="store_true", default=False)
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

    if args.custom_camera_paths is not None:
        rand_theta = 0
    else: 
        rand_theta = random.uniform(0, 2 * np.pi)
    
    # Initialize system state (RNG)
    # safe_state(args.quiet)

    print(f"use_dynamic: {args.use_dynamic}")

    # if args.custom_camera_path is not None:
    #     base_dir = os.path.dirname(args.custom_camera_path)
    #     with open(os.path.join(base_dir, "rand_theta.txt"), "w") as f:
    #         f.write(str(rand_theta))


    render_sets(model.extract(args), args.iteration, pipeline.extract(args), args.skip_train, args.skip_test, args.custom_camera_paths, args.object_color, rotation_theta=rand_theta, closed_loop_save_paths = args.closed_loop_save_paths, normalize_path=args.normalize_path, params_paths=args.params_paths, checkpoint_paths=args.checkpoint_paths, use_dynamic=args.use_dynamic)