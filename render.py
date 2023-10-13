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
from gym_pybullet_drones.examples.headless_hike import Simulator
# headless_hike = importlib.import_module("gym-pybullet-drones.gym_pybullet_drones.examples.headless_hike")

CLOSED_LOOP_NUM_FRAMES = 80

COLOR_MAP = {
    "R": "/home/makramchahine/repos/gaussian-splatting/output/solid_red_ball",
    "B": "/home/makramchahine/repos/gaussian-splatting/output/solid_blue_ball",
}

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


def dynamic_closed_loop_render_set(model_path, name, iteration, views, gaussians, pipeline, background, custom_camera_path=None, closed_loop_save_path=None, normalize_path=None, params_path=None, checkpoint_path=None, object_color=None, use_dynamic=False):
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

    # ! Generate random dimensions
    random_dist = random.uniform(1, 2)
    # random_height_offset = random.uniform(0.05, 0.25)
    random_yaw = 0 #random.choice([0.175 * np.pi, -0.175 * np.pi])
    sim = Simulator([object_color], [(random_dist, 0)], render_path, theta_offset=random_yaw)
    sim.setup_simulation()
    height_offset = sim.start_H - 0.1
    SCALING_FACTOR = 2.5

    # init stabilization
    vel_cmd = np.array([0, 0, 0, 0])
    sim.vel_cmd_world = vel_cmd
    sim.dynamic_step_simulation(vel_cmd)

    camera_dict = get_start_camera()
    camera_dict, _ = move_forward(camera_dict, 3.5 - (random_dist * SCALING_FACTOR), np.array([0, 0, 0, 0]))
    camera_dict, _ = rise_relative_to_camera(camera_dict, height_offset * SCALING_FACTOR, np.array([0, 0, 0, 0]))
    print(f"random_yaw: {random_yaw}")
    camera_dict, _ = rotate_camera_dict_about_up_direction(camera_dict, random_yaw, np.array([0, 0, 0, 0]))
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
        img = np.array(img).astype(np.float32) / 255.0

        # Get velocity labels from network inference
        img = img[None,:,:,0:3]
        inputs = [img, *hiddens]
        out = single_step_model.predict(inputs)
        unnormalized_vel_cmds.append(out[0][0])
        if normalize_path is not None:
            out[0][0] = out[0][0] * np_std + np_mean
        vel_cmd = out[0][0]  # shape: 1 x 8
        hiddens = out[1:]

        # Put into simulator
        sim.dynamic_step_simulation(vel_cmd)

        # move camera according to velocity commands
        displacement = sim.get_latest_displacement()
        camera_dict, _ = move_forward(camera_dict, displacement[0] * SCALING_FACTOR, np.array([0, 0, 0, 0]))
        camera_dict, _ = move_sideways(camera_dict, displacement[1] * SCALING_FACTOR, np.array([0, 0, 0, 0]))
        camera_dict, _ = rise_relative_to_camera(camera_dict, displacement[2] * SCALING_FACTOR, np.array([0, 0, 0, 0]))
        print(f"displacement[3]: {displacement[3]}")
        camera_dict, _ = rotate_camera_dict_about_up_direction(camera_dict, displacement[3], np.array([0, 0, 0, 0]))

        view = camera_from_dict(camera_dict)

    sim.export_plots()
    np.savetxt(os.path.join(render_path, "vel_cmds_unnorm.csv"), np.array(unnormalized_vel_cmds), delimiter=",")


def render_sets(dataset : ModelParams, iteration : int, pipeline : PipelineParams, skip_train : bool, skip_test : bool, custom_camera_path : str, object_color : str, rotation_theta: float = 0.0, closed_loop_save_path: str = None, normalize_path: str = None, params_path: str = None, checkpoint_path: str = None, use_dynamic: bool = False):
    object_path = COLOR_MAP[object_color]
    
    with torch.no_grad():
        gaussians = GaussianModel(dataset.sh_degree)
        scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False, object_path=object_path, rotation_theta=rotation_theta)

        bg_color = [1,1,1] if dataset.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

        if use_dynamic:
            dynamic_closed_loop_render_set(dataset.model_path, "closed_loop", scene.loaded_iter, None, gaussians, pipeline, background, custom_camera_path, closed_loop_save_path, normalize_path, params_path, checkpoint_path, object_color, use_dynamic)
        if closed_loop_save_path is not None:
            closed_loop_render_set(dataset.model_path, "closed_loop", scene.loaded_iter, None, gaussians, pipeline, background, custom_camera_path, closed_loop_save_path, normalize_path, params_path, checkpoint_path)
        elif custom_camera_path is not None:
            if not skip_train:
                render_set(dataset.model_path, "custom_train", scene.loaded_iter, parse_custom_cameras(custom_camera_path), gaussians, pipeline, background, custom_camera_path)
        else:
            if not skip_train:
                render_set(dataset.model_path, "train", scene.loaded_iter, scene.getTrainCameras(), gaussians, pipeline, background, custom_camera_path)

            if not skip_test:
                render_set(dataset.model_path, "test", scene.loaded_iter, scene.getTestCameras(), gaussians, pipeline, background)

closed_loop_eval = True
if closed_loop_eval:
    normalize_path = '/home/makramchahine/repos/drone_multimodal/clean_train_d0_filtered_300/mean_std.csv'
    # normalize_path = '/home/makramchahine/repos/drone_multimodal/clean_train_d0_300/mean_std.csv'
    # runner_checkpoint_path = "/home/makramchahine/repos/drone_multimodal/runner_models/filtered_g1_full_smoothest_300_og_1200sf/val/model-ctrnn_wiredcfccell_seq-64_lr-0.000100_epoch-1115_val-loss:0.0002_train-loss:0.0006_mse:0.0006_2023:10:05:22:55:55.hdf5"
    # runner_checkpoint_path = "/home/makramchahine/repos/drone_multimodal/runner_models/filtered_d0_300_og_600sf/val/model-ctrnn_wiredcfccell_seq-64_lr-0.000100_epoch-598_val-loss:0.0084_train-loss:0.0121_mse:0.0121_2023:10:11:19:50:51.hdf5"
    # runner_checkpoint_path = "/home/makramchahine/repos/drone_multimodal/runner_models/filtered_d0_300_og_600sf/recurrent/model-ctrnn_wiredcfccell_seq-64_lr-0.000100_epoch-100_val-loss:0.0437_train-loss:0.0728_mse:0.0728_2023:10:11:19:50:51.hdf5"
    # runner_checkpoint_path = "/home/makramchahine/repos/drone_multimodal/runner_models/filtered_d0_300_og_1200sf/val/model-ctrnn_wiredcfccell_seq-64_lr-0.000100_epoch-1189_val-loss:0.0061_train-loss:0.0062_mse:0.0062_2023:10:11:19:54:39.hdf5"
    # runner_checkpoint_path = "/home/makramchahine/repos/drone_multimodal/runner_models/filtered_g1_xyz_300_og_1200sf/val/model-ctrnn_wiredcfccell_seq-64_lr-0.000100_epoch-1195_val-loss:0.0027_train-loss:0.0023_mse:0.0023_2023:10:11:23:12:35.hdf5"
    # runner_checkpoint_path = "/home/makramchahine/repos/drone_multimodal/runner_models/filtered_g1_xyz_300_og_1200sf/recurrent/model-ctrnn_wiredcfccell_seq-64_lr-0.000100_epoch-100_val-loss:0.0516_train-loss:0.0688_mse:0.0688_2023:10:11:23:12:35.hdf5"
    runner_checkpoint_path = "/home/makramchahine/repos/drone_multimodal/runner_models/filtered_d0_filtered_300_og_600sf/val/model-ctrnn_wiredcfccell_seq-64_lr-0.000100_epoch-587_val-loss:0.0093_train-loss:0.0117_mse:0.0117_2023:10:12:22:23:30.hdf5"
    base_runner_folder = os.path.dirname(runner_checkpoint_path)
    params_path = os.path.join(base_runner_folder, "params.json")
else:
    normalize_path = None
    base_runner_folder = None
    runner_checkpoint_path = None
    params_path = None


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
    parser.add_argument("--custom_camera_path", default=None, type=str)
    parser.add_argument("--closed_loop_save_path", default=None, type=str)
    parser.add_argument("--use_dynamic", action="store_true", default=True)
    args = get_combined_args(parser)
    print("Rendering " + args.model_path)

    if not "custom_camera_path" in args:
        args.custom_camera_path = None
    if not "closed_loop_save_path" in args:
        args.closed_loop_save_path = None

    rand_theta = random.uniform(0, 2 * np.pi)
    
    # Initialize system state (RNG)
    # safe_state(args.quiet)

    print(f"use_dynamic: {args.use_dynamic}")

    if args.custom_camera_path is not None:
        base_dir = os.path.dirname(args.custom_camera_path)
        with open(os.path.join(base_dir, "rand_theta.txt"), "w") as f:
            f.write(str(rand_theta))


    render_sets(model.extract(args), args.iteration, pipeline.extract(args), args.skip_train, args.skip_test, args.custom_camera_path, args.object_color, rotation_theta=rand_theta, closed_loop_save_path = args.closed_loop_save_path, normalize_path=normalize_path, params_path=params_path, checkpoint_path=runner_checkpoint_path, use_dynamic=args.use_dynamic)