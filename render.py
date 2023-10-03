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
from tqdm import tqdm
from os import makedirs
from gaussian_renderer import render
import torchvision
from utils.general_utils import safe_state
from utils.camera_utils import parse_custom_cameras
from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams, get_combined_args
from gaussian_renderer import GaussianModel
import random
import numpy as np

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

def render_sets(dataset : ModelParams, iteration : int, pipeline : PipelineParams, skip_train : bool, skip_test : bool, custom_camera_path : str, object_path : str, rotation_theta: float = 0.0):
    with torch.no_grad():
        gaussians = GaussianModel(dataset.sh_degree)
        scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False, object_path=object_path, rotation_theta=rotation_theta)

        bg_color = [1,1,1] if dataset.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

        train_views = scene.getTrainCameras() if custom_camera_path is None else parse_custom_cameras(custom_camera_path)
        train_name = "train" if custom_camera_path is None else "custom_train"

        if not skip_train:
             render_set(dataset.model_path, train_name, scene.loaded_iter, train_views, gaussians, pipeline, background, custom_camera_path)

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
    parser.add_argument("--object_path", default="something", type=str)
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--custom_camera_path", default=None, type=str)
    args = get_combined_args(parser)
    print("Rendering " + args.model_path)
    print(f"Object_path: {args.object_path}")

    if not "custom_camera_path" in args:
        args.custom_camera_path = None

    rand_theta = random.uniform(0, 2 * np.pi)
    
    # Initialize system state (RNG)
    safe_state(args.quiet)

    if args.custom_camera_path is not None:
        base_dir = os.path.dirname(args.custom_camera_path)
        with open(os.path.join(base_dir, "rand_theta.txt"), "w") as f:
            f.write(str(rand_theta))


    render_sets(model.extract(args), args.iteration, pipeline.extract(args), args.skip_train, args.skip_test, args.custom_camera_path, args.object_path, rotation_theta=rand_theta)