
import numpy as np
import random
from PIL import Image
import os
import sys
import json

from drone_causality.keras_models import IMAGE_SHAPE
from drone_causality.preprocess.process_data_util import resize_and_crop
from gym_pybullet_drones.examples.schemas import InitConditionsClosedLoopInferenceSchema


def generate_init_conditions_closed_loop_inference(objects_color, PYBULLET_TO_GS_SCALING_FACTOR, closed_loop_save_paths) -> InitConditionsClosedLoopInferenceSchema:
    """
    Specific implementation with weighted probabilities

    Task: Single Object -- Approach and Turn

    """
    pybullet_rand_forward = random.uniform(1.5, 2)
    gs_rand_forward = pybullet_rand_forward * PYBULLET_TO_GS_SCALING_FACTOR
    gs_offsets_from_camera = [[0, 0, 0], [gs_rand_forward, 0, 0], [gs_rand_forward * 0.9, -gs_rand_forward if objects_color[0] == "R" else gs_rand_forward, 0]] # forward, right, up

    gs_offsets_xy = np.array(gs_offsets_from_camera)[1:, 0:2]
    gs_offsets_z = np.array(gs_offsets_from_camera)[1:, 2]

    max_yaw_offset = 0.175 * np.pi
    start_heights = [0.1 + 0.5]
    target_heights = ((0.1 + 0.5 + gs_offsets_z) / PYBULLET_TO_GS_SCALING_FACTOR).tolist()
    theta_offset = random.uniform(0, max_yaw_offset)
    if objects_color[0] == "R":
        theta_offset = -abs(theta_offset)
    else:
        theta_offset = abs(theta_offset)
    theta_environment = random.random() * 2 * np.pi

    # NOTE: y is opposite in pybullet compared to GS coordinates, so we flip it here:
    objects_relative = gs_offsets_xy / PYBULLET_TO_GS_SCALING_FACTOR * np.array([1, -1])

    init_conditions_schema = InitConditionsClosedLoopInferenceSchema()
    init_conditions = {
        "task_name": "closed_loop_inference",
        "start_heights": start_heights,
        "target_heights": target_heights,
        "start_dist": pybullet_rand_forward,
        "theta_offset": theta_offset,
        "theta_environment": theta_environment,
        "objects_relative": objects_relative,
        "objects_color": objects_color,
        "PYBULLET_TO_GS_SCALING_FACTOR": PYBULLET_TO_GS_SCALING_FACTOR,
        "gs_objects_relative": np.array(gs_offsets_from_camera)[1:, 0:2].tolist()
    }
    init_conditions = init_conditions_schema.load(init_conditions)
    for path in closed_loop_save_paths:
        os.makedirs(path, exist_ok=True)
        init_conditions_path = os.path.join(path, "init_conditions.json")
        with open(init_conditions_path, "w") as f:
            json.dump(init_conditions, f)

    return init_conditions

def transform_gs_img_to_network_input(rendering):
    # calculate next camera position with model velocity updates
    # transform rendering to (h, w, 4)-shaped array of uint8's containing the RBG(A) image
    img = rendering.cpu().numpy().transpose(1, 2, 0) * 255 

    img = Image.fromarray(img.astype(np.uint8))
    img = resize_and_crop(img, (IMAGE_SHAPE[1], IMAGE_SHAPE[0]))
    img = np.array(img).astype(np.uint8)

    # Get velocity labels from network inference
    img = img[None,:,:,0:3]
    return img