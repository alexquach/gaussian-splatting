
import numpy as np
import random
from PIL import Image
import os
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(SCRIPT_DIR, ".."))

from drone_multimodal.utils.model_utils import load_model_from_weights, generate_hidden_list, get_readable_name, \
    get_params_from_json
from drone_multimodal.keras_models import IMAGE_SHAPE
from drone_multimodal.preprocess.process_data_util import resize_and_crop

def generate_init_conditions(object_colors, gs_offsets, PYBULLET_TO_GS_SCALING_FACTOR):
    """Shared between training and CL inference"""
    start_H = 0.1 + 0.5 #random.choice([0, 1])
    # start_H = 0.1 + random.uniform(0, 1)
    gs_offsets_xy = np.array(gs_offsets)[:, 0:2]
    gs_offsets_z = np.array(gs_offsets)[:, 2]
    Theta = random.uniform(0, 2 * np.pi) # doesn't match the GS rotation, but should be fine because its arbitrary
    Theta_offset = random.uniform(0.175 * np.pi, -0.175 * np.pi) #random.choice([0.175 * np.pi, -0.175 * np.pi])
    if object_colors[0] == "R":
        Theta_offset = -abs(Theta_offset)
    else:
        Theta_offset = abs(Theta_offset)

    print(f"gs_offsets_xy / PYBULLET_TO_GS_SCALING_FACTOR: {gs_offsets_xy / PYBULLET_TO_GS_SCALING_FACTOR}")
    return {
        "start_H": start_H,
        "target_Hs": 0.6 + gs_offsets_z / PYBULLET_TO_GS_SCALING_FACTOR,
        "Theta": Theta,
        "Theta_offset": Theta_offset,
        "rel_obj": gs_offsets_xy / PYBULLET_TO_GS_SCALING_FACTOR * np.array([1, -1]), # y is flipped in pybullet
    }

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