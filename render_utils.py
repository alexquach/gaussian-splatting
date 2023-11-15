
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

COLOR_MAP = {
    "R": "/home/makramchahine/repos/gaussian-splatting/output/solid_red_ball",
    "B": "/home/makramchahine/repos/gaussian-splatting/output/solid_blue_ball",
}
PYBULLET_TO_GS_SCALING_FACTOR = 2.5
SECOND_BALL_SCALING_FACTOR = 2.7
GS_SIDEWAYS_OFFSET_RAND_VALUES = (2.5, 3)

def generate_init_conditions(object_color, pybullet_sideways_offset=0):
    """Shared between training and CL inference"""
    start_H = 0.1 + random.choice([0, 1])
    # start_H = 0.1 + random.uniform(0, 1)
    target_Hs = [0.1 + 0.5]
    Theta = 0 #random.random() * 2 * np.pi
    Theta_offset = random.uniform(0.175 * np.pi, -0.175 * np.pi) #random.choice([0.175 * np.pi, -0.175 * np.pi])
    if object_color == "R":
        Theta_offset = -abs(Theta_offset)
    else:
        Theta_offset = abs(Theta_offset)
    rel_obj = [(random.uniform(1, 2), 0)]

    pybullet_sideways_offset = pybullet_sideways_offset if object_color == "R" else -pybullet_sideways_offset
    rel_obj.append((rel_obj[0][0] - 0.2 / SECOND_BALL_SCALING_FACTOR, pybullet_sideways_offset))

    return {
        "start_H": start_H,
        "target_Hs": target_Hs,
        "Theta": Theta,
        "Theta_offset": Theta_offset,
        "rel_obj": rel_obj
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