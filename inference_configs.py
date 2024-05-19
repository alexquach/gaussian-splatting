from enum import Enum
from omegaconf import OmegaConf
import os
from render_utils import generate_init_conditions_closed_loop_inference_2choice, generate_init_conditions_closed_loop_inference_3choice_random

class Task(Enum):
    RB_2CHOICE = "2rb"
    COLOR_5 = "2colors"
    R_SHAPE = "2rshape"
    M_SHAPE = "2mshape"
    OPEN_DICT = "3open_dict"

task = Task(os.environ.get("TASK"))
env_name = os.environ.get("ENV_NAME")

env_tag = "p" if env_name == "samurai" else "a" if env_name == "arena" else ""
model_type = "modelconv"
model_tag = "vit" if model_type == "modello" else "conv" if model_type == "modelconv" else "lin"
model_path_prefix = "/home/makramchahine/Desktop"
output_folder_format = "./generated_paths2/cl_blip{}_" + model_tag + env_tag + "_" + task.value + "_{}"



if model_type == "modello":
    cfg = OmegaConf.load(f"{model_path_prefix}/modello")
elif model_type == "modellinear":
    cfg = OmegaConf.load(f"{model_path_prefix}/modellinear")
elif model_type == "modelconv":
    cfg = OmegaConf.load(f"{model_path_prefix}/modelconv")
else:
    raise ValueError("Invalid model type")

if task == Task.RB_2CHOICE:
    objects = ["red ball", "blue ball"]
    targets = ["red ball", "blue ball"]
    num_obj = 2
    generate_init_conditions = generate_init_conditions_closed_loop_inference_2choice
elif task == Task.COLOR_5:
    objects = ["red ball", "blue ball", "yellow ball", "green ball", "purple ball"]
    targets = ["red ball", "blue ball", "yellow ball", "green ball", "purple ball"]
    num_obj = 2
    generate_init_conditions = generate_init_conditions_closed_loop_inference_2choice
elif task == Task.R_SHAPE:
    objects = ["red ball", "red cube", "red pyramid"]
    targets = ["ball", "cube", "pyramid"]
    num_obj = 2
    generate_init_conditions = generate_init_conditions_closed_loop_inference_2choice
elif task == Task.M_SHAPE:
    objects = ["red ball", "blue ball", "yellow ball", "green ball", "purple ball", "red cube", "blue cube", "yellow cube", "green cube", "purple cube", "red pyramid", "blue pyramid", "yellow pyramid", "green pyramid", "purple pyramid"]
    targets = ["ball", "cube", "pyramid"]
    num_obj = 2
    generate_init_conditions = generate_init_conditions_closed_loop_inference_2choice
elif task == Task.OPEN_DICT:
    objects = ["red ball", "blue ball", "jeep", "horse", "dog", "palmtree", "watermelon", "rocket"]
    targets = ["red ball", "blue ball", "jeep", "horse", "dog", "palmtree", "watermelon", "rocket"]
    num_obj = 3
    generate_init_conditions = generate_init_conditions_closed_loop_inference_3choice_random
else:
    raise ValueError("Invalid task")