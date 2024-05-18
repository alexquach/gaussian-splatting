from enum import Enum
from omegaconf import OmegaConf

class Task(Enum):
    RB_2CHOICE = "2rb"
    COLOR_5 = "2colors"
    R_SHAPE = "2rshape"
    M_SHAPE = "2mshape"
    OPEN_DICT = "3open_dict"



task = Task.RB_2CHOICE
model_type = "modello"
model_tag = "vit" if model_type == "modello" else "lin"
model_path_prefix = "/home/makramchahine/Desktop"
output_folder_format = "./generated_paths2/cl_blip{}_" + model_tag + "p_" + task.value + "_{}"



if model_type == "modello":
    cfg = OmegaConf.load(f"{model_path_prefix}/modello")
elif model_type == "modellinear":
    cfg = OmegaConf.load(f"{model_path_prefix}/modellinear")
else:
    raise ValueError("Invalid model type")

if task == Task.RB_2CHOICE:
    objects = ["red ball", "blue ball"]
    targets = ["red ball", "blue ball"]
    num_obj = 2
elif task == Task.COLOR_5:
    objects = ["red ball", "blue ball", "yellow ball", "green ball", "purple ball"]
    targets = ["red ball", "blue ball", "yellow ball", "green ball", "purple ball"]
    num_obj = 2
elif task == Task.R_SHAPE:
    objects = ["red ball", "red cube"]
    targets = ["ball", "cube"]
    num_obj = 2
elif task == Task.M_SHAPE:
    objects = ["red ball", "blue ball", "yellow ball", "green ball", "purple ball", "red cube", "blue cube", "yellow cube", "green cube", "purple cube"]
    targets = ["ball", "cube"]
    num_obj = 2
elif task == Task.OPEN_DICT:
    objects = ["red ball", "blue ball", "jeep", "horse", "dog", "palmtree", "watermelon", "rocket"]
    targets = ["red ball", "blue ball", "jeep", "horse", "dog", "palmtree", "watermelon", "rocket"]
    num_obj = 3
else:
    raise ValueError("Invalid task")