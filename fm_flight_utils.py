from typing import List, Optional, Tuple

import hydra
import lightning as L
import pyrootutils
from lightning import Callback, LightningDataModule, LightningModule, Trainer
from lightning.pytorch.loggers import Logger
from omegaconf import DictConfig, OmegaConf
from hydra.core.hydra_config import HydraConfig
import torch

# make a dict conf called CFG
cfg = OmegaConf.load("/home/makramchahine/Desktop/modello")
# cfg = OmegaConf.load("/home/makramchahine/Desktop/modellinear")

model: LightningModule = hydra.utils.instantiate(cfg.model)
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device}")

if cfg.ckpt_path:
    # if its a list load the last one
    cpath = cfg.ckpt_path[0]
    print(f"Loading checkpoint: {cpath}")
    ckpt = torch.load(cpath, map_location=device)
    for dropped_key in ["net.extractor._clip_param", "net.extractor._model_param", "net.extractor._dino_param"]:
        if dropped_key in ckpt["state_dict"].keys():
            ckpt["state_dict"].pop(dropped_key) # HACK: remove param used for determining device
    model.load_state_dict(ckpt["state_dict"])

# make a copy of the model
modello = model

# with open("model_details.txt", "w") as file:
#     file.write(str(modello))
# Print model parameter count by layers
# for name, param in model.named_parameters():
#     if param.requires_grad:
#         print(f"{name}: {param.numel()}")


print("\n\n checkpoint loaded successfully \n\n")

from PIL import Image
from torchvision import transforms

image_path = "/home/makramchahine/repos/fm_flight/BLIP2_DATASET/eval/save-flight-02.13.2024_21.41.47.286898/000025a.png"

# load the image
img = Image.open(image_path)
# make the image have 3 channels
img = img.convert('RGB')

# resize the image to 224x224
img = img.resize((224, 224))

# convert the image to a tensor
img = transforms.ToTensor()(img).to(device)

text = "fly to red object"
# run inference
# preds = model.forward({"image": img, "text": text})
# print(preds)

# text = "fly to red target"
# # run inference
# preds = model.forward({"image": img, "text": text})
# print(preds)

# text = "fly to blue object"
# # run inference
# preds = model.forward({"image": img, "text": text})
# print(preds)