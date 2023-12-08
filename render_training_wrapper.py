import os
import subprocess
from tqdm import tqdm

from env_configs import ENV_CONFIGS

def run_script(m_path, s_path, custom_camera_paths, color):
    cmd = [
        "python",
        "render.py",
        "-m", m_path,
        "-s", s_path,
        "--object_color", color,
        "--custom_camera_paths",
    ] + custom_camera_paths
    subprocess.run(cmd)

# Paths and parameters
env_name = "holodeck"
m_path = ENV_CONFIGS[env_name]["m_path"]
s_path = ENV_CONFIGS[env_name]["s_path"]

base_dir = "/home/makramchahine/repos/gym-pybullet-drones/gym_pybullet_drones/examples/train_d6_ss2_400_3hzf_bm_px_td_nlsp_gn_nt"

red_folders = []
blue_folders = []

for folder in sorted(os.listdir(base_dir)):
    color_file = os.path.join(base_dir, folder, "colors.txt")
    if os.path.isfile(color_file):
        with open(color_file, "r") as f:
            color = f.readline().strip()
            if color == "R":
                red_folders.append(str(os.path.join(base_dir, folder, "path.json")))
            elif color == "B":
                blue_folders.append(str(os.path.join(base_dir, folder, "path.json")))

run_script(m_path, s_path, red_folders, "R")
run_script(m_path, s_path, blue_folders, "B")
