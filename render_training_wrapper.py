import os
import subprocess
from tqdm import tqdm

COLOR_MAP = {
    "R": "/home/makramchahine/repos/gaussian-splatting/output/solid_red_ball",
    "B": "/home/makramchahine/repos/gaussian-splatting/output/solid_blue_ball",
    "G": "/home/makramchahine/repos/gaussian-splatting/output/solid_blue_ball",
}

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
m_path = "/home/makramchahine/repos/gaussian-splatting/output/holodeck2"
s_path = "/home/makramchahine/repos/nerf/data/nerfstudio/custom/holodeck2/keyframes"

# Run the script multiple times
base_dir = "/home/makramchahine/repos/gym-pybullet-drones/gym_pybullet_drones/examples/train_d6_ss2_16_1_20hzf_td"

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

# for folder in tqdm(sorted(os.listdir(base_dir))):
#     if not os.path.isdir(os.path.join(base_dir, folder)):
#         continue
    
#     camera = os.path.join(base_dir, folder, "path.json")
#     with open(os.path.join(base_dir, folder, "colors.txt"), "r") as f:
#         color = f.readline().strip()
    
#     pics_dir = os.path.join(base_dir, folder, "pics0")
#     if not os.listdir(pics_dir):
#         run_script(m_path, s_path, camera, color)


