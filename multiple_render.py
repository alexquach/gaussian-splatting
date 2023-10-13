import os
import subprocess
from tqdm import tqdm

COLOR_MAP = {
    "R": "/home/makramchahine/repos/gaussian-splatting/output/solid_red_ball",
    "B": "/home/makramchahine/repos/gaussian-splatting/output/solid_blue_ball",
}

def run_script(m_path, s_path, custom_camera_path, color):
    cmd = [
        "python",
        "render.py",
        "-m", m_path,
        "-s", s_path,
        "--custom_camera_path", custom_camera_path,
        "--object_color", color,
    ]
    subprocess.run(cmd)

# Paths and parameters
m_path = "/home/makramchahine/repos/gaussian-splatting/output/holodeck2"
s_path = "/home/makramchahine/repos/nerf/data/nerfstudio/custom/holodeck2/keyframes"

# Run the script multiple times
base_dir = "/home/makramchahine/repos/gym-pybullet-drones/gym_pybullet_drones/examples/train_d0"
for folder in tqdm(sorted(os.listdir(base_dir))):
    if not os.path.isdir(os.path.join(base_dir, folder)):
        continue
    
    camera = os.path.join(base_dir, folder, "path.json")
    with open(os.path.join(base_dir, folder, "colors.txt"), "r") as f:
        color = f.readline().strip()
    run_script(m_path, s_path, camera, color)


