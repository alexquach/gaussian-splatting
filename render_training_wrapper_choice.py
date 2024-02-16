import os
import subprocess
from tqdm import tqdm

from env_configs import ENV_CONFIGS

# Paths and parameters
env_name = "holodeck"
m_path = ENV_CONFIGS[env_name]["m_path"]
s_path = ENV_CONFIGS[env_name]["s_path"]

def run_script(m_path, s_path, custom_camera_paths, colors, correct_side, start_dist, rand_theta):
    cmd = [
        "python",
        "render_choice.py",
        "-m", m_path,
        "-s", s_path,
        "--correct_side", correct_side,
        "--start_dist", str(start_dist),
        "--rand_theta", str(rand_theta),
        "--object_colors", *colors,
        "--custom_camera_paths",
    ] + custom_camera_paths
    subprocess.run(cmd)

# base_dir = "/home/makramchahine/repos/gym-pybullet-drones/gym_pybullet_drones/examples/train_d6_ss2_400_3hzf_bm_px_td_nlsp_gn_nt"
# base_dir = "/home/makramchahine/repos/gaussian-splatting/train_d6_ss2_10_3hzf_bm_px_td_nlsp_gn_nt_testing"
base_dir = "/home/makramchahine/repos/gaussian-splatting/train_blip_10"

for folder in sorted(os.listdir(base_dir)):
    if not os.path.isdir(os.path.join(base_dir, folder)):
        continue

    camera_path = []
    color_file = os.path.join(base_dir, folder, "colors.txt")
    if os.path.isfile(color_file):
        with open(color_file, "r") as f:
            color = f.readline().strip()
            camera_path.append(str(os.path.join(base_dir, folder, "path.json")))

    correct_file = os.path.join(base_dir, folder, "correct.txt")
    if os.path.isfile(correct_file):
        with open(correct_file, "r") as f:
            correct_side = f.readline().strip()

    with open(os.path.join(base_dir, folder, "start_dist.txt"), "r") as f:
        start_dist = float(f.readline().strip())

    with open(os.path.join(base_dir, folder, "rand_theta.txt"), "r") as f:
        rand_theta = float(f.readline().strip())

    if color == "R":
        colors = ["R", "B"]
    elif color == "B":
        colors = ["B", "R"]
    run_script(m_path, s_path, camera_path, colors, correct_side, start_dist, rand_theta)

