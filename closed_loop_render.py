import os
import subprocess
import random
from tqdm import tqdm

COLOR_MAP = {
    "R": "/home/makramchahine/repos/gaussian-splatting/output/solid_red_ball",
    "B": "/home/makramchahine/repos/gaussian-splatting/output/solid_blue_ball",
}

def run_script(m_path, s_path, color, folder_name):
    cmd = [
        "python",
        "render.py",
        "-m", m_path,
        "-s", s_path,
        # "--custom_camera_path", custom_camera_path,
        "--object_color", color,
        # "--closed_loop_save_path", f"/home/makramchahine/repos/gaussian-splatting/camera_assets/cl_g1_full_smoothest_300_og_600sf/{folder_name}"
        # "--closed_loop_save_path", f"/home/makramchahine/repos/gym-pybullet-drones/gym_pybullet_drones/examples/cl_d0_300_og_600sf_rot_last_fixcap_rec100_stablize/{folder_name}"
        "--closed_loop_save_path", f"/home/makramchahine/repos/gym-pybullet-drones/gym_pybullet_drones/examples/cl_d0_filtered_300_og_600sf/{folder_name}"
        # "--use_dynamic", "True"
    ]
    subprocess.run(cmd)


# Paths and parameters
m_path = "/home/makramchahine/repos/gaussian-splatting/output/holodeck2"
s_path = "/home/makramchahine/repos/nerf/data/nerfstudio/custom/holodeck2/keyframes"

NUM_SAMPLES = 10

colors = ["R", "B"] * (NUM_SAMPLES // 2)
random.shuffle(colors)
for i in tqdm(range(NUM_SAMPLES)):
    run_script(m_path, s_path, colors[i], f"run_{i}")


