import os
import subprocess
from tqdm import tqdm
import argparse

from env_configs import ENV_CONFIGS

def run_script(m_path, s_path, custom_camera_paths, full_path, env_name):
    cmd = [
        "python",
        "render_choice.py", # TODO fix
        "-m", m_path,
        "-s", s_path,
        "--full_path", full_path,
        "--env_name", env_name,
        "--custom_camera_paths",
    ] + custom_camera_paths
    subprocess.run(cmd)


def run_script_for_all_subfolders(base_dir, m_path, s_path, env_name):
    for folder in sorted(os.listdir(base_dir)):
        if not os.path.isdir(os.path.join(base_dir, folder)):
            continue

        full_path = os.path.join(base_dir, folder)
        camera_path = [str(os.path.join(base_dir, folder, "path.json"))]

        run_script(m_path, s_path, camera_path, full_path, env_name)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Provide base directory.')
    parser.add_argument('--base_dir', type=str, default="./generated_paths/train_fly_and_turn", help='Base directory for the script')
    parser.add_argument('--env_name', type=str, default="holodeck", help='Environment name')
    args = parser.parse_args()

    base_dir = args.base_dir
    env_name = args.env_name
    
    # Paths and parameters
    m_path = ENV_CONFIGS[env_name]["m_path"]
    s_path = ENV_CONFIGS[env_name]["s_path"]

    run_script_for_all_subfolders(base_dir, m_path, s_path, env_name)

