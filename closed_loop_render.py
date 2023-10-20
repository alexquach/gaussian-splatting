import os
import subprocess
import random
from tqdm import tqdm
import glob
import re
import itertools
import math
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import joblib
import subprocess

from render_folder import images_to_video, combine_videos

COLOR_MAP = {
    "R": "/home/makramchahine/repos/gaussian-splatting/output/solid_red_ball",
    "B": "/home/makramchahine/repos/gaussian-splatting/output/solid_blue_ball",
}

# Paths and parameters
M_PATH = "/home/makramchahine/repos/gaussian-splatting/output/holodeck2"
S_PATH = "/home/makramchahine/repos/nerf/data/nerfstudio/custom/holodeck2/keyframes"

DEFAULT_DURATION_SEC = 15
combined_video_filename = "combined_video.mp4"
video_filename = "rand.mp4"

# ! Adjustable Params
USE_DYNAMIC = True
        # "--closed_loop_save_path", f"/home/makramchahine/repos/gaussian-splatting/camera_assets/cl_g1_full_smoothest_300_og_600sf/{folder_name}"
        # "--closed_loop_save_path", f"/home/makramchahine/repos/gym-pybullet-drones/gym_pybullet_drones/examples/cl_d0_300_og_600sf_rot_last_fixcap_rec100_stablize/{folder_name}"
        # "--closed_loop_save_path", f"/home/makramchahine/repos/gym-pybullet-drones/gym_pybullet_drones/examples/cl_h0f_hr_pybullet_300_og_600sf_rec100_debug_fixh/{folder_name}"
MAIN_OUTPUT_FOLDER = "/home/makramchahine/repos/gym-pybullet-drones/gym_pybullet_drones/examples/cl_d2_300_ogf_300sf"
MAIN_CHECKPOINT_FOLDER = "/home/makramchahine/repos/drone_multimodal/runner_models/filtered_d2_300_ogf_300sf"
NORMALIZE_PATH = '/home/makramchahine/repos/drone_multimodal/clean_train_d0_300/mean_std.csv'
SAMPLES_PER_MODEL = 10
RUN_VAL = True
USE_EPOCH_FILTER = True
EPOCH_FILTER = [100, 200, 300, 400, 500, 600]#, 700, 800, 900, 1000, 1100, 1200]

def run_GS_render(color, run_absolute_path, params_path, checkpoint_path):
    cmd = [
        "python",
        "render.py",
        "-m", M_PATH,
        "-s", S_PATH,
        "--object_color", color,
        "--normalize_path", NORMALIZE_PATH,
        "--params_path", params_path,
        "--checkpoint_path", checkpoint_path,
        "--closed_loop_save_path", run_absolute_path
    ]
    if USE_DYNAMIC:
        cmd.append("--use_dynamic")
    subprocess.run(cmd)


def main():
    os.makedirs(MAIN_OUTPUT_FOLDER, exist_ok=True)
    evaluator = Evaluator(MAIN_OUTPUT_FOLDER, MAIN_CHECKPOINT_FOLDER, multi=False)

    if RUN_VAL:
        evaluator.config_val_eval()
    evaluator.config_recurrent_eval()

    evaluator.run_concurrent()
    evaluator.save_videos(dual_video=True)

class Evaluator():
    def __init__(self, main_output_folder, main_checkpoint_folder, multi=False):
        self.PRIMITIVE_OBJECTS = ["R", "B"]
        self.OBJECTS = []
        # OBJECTS = ["R", "R", "G", "B", "B"] * NUM_INITIALIZATIONS
        self.main_output_folder = main_output_folder
        self.main_checkpoint_folder = main_checkpoint_folder
        self.concurrent_params_paths = []
        self.concurrent_checkpoint_paths = []
        self.output_folder_full_paths = []
        self.multi = multi

    def config_val_eval(self):
        val_runner_folder = os.path.join(self.main_checkpoint_folder, 'val')
        hdf5_files = glob.glob(os.path.join(val_runner_folder, '*.hdf5'))
        json_files = glob.glob(os.path.join(val_runner_folder, '*.json'))
        if hdf5_files:
            hdf5_file_path = hdf5_files[0]  # get the first .hdf5 file
        else:
            print("No .hdf5 files found in the directory.")
        if json_files:
            json_file_path = json_files[0]  # get the first .json file
        else:
            print("No .json files found in the directory.")

        if not self.multi and hdf5_file_path and json_file_path and not os.path.exists(os.path.join(self.main_output_folder, 'val')):
            for i in range(SAMPLES_PER_MODEL):
                self.concurrent_checkpoint_paths.append(hdf5_file_path)
                self.concurrent_params_paths.append(json_file_path)
                self.output_folder_full_paths.append(os.path.join(self.main_output_folder, 'val', f'run_{i}'))

        self.append_initial_conditions()

    def config_recurrent_eval(self):
        recurrent_folder = os.path.join(self.main_checkpoint_folder, 'recurrent')
        hdf5_files = glob.glob(os.path.join(recurrent_folder, '*.hdf5'))

        for hdf5_file_path in hdf5_files:
            epoch_num = int(re.findall(r'epoch-(\d+)', hdf5_file_path)[0]) # parse "epoch-%d" from hdf5 filename
            print(epoch_num)

            if USE_EPOCH_FILTER and epoch_num not in EPOCH_FILTER:
                continue

            if os.path.exists(os.path.join(self.main_output_folder, f'recurrent{epoch_num}')):
                print(f"skipping epoch {epoch_num}")
                continue
            for i in range(SAMPLES_PER_MODEL):
                if os.path.exists(os.path.join(self.main_checkpoint_folder, 'recurrent', f'params{epoch_num}.json')):
                    self.concurrent_checkpoint_paths.append(hdf5_file_path)
                    self.concurrent_params_paths.append(os.path.join(self.main_checkpoint_folder, 'recurrent', f'params{epoch_num}.json'))
                    self.output_folder_full_paths.append(os.path.join(self.main_output_folder, f'recurrent{epoch_num}', f'run_{i}'))

            self.append_initial_conditions()

    def append_initial_conditions(self):
        """
        Assign initial conditions := [object colors]
        """
        if self.multi:
            PERMUTATIONS_COLORS = [list(perm) for perm in itertools.combinations_with_replacement(self.OBJECTS, 3)]
            OBJECTS = [random.sample(PERMUTATIONS_COLORS, 1)[0] for _ in range(SAMPLES_PER_MODEL)]
            LOCATIONS_REL = []
            for targets in OBJECTS:
                locations = []
                cur_point = (0, 0) #random.uniform(0.75, 1.5)
                cur_direction = 0 
                for target in targets:
                    cur_dist = random.uniform(1, 1.75) - 0.2
                    target_loc = (cur_point[0] + (cur_dist + 0.2) * math.cos(cur_direction), cur_point[1] + (cur_dist + 0.2) * math.sin(cur_direction))
                    cur_point = (cur_point[0] + cur_dist * math.cos(cur_direction), cur_point[1] + cur_dist * math.sin(cur_direction))
                    locations.append(target_loc)

                    if target[0] == 'R':
                        cur_direction += math.pi / 2
                    elif target[0] == 'G':
                        cur_direction += 0
                    elif target[0] == 'B':
                        cur_direction += -math.pi / 2
                LOCATIONS_REL.append(locations)
        else:
            OBJECTS = self.PRIMITIVE_OBJECTS * (SAMPLES_PER_MODEL // len(self.PRIMITIVE_OBJECTS))#[[random.choice(OBJECTS)] for _ in range(len(output_folder_paths))]
            # OBJECTS = [[x] for x in OBJECTS]
            LOCATIONS_REL = [[(random.uniform(0.75, 2.0), 0)] for _ in range(len(self.output_folder_full_paths))]

        self.OBJECTS.extend(OBJECTS)

    def run_concurrent(self, n_jobs=1):
        print(self.OBJECTS)
        joblib.Parallel(n_jobs=n_jobs)(joblib.delayed(run_GS_render)(color, run_absolute_path, params_path, checkpoint_path) for color, run_absolute_path, params_path, checkpoint_path in tqdm(zip(self.OBJECTS, self.output_folder_full_paths, self.concurrent_params_paths, self.concurrent_checkpoint_paths)))

    def save_videos(self, dual_video):
        for different_checkpoint_model in os.listdir(self.main_output_folder):
            for image_folder_name in sorted(os.listdir(os.path.join(self.main_output_folder, different_checkpoint_model))):
                if not os.path.isdir(os.path.join(self.main_output_folder, different_checkpoint_model, image_folder_name)):
                    continue
                image_folder = os.path.join(self.main_output_folder, different_checkpoint_model, image_folder_name)
                video_output = os.path.join(self.main_output_folder, different_checkpoint_model, image_folder_name, "video.mp4")
                print(image_folder)
                
                if f"video.mp4" in os.listdir(os.path.join(self.main_output_folder, different_checkpoint_model, image_folder_name)):
                    # os.system(f"rm -rf {video_output}")
                    continue
                images_to_video(image_folder, video_output, DUAL=dual_video)

            # Combine all videos in the directory
            combine_videos(os.path.join(self.main_output_folder, different_checkpoint_model))

if __name__ == "__main__":
    main()