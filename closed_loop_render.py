import os
import subprocess
import random
from tqdm import tqdm
import glob
import re
import itertools
import math
import numpy as np
import joblib
import subprocess
import matplotlib.pyplot as plt
import argparse
plt.rcParams.update({'figure.facecolor':'white'})

from render_folder import images_to_video, combine_videos

from env_configs import ENV_CONFIGS

# Paths and parameters
env_name = "holodeck"
M_PATH = ENV_CONFIGS[env_name]["m_path"]
S_PATH = ENV_CONFIGS[env_name]["s_path"]

combined_video_filename = "combined_video.mp4"
video_filename = "rand.mp4"

# ! Adjustable Params
USE_DYNAMIC = True
# tag = "d0_pybullet_300_ogf_600sf"
# tag = "d6_nonorm_ss2_600_1_10hzf_bm_td_srf_300sf_irreg2_64"
# tag = "d6_nonorm_ss2_600_1_10hzf_bm_pfff_td_srf_300sf_irreg2_64_hyp"
# tag = "d6_nonorm_ss2_200_9hzf_bm_px_td_nlsp_gn_nt_srf_150sf_irreg2_64_hyp_cfc"
# # tag = "d6_nonorm_ss2_600_3hzf_bm_px_td_nlsp_srf_300sf_irreg2_64"
# RECORD_HZ = 9
# # MAIN_OUTPUT_FOLDER = f"/home/makramchahine/repos/gym-pybullet-drones/gym_pybullet_drones/examples/cl_{tag}_mleno_{RECORD_HZ}hz_hypn"
# MAIN_OUTPUT_FOLDER = f"/home/makramchahine/repos/gym-pybullet-drones/gym_pybullet_drones/examples/cl_{tag}_mleno_{RECORD_HZ}hz_05sf"
# MAIN_CHECKPOINT_FOLDER = f"/home/makramchahine/repos/drone_multimodal/runner_models/filtered_{tag}"
NORMALIZE_PATH = '/home/makramchahine/repos/drone_multimodal/clean_train_d3_300/mean_std.csv'
SAMPLES_PER_MODEL = 10
RUN_VAL = True
USE_EPOCH_FILTER = True
EPOCH_FILTER = [] #[100, 200, 300, 400, 500, 600, 700, 800, 900, 1000, 1100, 1200]

def run_GS_render(color, run_absolute_paths, params_paths, checkpoint_paths):
    cmd = [
        "python",
        "render.py",
        "-m", M_PATH,
        "-s", S_PATH,
        "--record_hz", str(RECORD_HZ),
        "--object_color", color,
        "--normalize_path", NORMALIZE_PATH,
        "--params_paths", *params_paths,
        "--checkpoint_paths", *checkpoint_paths,
        "--closed_loop_save_paths", *run_absolute_paths
    ]
    if USE_DYNAMIC:
        cmd.append("--use_dynamic")
    subprocess.run(cmd)


def main():
    parser = argparse.ArgumentParser(description='Run script with different configurations.')
    parser.add_argument('--tag', type=str, help='Tag for the run')
    parser.add_argument('--record_hz', type=int, help='Record HZ')
    args = parser.parse_args()

    # Use the arguments in your script
    global tag, RECORD_HZ, MAIN_OUTPUT_FOLDER, MAIN_CHECKPOINT_FOLDER
    tag = args.tag
    RECORD_HZ = args.record_hz
    MAIN_OUTPUT_FOLDER = f"/home/makramchahine/repos/gym-pybullet-drones/gym_pybullet_drones/examples/cl_{tag}_mleno_{RECORD_HZ}hz_05sf_debugstraight"
    MAIN_CHECKPOINT_FOLDER = f"/home/makramchahine/repos/drone_multimodal/runner_models/filtered_{tag}"

    # os.makedirs(MAIN_OUTPUT_FOLDER, exist_ok=True)
    evaluator = Evaluator(MAIN_OUTPUT_FOLDER, MAIN_CHECKPOINT_FOLDER, multi=False)

    evaluator.config_eval_models(RUN_VAL)

    evaluator.run_concurrent()
    evaluator.save_videos(dual_video=True)
    evaluator.calculate_metrics()

class Evaluator():
    def __init__(self, main_output_folder, main_checkpoint_folder, multi=False):
        self.PRIMITIVE_OBJECTS = ["R", "B"]
        self.OBJECTS = []
        # OBJECTS = ["R", "R", "G", "B", "B"] * NUM_INITIALIZATIONS
        self.main_output_folder = main_output_folder
        self.main_checkpoint_folder = main_checkpoint_folder
        self.concurrent_params_paths = []
        self.concurrent_checkpoint_paths = []
        self.concurrent_output_folder_full_paths = []
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
                self.concurrent_output_folder_full_paths.append(os.path.join(self.main_output_folder, 'val', f'run_{i}'))

        self.append_initial_conditions()

    def config_eval_models(self, RUN_VAL):
        eval_models = []
        
        # ! Recurrent
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
            
            eval_models.append((hdf5_file_path, epoch_num))
        
        # ! Val
        if RUN_VAL:
            recurrent_folder = os.path.join(self.main_checkpoint_folder, 'val')
            hdf5_files = glob.glob(os.path.join(recurrent_folder, '*.hdf5'))
            eval_models.append((hdf5_files[0], 'val'))

        # ! Build concurrent runs
        for i in range(SAMPLES_PER_MODEL):
            concurrent_checkpoint_path = []
            concurrent_params_path = []
            concurrent_output_folder_full_path = []
            for hdf5_file_path, epoch_num in eval_models:
                concurrent_checkpoint_path.append(hdf5_file_path)
                if epoch_num != 'val':
                    concurrent_params_path.append(os.path.join(self.main_checkpoint_folder, 'recurrent', f'params{epoch_num}.json'))
                    concurrent_output_folder_full_path.append(os.path.join(self.main_output_folder, f'recurrent{epoch_num}', f'run_{i}'))
                else:
                    concurrent_params_path.append(os.path.join(self.main_checkpoint_folder, 'val', 'params.json'))
                    concurrent_output_folder_full_path.append(os.path.join(self.main_output_folder, 'val', f'run_{i}'))

            if len(concurrent_checkpoint_path) > 0:
                self.concurrent_checkpoint_paths.append(concurrent_checkpoint_path)
                self.concurrent_params_paths.append(concurrent_params_path)
                self.concurrent_output_folder_full_paths.append(concurrent_output_folder_full_path)

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
            LOCATIONS_REL = [[(random.uniform(0.75, 2.0), 0)] for _ in range(len(self.concurrent_output_folder_full_paths))]

        self.OBJECTS.extend(OBJECTS)

    def run_concurrent(self, n_jobs=1):
        print(self.OBJECTS)
        joblib.Parallel(n_jobs=n_jobs)(joblib.delayed(run_GS_render)(color, run_absolute_path, params_path, checkpoint_path) for color, run_absolute_path, params_path, checkpoint_path in tqdm(zip(self.OBJECTS, self.concurrent_output_folder_full_paths, self.concurrent_params_paths, self.concurrent_checkpoint_paths)))

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

    def calculate_metrics(self):
        eval_map = {}
        for different_checkpoint_model in os.listdir(self.main_output_folder):
            if not os.path.isdir(os.path.join(self.main_output_folder, different_checkpoint_model)):
                continue
            runs_history = []
            for image_folder_name in sorted(os.listdir(os.path.join(self.main_output_folder, different_checkpoint_model))):
                if not os.path.isdir(os.path.join(self.main_output_folder, different_checkpoint_model, image_folder_name)):
                    continue
                result_txt_path = os.path.join(self.main_output_folder, different_checkpoint_model, image_folder_name, "finish.txt")

                predictions_labels = np.genfromtxt(result_txt_path, delimiter=',', dtype=str, skip_header=0)
                runs_history.append(np.mean([row[0] == row[1] for row in predictions_labels]))
            eval_map[different_checkpoint_model] = np.mean(runs_history)
            print(f"model accuracy: {eval_map[different_checkpoint_model]}")

        print(eval_map)
        values_w_epochs = [(int(''.join(filter(str.isdigit, folder_tag))), value) for folder_tag, value in eval_map.items() if any(char.isdigit() for char in folder_tag)]
        values_wo_epochs = [(folder_tag, value) for folder_tag, value in eval_map.items() if not any(char.isdigit() for char in folder_tag)]
        epochs, values = zip(*sorted(values_w_epochs, key=lambda x: x[0]))
        plt.scatter(epochs, values)

        plt.scatter([0] * len(values_wo_epochs), [value for _, value in values_wo_epochs])
        max_acc = max(eval_map.values())
        plt.ylabel(f"Accuracy (16 runs); max={max_acc:.03f}")
        plt.xlabel("Epochs")
        plt.ylim(0, 1.1)
        plt.title(f"{self.main_output_folder}")
        plt.savefig(f"{self.main_output_folder}/accuracy.png")

if __name__ == "__main__":
    main()