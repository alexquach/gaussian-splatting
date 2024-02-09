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
NORMALIZE_PATH = None
# NORMALIZE_PATH = '/home/makramchahine/repos/drone_multimodal/clean_train_d3_300/mean_std.csv'
SAMPLES_PER_MODEL = 20
RUN_VAL = True
USE_EPOCH_FILTER = True
EPOCH_FILTER = [] #[100, 300] #[100, 200, 300, 400, 500, 600, 700, 800, 900, 1000, 1100, 1200]

def run_GS_render(color, run_absolute_paths, params_paths, checkpoint_paths, record_hzs, rand_theta):
    print(f"record_hzs: {record_hzs}")
    
    cmd = [
        "python",
        "render.py",
        "-m", M_PATH,
        "-s", S_PATH,
        "--object_color", *color,
        # "--normalize_path", NORMALIZE_PATH,
        "--record_hzs", *record_hzs,
        "--params_paths", *params_paths,
        "--checkpoint_paths", *checkpoint_paths,
        "--closed_loop_save_paths", *run_absolute_paths,
        "--rand_theta", str(rand_theta),
    ]
    if USE_DYNAMIC:
        cmd.append("--use_dynamic")
    subprocess.run(cmd)


def main():
    parser = argparse.ArgumentParser(description='Run script with different configurations.')
    parser.add_argument('--tags', nargs='*', type=str, help='Tags for the runs')
    parser.add_argument('--record_hzs', nargs='*', type=int, help='Record HZs')
    args = parser.parse_args()

    # Use the arguments in your script
    global tags, record_hzs, MAIN_OUTPUT_FOLDER, MAIN_CHECKPOINT_FOLDER
    tags = args.tags
    RECORD_HZS = args.record_hzs
    MAIN_OUTPUT_FOLDERS = [f"/home/makramchahine/repos/gym-pybullet-drones/gym_pybullet_drones/examples/cl_realgs_{tag}_mleno_{record_hz}hz_05sf_100act_doub_emergency" for tag, record_hz in zip(tags, RECORD_HZS)]
    MAIN_CHECKPOINT_FOLDERS = [f"/home/makramchahine/repos/drone_multimodal/runner_models/filtered_{tag}" for tag in tags]

    evaluator = Evaluator(MAIN_OUTPUT_FOLDERS, MAIN_CHECKPOINT_FOLDERS, RECORD_HZS, multi=True)

    evaluator.config_eval_models(RUN_VAL)
    evaluator.build_concurrent_runs()

    evaluator.run_concurrent()
    evaluator.save_videos(dual_video=True)
    try:
        evaluator.calculate_metrics()
    except Exception as e:
        print(e)

class Evaluator():
    def __init__(self, main_output_folders, main_checkpoint_folders, record_hzs, multi=False):
        self.PRIMITIVE_OBJECTS = ["R", "B"]
        self.OBJECTS = []

        self.main_output_folders = main_output_folders
        self.main_checkpoint_folders = main_checkpoint_folders
        self.record_hzs = record_hzs

        self.eval_models = []                           # models to evaluate
        self.concurrent_params_paths = []
        self.concurrent_checkpoint_paths = []
        self.concurrent_output_folder_full_paths = []
        self.concurrent_record_hzs = []
        self.multi = multi

    def config_eval_models(self, RUN_VAL):
        """
        Uses filters to figure out which models to run from recurrent and val loss 
        """
        for main_output_folder, main_checkpoint_folder, record_hz in zip(self.main_output_folders, self.main_checkpoint_folders, self.record_hzs):            
            # ! Recurrent
            recurrent_folder = os.path.join(main_checkpoint_folder, 'recurrent')
            hdf5_files = glob.glob(os.path.join(recurrent_folder, '*.hdf5'))
            for hdf5_file_path in hdf5_files:
                epoch_num = int(re.findall(r'epoch-(\d+)', hdf5_file_path)[0]) # parse "epoch-%d" from hdf5 filename
                print(epoch_num)

                if USE_EPOCH_FILTER and epoch_num not in EPOCH_FILTER:
                    continue

                if os.path.exists(os.path.join(main_output_folder, f'recurrent{epoch_num}')):
                    print(f"skipping epoch {epoch_num}")
                    continue
                
                self.eval_models.append((main_checkpoint_folder, main_output_folder, hdf5_file_path, epoch_num, record_hz))
            
            # ! Val
            if RUN_VAL:
                recurrent_folder = os.path.join(main_checkpoint_folder, 'val')
                hdf5_files = glob.glob(os.path.join(recurrent_folder, '*.hdf5'))
                self.eval_models.append((main_checkpoint_folder, main_output_folder, hdf5_files[0], 'val', record_hz))

    def build_concurrent_runs(self):
        """
            Samples per model
                x [models to evaluate per common run parameters]
        """
        for i in range(SAMPLES_PER_MODEL):
            concurrent_checkpoint_path = []
            concurrent_params_path = []
            concurrent_output_folder_full_path = []
            concurrent_record_hz = []

            for main_checkpoint_folder, main_output_folder, hdf5_file_path, epoch_num, record_hz in self.eval_models:
                concurrent_checkpoint_path.append(hdf5_file_path)
                if epoch_num != 'val':
                    concurrent_params_path.append(os.path.join(main_checkpoint_folder, 'recurrent', f'params{epoch_num}.json'))
                    concurrent_output_folder_full_path.append(os.path.join(main_output_folder, f'recurrent{epoch_num}', f'run_{i}'))
                    concurrent_record_hz.append(str(record_hz))
                else:
                    concurrent_params_path.append(os.path.join(main_checkpoint_folder, 'val', 'params.json'))
                    concurrent_output_folder_full_path.append(os.path.join(main_output_folder, 'val', f'run_{i}'))
                    concurrent_record_hz.append(str(record_hz))

            if len(concurrent_checkpoint_path) > 0:
                self.concurrent_checkpoint_paths.append(concurrent_checkpoint_path)
                self.concurrent_params_paths.append(concurrent_params_path)
                self.concurrent_output_folder_full_paths.append(concurrent_output_folder_full_path)
                self.concurrent_record_hzs.append(concurrent_record_hz)

        self.append_initial_conditions()

    def append_initial_conditions(self):
        """
        Assign initial conditions := [object colors]
        """
        if self.multi:
            PERMUTATIONS_COLORS = [list(perm) for perm in itertools.product(self.PRIMITIVE_OBJECTS, repeat=2)]
            OBJECTS = []
            for _ in range(SAMPLES_PER_MODEL // len(PERMUTATIONS_COLORS)):
                OBJECTS.extend(PERMUTATIONS_COLORS)
            OBJECTS.extend(PERMUTATIONS_COLORS[:SAMPLES_PER_MODEL % len(PERMUTATIONS_COLORS)])
            random.shuffle(OBJECTS)

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
            LOCATIONS_REL = [[(random.uniform(0.75, 2.0), 0)] for _ in range(len(self.concurrent_output_folder_full_paths))]

        self.OBJECTS.extend(OBJECTS)

    def run_concurrent(self, n_jobs=1):
        directory = "/home/makramchahine/repos/gym-pybullet-drones/gym_pybullet_drones/examples/cl_realgs_settings"
        if not os.path.exists(directory):
            os.makedirs(directory)
        for color, run_absolute_path, params_path, checkpoint_path, record_hz in tqdm(zip(self.OBJECTS, self.concurrent_output_folder_full_paths, self.concurrent_params_paths, self.concurrent_checkpoint_paths, self.concurrent_record_hzs)):
            # parse run_i
            print(run_absolute_path)
            run_i = int(re.findall(r'run_(\d+)', run_absolute_path[0])[0])
            print(run_i)
            rand_theta = random.uniform(0, 2 * np.pi)

            # store color array and rand_theta in file
            if not os.path.exists(os.path.join(directory, str(run_i))):
                os.makedirs(os.path.join(directory, str(run_i)))
            
            if not os.path.exists(os.path.join(directory, str(run_i), "color.txt")):
                color_array = np.array(color)
                np.savetxt(os.path.join(directory, str(run_i), "color.txt"), color_array, fmt="%s")
                np.savetxt(os.path.join(directory, str(run_i), "rand_theta.txt"), np.array([rand_theta]), fmt="%s")
            else:
                color_array = np.genfromtxt(os.path.join(directory, str(run_i), "color.txt"), dtype=str)
                rand_theta = np.genfromtxt(os.path.join(directory, str(run_i), "rand_theta.txt"), dtype=str)

                color = list(color_array)

            # if run_i > 9:

            # filtered_run_absolute_path = []
            # filtered_params_path = []
            # filtered_checkpoint_path = []
            # filtered_record_hz = []
            # for i, path in enumerate(run_absolute_path):
            #     if not os.path.exists(os.path.join(path, "finish.txt")):
            #         filtered_run_absolute_path.append(path)
            #         filtered_params_path.append(params_path[i])
            #         filtered_checkpoint_path.append(checkpoint_path[i])
            #         filtered_record_hz.append(record_hz[i])
            # run_GS_render(color, filtered_run_absolute_path, filtered_params_path, filtered_checkpoint_path, filtered_record_hz, rand_theta)

            if os.path.exists(os.path.join(run_absolute_path[0], "finish.txt")):
                continue
            run_GS_render(color, run_absolute_path, params_path, checkpoint_path, record_hz, rand_theta)
        # joblib.Parallel(n_jobs=n_jobs)(joblib.delayed(run_GS_render)(color, run_absolute_path, params_path, checkpoint_path, record_hz) for color, run_absolute_path, params_path, checkpoint_path, record_hz in tqdm(zip(self.OBJECTS, self.concurrent_output_folder_full_paths, self.concurrent_params_paths, self.concurrent_checkpoint_paths, self.concurrent_record_hzs)))

    def save_videos(self, dual_video):
        for main_output_folder in self.main_output_folders:
            for different_checkpoint_model in os.listdir(main_output_folder):
                for image_folder_name in sorted(os.listdir(os.path.join(main_output_folder, different_checkpoint_model))):
                    if not os.path.isdir(os.path.join(main_output_folder, different_checkpoint_model, image_folder_name)):
                        continue
                    image_folder = os.path.join(main_output_folder, different_checkpoint_model, image_folder_name)
                    video_output = os.path.join(main_output_folder, different_checkpoint_model, image_folder_name, "video.mp4")
                    print(image_folder)
                    
                    if f"video.mp4" in os.listdir(os.path.join(main_output_folder, different_checkpoint_model, image_folder_name)):
                        # os.system(f"rm -rf {video_output}")
                        continue
                    images_to_video(image_folder, video_output, DUAL=dual_video)

                # Combine all videos in the directory
                combine_videos(os.path.join(main_output_folder, different_checkpoint_model))

    def calculate_metrics(self):
        for main_output_folder in self.main_output_folders:
            eval_map = {}
            first_ball_eval_map = {}
            for different_checkpoint_model in os.listdir(main_output_folder):
                if not os.path.isdir(os.path.join(main_output_folder, different_checkpoint_model)):
                    continue
                runs_history = []
                first_ball_runs_history = []
                for image_folder_name in sorted(os.listdir(os.path.join(main_output_folder, different_checkpoint_model))):
                    if not os.path.isdir(os.path.join(main_output_folder, different_checkpoint_model, image_folder_name)):
                        continue
                    result_txt_path = os.path.join(main_output_folder, different_checkpoint_model, image_folder_name, "finish.txt")

                    predictions_labels = np.genfromtxt(result_txt_path, delimiter=',', dtype=str, skip_header=0)
                    runs_history.append(np.mean([row[0] == row[1] for row in predictions_labels]))
                    first_ball_runs_history.append(predictions_labels[0][0] == predictions_labels[0][1])
                eval_map[different_checkpoint_model] = np.mean(runs_history)
                first_ball_eval_map[different_checkpoint_model] = np.mean(first_ball_runs_history)
                print(f"model accuracy: {eval_map[different_checkpoint_model]}")
                print(f"first ball accuracy: {first_ball_eval_map[different_checkpoint_model]}")

            print(eval_map)
            values_w_epochs = [(int(''.join(filter(str.isdigit, folder_tag))), value) for folder_tag, value in eval_map.items() if any(char.isdigit() for char in folder_tag)]
            values_wo_epochs = [(folder_tag, value) for folder_tag, value in eval_map.items() if not any(char.isdigit() for char in folder_tag)]
            first_ball_values_w_epochs = [(int(''.join(filter(str.isdigit, folder_tag))), value) for folder_tag, value in first_ball_eval_map.items() if any(char.isdigit() for char in folder_tag)]
            first_ball_values_wo_epochs = [(folder_tag, value) for folder_tag, value in first_ball_eval_map.items() if not any(char.isdigit() for char in folder_tag)]
            
            if values_w_epochs:
                epochs, values = zip(*sorted(values_w_epochs, key=lambda x: x[0]))
                first_ball_epochs, first_ball_values = zip(*sorted(first_ball_values_w_epochs, key=lambda x: x[0]))
                plt.scatter(epochs, values)
                plt.scatter(first_ball_epochs, first_ball_values, color='red')

            plt.scatter([0] * len(values_wo_epochs), [value for _, value in values_wo_epochs])
            plt.scatter([0] * len(first_ball_values_wo_epochs), [value for _, value in first_ball_values_wo_epochs], color='red')
            max_acc = max(eval_map.values())
            plt.ylabel(f"Accuracy (16 runs); max={max_acc:.03f}")
            plt.xlabel("Epochs")
            plt.ylim(0, 1.1)
            plt.title(f"{main_output_folder}")
            plt.savefig(f"{main_output_folder}/accuracy.png")

if __name__ == "__main__":
    main()