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

from video_utils.render_folder import save_multi_layer_videos
from env_configs import ENV_CONFIGS

combined_video_filename = "combined_video.mp4"
video_filename = "rand.mp4"

# ! Adjustable Params
RUN_VAL = True
USE_EPOCH_FILTER = True
# which epoch checkpoints to run
EPOCH_FILTER = []

main_output_folder_format = "./generated_paths/cl_realgs_{}_mleno_{}hz_05sf_100act_doub"
main_checkpoint_folder_format = "./drone_causality/runner_models/filtered_{}"

def run_GS_render(env_name, colors, record_hzs, run_absolute_paths, params_paths, checkpoint_paths):
    M_PATH = ENV_CONFIGS[env_name]["m_path"]
    S_PATH = ENV_CONFIGS[env_name]["s_path"]
    
    cmd = [
        "python",
        "render_choice.py",
        "-m", M_PATH,
        "-s", S_PATH,
        "--is_closed_loop",
        "--objects_color", *colors,
        "--record_hzs", *record_hzs,
        "--params_paths", *params_paths,
        "--checkpoint_paths", *checkpoint_paths,
        "--closed_loop_save_paths", *run_absolute_paths,
    ]
    subprocess.run(cmd)


def main():
    parser = argparse.ArgumentParser(description='Run script with different configurations.')
    parser.add_argument('--tags', nargs='*', type=str, help='Tags for the runs')
    parser.add_argument('--record_hzs', nargs='*', type=int, help='Record HZs')
    parser.add_argument('--env_name', type=str, default="holodeck", help='Environment name')
    parser.add_argument('--num_objects_per_run', type=int, default=2, help='Number of objects per run')
    parser.add_argument('--samples_per_model', type=int, default=100, help='Number of samples per model')
    args = parser.parse_args()

    # Use the arguments in your script
    env_name = args.env_name
    num_objects_per_run = args.num_objects_per_run
    global tags, record_hzs, MAIN_OUTPUT_FOLDER, MAIN_CHECKPOINT_FOLDER, samples_per_model
    samples_per_model = args.samples_per_model
    tags = args.tags
    RECORD_HZS = args.record_hzs
    MAIN_OUTPUT_FOLDERS = [main_output_folder_format.format(tag, record_hz) for tag, record_hz in zip(tags, RECORD_HZS)]
    MAIN_CHECKPOINT_FOLDERS = [main_checkpoint_folder_format.format(tag) for tag in tags]

    print(f"MAIN_OUTPUT_FOLDERS: {MAIN_OUTPUT_FOLDERS}")
    print(f"MAIN_CHECKPOINT_FOLDERS: {MAIN_CHECKPOINT_FOLDERS}")

    evaluator = Evaluator(MAIN_OUTPUT_FOLDERS, MAIN_CHECKPOINT_FOLDERS, RECORD_HZS)

    evaluator.config_eval_models(RUN_VAL)
    evaluator.build_concurrent_runs(num_objects_per_run)

    evaluator.run_concurrent(env_name)
    save_multi_layer_videos(MAIN_OUTPUT_FOLDERS)
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

    def build_concurrent_runs(self, num_objects_per_run):
        """
            Samples per model
                x [models to evaluate per common run parameters]
        """
        for i in range(samples_per_model):
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

        self.append_initial_conditions(num_objects_per_run)

    def append_initial_conditions(self, num_objects_per_run):
        """
        Assign initial conditions := [object colors]
        """
        PERMUTATIONS_COLORS = [list(perm) for perm in itertools.product(self.PRIMITIVE_OBJECTS, repeat=num_objects_per_run)]

        OBJECTS = []
        for _ in range(samples_per_model // len(PERMUTATIONS_COLORS)):
            OBJECTS.extend(PERMUTATIONS_COLORS)
        OBJECTS.extend(PERMUTATIONS_COLORS[:samples_per_model % len(PERMUTATIONS_COLORS)])

        random.shuffle(OBJECTS)
        self.OBJECTS.extend(OBJECTS)

    def run_concurrent(self, env_name: str, n_jobs=2):
        # consistent_configs_dir = "./cl_consistent_configs"
        # if not os.path.exists(consistent_configs_dir):
        #     os.makedirs(consistent_configs_dir)
        
        # for colors, run_absolute_path, params_path, checkpoint_path, record_hz in tqdm(zip(self.OBJECTS, self.concurrent_output_folder_full_paths, self.concurrent_params_paths, self.concurrent_checkpoint_paths, self.concurrent_record_hzs)):
        #     run_i = int(re.findall(r'run_(\d+)', run_absolute_path[0])[0])

        #     if not os.path.exists(os.path.join(consistent_configs_dir, str(run_i))):
        #         os.makedirs(os.path.join(consistent_configs_dir, str(run_i)))

        #     if os.path.exists(os.path.join(run_absolute_path[0], "finish.txt")):
        #         continue
        #     run_GS_render(env_name, colors, record_hz, run_absolute_path, params_path, checkpoint_path)
        # run_i_list = [int(re.findall(r'run_(\d+)', run_absolute_path[0])[0]) for run_absolute_path in self.concurrent_output_folder_full_paths]
        joblib.Parallel(n_jobs=n_jobs)(joblib.delayed(run_GS_render)(env_name, colors, record_hz, run_absolute_path, params_path, checkpoint_path) for colors, run_absolute_path, params_path, checkpoint_path, record_hz in tqdm(zip(self.OBJECTS, self.concurrent_output_folder_full_paths, self.concurrent_params_paths, self.concurrent_checkpoint_paths, self.concurrent_record_hzs)))

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