Nice to haves:
- [ ] Script to put .urdfs in pybullet data
- [ ] Combined Pybullet Path Generation + GS Rendering Script

## Set up:
Install Cuda 11.8 + CuDNN on your machine with GPU\
(optional) colmap: https://colmap.github.io/install.html

`git clone https://github.com/alexquach/gaussian-splatting --recursive` \
`git checkout rss24` \
`conda env create -f environment.yml`\
`conda activate gs_pyb_comb`
`pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118`
`pip install -e ./submodules/diff-gaussian-rasterization && pip install -e ./submodules/simple-knn`
(https://github.com/graphdeco-inria/gaussian-splatting/issues/662 might be helpful)
<!-- `git submodule update --init --recursive` -->

because of the relative submodule paths, you may want to add alternate python paths:\
`export PYTHONPATH=$PYTHONPATH:./gym-pybullet-drones`\
`export PYTHONPATH=$PYTHONPATH:./drone_causality`

## Closed Loop Evaluation (with pretrained model):
To generate closed loop examples with a pretrained model and pretrained environment:
- Download and unzip `./data/holodeck2_data` and `./outputs/holodeck2_gs` from https://zenodo.org/records/10723785
- Run: `python closed_loop_wrapper.py`

## Training New Models:
Hyperparametertuning:\
`python hyperparameter_tuning.py wiredcfccell_objective <path/to/data> --n_trials 40 --batch_size 32 --storage_name sqlite:///<hypertuning_db>.db --timeout 32000 --study_name s2r_`\
Training Script:\
`python train_multiple.py wiredcfccell_objective <path/to/data> --seq_len 64 --n_epochs 300 --data_shift 16 --n_trains 5 --batch_size 32 --storage_name sqlite:///<hypertuning_db>.db --storage_type rdb --timeout 32000 --out_dir <path/to/output> --study_name s2r_`


## Creating New Training Data:
Generate PyBullet Trajectories:\
`python gym-pybullet-drones/gym_pybullet_drones/examples/simulator_wrapper.py`

Generate GS-Rendered Training Data from PyBullet Paths:\
`python utils/camera_generator.py `\
`python render_training_wrapper.py`

Preprocess data for training format:
`python ./drone_causality/preprocess/process_data_real.py --data_dir <data_dir> --out_dir <out_dir>`

## Generating New Gaussian Splatting Models:
Mostly follow the original instructions:
- Put data in form:
  - directory_name (ex: solid_red_ball)
    - input
      - image_0
      - image_1
      - ...
    - transforms_test.json
    - transforms_train.json
    - transforms_val.json
- `python train.py -s ./gaussian-splatting/data/solid_red_ball --densification_interval 500 --resolution 2`

### Generating New Objects (Blender)
- `pip install bpy==3.6.0`
- Use Blender to create images from all perspectives around:\
`python data/blender_script.py`
- Train a new model using the GS Training Method (`train.py`)
  - `python train.py -s ./data/purple_duck --densification_interval 500 --resolution 2`