import subprocess

def gen_tag(hz=110, pybullet=False, model_type="cfc"):
    """
    Generate model names (based on our paper's naming scheme)
    """
    if hz == 110:
        traj_hz = "600_1_10hzf"
        epochs = "300"
    elif hz == 3:
        traj_hz = "600_3hzf"
        epochs = "300"
    elif hz == 9:
        traj_hz = "200_9hzf"
        epochs = "150"

    pybullet = "_pybullet" if pybullet else ""

    tag = f"d6_nonorm_ss2_{traj_hz}_bm_px_td_nlsp_gn_nt{pybullet}_srf_{epochs}sf_irreg2_64_hyp_{model_type}"
    return tag

# Define your configurations
configurations = [
    {"tag": gen_tag(110, False, "cfc"), "record_hz": 3},
    # {"tag": gen_tag(110, False, "ltc"), "record_hz": 3},
    # {"tag": gen_tag(110, False, "lem4"), "record_hz": 3},
    # {"tag": gen_tag(3, False, "lstm"), "record_hz": 3},
]

def gen_custom_tag():
    return "trained_model"

tags = [config["tag"] for config in configurations]
record_hzs = [str(config["record_hz"]) for config in configurations]
env_name = "holodeck"

try: 
    command = [
        "python",
        "closed_loop_render.py",
        "--tags", *tags,
        "--record_hzs", *record_hzs,
        "--env_name", env_name
    ]

    subprocess.run(command)
except Exception as e:
    print(e)