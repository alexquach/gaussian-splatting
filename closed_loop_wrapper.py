import subprocess

# bm = bimodal
# px = fixed pybullet issue
# td = 
# nlsp = nonlinear speed
# gn = gaussian noise
# nt = no forward velocity while turning
# srf = 
# irreg2 = support for variable timestep-cfc
# 64 = batch size
# hyp = hyperparametertuned

def gen_tag(hz=110, pybullet=False, model_type="cfc"):
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
    # {"tag": gen_tag(110, True, "cfc"), "record_hz": 3},
    # {"tag": gen_tag(110, True, "cfc"), "record_hz": 9},
    # {"tag": gen_tag(3, True, "cfc"), "record_hz": 3},
    # {"tag": gen_tag(9, True, "cfc"), "record_hz": 9},
    # {"tag": gen_tag(9, True, "cfc"), "record_hz": 3},
    # {"tag": gen_tag(3, True, "lstm"), "record_hz": 3},
    # {"tag": gen_tag(9, True, "lstm"), "record_hz": 9},
    {"tag": gen_tag(110, False, "cfc"), "record_hz": 3},
    # {"tag": gen_tag(110, False, "cfc"), "record_hz": 9},
    # {"tag": gen_tag(3, False, "cfc"), "record_hz": 3},
    # {"tag": gen_tag(9, False, "cfc"), "record_hz": 9}, #this
    # {"tag": gen_tag(9, False, "cfc"), "record_hz": 3}, #this
    # {"tag": gen_tag(3, False, "lstm"), "record_hz": 3},
    # {"tag": gen_tag(9, False, "lstm"), "record_hz": 9},
]
print(configurations)

tags = [config["tag"] for config in configurations]
record_hzs = [str(config["record_hz"]) for config in configurations]

try: 
    command = [
        "python",
        "closed_loop_render.py",
        "--tags", *tags,
        "--record_hzs", *record_hzs,
    ]

    subprocess.run(command)
except Exception as e:
    print(e)