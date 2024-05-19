
import os

configs = [
    {"TASK": "2rb", "ENV_NAME": "samurai"},
    {"TASK": "2colors", "ENV_NAME": "samurai"},
    {"TASK": "2rshape", "ENV_NAME": "samurai"},
    {"TASK": "2mshape", "ENV_NAME": "samurai"},
    {"TASK": "3open_dict", "ENV_NAME": "samurai"},
    {"TASK": "2rb", "ENV_NAME": "arena"},
    {"TASK": "3open_dict", "ENV_NAME": "arena"},
]

for config in configs:
    # Set TASK environment variable
    os.environ["TASK"] = config["TASK"]
    os.environ["ENV_NAME"] = config["ENV_NAME"]
    
    # Run the script
    os.system("python queueing_script_test.py")
    os.system("python closed_loop_wrapper.py")
