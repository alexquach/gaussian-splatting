
import numpy as np

ENV_CONFIGS = {
    "holodeck": {
        "env_name": "holodeck",
        "m_path": "./output/holodeck2_gs",
        "s_path": "./data/holodeck2_data/keyframes",
        "ply_path": "./output/holodeck2_gs/point_cloud/iteration_30000/point_cloud.ply",
        "keycamera_path": "./camera_assets/key_cameras_4",
        "PYBULLET_TO_GS_SCALING_FACTOR": 1.5 #2 for training
    },
    "colosseum": {
        "env_name": "colosseum",
        "m_path": "/home/makramchahine/repos/gaussian-splatting/output/colosseum",
        "s_path": "/home/makramchahine/repos/nerf/data/phototourism/colosseum-exterior/dense",
        "ply_path": "/home/makramchahine/repos/gaussian-splatting/output/colosseum/point_cloud/iteration_30000/point_cloud.ply",
        "keycamera_path": "./camera_assets/colosseum_camera",
        "PYBULLET_TO_GS_SCALING_FACTOR": 1,
    }
}

TEMPLATE_CAMERA_JSON_PATH = "./output/holodeck2_gs/cameras.json"

OBJECT_MAP = {
    "red ball": {
        "ply_path": "./output/solid_red_ball/point_cloud/iteration_30000/point_cloud.ply",
        "urdf_path": "sphere2red.urdf",
        "scale": 0.2
        },
    
    "blue ball": {
        "ply_path": "./output/solid_blue_ball/point_cloud/iteration_30000/point_cloud.ply",
        "urdf_path": "sphere2blue.urdf",
        "scale": 0.2
    },
    "green ball": {
        "urdf_path": "sphere2green.urdf",
        "scale": 0.2
    },
    "yellow ball": {
        "urdf_path": "sphere2yellow.urdf",
        "scale": 0.2
    },
    "purple ball": {
        "urdf_path": "sphere2purple.urdf",
        "scale": 0.2
    },
    "Rc": {
        "urdf_path": "cube2red.urdf",
        "scale": 0.2
    },
    "Bc": {
        "urdf_path": "cube2blue.urdf",
        "scale": 0.2
    },
    "Yc": {
        "urdf_path": "cube2yellow.urdf",
        "scale": 0.2
    },
    "Pc": {
        "urdf_path": "cube2purple.urdf",
        "scale": 0.2
    },
    "Gc": {
        "urdf_path": "cube2green.urdf",
        "scale": 0.2
    },
    "jeep": {
        "urdf_path": "jeep.urdf",
        "scale": 0.2 / 1.5,
        "orientation": [np.pi / 2, 0, 0]
    },
    "horse": {
        "urdf_path": "horse.urdf",
        "scale": 1 / 5000
    },
    "dog": {
        "urdf_path": "dog.urdf",
        "scale": 1 / 100
    },
    "palmtree": {
        "urdf_path": "palmtree.urdf",
        "scale": 1 / 2000
    },
    "watermelon": {
        "urdf_path": "watermelon.urdf",
        "scale": 1 / 100
    },
    "traffic_light": {
        "urdf_path": "traffic_light.urdf",
        "scale": 1 / 100,
        "orientation": [np.pi / 2, 0, 0]
    },
    "robot": {
        "urdf_path": "robot.urdf",
        "scale": 1 / 100,
    },
    "rocket": {
        "urdf_path": "rocket.urdf",
        "scale": 1 / 1000,
    },
    "pikachu": {
        "urdf_path": "pikachu.urdf",
        "scale": 1 / 10,
    },
    "house_interior": {
        "urdf_path": "house_interior.urdf",
        "scale": 1 / 1000,
    }
}

COLOR_MAP = {key: value.get("ply_path", "./output/solid_blue_ball/point_cloud/iteration_30000/point_cloud.ply") for key, value in OBJECT_MAP.items()}
URDF_MAP = {key: value["urdf_path"] for key, value in OBJECT_MAP.items()}
ORIENTATION_MAP = {key: value.get("orientation", [0, 0, 0]) for key, value in OBJECT_MAP.items()}
SCALING_MAP = {key: value["scale"] for key, value in OBJECT_MAP.items()}