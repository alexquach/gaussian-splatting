
ENV_CONFIGS = {
    "holodeck": {
        "env_name": "holodeck",
        "m_path": "/home/makramchahine/repos/gaussian-splatting/output/holodeck2",
        "s_path": "/home/makramchahine/repos/nerf/data/nerfstudio/custom/holodeck2/keyframes",
        "ply_path": "/home/makramchahine/repos/gaussian-splatting/output/holodeck2/point_cloud/iteration_30000/point_cloud.ply",
        "keycamera_path": "./camera_assets/key_cameras_4",
        "PYBULLET_TO_GS_SCALING_FACTOR": 2.525,
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

TEMPLATE_CAMERA_JSON_PATH = "/home/makramchahine/repos/gaussian-splatting/output/holodeck2/cameras.json"

COLOR_MAP = {
    "R": "/home/makramchahine/repos/gaussian-splatting/output/solid_red_ball/point_cloud/iteration_30000/point_cloud.ply",
    "B": "/home/makramchahine/repos/gaussian-splatting/output/solid_blue_ball/point_cloud/iteration_30000/point_cloud.ply",
    "G": "/home/makramchahine/repos/gaussian-splatting/output/solid_blue_ball/point_cloud/iteration_30000/point_cloud.ply",
}