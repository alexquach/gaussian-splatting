import numpy as np
import json

from env_configs import TEMPLATE_CAMERA_JSON_PATH

RENDER_WIDTH = 256
RENDER_HEIGHT = 144

def get_template_camera_structure(camera_json_path=TEMPLATE_CAMERA_JSON_PATH):
    with open(camera_json_path) as f:
        cameras = json.load(f)
    return cameras


def get_pos_rot(camera_dict):
    return np.array(camera_dict['position'].copy()), np.array(camera_dict['rotation'].copy())

# ! Keycamera processing
MANUAL_ADJUSTMENT = np.array([-0.035, 0.01, 0])
# MANUAL_ADJUSTMENT = np.array([0, 0, 0])
def process_keycamera_to_W2C(keycamera_dict):
    origin = np.array(keycamera_dict['origin'])
    up = np.array(keycamera_dict['up'])
    print(f"up_base: {up}")
    target = np.array(keycamera_dict['target'])

    forward_direction = target - origin + MANUAL_ADJUSTMENT
    forward_direction = forward_direction / np.linalg.norm(forward_direction)

    right_direction = np.cross(up, forward_direction)
    right_direction = right_direction / np.linalg.norm(right_direction)

    up_direction = np.cross(forward_direction, right_direction)
    up_direction = up_direction / np.linalg.norm(up_direction)
    print(f"up_direction: {up_direction}")

    rot = np.array([up_direction, right_direction, forward_direction])

    # project origin onto plane defined by up direction
    origin = origin - np.dot(origin, up) * up

    # print(f"long: {np.array(rotate_about_forward_direction({'rotation': rot, 'position': origin.tolist()}, np.pi/2)['rotation'])[:, 1]}")

    return {
        'position': origin.tolist(),
        'rotation': rot.tolist()
    }

def parse_keycamera(file_path):
    with open(file_path, 'r') as file:
        data = file.readlines()
    parsed_data = []
    for line in data:
        line = line.strip().split('-D')
        line_dict = {}
        for item in line:
            output = item.split('=')
            if len(output) == 2:
                key, value = output
                key = key.strip()
                # if value has commas, convert it to a list
                if ',' in value:
                    value = value.split(',')
                    value = [float(v) for v in value]
                else:
                    value = float(value)
                
                line_dict[key] = value
        parsed_data.append(line_dict)
    return parsed_data

def get_keycameras(file_path):
    parsed_data = parse_keycamera(file_path)
    keycameras = [process_keycamera_to_W2C(keycamera) for keycamera in parsed_data]

    return keycameras

def replace_w2c(camera_dict, keycamera_dict):
    new_dict = camera_dict.copy()
    new_dict['position'] = keycamera_dict['position']
    new_dict['rotation'] = keycamera_dict['rotation']
    return new_dict

def get_start_camera(keycamera_path):
    ref_camera = get_template_camera_structure()[185]

    # Calculate the new focal lengths based on the new dimensions
    old_fx = ref_camera['fx']
    old_fy = ref_camera['fy']
    old_width = ref_camera['width']
    old_height = ref_camera['height']

    new_fx = old_fx * RENDER_WIDTH / old_width
    new_fy = old_fy * RENDER_HEIGHT / old_height

    # Update the camera parameters
    ref_camera['width'] = RENDER_WIDTH
    ref_camera['height'] = RENDER_HEIGHT
    ref_camera['fx'] = new_fx
    ref_camera['fy'] = new_fy

    # ! Set position and rotation from keycamera
    keycamera = get_keycameras(keycamera_path)[0]
    start_dict = replace_w2c(ref_camera, keycamera)

    # ? Rotate camera to correct orientation in the forward axis -- artifact of camera
    start_dict = rotate_about_forward_direction(start_dict, np.pi/2)
    return start_dict

# ! Camera manipulation

def get_forward_direction(camera_dict):
    return np.array(camera_dict['rotation'][2]).copy()

def move_forward(start_dict, distance, delta):
    new_dict = start_dict.copy()
    pos, rot = get_pos_rot(start_dict)
    
    forward_direction = rot[:, 2]
    pos = pos + forward_direction * distance
    new_dict['position'] = pos.tolist()
    delta = delta + np.array([distance, 0, 0, 0])
    return new_dict, delta

def move_sideways(start_dict, distance, delta):
    new_dict = start_dict.copy()
    pos, rot = get_pos_rot(start_dict)
    
    forward_direction = rot[:, 0]
    pos = pos + forward_direction * distance
    new_dict['position'] = pos.tolist()
    delta = delta + np.array([0, distance, 0, 0])
    return new_dict, delta

def rise_relative_to_camera(start_dict, distance, delta):
    new_dict = start_dict.copy()
    pos, rot = get_pos_rot(start_dict)
    
    up_direction = rot[:, 1]
    pos = pos + up_direction * distance
    new_dict['position'] = pos.tolist()
    delta = delta + np.array([0, 0, distance, 0])
    return new_dict, delta

def point_camera_at(start_dict, point):
    new_dict = start_dict.copy()
    rot = np.array(new_dict['rotation'])
    pos = np.array(new_dict['position'])
    
    forward_direction = point - pos
    forward_direction = forward_direction / np.linalg.norm(forward_direction)

    up_direction = np.array([1, 0, 0])
    right_direction = np.cross(up_direction, forward_direction)
    right_direction = right_direction / np.linalg.norm(right_direction)

    up_direction = np.cross(forward_direction, right_direction)
    up_direction = up_direction / np.linalg.norm(up_direction)
    
    rot = np.array([up_direction, right_direction, forward_direction])
    new_dict['rotation'] = rot.tolist()
    return new_dict

def point_camera_at_origin(start_dict):
    return point_camera_at(start_dict, np.array([0, 0, 0]))

def place_camera_at(start_dict, point):
    new_dict = start_dict.copy()
    new_dict['position'] = point.tolist()
    return new_dict

def rotate_camera(start_dict):
    new_dict = start_dict.copy()
    pos, rot = get_pos_rot(start_dict)
    
    up_vector = -rot[:, 0]
    up_plane = np.array([1, 0, 0])  # Assuming the up plane is the y-axis
    
    # Calculate the angle between the camera's up vector and the up plane
    angle = np.arccos(np.dot(up_vector, up_plane) / (np.linalg.norm(up_vector) * np.linalg.norm(up_plane)))
    
    # Rotate the camera by the calculated angle around the cross product of the up vector and the up plane
    rotation_axis = np.cross(up_vector, up_plane)
    rotation_matrix = np.array([[np.cos(angle) + rotation_axis[0]**2 * (1 - np.cos(angle)), rotation_axis[0] * rotation_axis[1] * (1 - np.cos(angle)) - rotation_axis[2] * np.sin(angle), rotation_axis[0] * rotation_axis[2] * (1 - np.cos(angle)) + rotation_axis[1] * np.sin(angle)],
                                [rotation_axis[1] * rotation_axis[0] * (1 - np.cos(angle)) + rotation_axis[2] * np.sin(angle), np.cos(angle) + rotation_axis[1]**2 * (1 - np.cos(angle)), rotation_axis[1] * rotation_axis[2] * (1 - np.cos(angle)) - rotation_axis[0] * np.sin(angle)],
                                [rotation_axis[2] * rotation_axis[0] * (1 - np.cos(angle)) - rotation_axis[1] * np.sin(angle), rotation_axis[2] * rotation_axis[1] * (1 - np.cos(angle)) + rotation_axis[0] * np.sin(angle), np.cos(angle) + rotation_axis[2]**2 * (1 - np.cos(angle))]])
    
    # Apply the rotation to the camera's position and up vector
    new_dict['position'] = (np.dot(rotation_matrix, pos)).tolist()
    new_dict['rotation'] = (rotation_matrix @ rot).tolist()
    # rot = np.dot(rotation_matrix, up_vector)
    # rot = rot / np.linalg.norm(rot)

    # new_dict['rotation'] = np.array([rot, np.cross(rot, np.array([0, 1, 0])), np.array([0, 1, 0])]).tolist()

    return new_dict



def rotate_about_forward_direction(start_dict, angle):
    new_dict = start_dict.copy()
    pos, rot = get_pos_rot(start_dict)

    # Define the rotation matrix for counterclockwise rotation about the forward direction
    rotation_matrix = np.array([[np.cos(angle), -np.sin(angle), 0], 
                                [np.sin(angle), np.cos(angle), 0],
                                [0, 0, 1]])

    # Apply the rotation to the camera's rotation matrix
    new_dict['rotation'] = (rotation_matrix @ rot).tolist()

    return new_dict


def rotate_about_up_direction(rot, angle):
    # Define the rotation matrix for counterclockwise rotation about the up direction
    rotation_matrix = np.array([
                                # [1, 0, 0],
                                # [0, np.cos(angle), -np.sin(angle)], 
                                # [0, np.sin(angle), np.cos(angle)],
                                [np.cos(angle), 0, -np.sin(angle)], 
                                [0, 1, 0],
                                [np.sin(angle), 0, np.cos(angle)],
                                ])

    # rotate on first row, is close to up direction
    # rotate on second row, is close to right direction
    # rotate on third row, is forward direction

    # Apply the rotation to the camera's rotation matrix
    new_rot = (rot @ rotation_matrix)
    # print(new_rot)
    return new_rot

def rotate_camera_dict_about_up_direction(camera_dict, angle, delta):
    new_dict = camera_dict.copy()
    pos, rot = get_pos_rot(camera_dict)

    new_dict['rotation'] = rotate_about_up_direction(rot, angle).tolist()
    delta = delta + np.array([0, 0, 0, angle])
    return new_dict, delta

def dist_from_origin(camera_dict):
    pos, rot = get_pos_rot(camera_dict)
    return np.linalg.norm(pos)

def flip_camera(start_dict):
    new_dict = start_dict.copy()
    pos, rot = get_pos_rot(start_dict)

    # Define the rotation matrix for counterclockwise rotation about the forward direction
    rotation_matrix = np.array([[1, 0, 0], 
                                [0, 1, 0],
                                [0, 0, -1]])

    # Apply the rotation to the camera's rotation matrix
    new_dict['rotation'] = (rot @ rotation_matrix).tolist()

    return new_dict

def camera_diff(camera1, camera2):
    pos1, rot1 = get_pos_rot(camera1)
    pos2, rot2 = get_pos_rot(camera2)

    pos_diff = pos1 - pos2
    rot_diff = rot1 - rot2

    print("Position difference: ", pos_diff)
    print("Rotation difference: ", rot_diff)

def get_yaw_diff_relative_to_origin(camera_dict):
    # global yaw
    pos, rot = get_pos_rot(camera_dict)
    forward_direction = rot[:, 2]
    forward_direction = forward_direction / np.linalg.norm(forward_direction)
    yaw = np.arctan2(forward_direction[2], forward_direction[0])

    # global yaw of optimal orientation directly facing origin from pos
    theta = np.arctan2(pos[2], pos[0]) + np.pi

    diff = yaw - theta

    #normalize to [-pi, pi]
    diff = (diff + np.pi) % (2 * np.pi) - np.pi
    return diff, theta
