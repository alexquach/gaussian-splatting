import numpy as np

def rotate_around_vector(xyz, vector, angle):
    # https://en.wikipedia.org/wiki/Rotation_matrix#Rotation_matrix_from_axis_and_angle
    vector = vector / np.linalg.norm(vector)
    ux, uy, uz = vector
    c = np.cos(angle)
    s = np.sin(angle)
    R = np.array([[c + ux**2 * (1 - c), ux * uy * (1 - c) - uz * s, ux * uz * (1 - c) + uy * s],
                [uy * ux * (1 - c) + uz * s, c + uy**2 * (1 - c), uy * uz * (1 - c) - ux * s],
                [uz * ux * (1 - c) - uy * s, uz * uy * (1 - c) + ux * s, c + uz**2 * (1 - c)]])
    return xyz @ R.transpose()

def move_in_direction(xyz, vector, distance):
    return xyz + distance * vector