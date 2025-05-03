import numpy as np
from scipy.spatial.transform import Rotation as Rot

def rot_mat_2d(angle):
    """Return a 2D rotation matrix for a given angle."""
    return Rot.from_euler('z', angle).as_matrix()[0:2, 0:2]

def angle_mod(x, zero_2_2pi=False, degree=False):
    if isinstance(x, float):
        is_float = True
    else:
        is_float = False
    x = np.asarray(x).flatten()
    if degree:
        x = np.deg2rad(x)
    if zero_2_2pi:
        mod_angle = x % (2 * np.pi)
    else:
        mod_angle = (x + np.pi) % (2 * np.pi) - np.pi
    if degree:
        mod_angle = np.rad2deg(mod_angle)
    if is_float:
        return mod_angle.item()
    else:
        return mod_angle