import numpy as np
import numpy.linalg as npl


def mgrid_to_points(mgrid):
    """
    Takes a NxD1xD2xD3 meshgrid or tuple(meshgrid) and outputs a D1*D2*D3xN
    matrix of coordinate points
    """
    points = np.empty(shape=(np.prod(mgrid[0].shape), len(mgrid)),
                      dtype=mgrid[0].dtype)
    for i in range(len(mgrid)):
        points[:, i] = mgrid[i].ravel()
    return points


def points_to_mgrid(points, grid_shape):

    mgrid = np.empty(shape=((points.shape[1],) + tuple(grid_shape)),
                     dtype=points.dtype)
    for i in range(points.shape[1]):
        mgrid[i] = points[:, i].reshape(grid_shape)

    return mgrid


def get_angle(v1, v2):
    v1_u = v1 / npl.norm(v1)
    v2_u = v2 / npl.norm(v2)
    return np.rad2deg(np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0)))


def get_rotation_matrix(axis, angle_deg=None, angle_rad=None):
    """
    Modified from:
    https://stackoverflow.com/questions/6802577/python-rotation-of-3d-vector

    Return the rotation matrix associated with counterclockwise rotation about
    the given axis by theta degrees.
    """
    theta = angle_rad or np.deg2rad(angle_deg)
    axis = np.asarray(axis).ravel()
    axis = axis/np.linalg.norm(axis)
    a = np.cos(theta/2.0)
    b, c, d = -axis*np.sin(theta/2.0)
    aa, bb, cc, dd = a*a, b*b, c*c, d*d
    bc, ad, ac, ab, bd, cd = b*c, a*d, a*c, a*b, b*d, c*d

    return np.array([[aa+bb-cc-dd, 2*(bc+ad), 2*(bd-ac)],
                     [2*(bc-ad), aa+cc-bb-dd, 2*(cd+ab)],
                     [2*(bd+ac), 2*(cd-ab), aa+dd-bb-cc]])


def _rotate_grid(grid, rot_mat):
    """
    Rotates a grid around its center by 'angle_deg' degrees counter clockwise
    around the vector 'axis'.
    """
    points = mgrid_to_points(grid)

    # Center, rotate and bring back the grid
    center = np.mean(points, axis=0).reshape((1, 3))
    c_points = points - center
    rotated = np.dot(rot_mat, c_points.T).T.astype(np.float32) + center

    return rotated
