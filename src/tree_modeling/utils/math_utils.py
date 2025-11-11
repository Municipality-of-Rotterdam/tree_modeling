# PointCloud_Tree_Modelling by Amsterdam Intelligence, GPL-3.0 license

"""Math utility methods - Module (Python)

The module is adapted from:
https://github.com/Amsterdam-AI-Team/Urban_PointCloud_Processing

The method rodrigues_rot() is adapted from:
https://github.com/SKrisanski/FSCT/blob/main/scripts/measure.py
"""

import numpy as np
from numba import jit
from typing import Union


@jit(nopython=True, cache=True, parallel=True)  # type: ignore[misc]
def vector_angle(u: np.ndarray, v: np.ndarray = np.array([0.0, 0.0, 1.0])) -> float:
    """
    Compute the angle in degrees between vectors 'u' and 'v'.

    If only 'u' is provided, the angle between 'u' and the vertical axis is returned.

    Args:
        u (np.ndarray): Input vector.
        v (np.ndarray, optional): Reference vector (default: [0, 0, 1]).

    Returns:
        float: Angle in degrees between the vectors.
    """
    # see https://stackoverflow.com/a/2827466/425458
    c = np.dot(u / np.linalg.norm(u), v / np.linalg.norm(v))
    clip = np.minimum(1, np.maximum(c, -1))
    return float(np.rad2deg(np.arccos(clip)))


def vector_bearing(vector: Union[np.ndarray, list[float]]) -> float:
    """
    Convert a 2D vector (x, y) into a bearing angle in the range [0, 360] degrees.

    Args:
        vector (Union[np.ndarray, list]): A 2D vector.

    Returns:
        float: Bearing angle in degrees.
    """
    v_u = np.array(vector) / np.linalg.norm(vector)
    initial_bearing = np.rad2deg(np.arctan2(v_u[1], -v_u[0]) - np.arctan2(1, 0))
    compass_bearing: float = (initial_bearing + 360) % 360
    return compass_bearing


def rodrigues_rot(points: np.ndarray, vector1: np.ndarray, vector2: np.ndarray) -> np.ndarray:
    """
    Perform Rodrigues' rotation formula to rotate a set of points.

    Given a set of points and two vectors, this function computes the rotation
    that aligns `vector1` to `vector2` and applies the transformation to `points`.

    Args:
        points (np.ndarray): Array of shape (N, 3) or (3,) representing the points to be rotated.
        vector1 (np.ndarray): Initial direction vector.
        vector2 (np.ndarray): Target direction vector.

    Returns:
        np.ndarray: Rotated points of the same shape as the input.
    """
    if points.ndim == 1:
        points = points[np.newaxis, :]

    vector1 = vector1 / np.linalg.norm(vector1)
    vector2 = vector2 / np.linalg.norm(vector2)
    k = np.cross(vector1, vector2)
    if np.sum(k) != 0:
        k = k / np.linalg.norm(k)
    theta = np.arccos(np.dot(vector1, vector2))

    # MATRIX MULTIPLICATION
    P_rot = np.zeros((len(points), 3))
    for i in range(len(points)):
        P_rot[i] = (
            points[i] * np.cos(theta)
            + np.cross(k, points[i]) * np.sin(theta)
            + k * np.dot(k, points[i]) * (1 - np.cos(theta))
        )
    return P_rot


def line_plane_intersection(
    plane_point: np.ndarray,
    plane_normal: np.ndarray,
    line_point: np.ndarray,
    line_direction: np.ndarray,
    epsilon: float = 1e-6,
) -> np.ndarray:
    """
    Compute the intersection point of a line with a plane.

    Args:
        plane_point (np.ndarray): A point on the plane.
        plane_normal (np.ndarray): The normal vector of the plane.
        line_point (np.ndarray): A point on the line.
        line_direction (np.ndarray): The direction vector of the line.
        epsilon (float, optional): Small value to avoid numerical instability (default: 1e-6).

    Returns:
        np.ndarray: The intersection point.

    Raises:
        RuntimeError: If the line is parallel to the plane (no intersection).
    """
    ndotu = plane_normal.dot(line_direction)
    if abs(ndotu) < epsilon:
        raise RuntimeError("no intersection or line is within plane")

    w = line_point - plane_point
    si = -plane_normal.dot(w) / ndotu
    intersection_point = w + si * line_direction + plane_point

    return intersection_point
