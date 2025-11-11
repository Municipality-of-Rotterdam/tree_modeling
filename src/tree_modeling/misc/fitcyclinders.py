# PointCloud_Tree_Modelling by Amsterdam Intelligence, GPL-3.0 license

"""
Cylinder fit methods - Module (Python)

The module is adapted from:
https://github.com/SKrisanski/FSCT/blob/main/scripts/measure.py
"""

import warnings

import numpy as np
from scipy.optimize import leastsq
from scipy.spatial.transform import Rotation as R
from typing import Tuple, Any

import tree_modeling.utils.math_utils as math_utils


def circumferential_completeness_index(
    fitted_circle_centre: np.ndarray, estimated_radius: float, slice_points: np.ndarray
) -> float:
    """
    Computes the Circumferential Completeness Index (CCI) of a fitted circle.

    Args:
        fitted_circle_centre (np.ndarray): The x, y coordinates of the circle's center.
        estimated_radius (float): The radius of the circle.
        slice_points (np.ndarray): The points the circle was fitted to.

    Returns:
        float: The Circumferential Completeness Index (CCI).
    """

    sector_angle = 4.5  # degrees
    num_sections = int(np.ceil(360 / sector_angle))
    sectors = np.linspace(-180, 180, num=num_sections, endpoint=False)

    centre_vectors = slice_points[:, :2] - fitted_circle_centre
    norms = np.linalg.norm(centre_vectors, axis=1)

    centre_vectors = centre_vectors / np.atleast_2d(norms).T
    centre_vectors = centre_vectors[np.logical_and(norms >= 0.8 * estimated_radius, norms <= 1.2 * estimated_radius)]

    sector_vectors = np.vstack((np.cos(sectors), np.sin(sectors))).T
    CCI: float = (
        np.sum(
            [
                np.any(
                    np.degrees(
                        np.arccos(np.clip(np.einsum("ij,ij->i", np.atleast_2d(sector_vector), centre_vectors), -1, 1))
                    )
                    < sector_angle / 2
                )
                for sector_vector in sector_vectors
            ]
        )
        / num_sections
    )

    return CCI


def fit_vertical_cylinder_3D(xyz: np.ndarray, th: float) -> Tuple[np.ndarray, np.ndarray, float, np.ndarray, float]:
    """
        This is a fitting for a vertical cylinder fitting
        Reference:
        http://www.int-arch-photogramm-remote-sens-spatial-inf-sci.net/XXXIX-B5/169/2012/isprsarchives-XXXIX-B5-169-2012.pdf

        xyz is a matrix contain at least 5 rows, and each row stores x y z of a cylindrical surface
        p is initial values of the parameter;
        p[0] = Xc, x coordinate of the cylinder centre
        P[1] = Yc, y coordinate of the cylinder centre
        P[2] = alpha, rotation angle (radian) about the x-axis
        P[3] = beta, rotation angle (radian) about the y-axis
        P[4] = r, radius of the cylinder

        th, threshold for the convergence of the least squares

    Args:
        xyz (np.ndarray): Matrix with at least 5 rows, each storing x, y, z of a cylindrical surface.
        th (float): Threshold for the convergence of the least squares optimization.

    Returns:
        Tuple[np.ndarray, np.ndarray, float, np.ndarray, float]:
            - center (np.ndarray): The fitted cylinder's center (x, y, z).
            - axis (np.ndarray): The fitted cylinder's axis direction.
            - radius (float): The fitted cylinder's radius.
            - inliers (np.ndarray): Indices of the inliers after fitting.
            - CCI (float): The Circumferential Completeness Index (CCI) for the fitted cylinder.
    """
    xyz_mean = np.mean(xyz, axis=0)
    xyz_centered = xyz - xyz_mean
    x = xyz_centered[:, 0]
    y = xyz_centered[:, 1]
    z = xyz_centered[:, 2]

    # init parameters
    p = [0, 0, 0, 0, max(np.abs(y).max(), np.abs(x).max())]

    def fitfunc(p: np.ndarray, x: np.ndarray, y: np.ndarray, z: np.ndarray) -> np.ndarray:
        """Calculate the fitting function."""
        return (
            -np.cos(p[3]) * (p[0] - x) - z * np.cos(p[2]) * np.sin(p[3]) - np.sin(p[2]) * np.sin(p[3]) * (p[1] - y)
        ) ** 2 + (z * np.sin(p[2]) - np.cos(p[2]) * (p[1] - y)) ** 2

    # Define the error function
    def errfunc(p: np.ndarray, x: np.ndarray, y: np.ndarray, z: np.ndarray) -> np.ndarray:
        """Calculate the error function."""
        return fitfunc(p, x, y, z) - p[4] ** 2  # error function

    # fit
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore")
        est_p = leastsq(errfunc, p, args=(x, y, z), maxfev=1000)[0]
        inliers = np.where(errfunc(est_p, x, y, z) < th)[0]

    # convert
    center = np.array([est_p[0], est_p[1], 0]) + xyz_mean
    radius = est_p[4]

    rotation = R.from_rotvec([est_p[2], 0, 0])
    axis = rotation.apply([0, 0, 1])
    rotation = R.from_rotvec([0, est_p[3], 0])
    axis = rotation.apply(axis)

    # circumferential completeness index (CCI)
    P_xy = math_utils.rodrigues_rot(xyz_centered, axis, [0, 0, 1])
    CCI = circumferential_completeness_index([est_p[0], est_p[1]], radius, P_xy)

    return center, axis, radius, inliers, CCI


def fit_cylinders_to_stem(stem_cloud: Any, slice_thickness: float) -> Tuple[np.ndarray, np.ndarray]:
    """
    Fits a 3D line to the skeleton points cluster provided, using this line as the major axis/axial vector of the cylinder.
    Fits a series of cylinders perpendicular to this axis to the point cloud of this particular stem segment.

    Args:
        stem_cloud (Any): The point cloud representing the segment of the stem.
        slice_thickness (float): The thickness of the slices to be used for cylinder fitting.

    Returns:
        Tuple[np.ndarray, np.ndarray]:
            - cyl_array (np.ndarray): A numpy array containing the fitted cylinders (center, radius, CCI).
            - dist_to_stem_center (np.ndarray): Distances of the fitted cylinders to the stem center.
    """
    # fit 3D cylinder
    stem_cloud_sampled = stem_cloud.voxel_down_sample(0.01)
    points = np.asarray(stem_cloud_sampled.points)[:, :3]
    min_z, max_z = points[:, 2].min(), points[:, 2].max()
    center, axis, _, _, _ = fit_vertical_cylinder_3D(points, 0.1)

    # slice cylinder
    b_center = (center + axis * (min_z - center[2] + slice_thickness / 2),)
    t_center = center + axis * (max_z - center[2] - slice_thickness / 2)
    length = np.linalg.norm(t_center - b_center)
    line_centers = np.linspace(b_center, t_center, int(length / slice_thickness))

    # fit cylinders per slice
    cyl_array = np.zeros((0, 5))
    dist_to_stem_center = []
    for line_center in line_centers:
        plane_slice = points[np.linalg.norm(abs(axis * (points - line_center)), axis=1) < (slice_thickness / 2)]
        if plane_slice.shape[0] > 0:
            cyl_center, _, cyl_radius, _, cyl_cci = fit_vertical_cylinder_3D(plane_slice, 0.03)
            cyl_array = np.vstack((cyl_array, np.array([*cyl_center, cyl_radius, cyl_cci])))
            dist_to_stem_center.append(np.linalg.norm(cyl_center - line_center[0]))  # should be < 0.1

    return cyl_array, np.array(dist_to_stem_center)
