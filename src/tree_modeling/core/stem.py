# tree_modeling/core/stem.py
from __future__ import annotations

import numpy as np
import open3d as o3d
import trimesh
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from scipy.interpolate import LinearNDInterpolator
from scipy.spatial import Delaunay
from numba import jit

from tree_modeling.misc.fitcyclinders import fit_vertical_cylinder_3D
from tree_modeling.logger import logger
from tree_modeling.utils import math_utils, o3d_utils
from .constants import BREAST_HEIGHT_M


def fit_cylinder(
    stem_cloud: o3d.geometry.PointCloud, additional_radius: float
) -> tuple[np.ndarray | None, np.ndarray | None, float | None, tuple[float, float, float]]:
    """Fit a vertical cylinder to stem points and return center, axis, radius, and ray direction.

    The cylinder axis is normalized and the radius is expanded by ``additional_radius``.
    The ray direction points roughly downward (opposite the sign of ``axis.z``).
    """
    stem_voxeld = stem_cloud.voxel_down_sample(0.04)
    stem_points = np.array(stem_voxeld.points)
    try:
        cyl_center, cyl_axis, cyl_radius = fit_vertical_cylinder_3D(stem_points, 0.05)[:3]
        cyl_radius += additional_radius
        cyl_axis = cyl_axis / np.linalg.norm(cyl_axis)
        ray_direction = -np.sign(cyl_axis[2]) * cyl_axis
        return cyl_center, cyl_axis, cyl_radius, tuple(ray_direction.tolist())
    except Exception as e:
        logger.error("Exception in fit_cylinder: %s", e)
        cyl_center, cyl_axis, cyl_radius = (None, None, None)
        ray_direction = (np.nan, np.nan, np.nan)
        return cyl_center, cyl_axis, cyl_radius, ray_direction


def calculate_rotation_matrix(cyl_axis: np.ndarray) -> np.ndarray:
    """Compute a rotation matrix that maps the +Z axis onto ``cyl_axis``."""
    z_axis = np.array([0, 0, 1])
    v = np.cross(z_axis, cyl_axis)
    s = np.linalg.norm(v)
    c = np.dot(z_axis, cyl_axis)
    if s != 0:
        vx = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
        rotation_matrix = np.eye(3) + vx + np.matmul(vx, vx) * ((1 - c) / (s**2))
    else:
        rotation_matrix = np.eye(3)
    return rotation_matrix


def get_edge_points(
    cyl_center: np.ndarray, cyl_axis: np.ndarray, cyl_radius: float, num_edge_points: int
) -> np.ndarray:
    """Sample evenly spaced points on the cylinder rim in global coordinates."""
    rotation_matrix = calculate_rotation_matrix(cyl_axis)
    theta = np.linspace(0, 2 * np.pi, num_edge_points, endpoint=False)
    local_edge_points = np.array([[np.cos(t), np.sin(t), 0] for t in theta]) * cyl_radius
    return np.array([cyl_center + rotation_matrix @ pt for pt in local_edge_points])


def generate_cylinder_points(
    cyl_center: np.ndarray, cyl_axis: np.ndarray, cyl_radius: float, height: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Generate a small cylindrical surface aligned to an arbitrary axis."""
    theta = np.linspace(0, 2 * np.pi, 100)
    z = np.linspace(0, height, 50)
    theta, z = np.meshgrid(theta, z)
    x = cyl_radius * np.cos(theta)
    y = cyl_radius * np.sin(theta)
    cyl_points = np.array([x.flatten(), y.flatten(), z.flatten()])
    rotation_matrix = calculate_rotation_matrix(cyl_axis)
    rotated_points = np.dot(rotation_matrix, cyl_points).T + cyl_center
    return (
        rotated_points[:, 0].reshape(x.shape),
        rotated_points[:, 1].reshape(y.shape),
        rotated_points[:, 2].reshape(z.shape),
    )


def stem_height(stem_cloud: o3d.geometry.PointCloud, ground_level: float = 0) -> float | None:
    """
    Compute stem height as ``max_z - ground_level``.

    Parameters
    ----------
    stem_cloud : o3d.geometry.PointCloud
        Stem point cloud.
    ground_level : float, default=0
        Reference ground elevation.

    Returns
    -------
    float or None
        Estimated stem height, or ``None`` on error.
    """
    try:
        height: float = stem_cloud.get_max_bound()[2] - ground_level
        return height
    except Exception as e:
        logger.info("Error at %s", "tree_utils error", exc_info=e)
        return None


def stem_angle(stem_cylinders: np.ndarray) -> float | None:
    """
    Estimate the 3D angle of the stem axis from fitted cylinder centers.

    Parameters
    ----------
    stem_cylinders : np.ndarray
        Array where each row contains a cylinder center (at least first 3 columns).

    Returns
    -------
    float or None
        Angle in radians (0 = vertical), or ``None`` on error.
    """
    try:
        angle: float = math_utils.vector_angle(stem_cylinders[-1, :3] - stem_cylinders[0, :3])
        return angle
    except Exception as e:
        logger.info("Error at %s", "tree_utils error", exc_info=e)
        return None


def stem_bearing(stem_cylinders: np.ndarray) -> float | None:
    """
    Estimate planar bearing (XY) from first and last fitted cylinder centers.

    Parameters
    ----------
    stem_cylinders : np.ndarray
        Array where each row contains a cylinder center; uses first two columns.

    Returns
    -------
    float or None
        Bearing angle in radians (e.g., from +X), or ``None`` on error.
    """
    try:
        angle_bearing: float = math_utils.vector_bearing(stem_cylinders[-1, :2] - stem_cylinders[0, :2])
        return angle_bearing
    except Exception as e:
        logger.info("Error at %s", "tree_utils error", exc_info=e)
        return None


def diameter_at_breastheight(stem_cloud: o3d.geometry.PointCloud, ground_level: float = 0) -> float | None:
    """
    Estimate the diameter at breast height (DBH) via vertical cylinder fitting.

    A slice centered at ``ground_level + 1.3`` m with ±0.15 m thickness is extracted.
    A vertical cylinder is fitted and its diameter returned.

    Parameters
    ----------
    stem_cloud : o3d.geometry.PointCloud
        Stem point cloud.
    ground_level : float, default=0
        Ground elevation (z).

    Returns
    -------
    float or None
        Diameter at breast height in meters, or ``None`` if insufficient data.
    """
    try:
        stem_points = np.asarray(stem_cloud.points)
        z = ground_level + BREAST_HEIGHT_M

        # clip slice
        mask = axis_clip(stem_points, 2, z - 0.15, z + 0.15)
        stem_slice = stem_points[mask]
        if len(stem_slice) < 20:
            return None

        # fit cylinder
        radius = fit_vertical_cylinder_3D(stem_slice, 0.04)[2]
        diameter: float = 2 * radius

        return diameter
    except Exception as e:
        logger.info("Error at %s", "tree_utils error", exc_info=e)
        return None


def calculate_intersections(
    ground_trimesh: o3d.geometry.TriangleMesh | trimesh.Trimesh,
    edge_points: np.ndarray,
    ray_direction: np.ndarray,
) -> np.ndarray:
    """Intersect rays (one per edge point) with the ground trimesh. Returns highest-z hits."""
    # Support both incoming O3D mesh (convert) and trimesh
    if isinstance(ground_trimesh, o3d.geometry.TriangleMesh):
        tm = o3d_utils.to_trimesh(ground_trimesh)
        if tm is None:
            return np.empty((0, 3))
        ground_trimesh = tm

    intersections: list[np.ndarray] = []
    for edge_point in edge_points:
        if not isinstance(edge_point, np.ndarray) or edge_point.shape != (3,):
            logger.warning("Skipping invalid edge point: %s", edge_point)
            continue
        if edge_point.size == 0 or np.any(~np.isfinite(edge_point)):
            logger.warning("Skipping non-finite edge point: %s", edge_point)
            continue
        if np.all(ray_direction == 0) or np.any(~np.isfinite(ray_direction)):
            logger.warning("Invalid ray direction: %s", ray_direction)
            break

        locations, _, _ = ground_trimesh.ray.intersects_location([edge_point], [ray_direction])
        if len(locations) > 0:
            intersections.append(locations[np.argmax(locations[:, 2])])
    return np.array(intersections)


def reconstruct_trimesh(ground_trimesh: trimesh.Trimesh, linspace_pts: int = 100, buffer: int = 5) -> trimesh.Trimesh:
    """Reconstruct a triangulated mesh over a height field via LinearNDInterpolation."""
    vertices = ground_trimesh.vertices
    tri = Delaunay(vertices[:, 0:2])

    xmin, xmax = np.min(vertices[:, 0]), np.max(vertices[:, 0])
    ymin, ymax = np.min(vertices[:, 1]), np.max(vertices[:, 1])
    bbox = np.array([[xmin - buffer, xmax + buffer], [ymin - buffer, ymax + buffer]]).T
    x_grid, y_grid = np.meshgrid(
        np.linspace(bbox[0][0], bbox[1][0], linspace_pts), np.linspace(bbox[0][1], bbox[1][1], linspace_pts)
    )
    z_interp = LinearNDInterpolator(tri, vertices[:, 2])
    x = x_grid.ravel()
    y = y_grid.ravel()
    z = z_interp(x, y)

    vertices_with_z = np.c_[np.column_stack((x, y)), z]
    tri_reconstructed = Delaunay(vertices_with_z[:, 0:2])
    return trimesh.Trimesh(vertices=vertices_with_z, faces=tri_reconstructed.simplices)


def plot_cylinder_with_rays(
    stem_cloud: o3d.geometry.PointCloud,
    ground_mesh: trimesh.Trimesh,
    cyl_center: np.ndarray,
    cyl_axis: np.ndarray,
    cyl_radius: float,
    edge_points: np.ndarray,
    ground_intersections: list[np.ndarray],
    start_point: np.ndarray,
    end_point: np.ndarray,
    ray_direction: np.ndarray,
    figname: str,
    ray_length: float,
    height: float = 0.01,
) -> None:
    """Visualize a fitted stem cylinder, cast rays to the ground mesh, and save a figure."""
    stem_points = np.asarray(stem_cloud.points)
    ground_vertices = np.asarray(ground_mesh.vertices)
    ground_faces = np.asarray(ground_mesh.faces)

    x_cyl, y_cyl, z_cyl = generate_cylinder_points(cyl_center, cyl_axis, cyl_radius, height)

    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection="3d")

    ax.plot_surface(x_cyl, y_cyl, z_cyl, color="lightblue", alpha=0.5, edgecolor="blue", label="Cylinder", zorder=3)
    ax.scatter(edge_points[:, 0], edge_points[:, 1], edge_points[:, 2], color="grey", label="Edge Points", zorder=3)

    for edge_point in edge_points:
        ray_end = edge_point + ray_direction * ray_length
        ax.plot(
            [edge_point[0], ray_end[0]],
            [edge_point[1], ray_end[1]],
            [edge_point[2], ray_end[2]],
            color="gray",
            zorder=2,
        )

    ground_intersections_arr = np.array(ground_intersections)
    ax.scatter(
        ground_intersections_arr[:, 0],
        ground_intersections_arr[:, 1],
        ground_intersections_arr[:, 2],
        color="red",
        label="Ground Intersections",
        zorder=4,
    )

    ax.scatter(*start_point, color="black", marker="*", s=50, label="Start Point", zorder=6)
    ax.scatter(*end_point, color="black", marker="*", s=50, label="End Point", zorder=6)

    ax.scatter(
        stem_points[:, 0], stem_points[:, 1], stem_points[:, 2], s=2, color="green", label="Stem Cloud", zorder=5
    )

    face_colors = (ground_vertices[ground_faces].mean(axis=1)[:, 2] - ground_vertices[:, 2].min()) / (
        ground_vertices[:, 2].ptp()
    )
    mesh = Poly3DCollection(
        ground_vertices[ground_faces],
        array=face_colors,
        cmap="viridis",
        edgecolor="w",
        linewidth=0.01,
        alpha=0.6,
        zorder=1,
    )
    ax.add_collection3d(mesh)

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_title("Cylinder, Edge Points, and Rays")
    plt.colorbar(mesh, ax=ax, shrink=0.5, aspect=10, label="Relative ground height (z-value)")
    ax.legend()

    plt.savefig(figname)
    plt.close()


@jit(nopython=True, cache=True)  # type: ignore[misc]
def axis_clip(points: np.ndarray, axis: int, lower: float | int = -np.inf, upper: float | int = np.inf) -> np.ndarray:
    """
    Clip points within specified bounds along a given axis.

    Parameters
    ----------
    points : np.ndarray of shape (n_points, n_dims)
        The array of points, where each row represents a point in an n-dimensional space.
    axis : int
        The axis along which to apply the clipping.
    lower : float or int, optional (default: -inf)
        The lower bound for clipping along the specified axis.
    upper : float or int, optional (default: inf)
        The upper bound for clipping along the specified axis.

    Returns
    -------
    np.ndarray of shape (n_points,)
        A boolean mask with True for points within the specified bounds along the given axis.
    """
    clip_mask = (points[:, axis] <= upper) & (points[:, axis] >= lower)
    return clip_mask
