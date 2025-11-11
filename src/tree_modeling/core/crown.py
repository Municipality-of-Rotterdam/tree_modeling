# tree_modeling/core/crown.py
from __future__ import annotations

import numpy as np
import open3d as o3d
import trimesh
import pymeshfix
from alphashape import alphashape

from .constants import TREE_COLORS
from .types import Labels
from tree_modeling.misc.smallestenclosingcircle import make_circle
from tree_modeling.utils import o3d_utils
from tree_modeling.logger import logger


def crown_to_mesh(
    crown_cloud: o3d.geometry.PointCloud, method: str, alpha: float = 0.8
) -> tuple[o3d.geometry.TriangleMesh | None, float | None]:
    """Convert a crown cloud to a mesh (alpha shape or convex hull) and compute volume.

    Parameters
    ----------
    crown_cloud : o3d.geometry.PointCloud
        Crown point cloud.
    method : str
        Either ``"alphashape"`` or ``"convex_hull"``.
    alpha : float, default=0.8
        Alpha parameter for the alpha-shape meshing (if method is ``"alphashape"``).

    Returns
    -------
    tuple[o3d.geometry.TriangleMesh | None, float | None]
        ``(mesh, volume)`` where either may be ``None`` on failure.
    """
    try:
        if method == "alphashape":
            crown_cloud_sampled = crown_cloud.voxel_down_sample(0.4)
            pts = np.asarray(crown_cloud_sampled.points)
            mesh = alphashape(pts, alpha)
            clean_points, clean_faces = pymeshfix.clean_from_arrays(mesh.vertices, mesh.faces)
            mesh_tm = trimesh.base.Trimesh(clean_points, clean_faces)
            mesh_tm.fix_normals()
            o3d_mesh = mesh_tm.as_open3d
        else:
            crown_cloud_sampled = crown_cloud.voxel_down_sample(0.2)
            o3d_mesh, _ = crown_cloud_sampled.compute_convex_hull()

        o3d_mesh.compute_vertex_normals()
        o3d_mesh.paint_uniform_color(TREE_COLORS["foliage"])
        if len(np.asarray(o3d_mesh.vertices)) == 0 or len(np.asarray(o3d_mesh.triangles)) == 0:
            logger.error("Invalid crown mesh (no vertices/triangles).")
            return None, None

        return o3d_mesh, o3d_mesh.get_volume()

    except Exception as e:
        logger.info("Error in crown_to_mesh", exc_info=e)
        return None, None


def crown_diameter(crown_cloud: o3d.geometry.PointCloud) -> float | None:
    """Compute crown diameter from a 2D projection using the smallest enclosing circle.

    Parameters
    ----------
    crown_cloud : o3d.geometry.PointCloud
        Crown point cloud.

    Returns
    -------
    float | None
        Crown diameter in meters, or ``None`` on failure.
    """
    try:
        proj_pts = o3d_utils.project(crown_cloud, 2, 0.2)
        radius = make_circle(proj_pts)[2]
        return float(radius * 2)
    except Exception as e:
        logger.info("Error in crown_diameter", exc_info=e)
        return None


def crown_shape(crown_cloud: o3d.geometry.PointCloud) -> str | None:
    """Classify crown shape as conical, inverse-conical, spherical, or cylindrical.

    Parameters
    ----------
    crown_cloud : o3d.geometry.PointCloud
        Crown point cloud.

    Returns
    -------
    str | None
        One of ``{"cylindrical","conical","inverse_conical","spherical"}`` (per :class:`Labels`),
        or ``None`` on failure.
    """
    try:
        crown_sampled = crown_cloud.voxel_down_sample(0.05)
        pts = np.asarray(crown_sampled.points)[:, :3]
        min_z, max_z = pts[:, 2].min(), pts[:, 2].max()
        step_size = (max_z - min_z) / 100
        bins = np.arange(min_z, max_z, step_size)
        slice_ind = np.digitize(pts[:, 2], bins, right=True)

        slice_pts = pts[slice_ind >= 95, :2]
        a_rd = make_circle(slice_pts)[2]
        slice_pts = pts[(slice_ind >= 45) & (slice_ind <= 55), :2]
        b_rd = make_circle(slice_pts)[2]
        slice_pts = pts[slice_ind <= 5, :2]
        c_rd = make_circle(slice_pts)[2]

        shape = Labels.CYLINDRICAL
        if abs(a_rd - b_rd) < 0.2 and abs(b_rd - c_rd) < 0.2 and abs(a_rd - c_rd) < 0.2:
            shape = Labels.CYLINDRICAL
        elif a_rd > b_rd > c_rd:
            shape = Labels.CONICAL
        elif a_rd < b_rd < c_rd:
            shape = Labels.INVERSE_CONICAL
        elif a_rd < b_rd > c_rd:
            shape = Labels.SPHERICAL

        return Labels.get_str(shape)
    except Exception as e:
        logger.info("Error in crown_shape", exc_info=e)
        return None


def crown_height(crown_cloud: o3d.geometry.PointCloud) -> float | None:
    """Compute crown height from the provided point cloud.

    Parameters
    ----------
    crown_cloud : o3d.geometry.PointCloud
        Crown point cloud.

    Returns
    -------
    float | None
        Crown height in meters, or ``None`` on error.
    """
    try:
        return float(o3d_utils.cloud_height(crown_cloud))
    except Exception as e:
        logger.info("Error in crown_height", exc_info=e)
        return None


def crown_base_height(crown_cloud: o3d.geometry.PointCloud, ground_level: float = 0) -> float | None:
    """Compute base height of the crown (min Z minus ground level).

    Parameters
    ----------
    crown_cloud : o3d.geometry.PointCloud
        Crown point cloud.
    ground_level : float, default=0
        Reference ground elevation.

    Returns
    -------
    float | None
        Height in meters, or ``None`` on error.
    """
    try:
        return float(crown_cloud.get_min_bound()[2] - ground_level)
    except Exception as e:
        logger.info("Error in crown_base_height", exc_info=e)
        return None


def find_lowest_crown_points_by_quadrant(
    crown_cloud: o3d.geometry.PointCloud, stem_basepoint: np.ndarray
) -> dict[str, tuple[float, float, float] | None]:
    """Find the lowest crown point in each quadrant relative to the stem base.

    Parameters
    ----------
    crown_cloud : o3d.geometry.PointCloud
        Crown point cloud.
    stem_basepoint : np.ndarray
        ``(x, y, z)`` base of the stem used as quadrant origin.

    Returns
    -------
    dict[str, tuple[float, float, float] | None]
        Mapping with keys ``{"NE","SE","SW","NW"}`` → lowest point tuple or ``None`` if absent.
    """
    points = np.asarray(crown_cloud.points)

    rel_x = points[:, 0] - stem_basepoint[0]
    rel_y = points[:, 1] - stem_basepoint[1]

    mask_NE = (rel_x >= 0) & (rel_y >= 0)
    mask_SE = (rel_x >= 0) & (rel_y < 0)
    mask_SW = (rel_x < 0) & (rel_y < 0)
    mask_NW = (rel_x < 0) & (rel_y >= 0)

    out: dict[str, tuple[float, float, float] | None] = {}
    for quadrant, mask in zip(["NE", "SE", "SW", "NW"], [mask_NE, mask_SE, mask_SW, mask_NW]):
        if np.any(mask):
            min_idx = np.argmin(points[mask][:, 2])
            out[quadrant] = tuple(points[mask][min_idx])
        else:
            out[quadrant] = None
    return out
