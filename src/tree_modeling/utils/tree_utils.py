"""Point cloud tree processing methods.

This module provides methods to process point clouds of trees:
- Skeleton reconstruction and filtering
- Crown and stem separation/metrics
- Cylinder fitting and ground intersection utilities
- Slice-based curation helpers for stem/crown
"""

from __future__ import annotations

import os

import laspy
import numpy as np
import open3d as o3d

from tree_modeling.utils import math_utils, o3d_utils
from tree_modeling.logger import logger
from tree_modeling.core.stem import (
    fit_cylinder,
    get_edge_points,
    calculate_intersections,
    reconstruct_trimesh,
    plot_cylinder_with_rays,
)
from tree_modeling.core.skeleton import (
    leafwood_classification,
    reconstruct_skeleton,
    skeleton_split,
)
from tree_modeling.core.types import SkeletonData, Labels
from tree_modeling.core.constants import (
    STEM_ROI_BUFFER_M,
    CYL_ADDITIONAL_RADIUS_M,
    EDGE_POINTS_DEFAULT,
    MIN_GROUND_POINTS,
    MIN_Z_AXIS_ALIGNMENT,
)


def tree_separate(
    tree_cloud: o3d.geometry.PointCloud,
    adtree_exe: str,
    outdirname: str,
    debug: bool,
    filter_leaves: str | None = None,
) -> tuple[o3d.geometry.PointCloud, o3d.geometry.PointCloud, o3d.geometry.PointCloud]:
    """
    Separate the stem and crown regions from a tree point cloud.

    Steps
    -----
    1. Optionally classify and filter leaves.
    2. Reconstruct the skeleton using the adTree executable.
    3. Split the cloud into stem and crown based on skeleton proximity.

    Parameters
    ----------
    tree_cloud : o3d.geometry.PointCloud
        The complete point cloud of the tree (stem + crown).
    adtree_exe : str
        Path to the adTree skeleton reconstruction executable.
    outdirname : str
        Output directory for skeleton and debug artifacts.
    debug : bool
        If ``True``, saves intermediate files and plots.
    filter_leaves : str or None, optional
        Leaf filtering method, one of ``'curvature'`` or ``'surface_variation'``.

    Returns
    -------
    tuple[o3d.geometry.PointCloud, o3d.geometry.PointCloud, o3d.geometry.PointCloud]
        The tuple ``(stem_cloud, crown_cloud, wood_cloud)``.
    """
    # 1. Classify and filter leaves (optional)
    labels = np.ones(len(tree_cloud.points), dtype=int)
    wood_cloud = tree_cloud
    if filter_leaves:
        logger.info(f"Leaf-wood classification using `{filter_leaves}` method...")
        labels = leafwood_classification(tree_cloud, method=filter_leaves)
        wood_cloud = tree_cloud.select_by_index(np.where(labels == Labels.WOOD)[0])
        logger.info(f"Done. {np.sum(labels==Labels.WOOD)}/{len(labels)} points wood.")

        # Early exit if no wood points → also 0 stem, 0 crown
        if len(wood_cloud.points) == 0:
            logger.info("No wood points detected. Returning empty stem and crown.")
            empty = tree_cloud.select_by_index([])  # empty point cloud
            return empty, empty, empty

    # 2. Skeleton reconstruction
    logger.info("Reconstructing tree skeleton...")
    skeleton: SkeletonData
    skeleton, skeleton_path = reconstruct_skeleton(wood_cloud, adtree_exe, outdirname, overwrite=True, debug=debug)
    logger.info(f"Done. Skeleton constructed containing {len(skeleton['vertices'])} nodes.")
    logger.info(f"Skeleton written to {skeleton_path}.")

    # 3. Stem-crown splitting
    logger.info("Splitting stem from crown...")
    outdirname_dbg = None if not debug else outdirname
    mask = skeleton_split(tree_cloud, skeleton["graph"], outdirname_dbg)
    labels[mask] = Labels.STEM
    logger.info(f"Done. {np.sum(mask)}/{len(labels)} points labeled as stem.")

    stem_cloud = tree_cloud.select_by_index(np.where(mask)[0])
    crown_cloud = tree_cloud.select_by_index(np.where(mask)[0], invert=True)

    return stem_cloud, crown_cloud, wood_cloud


def get_pc_bbox(pc: o3d.geometry.PointCloud) -> tuple[float, float, float, float]:
    """
    Compute the 2D bounding box of a point cloud.

    Parameters
    ----------
    pc : o3d.geometry.PointCloud
        Input point cloud.

    Returns
    -------
    tuple[float, float, float, float]
        ``(minx, miny, maxx, maxy)`` in the XY plane.
    """
    pc_pts = np.asarray(pc.points)
    bbox = (min(pc_pts[:, 0]), min(pc_pts[:, 1]), max(pc_pts[:, 0]), max(pc_pts[:, 1]))
    return bbox


def calculate_reference_height(
    ground_cloud: laspy.lasdata.LasData, ref_pavement: tuple[float, float]
) -> tuple[float, float, float]:
    """
    Compute reference pavement elevation statistics.

    Parameters
    ----------
    ground_cloud : laspy.lasdata.LasData
        Ground LAS data with ``AssetInstance``, ``AssetType``, and ``z`` fields.
    ref_pavement : tuple[float, float]
        Pavement identifier as ``(instance, class)``.

    Returns
    -------
    tuple[float, float, float]
        ``(median_z, min_z, max_z)`` or ``(nan, nan, nan)`` if unavailable.
    """
    if ref_pavement[0] != 0:
        ref_pavement_las = ground_cloud[
            (ground_cloud.AssetInstance == ref_pavement[0]) & (ground_cloud.AssetType == ref_pavement[1])
        ]
        return (
            np.median(ref_pavement_las.z),
            np.min(ref_pavement_las.z),
            np.max(ref_pavement_las.z),
        )
    return (np.nan, np.nan, np.nan)


def get_pc_subsection(
    ground_las: laspy.lasdata.LasData, bounds: tuple[float, float, float, float]
) -> tuple[o3d.geometry.PointCloud, tuple[float, float]]:
    """Get a subsection of the ground point cloud within bounds and its reference pavement.

    Args:
        ground_las: Ground points (LAS).
        bounds: (minx, miny, maxx, maxy) defining the subsection.

    Returns:
        ``(pc_subsection, ref_pavement)`` where ``ref_pavement=(instance, class)``.
    """
    REF_HEIGHT_BUFFER = 2  # buffer around tree region; trees often have their own BGT object
    region_mask = (
        (ground_las.x > bounds[0])
        & (ground_las.y > bounds[1])
        & (ground_las.x < bounds[2])
        & (ground_las.y < bounds[3])
    )
    points = np.vstack(
        (ground_las.points[region_mask].x, ground_las.points[region_mask].y, ground_las.points[region_mask].z)
    ).transpose()

    pc_subsection = o3d.geometry.PointCloud()
    pc_subsection.points = o3d.utility.Vector3dVector(points)

    # Get the most common pavement instance to determine the reference height
    region_mask_ref_height = (
        (ground_las.x > bounds[0] - REF_HEIGHT_BUFFER)
        & (ground_las.y > bounds[1] - REF_HEIGHT_BUFFER)
        & (ground_las.x < bounds[2] + REF_HEIGHT_BUFFER)
        & (ground_las.y < bounds[3] + REF_HEIGHT_BUFFER)
    )
    pav_inst_values, pav_inst_counts = np.unique(
        ground_las.points[region_mask_ref_height].AssetInstance, return_counts=True
    )
    ref_instance = pav_inst_values[np.argmax(pav_inst_counts)]
    pav_cls_values, pav_cls_counts = np.unique(
        ground_las[region_mask_ref_height][ground_las[region_mask_ref_height].AssetInstance == ref_instance].AssetType,
        return_counts=True,
    )
    ref_class = pav_cls_values[np.argmax(pav_cls_counts)]
    ref_pavement = (ref_instance, ref_class)

    return pc_subsection, ref_pavement


def get_stem_endpoints(
    stem_cloud: o3d.geometry.PointCloud,
    ground_cloud: laspy.lasdata.LasData,
    roi_buffer: float = STEM_ROI_BUFFER_M,
    additional_radius: float = CYL_ADDITIONAL_RADIUS_M,
    num_edge_points: int = EDGE_POINTS_DEFAULT,
    debug: bool = False,
    outdirname: str | None = None,
) -> tuple[np.ndarray, np.ndarray, tuple[float, float, float], tuple[float, float, float]]:
    """
    Estimate the stem's base and top points via cylinder fitting and ground intersection.

    Parameters
    ----------
    stem_cloud : o3d.geometry.PointCloud
        Point cloud of the stem region.
    ground_cloud : laspy.lasdata.LasData
        Ground-surface LAS data for intersection and elevation reference.
    roi_buffer : float, default=1.5
        Buffer distance around the stem bounding box for ground extraction.
    additional_radius : float, default=1.0
        Additional radius added to the fitted cylinder.
    num_edge_points : int, default=20
        Number of evenly spaced edge points generated around the cylinder.
    debug : bool, default=False
        If ``True``, generates a 3D visualization saved to ``outdirname``.
    outdirname : str or None, optional
        Directory path to save the visualization if ``debug=True``.

    Returns
    -------
    tuple[np.ndarray, np.ndarray, tuple[float, float, float], tuple[float, float, float]]
        ``(start_point, end_point, ref_pavement_height, cyl_axis)`` where:

        - **start_point** : ndarray of shape (3,), base of the stem.
        - **end_point** : ndarray of shape (3,), top of the stem.
        - **ref_pavement_height** : (median, min, max) of ground elevations nearby.
        - **cyl_axis** : Fitted cylinder axis direction vector.
    """
    horizontal_distance = np.linalg.norm(stem_cloud.get_min_bound() - stem_cloud.get_max_bound())
    logger.info("Horizontal distance between stem endpoints: %.2f m", horizontal_distance)

    # Default values if processing fails
    start_point, end_point = stem_cloud.get_min_bound(), stem_cloud.get_max_bound()
    ref_pavement_height = (np.nan, np.nan, np.nan)

    # Fit the cylinder
    cyl_center, cyl_axis, cyl_radius, ray_direction = fit_cylinder(stem_cloud, additional_radius)

    # Extract ground region of interest (ROI)
    bbox = np.array(get_pc_bbox(stem_cloud))
    buffer = np.array([roi_buffer * elem for elem in (-1, -1, 1, 1)])
    cloud_roi: tuple[float, float, float, float] = tuple(np.add(bbox, buffer))
    ground_subset, ref_pavement = get_pc_subsection(ground_las=ground_cloud, bounds=cloud_roi)
    ref_pavement_height = calculate_reference_height(ground_cloud, ref_pavement)

    if np.abs(ray_direction[2]) <= MIN_Z_AXIS_ALIGNMENT or np.isnan(ray_direction[2]):
        logger.error("Cylinder axis not aligned with Z-direction. Fit likely failed.")
        logger.error("Determining base point with ref pavement directly by taking its median value.")
        start_point = np.array([0, 0, ref_pavement_height[0]])
        return stem_cloud.get_min_bound(), stem_cloud.get_max_bound(), ref_pavement_height, ray_direction

    if len(np.asarray(ground_subset.points)) <= MIN_GROUND_POINTS:
        logger.error("Insufficient ground points. Returning default values.")
        return start_point, end_point, ref_pavement_height, ray_direction

    # Create edge points and intersect with ground mesh
    ground_mesh = o3d_utils.surface_mesh_creation(ground_subset)
    if ground_mesh is None:
        logger.error("Failed to create ground surface mesh.")
        return start_point, end_point, ref_pavement_height, ray_direction

    ground_trimesh = o3d_utils.to_trimesh(ground_mesh)
    if ground_trimesh is None or len(ground_trimesh.vertices) == 0 or len(ground_trimesh.faces) == 0:
        logger.error("Failed to convert ground mesh to Trimesh or the mesh is empty.")
        return start_point, end_point, ref_pavement_height, ray_direction

    edge_points = get_edge_points(cyl_center, cyl_axis, cyl_radius, num_edge_points=num_edge_points)
    if len(edge_points) == 0:
        logger.error("No edge points generated for intersection calculation.")
        return start_point, end_point, ref_pavement_height, ray_direction

    ground_intersections = calculate_intersections(ground_trimesh, edge_points, np.array(ray_direction))

    if len(ground_intersections) != len(edge_points):
        logger.info("Reconstructing ground mesh due to incomplete intersections.")
        ground_trimesh = reconstruct_trimesh(ground_trimesh)
        if ground_trimesh is None or len(ground_trimesh.vertices) == 0 or len(ground_trimesh.faces) == 0:
            logger.error("Error: Invalid ground_trimesh after reconstruction (None, no vertices, or no faces).")
            return start_point, end_point, ref_pavement_height, ray_direction

        ground_intersections = calculate_intersections(ground_trimesh, edge_points, np.array(ray_direction))

        if len(ground_intersections) == 0:
            logger.error("Failed to intersect rays with ground mesh after reconstruction.")
            return start_point, end_point, ref_pavement_height, ray_direction

    # Calculate start and end points
    ground_z = np.median(ground_intersections[:, 2])
    start_point = math_utils.line_plane_intersection(
        plane_point=np.array([0, 0, ground_z]),
        plane_normal=np.array([0, 0, 1]),
        line_point=cyl_center,
        line_direction=-np.array(cyl_axis),
    )
    end_point = math_utils.line_plane_intersection(
        plane_point=np.array([0, 0, stem_cloud.get_max_bound()[2]]),
        plane_normal=np.array([0, 0, 1]),
        line_point=cyl_center,
        line_direction=np.array(cyl_axis),
    )

    if debug and outdirname:
        outfname = os.path.join(outdirname, "stem_base.png")
        ray_length = np.mean(edge_points, axis=0)[2] - np.mean(ground_intersections, axis=0)[2] + 1
        plot_cylinder_with_rays(
            stem_cloud=stem_cloud,
            ground_mesh=o3d_utils.to_trimesh(ground_mesh),
            cyl_center=np.array(cyl_center),
            cyl_axis=np.array(cyl_axis),
            cyl_radius=float(cyl_radius),
            edge_points=edge_points,
            ground_intersections=list(ground_intersections),
            start_point=start_point,
            end_point=end_point,
            ray_direction=np.array(ray_direction),
            figname=outfname,
            ray_length=float(ray_length),
        )

    return start_point, end_point, ref_pavement_height, ray_direction
