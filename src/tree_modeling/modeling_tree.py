"""Tree modeling: crown/stem separation, curation, meshing, and reporting.

This module orchestrates tree reconstruction from processed point clouds. It
separates crown and stem, performs geometric curation, generates meshes and
metrics, and writes per-tree and per-tile summaries.
"""

from __future__ import annotations

import argparse
import json
import multiprocessing
import os

import laspy
import numpy as np
import open3d as o3d
from scipy.spatial import cKDTree
import trimesh
from tqdm import tqdm
from sklearn.cluster import DBSCAN

from tree_modeling.config import Paths
from tree_modeling.determine_pruning_needs import (
    create_pavement_zones_and_find_tree_intersections,
)
from tree_modeling.helper_functions import (
    write_stem_crown_cloud,
    divide_pointcloud_into_slices,
    debug_stem_crown_slices,
    plot_pointcloud,
    plot_stem_crown,
    match_segment_id_to_tree,
    combine_json_files,
    load_processed_files,
    write_tree_stats_segments_gpkg,
)
from tree_modeling.misc.fitcyclinders import (
    fit_cylinders_to_stem,
    fit_vertical_cylinder_3D,
)
from tree_modeling.logger import logger
from tree_modeling.utils import o3d_utils
from tree_modeling.utils.tree_utils import tree_separate, get_stem_endpoints
from tree_modeling.core.curation import (
    curate_stem_cloud,
    curate_crown_cloud,
)
from tree_modeling.core.crown import (
    crown_height,
    crown_base_height,
    find_lowest_crown_points_by_quadrant,
    crown_diameter,
    crown_shape,
    crown_to_mesh,
)
from tree_modeling.core.stem import (
    stem_angle,
    stem_bearing,
    diameter_at_breastheight,
)
from tree_modeling.core.constants import TREE_COLORS
from tree_modeling.core.types import CrownStats

# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _process_point_cloud_star(
    args: tuple[
        str,  # pc_path
        dict[str, str | list[str]],  # pc_ops_files
        str,  # bgt_pavements_path
        argparse.Namespace,  # args
    ],
) -> dict[str, dict[str, str]] | None:
    """Star-unpack wrapper for :func:`process_point_cloud` for pool workers.

    The expected signature is derived from :func:`modeling_tree`, which builds
    tuples like ``(pc_path, pc_ops_files, bgt_pavements_path, args)`` where
    ``pc_ops_files`` matches the structure returned by ``load_processed_files``.
    """
    return process_point_cloud(*args)


# ---------------------------------------------------------------------------
# Crown / stem processing
# ---------------------------------------------------------------------------


def process_crown(
    crown_cloud: o3d.geometry.PointCloud,
    stem_basepoint: np.ndarray,
    outdirname: str,
) -> tuple[CrownStats, o3d.geometry.TriangleMesh]:
    """Process the tree crown.

    Computes descriptive statistics, derives a crown mesh (alpha shape), and
    writes crown artifacts to disk.

    Args:
        crown_cloud: Crown point cloud.
        stem_basepoint: Base of the stem (``[x, y, z]``).
        outdirname: Output directory for artifacts.

    Returns:
        A tuple ``(crown_stats, crown_mesh)`` where ``crown_stats`` contains
        geometric properties and ``crown_mesh`` is an ``open3d`` triangle mesh.
    """
    crown_center = tuple(crown_cloud.get_center()[:2])
    crown_basepoint = tuple(crown_cloud.get_min_bound())
    crown_toppoint = tuple(crown_cloud.get_max_bound())
    crown_size = crown_height(crown_cloud=crown_cloud)

    if isinstance(stem_basepoint, np.ndarray):
        crown_base_h = crown_base_height(
            crown_cloud=crown_cloud, ground_level=stem_basepoint[2]
        )  # distance crown lowest point to ground
        crown_top_h = crown_base_h + crown_size
        lowest_crown_points = find_lowest_crown_points_by_quadrant(
            crown_cloud=crown_cloud, stem_basepoint=stem_basepoint
        )
    else:
        crown_base_h = None
        crown_top_h = None
        lowest_crown_points = None

    crown_diam = crown_diameter(crown_cloud=crown_cloud)
    crown_sh = crown_shape(crown_cloud=crown_cloud)
    crown_mesh_alpha, crown_volume = crown_to_mesh(crown_cloud=crown_cloud, method="alphashape")

    logger.info("Writing crown mesh and point cloud to file.")
    try:
        o3d.io.write_point_cloud(os.path.join(outdirname, "crown_curate.ply"), crown_cloud)
        o3d.io.write_triangle_mesh(
            os.path.join(outdirname, "crown_mesh.ply"),
            crown_mesh_alpha,
            write_ascii=False,
        )
    except Exception:
        logger.exception("An error occurred during writing crown mesh and point cloud to file.")

    crown_stats: CrownStats = {
        "crown_center": crown_center,
        "crown_basepoint": crown_basepoint,
        "crown_toppoint": crown_toppoint,
        "crown_lowest_points": lowest_crown_points,
        "crown_size": crown_size,
        "crown_base_height": crown_base_h,
        "crown_top_height": crown_top_h,
        "crown_diameter": crown_diam,
        "crown_shape": crown_sh,
        "crown_volume": crown_volume,
    }
    return crown_stats, crown_mesh_alpha


def process_stem(
    stem_cloud: o3d.geometry.PointCloud,
    stem_basepoint: np.ndarray,
    stem_toppoint: np.ndarray,
    outdirname: str,
) -> tuple[dict[str, object], o3d.geometry.TriangleMesh | None]:
    """Process the stem (wood) of the tree.

    Fits cylinders per slice for a coarse stem mesh, estimates angle/bearing,
    and (optionally) writes a breast-height cylinder.

    Args:
        stem_cloud: Point cloud representing the stem.
        stem_basepoint: Base of the stem (``[x, y, z]``).
        stem_toppoint: Top of the stem (``[x, y, z]``).
        outdirname: Output directory for artifacts.

    Returns:
        A tuple ``(stem_stats, stem_mesh)``. On failure, ``stem_mesh`` may be
        ``None`` and stats will contain ``None`` values for affected fields.
    """
    MAX_STEM_OFFSET = 0.1  # max distance between fitted cylinder center and stem center
    try:
        stem_cylinders, dist_to_stem = fit_cylinders_to_stem(stem_cloud=stem_cloud, slice_thickness=0.25)
        stem_mesh = o3d_utils.mesh_from_cylinders(
            cyl_array=stem_cylinders[dist_to_stem < MAX_STEM_OFFSET],
            color=TREE_COLORS["stem"],
        )
    except Exception:
        logger.exception("Could not fit cylinder to stem. Aborting.")
        return (
            {
                "stem_height": None,
                "diameter_at_breast_height": None,
                "stem_CCI (med, min, max)": None,
                "stem_angle": None,
                "compas_bearing": None,
                "stem_basepoint": None,
            },
            None,
        )

    logger.info("Writing stem mesh to file.")
    try:
        o3d.io.write_point_cloud(os.path.join(outdirname, "stem_curate.ply"), stem_cloud, write_ascii=False)
        o3d.io.write_triangle_mesh(os.path.join(outdirname, "stem_mesh.ply"), stem_mesh, write_ascii=False)
    except Exception:
        logger.exception("An error occurred during writing stem mesh and point cloud to file.")

    stem_CCI = (
        float(np.median(stem_cylinders[:, 4])),
        float(np.min(stem_cylinders[:, 4])),
        float(np.max(stem_cylinders[:, 4])),
    )
    stem_ang = stem_angle(stem_cylinders=stem_cylinders)
    compas_bearing = stem_bearing(stem_cylinders=stem_cylinders)

    # Estimate stem height
    stem_height = float(stem_toppoint[2] - stem_basepoint[2]) if isinstance(stem_basepoint, np.ndarray) else None

    breast_height = 1.3
    if isinstance(stem_basepoint, np.ndarray):
        dbh = diameter_at_breastheight(stem_cloud=stem_cloud, ground_level=float(stem_basepoint[2]))
    else:
        dbh = None

    # Breast-height fit (auxiliary visualization)
    if isinstance(stem_basepoint, np.ndarray):
        stem_pts = np.asarray(stem_cloud.points)
        breast_mask = np.where(stem_pts[:, 2] < stem_basepoint[2] + breast_height)[0]
        breast_cloud = stem_cloud.select_by_index(breast_mask)
        breast_pts = np.array(breast_cloud.points)
        if len(breast_pts) > 0:
            try:
                cyl_center, cyl_axis, cyl_radius = fit_vertical_cylinder_3D(xyz=breast_pts, th=0.05)[:3]
                breast_cylinder = trimesh.creation.cylinder(
                    radius=float(cyl_radius),
                    sections=20,
                    segment=(
                        cyl_center - cyl_axis * breast_height / 2,
                        cyl_center + cyl_axis * breast_height / 2,
                    ),
                ).as_open3d
                o3d.io.write_triangle_mesh(
                    os.path.join(outdirname, "breast_mesh.ply"),
                    breast_cylinder,
                    write_ascii=False,
                )
            except Exception:
                logger.exception("An error occurred when fitting cylinder 3D.")

    return (
        {
            "stem_height": stem_height,
            "diameter_at_breast_height": dbh,
            "stem_CCI (med, min, max)": stem_CCI,
            "stem_angle": stem_ang,
            "compas_bearing": compas_bearing,
            "stem_basepoint": tuple(stem_basepoint) if isinstance(stem_basepoint, np.ndarray) else None,
        },
        stem_mesh,
    )


def calculate_stem_crown_distance(
    stem_cloud: o3d.geometry.PointCloud,
    crown_cloud: o3d.geometry.PointCloud,
) -> float:
    """Horizontal distance between stem and crown centers.

    Args:
        stem_cloud: Stem point cloud.
        crown_cloud: Crown point cloud.

    Returns:
        Euclidean distance in the XY-plane between mean coordinates of the
        stem and crown.
    """
    stem_mean = np.asarray(stem_cloud.points).mean(axis=0)
    crown_mean = np.asarray(crown_cloud.points).mean(axis=0)
    return float(np.linalg.norm((stem_mean - crown_mean)[:2]))


def min_distance_to_tree_crown(cluster_points: np.ndarray, tree_points: np.ndarray) -> float:
    """Minimum distance between a small cluster and a reference cluster.

    Used to decide whether a small cluster should be assigned to the crown.

    Args:
        cluster_points: ``(N, 3)`` array of cluster points.
        tree_points: ``(M, 3)`` array for the reference tree cluster.

    Returns:
        Smallest nearest-neighbor distance between the two sets.
    """
    # Build a KDTree for efficient nearest neighbor search
    tree = cKDTree(tree_points)

    # Compute the minimal distance from each point in the small cluster to the tree cluster
    distances, _ = tree.query(cluster_points, k=1)
    min_dist: float = float(np.min(distances))

    return min_dist


def recover_crown_cloud(
    stem_cloud_filtered: o3d.geometry.PointCloud,
    crown_cloud: o3d.geometry.PointCloud,
    remaining_points: np.ndarray,
    outdirname: str,
    debug: bool,
    distance_threshold: float = 0.2,
) -> o3d.geometry.PointCloud:
    """Recover additional crown points from leftover stem-candidate points.

    Clusters ``remaining_points`` using DBSCAN and appends clusters that are
    closer to the crown than to the stem (and within a distance threshold).

    Args:
        stem_cloud_filtered: Filtered stem point cloud.
        crown_cloud: Existing crown point cloud (may be augmented).
        remaining_points: ``(N, 3)`` array of points to evaluate.
        outdirname: Directory for debug visualizations/exports.
        debug: If ``True``, write debug plots and point clouds.
        distance_threshold: Max distance for a cluster to be considered crown.

    Returns:
        Point cloud of additional crown points detected; may be empty.
    """
    MIN_CLUSTER_SIZE = 4

    # Initialize an empty point cloud to accumulate additional crown points
    additional_crown_cloud = o3d.geometry.PointCloud()

    if len(remaining_points) < MIN_CLUSTER_SIZE:
        logger.info("Too few points to recover crown cloud.")
        return additional_crown_cloud

    # Cluster the remaining points using DBSCAN
    clusterer = DBSCAN(eps=0.1, min_samples=MIN_CLUSTER_SIZE)  # tune eps for dataset
    cluster_labels_values = clusterer.fit_predict(remaining_points)

    # ---- Order clusters by MAX z (descending), ignoring label -1 (noise) ----
    z = remaining_points[:, 2]
    u, inv = np.unique(cluster_labels_values, return_inverse=True)  # unique labels and per-point group indices
    scores = np.full(u.shape, -np.inf)
    np.maximum.at(scores, inv, z)  # groupwise max in O(N)
    valid = u != -1
    ordered_labels = u[valid][np.argsort(-scores[valid])]  # descending by max z
    # -------------------------------------------------------------------------

    # Iterate over clusters in the desired order
    for label in ordered_labels:
        # Get the indices of points belonging to the current cluster
        cluster_indices = np.where(cluster_labels_values == label)[0]
        if cluster_indices.size == 0:
            continue
        cluster_points = remaining_points[cluster_indices]

        # Distances to crown and stem
        dist_to_crown = min_distance_to_tree_crown(
            cluster_points=cluster_points, tree_points=np.asarray(crown_cloud.points)
        )
        dist_to_stem = min_distance_to_tree_crown(
            cluster_points=cluster_points, tree_points=np.asarray(stem_cloud_filtered.points)
        )

        # Keep if closer to crown (or within threshold)
        if (dist_to_crown > dist_to_stem and dist_to_crown > 0.05) or (dist_to_crown > distance_threshold):
            continue

        # Append the cluster to crown
        cluster_point_cloud = o3d.geometry.PointCloud()
        cluster_point_cloud.points = o3d.utility.Vector3dVector(cluster_points)
        additional_crown_cloud += cluster_point_cloud
        crown_cloud += cluster_point_cloud

    if debug and len(additional_crown_cloud.points) > 0:
        additional_crown_mask = np.array(
            [True] * len(additional_crown_cloud.points) + [False] * len(stem_cloud_filtered.points)
        )
        pcfigname = os.path.join(outdirname, "additional_crown_pts.png")
        plot_pointcloud(
            pcd=(additional_crown_cloud + stem_cloud_filtered),
            outfname=pcfigname,
            mask=additional_crown_mask,
        )
        o3d.io.write_point_cloud(os.path.join(outdirname, "additional_crown.ply"), additional_crown_cloud)

    return additional_crown_cloud


def curate_crown(crown_cloud: o3d.geometry.PointCloud, outdirname: str, debug: bool) -> o3d.geometry.PointCloud:
    """Curate the crown point cloud by analyzing vertical slices and filtering out invalid segments.


    This function divides the crown point cloud into vertical height slices and
    evaluates each slice to determine whether it contains valid crown points.
    Invalid slices (e.g., noise or outliers) are filtered out to refine the final crown cloud.


    Args:
    crown_cloud: The full crown point cloud obtained after stem/crown separation.
    outdirname: The directory path where debug visualizations and intermediate results
        will be saved if debug mode is enabled.
    debug: Whether to enable debug mode, which produces visual diagnostics such as slice plots
        and validation overlays for inspection.


    Returns:
    A filtered ``open3d.geometry.PointCloud`` containing only valid crown points.
    """
    CROWN_HEIGHT_SEGMENT = 0.4
    crown_slices = divide_pointcloud_into_slices(pcd=crown_cloud, height_segment=CROWN_HEIGHT_SEGMENT)
    valid_crown_points, valid_crown_slices = curate_crown_cloud(
        pcd=crown_cloud, slices=crown_slices, height_segment=CROWN_HEIGHT_SEGMENT
    )
    crown_cloud_filtered = crown_cloud.select_by_index(np.where(valid_crown_points)[0])

    if debug:
        debug_stem_crown_slices(
            pointcloud=crown_cloud,
            slices=crown_slices,
            valid_slices=valid_crown_slices,
            valid_points=valid_crown_points,
            outdirname=outdirname,
            label="crown",
        )

    return crown_cloud_filtered


def curate_stem(
    stem_cloud: o3d.geometry.PointCloud, outdirname: str, debug: bool
) -> tuple[o3d.geometry.PointCloud, np.ndarray]:
    """Curate the stem point cloud by height-slice validation and filtering.


    The function divides the stem point cloud into vertical slices, analyzes them for geometric
    consistency, and retains only valid portions of the stem. Any remaining points identified as
    invalid are returned separately to allow possible reuse for crown recovery.


    Args:
    stem_cloud: The original stem point cloud obtained after segmentation.
    outdirname: The directory path for saving diagnostic visualizations and filtered outputs
        when debug mode is enabled.
    debug: Whether to enable debug mode, which triggers creation of slice-level visualizations
        to illustrate which parts of the stem were retained or rejected.


    Returns:
    A tuple ``(stem_cloud_filtered, remaining_points)`` where:


    - ``stem_cloud_filtered`` is an ``open3d.geometry.PointCloud`` containing only valid stem points.
    - ``remaining_points`` is a NumPy array of invalid or filtered-out points, typically used
    for recovering missed crown data.
    """
    STEM_HEIGHT_SEGMENT = 0.1
    stem_slices = divide_pointcloud_into_slices(pcd=stem_cloud, height_segment=STEM_HEIGHT_SEGMENT)
    valid_stem_points, valid_stem_slices = curate_stem_cloud(
        pcd=stem_cloud, slices=stem_slices, height_segment=STEM_HEIGHT_SEGMENT
    )

    if valid_stem_points is not None and valid_stem_points.any():
        stem_cloud_filtered = stem_cloud.select_by_index(np.where(valid_stem_points)[0])
        remaining_points = np.asarray(stem_cloud.select_by_index(np.where(~valid_stem_points)[0]).points)
    else:
        stem_cloud_filtered = o3d.geometry.PointCloud()  # empty
        remaining_points = np.array([])

    if debug:
        debug_stem_crown_slices(
            pointcloud=stem_cloud,
            slices=stem_slices,
            valid_slices=valid_stem_slices,
            valid_points=valid_stem_points,
            outdirname=outdirname,
            label="stem",
        )

    return stem_cloud_filtered, remaining_points


def separate_stem_and_crown(
    tree_cloud: o3d.geometry.PointCloud,
    adtree_exe: str,
    filter_leaves: str,
    outdirname: str,
    debug: bool,
) -> tuple[o3d.geometry.PointCloud | None, o3d.geometry.PointCloud | None]:
    """Separate stem and crown using AdTree.

    Args:
        tree_cloud: Full tree point cloud.
        adtree_exe: Path to AdTree binary.
        filter_leaves: Leaf filtering strategy.
        outdirname: Output directory for artifacts.
        debug: If ``True``, write intermediate files.

    Returns:
        ``(stem_cloud, crown_cloud)`` or ``(None, None)`` if either part is
        too small and processing should be skipped.
    """
    MIN_STEM_PTS = 25  # min number of points for a valid stem
    MIN_CROWN_PTS = 100  # min number of points for a valid crown

    stem_cloud, crown_cloud, _ = tree_separate(
        tree_cloud=tree_cloud,
        adtree_exe=adtree_exe,
        outdirname=outdirname,
        debug=debug,
        filter_leaves=filter_leaves,
    )
    if debug:
        write_stem_crown_cloud(
            stem_cloud=stem_cloud,
            crown_cloud=crown_cloud,
            outdirname=outdirname,
        )

    if len(stem_cloud.points) < MIN_STEM_PTS or len(crown_cloud.points) < MIN_CROWN_PTS:
        logger.error("Too few points for either stem or crown. Skipping.")
        return None, None

    return stem_cloud, crown_cloud


# ---------------------------------------------------------------------------
# Orchestration per tree & per tile
# ---------------------------------------------------------------------------


def process_point_cloud(
    pc_path: str,
    pc_ops_files: dict[str, str | list[str]],
    bgt_pavements_path: str,
    args: argparse.Namespace,
) -> dict[str, dict[str, str]] | None:
    """Process a single point cloud tile and its tree segments.

    Steps: load ops metadata, separate/curate crown & stem, mesh, compute
    metrics, write per-tree JSONs, combine into per-tile outputs, and compute
    tree–pavement intersections.

    Args:
        pc_path: Key/name for the tile (from metadata).
        pc_ops_files: Mapping with paths for tile and per-tree files.
        bgt_pavements_path: Path to BGT pavements for this tile (relative to ``args.bgt_dir``).
        args: Parsed CLI arguments.

    Returns:
        Mapping ``{tile_dir: {"tree_stats_json": path, "tree_stats_gpkg": path}}``
        or ``None`` when no trees were processed.
    """
    GROUND_POINTS_INDEX = 2
    logger.info("Processing point cloud: %s", pc_path)
    adtree_exe_path = Paths.get_adtree()

    tile_path = os.path.join(args.ops_dir, str(pc_ops_files["tile_path"]))
    tree_gpkg_path = os.path.join(args.ops_dir, str(pc_ops_files["tree_segments_path"]))

    las = laspy.read(tile_path)
    ground_cloud_las = las[las.classification == GROUND_POINTS_INDEX]

    tree_stat_files: list[str] = []
    tree_paths = pc_ops_files["tree_file_paths"]
    for tree_path in tree_paths:
        obsurv_tree = os.path.basename(os.path.dirname(tree_path))
        logger.info("Processing tree %s", obsurv_tree)
        outdirname = os.path.join(args.modeling_dir, os.path.dirname(tree_path))
        os.makedirs(outdirname, exist_ok=True)
        tree_stats_fname = os.path.join(outdirname, "tree_stats.json")
        if os.path.exists(tree_stats_fname) and not args.overwrite:
            logger.info("Tree %s already processed, skipping", obsurv_tree)
            tree_stat_files.append(tree_stats_fname)
            continue

        tree_cloud = o3d.io.read_point_cloud(os.path.join(args.ops_dir, tree_path))

        stem_cloud, crown_cloud = separate_stem_and_crown(
            tree_cloud=tree_cloud,
            adtree_exe=adtree_exe_path,
            filter_leaves="surface_variation",
            outdirname=outdirname,
            debug=args.debug,
        )

        if stem_cloud is None or crown_cloud is None:
            continue

        stem_cloud_filtered, remaining_points = curate_stem(
            stem_cloud=stem_cloud, outdirname=outdirname, debug=args.debug
        )

        if len(remaining_points) > 0:
            # Recover additional crown points from the remaining points
            additional_crown_cloud = recover_crown_cloud(
                stem_cloud_filtered=stem_cloud_filtered,
                crown_cloud=crown_cloud,
                remaining_points=remaining_points,
                outdirname=outdirname,
                debug=args.debug,
            )
            # Add the additional points to the crown
            crown_cloud += additional_crown_cloud

        crown_cloud_filtered = curate_crown(crown_cloud=crown_cloud, outdirname=outdirname, debug=args.debug)

        if (
            not stem_cloud_filtered
            or not crown_cloud_filtered
            or len(stem_cloud_filtered.points) == 0
            or len(crown_cloud_filtered.points) == 0
        ):
            logger.error("Preprocessing failed for file %s. Skipping.", tree_path)
            continue

        # Validate stem and crown
        stem_crown_dist = calculate_stem_crown_distance(
            stem_cloud=stem_cloud_filtered, crown_cloud=crown_cloud_filtered
        )

        # Determine ground points corresponding to the tree
        stem_basepoint, stem_toppoint, ref_height, cyl_axis = get_stem_endpoints(
            stem_cloud=stem_cloud_filtered, ground_cloud=ground_cloud_las, debug=args.debug, outdirname=outdirname
        )

        # Process the crown and stem data
        crown_data: CrownStats
        crown_data, crown_mesh = process_crown(
            crown_cloud=crown_cloud_filtered, stem_basepoint=stem_basepoint, outdirname=outdirname
        )
        stem_data, stem_mesh = process_stem(
            stem_cloud=stem_cloud_filtered,
            stem_basepoint=stem_basepoint,
            stem_toppoint=stem_toppoint,
            outdirname=outdirname,
        )

        # Combine and enrich tree data
        tree_data = {**crown_data, **stem_data}
        tree_data["pc_filename"] = tile_path

        # Name of the saved tree file from pc_ops is the obsurv_id
        obsurv_id = os.path.splitext(os.path.basename(tree_path))[0]
        tree_data["obsurv_id"] = obsurv_id

        tree_data["segment_id"], tree_data["stem_obsurv_distance"] = match_segment_id_to_tree(
            tree_data=tree_data, tree_gpkg_path=tree_gpkg_path
        )
        tree_data["stem_cyl_axis"] = list(cyl_axis)
        tree_data["stem_crown_dist"] = stem_crown_dist
        tree_data["ref_pavement_height (med, min, max)"] = ref_height

        if args.debug:
            plot_stem_crown(
                stem_cloud=stem_cloud_filtered,
                stem_mesh=stem_mesh,
                crown_cloud=crown_cloud_filtered,
                crown_mesh=crown_mesh,
                outdirname=outdirname,
            )

        # Write tree statistics to a JSON file
        tree_stat_files.append(tree_stats_fname)
        with open(tree_stats_fname, "w") as json_file:
            json.dump(tree_data, json_file)
        logger.info("Tree statistics written to %s.", tree_stats_fname)

    if len(tree_paths) > 0:
        modeling_out_dir = os.path.join(args.modeling_dir, os.path.dirname(str(pc_ops_files["tile_path"])))
        bgt_pavements_path = os.path.join(args.bgt_dir, bgt_pavements_path)
        logger.info("Combining tree stat files..")
        tree_stats_combined_fname = os.path.join(modeling_out_dir, "tree_stats_combined.json")
        combined_data = combine_json_files(
            tree_stat_filenames=tree_stat_files, output_file_path=tree_stats_combined_fname
        )
        logger.info("Finding trees in free pavement zones.")

        pavement_tree_intersections = create_pavement_zones_and_find_tree_intersections(
            pavement_input_path=bgt_pavements_path,
            input_file_path=tile_path,
            tree_stat_file_path=tree_stats_combined_fname,
            output_path=modeling_out_dir,
        )

        logger.info("Writing pavement-tree intersections to file.")
        intersect_outfname = os.path.join(modeling_out_dir, "pavement_tree_intersections.json")
        with open(intersect_outfname, "w") as json_file:
            json.dump(pavement_tree_intersections, json_file)

        out_gpkg_path = os.path.join(modeling_out_dir, "tree_stats_segments.gpkg")
        write_tree_stats_segments_gpkg(
            combined_data=combined_data,
            tree_gpkg_path=tree_gpkg_path,
            output_gpkg_path=out_gpkg_path,
            pavement_tree_intersections=pavement_tree_intersections,
            seg_col="segment_ids",
            overhang_dim="3D",
        )
        return {
            os.path.dirname(str(pc_ops_files["tile_path"])): {
                "tree_stats_json": tree_stats_combined_fname,
                "tree_stats_gpkg": out_gpkg_path,
            }
        }
    else:
        return None


def modeling_tree(args: argparse.Namespace) -> dict[str, dict[str, str]]:
    """Run the complete tree modeling pipeline for all processed tiles using multiprocessing.


    This function coordinates modeling for all input tiles listed in the provided metadata files.
    Each tile is processed in parallel using multiple worker processes. The results from all tiles
    are merged into a consolidated JSON output containing paths to per-tile artifacts.


    Args:
    args: The parsed command-line arguments specifying configuration and file paths. Must include:


    - ``ops_metadata``: Path to JSON metadata describing processed point cloud tiles and trees.
    - ``bgt_metadata``: Path to JSON metadata describing BGT pavement layers.
    - ``ops_dir``: Base directory containing processed operation files.
    - ``bgt_dir``: Base directory containing pavement data.
    - ``modeling_dir``: Directory for saving modeling outputs and reports.
    - ``num_workers``: Number of parallel processes to use (optional).
    - ``overwrite``: Whether to overwrite existing results.
    - ``debug``: Enable debug mode with additional plots and logs.


    Returns:
    A dictionary mapping each processed tile directory to another dictionary containing:


    - ``tree_stats_json``: Path to the combined per-tile tree statistics JSON file.
    - ``tree_stats_gpkg``: Path to the GeoPackage file containing spatially joined tree metrics.
    """
    logger.info("Loading processed files.")
    ops_metadata, bgt_metadata = load_processed_files(args=args)

    process_args = [(pc_path, tree_paths, bgt_metadata[pc_path], args) for pc_path, tree_paths in ops_metadata.items()]

    # Use multiprocessing to parallelize point cloud processing
    cpu_minus_one = max(1, multiprocessing.cpu_count() - 1)
    if args.num_workers is None:
        num_workers = cpu_minus_one
    elif 0 < args.num_workers <= cpu_minus_one:
        num_workers = args.num_workers
    else:
        logger.error(
            "Invalid num_workers arg: %s. \
            Defaulting to cpu_count - 1 (%s)",
            args.num_workers,
            cpu_minus_one,
        )
        num_workers = cpu_minus_one
    logger.info("Processing with %s parallel workers", num_workers)

    result_dict: dict[str, dict[str, str]] = {}
    with multiprocessing.Pool(processes=num_workers) as pool:
        results = list(
            tqdm(
                pool.imap_unordered(_process_point_cloud_star, process_args, chunksize=1),  # type: ignore[arg-type]
                total=len(process_args),
                desc="Processing point clouds",
            )
        )

    # Merge per-task dicts into a single mapping
    for r in results:
        if isinstance(r, dict) and r:
            # r is like {"<tile_dir>": {"tree_stats_json": "...", "tree_stats_gpkg": "..."}}
            result_dict.update(r)

    # Write results to a JSON file
    modeling_results_path = os.path.join(args.modeling_dir, "tree_modeling_results.json")
    os.makedirs(os.path.dirname(modeling_results_path), exist_ok=True)
    with open(modeling_results_path, "w", encoding="utf-8") as json_file:
        json.dump(result_dict, json_file, indent=4)

    logger.info("Results written to %s", modeling_results_path)
    return result_dict


def configure_arg_parser() -> argparse.Namespace:
    """Configure the argument parser for the CLI.

    Returns:
        Parsed ``argparse.Namespace`` containing all CLI options.
    """
    parser = argparse.ArgumentParser(description="Tree modeling")
    parser.add_argument(
        "--ops_metadata",
        type=str,
        help="Path to the JSON file containing pc operations files dictionary.",
    )
    parser.add_argument(
        "--bgt_metadata",
        type=str,
        help="Path to the JSON file containing BGT pavements files dictionary.",
    )
    parser.add_argument(
        "--ops_dir",
        type=str,
        help="Path to the mount location of the pc_ops files.",
    )
    parser.add_argument(
        "--bgt_dir",
        type=str,
        help="Path to the mount location of the BGT pavements files.",
    )
    parser.add_argument("--modeling_dir", type=str, help="Path to write the tree modeling results.")
    parser.add_argument(
        "--num_workers",
        type=int,
        help="Number of parallel workers to use.",
    )
    parser.add_argument(
        "--overwrite",
        type=lambda s: s.lower() == "true",
        default=False,
        help="Flag to overwrite output files if they already exist.",
    )
    parser.add_argument(
        "--debug",
        type=lambda s: s.lower() == "true",
        default=False,
        help="Specify whether to run debug mode.",
    )
    return parser.parse_args()


def main() -> None:
    """Entrypoint for running tree modeling from the command line."""
    args = configure_arg_parser()
    modeling_tree(args=args)


if __name__ == "__main__":
    multiprocessing.set_start_method("spawn", force=True)
    main()
