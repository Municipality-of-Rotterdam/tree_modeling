from __future__ import annotations

import os
import subprocess

import networkx as nx
import numpy as np
import open3d as o3d
from scipy.interpolate import interp1d
from scipy.spatial import KDTree

from tree_modeling.logger import logger
from tree_modeling.utils import o3d_utils
from .graph import filter_skeleton_nodes, read_ply, path_till_split, plot_graph_with_path
from .types import SkeletonData, Labels


def leafwood_classification(tree_cloud: o3d.geometry.PointCloud, method: str) -> np.ndarray:
    """
    Classify points in a tree point cloud as leaf or wood.

    Parameters
    ----------
    tree_cloud : o3d.geometry.PointCloud
        Input tree cloud.
    method : str
        'curvature' or 'surface_variation'.

    Returns
    -------
    np.ndarray
        Integer labels (Labels.LEAF / Labels.WOOD) per original point.
    """
    labels = np.full(len(tree_cloud.points), Labels.LEAF, dtype=int)

    # Downsample + trace
    pcd_down, _, trace = tree_cloud.voxel_down_sample_and_trace(
        0.02, tree_cloud.get_min_bound(), tree_cloud.get_max_bound()
    )

    # Outlier filter on downsampled
    pcd_inliers, ind_inliers = pcd_down.remove_statistical_outlier(nb_neighbors=16, std_ratio=2.0)
    ind_inliers = np.asarray(ind_inliers)

    # Classify on inlier cloud
    if method == "curvature":
        mask = o3d_utils.curvature_filter(pcd_inliers, 0.075, min1=20, min2=35)
    else:
        mask = o3d_utils.surface_variation_filter(pcd_inliers, 0.1, 0.15)

    # Map back to original indices
    chosen_down_idx = ind_inliers[mask] if mask is not None else np.array([], dtype=int)
    pieces = [np.asarray(trace[i]) for i in chosen_down_idx]
    if pieces:
        ind = np.concatenate(pieces)
        labels[ind] = Labels.WOOD

    return labels


def reconstruct_skeleton(
    tree_cloud: o3d.geometry.PointCloud,
    exe_path: str,
    outdirname: str,
    overwrite: bool = False,
    debug: bool = False,
) -> tuple[SkeletonData, str]:
    """
    Reconstruct a tree skeleton from an Open3D point cloud using adTree.

    Returns
    -------
    tuple[dict[str, nx.DiGraph | np.ndarray | None], str]
        Filtered skeleton dictionary and the output PLY path.
    """
    filename = os.path.basename(outdirname)

    # create input file system
    tmp_folder = "./tmp"
    in_file = os.path.join(tmp_folder, filename + ".xyz")
    out_file = os.path.join(outdirname, "skeleton.ply")
    os.makedirs(os.path.dirname(out_file), exist_ok=True)
    if not os.path.exists(tmp_folder):
        os.mkdir(tmp_folder)

    # Initialize defaults
    graph: nx.DiGraph | None = None
    vertices: np.ndarray = np.array([])
    edges: np.ndarray = np.array([])

    try:
        tree_cloud_sampled = tree_cloud.voxel_down_sample(0.02)
        o3d.io.write_point_cloud(in_file, tree_cloud_sampled)  # write input file
        if os.path.exists(out_file) and not overwrite:
            logger.info("Skeleton exists and overwrite=False. Reusing.")
        else:
            if os.path.exists(out_file):
                os.remove(out_file)
            result = subprocess.run([exe_path, in_file, out_file], capture_output=True, text=True, check=True)
            if debug and result.stdout:
                logger.info("adTree stdout:\n%s", result.stdout)
            if result.stderr:
                logger.info("adTree stderr:\n%s", result.stderr)

        # read output graph
        graph, vertices, edges = read_ply(out_file)

    except subprocess.CalledProcessError as error_msg:
        logger.error("Failed reconstructing tree:\n %s", error_msg.stderr)
    except Exception as error_msg:
        logger.error("Failed:\n %s", error_msg)

    # clean filesystem
    if os.path.exists(in_file):
        os.remove(in_file)

    skeleton: SkeletonData = {"graph": graph, "vertices": vertices, "edges": edges}
    if graph is not None:
        skeleton_filtered, _ = filter_skeleton_nodes(skeleton=skeleton, out_file=out_file, debug=debug)
    else:
        skeleton_filtered = skeleton

    return skeleton_filtered, out_file


def skeleton_split(
    tree_cloud: o3d.geometry.PointCloud, skeleton_graph: nx.DiGraph, outdirname: str | None = None
) -> np.ndarray | None:
    """
    Split the stem from the crown by following the reconstructed skeleton.

    Returns
    -------
    np.ndarray | None
        Boolean mask over `tree_cloud` points (True = stem), or None on failure.
    """
    if skeleton_graph is None or len(skeleton_graph.nodes) == 0:
        logger.error("Invalid skeleton graph.")
        return None

    # get start node and retrieve path
    z_values = nx.get_node_attributes(skeleton_graph, "z")
    start_node = min(z_values, key=z_values.get)
    path, visited_nodes = path_till_split(skeleton_graph, start_node)
    skeleton_pts = np.array([list(skeleton_graph.nodes[node].values()) for node in path])

    if len(skeleton_pts) < 2:
        logger.error("Not enough skeleton points to perform split.")
        return None

    if outdirname:
        out_file = os.path.join(outdirname, "graph_path.png")
        plot_graph_with_path(skeleton_graph, path, visited_nodes, out_file)

    # Filter cloud for stem points
    tree_points = np.array(tree_cloud.points)
    labels = np.zeros(len(tree_points), dtype=bool)
    mask_idx = np.where(tree_points[:, 2] < skeleton_pts[:, 2].max())[0]

    # Build KDTree over relevant points
    tree = KDTree(tree_points[mask_idx])
    selection: set[int] = set()

    # Dense interpolation along the skeleton path
    cumulative_distance = np.cumsum(np.linalg.norm(np.diff(skeleton_pts, axis=0), axis=1))
    cumulative_distance = np.insert(cumulative_distance, 0, 0)
    interp = interp1d(cumulative_distance, skeleton_pts, axis=0, kind="linear")

    num_points = max(2, int(cumulative_distance[-1] / 0.05))  # ~ one sample per 5 cm
    dense_skeleton_pts = interp(np.linspace(0, cumulative_distance[-1], num=num_points))
    skeleton_pts = dense_skeleton_pts

    for result in tree.query_ball_point(skeleton_pts, r=0.75):
        selection.update(result)
    selection = mask_idx[list(selection)]
    labels[selection] = True

    return labels
