"""Helper functions for tree modeling utilities.

This module provides small utilities used across the tree modeling pipeline,
including metadata I/O, plotting helpers, GeoPackage writing, and small
geometry operations. Logic is preserved; updates focus on docstrings, type
hints (native unions), and readability.
"""

from __future__ import annotations

import argparse
import json
import os
from collections import defaultdict

import geopandas as gpd
import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from shapely.geometry import box, Point, Polygon
import networkx as nx
import pandas as pd

from tree_modeling.logger import logger


def _xy(G: nx.Graph, n: int) -> tuple[float, float]:
    """Coordinates (x, y) for a node from attributes.

    Expects each node to carry either ``x`` and ``y`` attributes.

    Args:
        G: Graph whose nodes have ``x`` and ``y`` attributes.
        n: Node identifier.

    Returns:
        The node's ``(x, y)`` coordinates as floats.

    Raises:
        ValueError: If the node is missing ``x`` and ``y`` attributes.
    """
    nd = G.nodes[n]
    if "x" in nd and "y" in nd:
        return float(nd["x"]), float(nd["y"])
    raise ValueError(f"Node {n} missing 'x','y' attributes.")


def max_successor_distance(G: nx.Graph, successors: list[int]) -> float:
    """Maximum pairwise Euclidean distance among successors in (x, y).

    Pure NumPy implementation used as a fallback when
    only node attributes are available.

    Args:
        G: Graph with nodes carrying ``x`` and ``y`` coordinates.
        successors: Node IDs whose spatial spread will be measured.

    Returns:
        The maximum pairwise distance among the provided nodes. Returns
        ``0.0`` when ``successors`` has length ``0`` or ``1``.
    """
    if len(successors) < 2:
        return 0.0
    pts = np.array([_xy(G, n) for n in successors], dtype=float)
    diffs = pts[:, None, :] - pts[None, :, :]
    dists = np.hypot(diffs[..., 0], diffs[..., 1])
    return float(np.max(np.triu(dists, k=1)))


def load_processed_files(
    args: argparse.Namespace,
) -> tuple[dict[str, dict[str, str | list[str]]], dict[str, dict[str, str]]]:
    """Load preprocessed metadata JSON files for ops and BGT pavements.

    Args:
        args: Parsed CLI arguments. Must provide ``args.ops_metadata`` and
            ``args.bgt_metadata`` pointing to JSON files.

    Returns:
        A tuple ``(ops_metadata, bgt_metadata)`` where:

        - ``ops_metadata`` maps tile keys to a dict containing paths for
          ``"tile_path"``, ``"tree_segments_path"``, and ``"tree_file_paths"``
          (the last being a list of per-tree files).
        - ``bgt_metadata`` maps tile keys to a dict with pavement file paths.
    """
    logger.info("Loading JSON files.")
    with open(args.ops_metadata) as json_file:
        ops_metadata = json.load(json_file)

    with open(args.bgt_metadata) as json_file:
        bgt_metadata = json.load(json_file)

    return ops_metadata, bgt_metadata


def combine_json_files(
    tree_stat_filenames: list[str], output_file_path: str
) -> dict[str, dict[str, float | str | np.ndarray]]:
    """Combine multiple per-tree JSONs into a single mapping and write it.

    Each input file's contents are stored under its filename key.

    Args:
        tree_stat_filenames: List of per-tree JSON file paths to combine.
        output_file_path: Destination path for the combined JSON.

    Returns:
        The combined mapping that was written to ``output_file_path``.
    """
    combined_data: dict[str, dict[str, float | str | np.ndarray]] = {}

    for json_file in tree_stat_filenames:
        if json_file is not None:
            with open(json_file, "r") as file:
                data = json.load(file)
                combined_data[json_file] = data

    # Write the combined data to the output file
    with open(output_file_path, "w") as output_file:
        json.dump(combined_data, output_file)
    return combined_data


def write_stem_crown_cloud(
    stem_cloud: o3d.geometry.PointCloud,
    crown_cloud: o3d.geometry.PointCloud,
    outdirname: str,
) -> None:
    """Write stem and crown point clouds to PLY for debugging.

    Args:
        stem_cloud: Stem point cloud to be written when non-empty.
        crown_cloud: Crown point cloud to be written when non-empty.
        outdirname: Directory where ``stem.ply`` and ``crown.ply`` will be saved.
    """
    if len(stem_cloud.points) > 0:
        stem_name = os.path.join(outdirname, "stem.ply")
        o3d.io.write_point_cloud(stem_name, stem_cloud)

    if len(crown_cloud.points) > 0:
        crown_name = os.path.join(outdirname, "crown.ply")
        o3d.io.write_point_cloud(crown_name, crown_cloud)


def _invert_pavement_index(
    pavement_tree_intersections: dict[str, dict[str, list[str]]], dim: str = "3D"
) -> dict[str, list[str]]:
    """Create ``obsurv_id -> [pavement names]`` for a given dimension.

    Input example::

        {"1_1.ply": {"2D": ["188637"], "3D": ["188637"]}, ...}

    Args:
        pavement_tree_intersections: Mapping from pavement name to a dict
            containing keys ``"2D"`` and/or ``"3D"`` listing obsurv IDs.
        dim: Which dimension key (``"2D"`` or ``"3D"``) to use.

    Returns:
        A mapping from ``obsurv_id`` to a sorted, de-duplicated list of
        pavement names where the tree overlaps in the chosen dimension.
    """
    idx: dict[str, list[str]] = defaultdict(list)
    for pavement, dims in pavement_tree_intersections.items():
        for obs_id in map(str, dims.get(dim, []) or []):
            idx[obs_id].append(pavement)
    # de-dup + sort for determinism
    return {k: sorted(set(v)) for k, v in idx.items()}


def write_tree_stats_segments_gpkg(
    combined_data: dict[str, dict[str, float | str | np.ndarray]],
    tree_gpkg_path: str,
    output_gpkg_path: str,
    pavement_tree_intersections: dict[str, dict[str, list[str]]],
    seg_col: str = "segment_ids",
    overhang_dim: str = "3D",
) -> str | None:
    """Join modeled tree metrics to segment polygons and write a GPKG.

    Matches each tree entry in ``combined_data`` to a polygon in
    ``tree_gpkg_path`` using string equality on the ``seg_col`` (e.g.,
    ``"(1,)"``). Adds an ``overhangs_pavement`` column derived from
    ``pavement_tree_intersections`` for the specified dimension.

    Args:
        combined_data: Mapping of tree identifiers to metric dictionaries (from
            ``tree_stats_combined.json``).
        tree_gpkg_path: Path to the GeoPackage containing segment polygons.
        output_gpkg_path: Destination path for the merged GeoPackage.
        pavement_tree_intersections: Pavement-tree overlaps in the format::

            {
                "<pavement_name>": {
                    "2D": ["obsurv_id1", ...],
                    "3D": ["obsurv_id1", ...]
                },
                ...
            }

        seg_col: Column in the GPKG that stores the segment identifiers.
        overhang_dim: Dimension key (``"2D"`` or ``"3D"``) to use when
            populating ``overhangs_pavement``.

    Returns:
        The path to the written GeoPackage, or ``None`` if nothing matched.
    """

    seg_gdf = gpd.read_file(tree_gpkg_path)
    if seg_col not in seg_gdf.columns:
        raise KeyError(f"Expected column '{seg_col}' in {tree_gpkg_path}")

    # Make sure we compare as strings
    seg_gdf[seg_col] = seg_gdf[seg_col].astype(str)

    # segment_ids string -> row index
    seg_index = {str(v): i for i, v in enumerate(seg_gdf[seg_col])}

    # Build obsurv_id -> [pavements] for chosen dimension (3D by default)
    overhang_idx = _invert_pavement_index(pavement_tree_intersections, dim=overhang_dim)

    rows: list[dict[str, float | str | np.ndarray]] = []
    geoms: list[Polygon] = []
    total = matched = 0

    for tree_key, metrics in combined_data.items():
        total += 1
        if not isinstance(metrics, dict):
            continue

        seg_id = str(metrics.get("segment_id", "")).strip()
        if not seg_id:
            continue

        row_idx = seg_index.get(seg_id)
        if row_idx is None:
            logger.debug("No segment match for tree_id=%s (segment_id=%r)", tree_key, seg_id)
            continue

        matched += 1
        geom = seg_gdf.geometry.iloc[row_idx]

        obs_id = str(metrics.get("obsurv_id", "")).strip()
        pavements_for_obs = overhang_idx.get(obs_id, [])

        rows.append(
            {
                "tree_id": str(tree_key),
                "overhangs_pavement": pavements_for_obs,
                **metrics,
            }
        )
        geoms.append(geom)

    if not rows:
        logger.warning("No trees matched segments (0/%d). Not writing GPKG.", total)
        return None

    out_df = pd.DataFrame(rows)
    out_gdf = gpd.GeoDataFrame(out_df, geometry=geoms, crs=seg_gdf.crs)

    os.makedirs(os.path.dirname(output_gpkg_path), exist_ok=True)

    out_gdf.to_file(output_gpkg_path, driver="GPKG")
    logger.info(
        "Wrote %d matched trees (of %d) to %s with 'overhangs_pavement' (%s overlaps).",
        matched,
        total,
        output_gpkg_path,
        overhang_dim,
    )
    return output_gpkg_path


def match_segment_id_to_tree(
    tree_data: dict[str, object],
    tree_gpkg_path: str,
    max_valid_distance: float = 2.5,
) -> tuple[str | None, float | None]:
    """Match a segment ID to a modeled tree based on nearest geometry.

    The tree's stem basepoint (``x, y, z``) is compared to geometries in the
    GeoPackage, and the closest segment within ``max_valid_distance`` is
    selected.

    Args:
        tree_data: Dictionary containing attributes for a modeled tree. Must
            include ``"stem_basepoint"`` as a tuple ``(x, y, z)`` or ``None``.
        tree_gpkg_path: Path to the GeoPackage with tree segment geometries.
        max_valid_distance: Maximum allowed distance (in meters) for a valid
            match.

    Returns:
        A tuple ``(segment_id, distance_m)`` where values are ``None`` if no
        valid match is found.

    Note:
        Segment uniqueness (assign-once) is not enforced yet.
    """
    if tree_data["stem_basepoint"] is None:
        return None, None
    tree_gdf = gpd.read_file(tree_gpkg_path)
    stem_location = Point(tree_data["stem_basepoint"][0:2])  # type: ignore[index]
    distances = tree_gdf.geometry.distance(stem_location)
    if distances.min() > max_valid_distance:
        logger.error(
            "Modeled tree is too far away from known trees. Distance = %s m",
            distances.min(),
        )
        return None, None

    if "ID" in tree_gdf.columns:
        matching_id = tree_gdf.loc[distances.argmin(), "ID"]
    elif "segment_id" in tree_gdf.columns:
        matching_id = tree_gdf.loc[distances.argmin(), "segment_id"]
    elif "segment_ids" in tree_gdf.columns:
        matching_id = tree_gdf.loc[distances.argmin(), "segment_ids"]
    else:
        logger.error("Neither 'ID' nor 'segment_id' columns found in %s", tree_gpkg_path)
        return None, None

    return matching_id, float(distances.min())


def divide_pointcloud_into_slices(
    pcd: o3d.geometry.PointCloud, height_segment: float = 0.5
) -> list[tuple[np.ndarray, Polygon | None, float]]:
    """Divide a point cloud into horizontal slices by height interval.

    Args:
        pcd: The input point cloud to slice.
        height_segment: Vertical step in meters for slicing (default ``0.5``).

    Returns:
        A list of tuples ``(slice_points, bbox, z_min)`` for each slice where:

        - ``slice_points`` is a ``(N, 3)`` NumPy array of points within the
          slice's height range.
        - ``bbox`` is the 2D bounding box (Shapely ``Polygon``) in XY of the
          slice, or ``None`` if the slice is empty.
        - ``z_min`` is the lower Z bound for the slice.
    """
    points = np.asarray(pcd.points)

    # Determine the minimum and maximum Z values in the point cloud
    min_z = np.min(points[:, 2])
    max_z = np.max(points[:, 2])

    # Calculate the number of slices needed
    num_slices = int(np.ceil((max_z - min_z) / height_segment))

    # Create a list to store the slices with bbox
    slices_with_bbox: list[tuple[np.ndarray, Polygon | None, float]] = []

    # Divide the point cloud into slices
    for i in range(num_slices):
        z_min = float(min_z + i * height_segment)
        z_max = float(min_z + (i + 1) * height_segment)
        mask = (points[:, 2] >= z_min) & (points[:, 2] < z_max)
        slice_points = points[mask]

        # Calculate the bounding box for the slice points
        if len(slice_points) > 0:
            x_min, y_min = np.min(slice_points[:, :2], axis=0)
            x_max, y_max = np.max(slice_points[:, :2], axis=0)
            bbox: Polygon | None = box(x_min, y_min, x_max, y_max)
        else:
            # If there are no points in the slice, create an empty bounding box
            bbox = None

        slices_with_bbox.append((slice_points, bbox, z_min))

    return slices_with_bbox


def debug_stem_crown_slices(
    pointcloud: o3d.geometry.PointCloud,
    slices: list[tuple[np.ndarray, Polygon | None, float]],
    valid_slices: list[bool] | None,
    valid_points: np.ndarray | None,
    outdirname: str,
    label: str,
) -> None:
    """Generate slice and point-cloud debug plots for stem/crown curation.

    Args:
        pointcloud: The original point cloud (stem or crown).
        slices: Output of :func:`divide_pointcloud_into_slices`.
        valid_slices: Boolean-like list marking valid slices; if ``None``, all
            slices are considered invalid.
        valid_points: Boolean mask per point indicating validity; if ``None``,
            all points are considered invalid.
        outdirname: Directory to write debug images.
        label: Label prefix for output filenames (e.g., ``"stem"`` or
            ``"crown"``).
    """
    slicefigname = os.path.join(outdirname, f"{label}_slices.png")
    pcfigname = os.path.join(outdirname, f"{label}_filter.png")
    if valid_slices is None:
        valid_slices = [False for _ in slices]
        valid_points = np.array([False] * len(pointcloud.points))
    plot_slices(slices=slices, validity=valid_slices, outfname=slicefigname)
    plot_pointcloud(pcd=pointcloud, outfname=pcfigname, mask=valid_points)


def plot_slices(
    slices: list[tuple[np.ndarray, Polygon | None, float]],
    validity: list[bool],
    outfname: str,
    highlight_segment: int | None = None,
) -> None:
    """Plot 3D rectangle outlines of slice bounding boxes by Z height.

    Optionally color-code by validity and highlight a specific slice with a
    vertical dotted line.

    Args:
        slices: Sequence of tuples ``(points, bbox, z_min)`` as returned by
            :func:`divide_pointcloud_into_slices`.
        validity: List marking each slice as valid (``True``) or invalid
            (``False``). Non-boolean truthy values are also accepted.
        outfname: Path where the rendered plot image will be saved.
        highlight_segment: Index of a slice to highlight with a dotted line.
    """
    # Plotting the polygons in a 3D plot
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")

    for i, ((_, bbox, height), valid) in enumerate(zip(slices, validity)):
        if bbox is None:
            continue
        # Get the coordinates of the polygon
        coords = np.array(bbox.exterior.coords[:])
        x, y = coords[:, 0], coords[:, 1]
        z = np.full(len(coords), height)

        # Plot each rectangle as a series of line segments (quads)
        color = "b" if valid else "r"
        ax.plot([x[0], x[1]], [y[0], y[1]], z[0:2], color=color)
        ax.plot([x[1], x[2]], [y[1], y[2]], z[0:2], color=color)
        ax.plot([x[2], x[3]], [y[2], y[3]], z[0:2], color=color)
        ax.plot([x[3], x[0]], [y[3], y[0]], z[0:2], color=color)

        # Add a dotted line along the y-axis if this is the segment to highlight
        if highlight_segment is not None and i == highlight_segment:
            y_min, y_max = ax.get_ylim()
            ax.plot(
                [x.mean()] * 2,
                [y_min - 0.5, y_max + 0.5],
                [height, height],
                linestyle="dotted",
                color="k",
                linewidth=1,
            )

    # Axis labels and view
    ax.set_xlabel("X-axis")
    ax.set_ylabel("Y-axis")
    ax.set_zlabel("Z-axis")

    ax.view_init(0, 0)
    plt.title("3D Plot of Rectangles with Z-values")
    plt.savefig(outfname)
    plt.close()


def plot_pointcloud(pcd: o3d.geometry.PointCloud, outfname: str, mask: np.ndarray | None = None) -> None:
    """Render a simple 3D scatter of a point cloud and save to file.

    Args:
        pcd: The point cloud to render.
        outfname: Destination image filename.
        mask: Optional boolean mask marking valid/invalid points. Valid points
            are colored green; invalid points are red. If ``None``, all points
            are rendered in blue.
    """
    # Extract data
    points = np.asarray(pcd.points)

    if mask is not None:
        colors = np.where(mask, "green", "red")
    else:
        colors = np.array(["blue" for _ in range(len(points))])

    # Plotting the point cloud
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(points[:, 0], points[:, 1], points[:, 2], c=colors, s=0.5)

    # Set labels and title
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_title("Point Cloud Visualization")

    # Set the view axis to be horizontal
    ax.view_init(15, 45)

    plt.savefig(outfname)
    plt.close()


def plot_stem_crown(
    stem_cloud: o3d.geometry.PointCloud,
    stem_mesh: o3d.geometry.TriangleMesh | None,
    crown_cloud: o3d.geometry.PointCloud,
    crown_mesh: o3d.geometry.TriangleMesh | None,
    outdirname: str,
) -> None:
    """Plot combined stem+crown point clouds and their meshes to images.

    Args:
        stem_cloud: Point cloud of the stem.
        stem_mesh: Triangle mesh representing the stem (may be ``None``).
        crown_cloud: Point cloud of the crown.
        crown_mesh: Triangle mesh representing the crown (may be ``None``).
        outdirname: Directory where the output images will be saved.
    """
    # Create a combined point cloud from stem and crown clouds
    stem_points = np.asarray(stem_cloud.points)
    crown_points = np.asarray(crown_cloud.points)
    stem_crown_cloud = o3d.geometry.PointCloud()
    stem_crown_cloud.points = o3d.utility.Vector3dVector(np.concatenate([stem_points, crown_points]))

    # Generate filenames for output images
    pcfigname = os.path.join(outdirname, "stem_crown_separation.png")
    meshfigname = os.path.join(outdirname, "stem_crown_mesh.png")

    # Create a mask to distinguish stem and crown points in the combined point cloud
    stem_crown_mask = [False] * len(stem_points) + [True] * len(crown_points)

    # Plot the combined point cloud with mask applied
    plot_pointcloud(pcd=stem_crown_cloud, outfname=pcfigname, mask=np.array(stem_crown_mask, dtype=bool))

    # Plot the mesh images
    plot_meshes(stem_mesh=stem_mesh, crown_mesh=crown_mesh, outfname=meshfigname)


def plot_meshes(
    stem_mesh: o3d.geometry.TriangleMesh | None,
    crown_mesh: o3d.geometry.TriangleMesh | None,
    outfname: str,
) -> None:
    """Plot stem and crown meshes in a single 3D figure and save to file.

    Args:
        stem_mesh: Stem mesh to plot (may be ``None``).
        crown_mesh: Crown mesh to plot (may be ``None``).
        outfname: Destination image filename.
    """
    # Create a figure and a 3D axis
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection="3d")

    if stem_mesh is not None:
        # Plot stem_mesh in brown
        stem_vertices = np.asarray(stem_mesh.vertices)
        stem_faces = (
            np.asarray(stem_mesh.triangles)
            if isinstance(stem_mesh, o3d.geometry.TriangleMesh)
            else np.asarray(stem_mesh.faces)
        )
        stem_poly3d = Poly3DCollection(
            stem_vertices[stem_faces], color="saddlebrown", alpha=0.3, edgecolor="brown", linewidth=0.5
        )
        ax.add_collection3d(stem_poly3d)
        ax.scatter(stem_vertices[:, 0], stem_vertices[:, 1], stem_vertices[:, 2], color="saddlebrown", s=10)

    if crown_mesh is not None:
        # Plot crown_mesh in green
        crown_vertices = np.asarray(crown_mesh.vertices)
        crown_faces = np.asarray(crown_mesh.triangles)
        crown_poly3d = Poly3DCollection(
            crown_vertices[crown_faces], color="forestgreen", alpha=0.3, edgecolor="green", linewidth=0.5
        )
        ax.add_collection3d(crown_poly3d)
        ax.scatter(crown_vertices[:, 0], crown_vertices[:, 1], crown_vertices[:, 2], color="forestgreen", s=10)
    else:
        logger.error("Cannot plot meshes due to invalid crown mesh. Returning.")
        return

    # Set axis properties
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_title("3D Plot of stem and crown mesh")

    # Set the limits for better visualization
    if stem_mesh is not None:
        all_vertices = np.vstack((stem_vertices, crown_vertices))
    else:
        all_vertices = crown_vertices
        ax.set_title("3D Plot of crown mesh")
    ax.set_xlim([all_vertices[:, 0].min(), all_vertices[:, 0].max()])
    ax.set_ylim([all_vertices[:, 1].min(), all_vertices[:, 1].max()])
    ax.set_zlim([all_vertices[:, 2].min(), all_vertices[:, 2].max()])

    ax.view_init(15, 45)

    plt.savefig(outfname)
    plt.close()
