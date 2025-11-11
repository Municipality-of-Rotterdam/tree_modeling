"""Script for determining the pruning needs."""

import argparse
import json
import os

import geopandas as gpd
import numpy as np
import laspy
import trimesh
from shapely.geometry import MultiPolygon, Point, Polygon
from trimesh.collision import CollisionManager

from tree_modeling.logger import logger
from tree_modeling.create_pavement_meshes import (
    create_mesh,
    create_vertices,
    export_mesh,
    get_coordinates,
    update_mesh,
)


def configure_arg_parser() -> argparse.ArgumentParser:
    """Configure the argument parser.

    Returns
    -------
    argparse.ArgumentParser
        Configured argument parser.
    """
    parser = argparse.ArgumentParser(description="Determine tree–pavement pruning needs.")
    parser.add_argument("--pavement_input", type=str, required=True, help="Path to the BGT pavements GeoPackage.")
    parser.add_argument(
        "--input_file",
        type=str,
        required=True,
        help="Path to the LAS/LAZ file with ground points and BGT attributes.",
    )
    parser.add_argument(
        "--tree_stat_file",
        type=str,
        required=True,
        help="Path to the combined per-tile tree statistics JSON (tree_stats_combined.json).",
    )
    parser.add_argument("--output_path", type=str, required=True, help="Directory to write outputs.")
    return parser


def check_mesh_intersection(crown_mesh_path: str, pavement_mesh_path: str) -> bool:
    """Check for intersection between a tree crown mesh and a pavement mesh.
    This function loads the specified crown and pavement meshes from their file paths,
    adds them to a collision manager, and checks if there is any intersection between them.


    Parameters
    ----------
    crown_mesh_path : str
        File path to the tree crown mesh (.ply).
    pavement_mesh_path : str
        File path to the pavement mesh (.ply).

    Returns
    -------
    bool
        True if meshes intersect; otherwise False.
    """
    if not os.path.exists(crown_mesh_path):
        logger.error("Cannot determine intersection: crown mesh not found at %s", crown_mesh_path)
        return False

    crown_mesh = trimesh.load(crown_mesh_path)
    pavement_mesh = trimesh.load(pavement_mesh_path)

    if not isinstance(crown_mesh, trimesh.Trimesh):
        logger.error("Crown mesh is not a valid Trimesh object: %s", crown_mesh_path)
        return False

    if not isinstance(pavement_mesh, trimesh.Trimesh):
        logger.error("Pavement mesh is not a valid Trimesh object: %s", pavement_mesh_path)
        return False

    collision_manager = CollisionManager()
    collision_manager.add_object("crown", crown_mesh)
    collision_manager.add_object("pavement", pavement_mesh)

    return bool(collision_manager.in_collision_internal())


def create_crown_circle(center: tuple[float, float] | list[float], diameter: float) -> Polygon:
    """Create a circular polygon representing the tree crown.

    Parameters
    ----------
    center : tuple[float, float] | list[float]
        (x, y) center of the crown.
    diameter : float
        Crown diameter.

    Returns
    -------
    shapely.geometry.Polygon
        Circular polygon approximating the crown footprint.
    """
    radius = diameter / 2.0
    return Point(center).buffer(radius)


def check_intersection(polygon: Polygon | MultiPolygon, tree_crowns: dict[str, Polygon]) -> list[str]:
    """Return obsurv_ids of crowns that intersect a given polygon.

    Parameters
    ----------
    polygon : Polygon | MultiPolygon
        Pavement geometry.
    tree_crowns : dict[str, Polygon]
        Mapping obsurv_id -> crown footprint polygon.

    Returns
    -------
    list[str]
        obsurv_ids whose crown footprints intersect the polygon.
    """
    intersecting: list[str] = []
    for obsurv_id, crown_poly in tree_crowns.items():
        if polygon.intersects(crown_poly):
            intersecting.append(obsurv_id)
    return intersecting


def create_pavement_zones_and_find_tree_intersections(
    pavement_input_path: str, input_file_path: str, tree_stat_file_path: str, output_path: str
) -> dict[str, dict[str, list[str]]]:
    """Create 3D pavement zones and find intersecting tree crowns (2D and 3D).
    This entails creating a 3D mesh (i.e., zone) per pavement that should be obstacle-free.
    After that, we check which crown meshes intersect with which pavement zones and write the results to a file.

    Workflow:
    1) Load pavements (BGT), ground LAS/LAZ, and combined per-tree stats.
    2) Build simple 2D crown circles from `crown_center` and `crown_diameter`.
    3) For each BGT polygon (by class), compute 2D intersections (polygon vs crown circles).
    4) If any 2D intersections exist, build a 3D pavement mesh and write it to disk.
    5) For each 2D-intersecting tree, test 3D collision (mesh vs crown mesh) and record hits.

    Parameters
    ----------
    pavement_input_path : str
        Path to BGT pavements GeoPackage.
    input_file_path : str
        Path to the LAS/LAZ file with ground points and BGT attributes.
    tree_stat_file_path : str
        Path to the combined per-tile tree stats JSON (tree_stats_combined.json).
    output_path : str
        Directory to write 'pavements/' and where tree meshes are expected under 'trees/<obsurv_id>/'.

    Returns
    -------
    dict[str, dict[str, list[str]]]
        Mapping of pavement mesh filename -> {"2D": [...], "3D": [...]} lists of obsurv_ids.
    """
    logger.info("Reading pavement and tree data.")
    pavement_gdf = gpd.read_file(pavement_input_path)
    input_laz = laspy.read(input_file_path)

    with open(tree_stat_file_path, "r", encoding="utf-8") as f:
        combined_tree_stats: dict[str, dict[str, object]] = json.load(f)

    # Build 2D crown footprints (skip entries missing center/diameter)
    tree_crowns: dict[str, Polygon] = {}
    for _, stats in combined_tree_stats.items():
        try:
            obsurv_id = str(stats["obsurv_id"])
            center = stats.get("crown_center")
            diameter: float = stats.get("crown_diameter")  # type: ignore[assignment]
            if center is None or diameter is None:
                continue
            cx, cy = float(center[0]), float(center[1])  # type: ignore[index]
            d = float(diameter)
            if not np.isfinite(d) or d <= 0:
                continue
            tree_crowns[obsurv_id] = create_crown_circle(center=(cx, cy), diameter=d)
        except Exception as exc:  # robust against malformed entries
            logger.debug("Skipping tree with invalid stats: %s (%s)", stats, exc)

    pavement_tree_intersections: dict[str, dict[str, list[str]]] = {}
    pavements_outdir = os.path.join(output_path, "pavements")
    os.makedirs(pavements_outdir, exist_ok=True)

    # Only process relevant BGT classes (1,2,3) like original implementation
    for class_id in [1, 2, 3]:
        logger.info("Processing pavements with class=%s.", class_id)
        for coordinates, polygon, instance in get_coordinates(gdf=pavement_gdf, classification=class_id):
            # 2D filter first (cheap)
            intersecting_crowns = check_intersection(polygon, tree_crowns)
            if not intersecting_crowns:
                continue

            outfname = os.path.join(pavements_outdir, f"{class_id}_{instance}.ply")
            pavement_tree_intersections[os.path.basename(outfname)] = {"2D": intersecting_crowns, "3D": []}

            # Build 3D pavement mesh
            vertices = create_vertices(
                coordinates=coordinates, class_id=class_id, instance=instance, input_laz=input_laz
            )
            if vertices is None:
                logger.error(
                    "Could not determine vertices for class=%s, instance=%s. Removing pavement from intersections.",
                    class_id,
                    instance,
                )
                del pavement_tree_intersections[os.path.basename(outfname)]
                continue

            mesh = create_mesh(vertices)
            if mesh is None:
                logger.error(
                    "Invalid mesh for class=%s, instance=%s. Removing pavement from intersections.", class_id, instance
                )
                del pavement_tree_intersections[os.path.basename(outfname)]
                continue

            mesh = update_mesh(mesh, polygon)
            logger.info("Writing pavement mesh %s", outfname)
            export_mesh(mesh, outfname)

    # 3D collision checks (only for 2D-positive pairs)
    for obsurv_id in tree_crowns:
        for pavement_mesh_name, match_dict in pavement_tree_intersections.items():
            if obsurv_id not in match_dict["2D"]:
                continue
            full_pavement_path = os.path.join(pavements_outdir, pavement_mesh_name)
            crown_mesh_path = os.path.join(output_path, "trees", obsurv_id, "crown_mesh.ply")
            if check_mesh_intersection(crown_mesh_path=crown_mesh_path, pavement_mesh_path=full_pavement_path):
                logger.info("3D intersection found between %s and %s", crown_mesh_path, pavement_mesh_name)
                match_dict["3D"].append(obsurv_id)

    return pavement_tree_intersections


def main() -> None:
    """Entry point."""
    parser = configure_arg_parser()
    args = parser.parse_args()

    pavement_tree_intersections = create_pavement_zones_and_find_tree_intersections(
        pavement_input_path=args.pavement_input,
        input_file_path=args.input_file,
        tree_stat_file_path=args.tree_stat_file,
        output_path=args.output_path,
    )

    logger.info("Writing pavement-tree intersections to file.")
    outfname = os.path.join(args.output_path, "pavement_tree_intersections.json")
    with open(outfname, "w", encoding="utf-8") as f:
        json.dump(pavement_tree_intersections, f, indent=2)


if __name__ == "__main__":
    main()
