"""Creating pavement meshes."""

import argparse
import os
from collections.abc import Generator

import geopandas as gpd
import laspy
import numpy as np
import trimesh
from scipy.interpolate import LinearNDInterpolator
from scipy.spatial import Delaunay
from shapely.geometry import MultiPolygon, Point, Polygon

from tree_modeling.logger import logger

branch_free_height: dict[int, float] = {
    1: 2.5,
    2: 2.5,
    3: 4.5,
    4: 6.5,  # extra vrije doorgang
}
NEGATIVE_BUFFER: float = -0.0001  # Polygon buffer to create exterior ring
GROUND_POINTS_INDEX = 2


def get_coordinates(
    gdf: gpd.GeoDataFrame, classification: int
) -> Generator[tuple[list[tuple[float, float]], Polygon | MultiPolygon, int | str], None, None]:
    """
    Extract border coordinates for each BGT polygon of a given classification.

    Parameters
    ----------
    gdf : geopandas.GeoDataFrame
        Input GeoDataFrame with columns: 'classification', 'instance', and Polygon/MultiPolygon geometries.
    classification : int
        Asset class identifier to filter rows (e.g., pavement class).

    Yields
    ------
    tuple[list[tuple[float, float]], Polygon | MultiPolygon, int | str]
        - A flat list of (x, y) coordinates along the (slightly contracted) exterior ring(s),
        - The original polygon geometry (Polygon or MultiPolygon),
        - The 'instance' identifier (int or string).
    """
    for _, row in gdf[gdf["classification"] == classification].iterrows():
        all_coordinates: list[tuple[float, float]] = []
        bgt_geometry = row.geometry
        if isinstance(bgt_geometry, MultiPolygon):
            for polygon in bgt_geometry.geoms:
                exterior_ring = polygon.buffer(NEGATIVE_BUFFER).exterior
                all_coordinates.extend(list(exterior_ring.coords))
        elif isinstance(bgt_geometry, Polygon):
            exterior_ring = bgt_geometry.buffer(NEGATIVE_BUFFER).exterior
            all_coordinates.extend(list(exterior_ring.coords))
        yield all_coordinates, bgt_geometry, row.instance


def create_vertices(
    coordinates: list[tuple[float, float]],
    class_id: int,
    instance: int | str,
    input_laz: laspy.lasdata.LasData,
    min_points: int = 250,
) -> list[tuple[float, float, float]] | None:
    """
    Lift 2D boundary coordinates to 3D by interpolating Z from classified ground points.

    For each (x, y) on the polygon border, two vertices are generated:
    one at ground Z and one at ground Z + branch_free_height[class_id].

    Parameters
    ----------
    coordinates : list[tuple[float, float]]
        Exterior border coordinates (x, y) of the polygon.
    class_id : int
        Asset class id (used to choose the free height offset).
    instance : int | str
        Asset instance identifier in the LAS attributes.
    input_laz : laspy.lasdata.LasData
        LAS source containing ground points with attributes 'classification', 'AssetType', 'AssetInstance'.
    min_points : int, default=250
        Minimum required points in the subset to build a stable interpolator.

    Returns
    -------
    list[tuple[float, float, float]] | None
        3D vertices as (x, y, z) if enough points are available, otherwise None.
    """
    pavement_pc = input_laz[
        (input_laz.classification == GROUND_POINTS_INDEX)
        & (input_laz.AssetType == class_id)
        & (input_laz.AssetInstance == instance)
    ]
    if len(pavement_pc.points) < min_points:
        logger.info(
            "Not enough pointcloud points in current BGT pavement instance. Num points = %s. Returning.",
            len(pavement_pc.points),
        )
        return None

    fill_value = float(np.median(np.asarray(pavement_pc.z)))
    lin_interp = create_interpolator_from_pc(pc=pavement_pc, fill_value=fill_value)

    verts: list[tuple[float, float, float]] = []
    free_h = branch_free_height.get(class_id, 0.0)
    for x, y in coordinates:
        z0 = float(lin_interp(x, y))
        verts.append((x, y, z0))
        verts.append((x, y, z0 + free_h))
    return verts


def create_interpolator_from_pc(pc: laspy.lasdata.LasData, fill_value: float) -> LinearNDInterpolator:
    """
    Create a LinearNDInterpolator (z = f(x, y)) from LAS ground points via Delaunay.

    Parameters
    ----------
    pc : laspy.lasdata.LasData
        LAS subset with x, y, z arrays.
    fill_value : float
        Value used when (x, y) lies outside the convex hull.

    Returns
    -------
    scipy.interpolate.LinearNDInterpolator
        Interpolator such that interp(x, y) -> z (float).
    """
    tri = Delaunay(np.stack((pc.x, pc.y), axis=-1))
    return LinearNDInterpolator(points=tri, values=pc.z, fill_value=fill_value)


def create_mesh(vertices: list[tuple[float, float, float]]) -> trimesh.Trimesh | None:
    """
    Triangulate the lifted boundary vertices into a surface mesh.

    Parameters
    ----------
    vertices : list[tuple[float, float, float]]
        3D vertices along the polygon border(s).

    Returns
    -------
    trimesh.Trimesh | None
        Triangulated mesh or None if Delaunay fails.
    """
    try:
        triangulation = Delaunay(vertices)
    except Exception:
        logger.exception("An error occured during Delaunay triangulation.")
        return None
    return trimesh.Trimesh(vertices, triangulation.simplices)


def update_mesh(mesh: trimesh.Trimesh, polygon: Polygon | MultiPolygon) -> trimesh.Trimesh:
    """
    Mask faces whose triangle centroids fall outside the target polygon.

    Parameters
    ----------
    mesh : trimesh.Trimesh
        Input mesh produced from lifted border vertices.
    polygon : Polygon | MultiPolygon
        Area of interest; a small positive buffer is applied for robustness.

    Returns
    -------
    trimesh.Trimesh
        Mesh with invalid faces removed (in-place update and returned).
    """
    gs = gpd.GeoSeries([Point(*xy) for xy in mesh.triangles_center[:, :2]])
    valid = gs.within(gpd.GeoSeries([polygon]).buffer(0.1).iloc[0])
    mesh.update_faces(mask=valid.to_numpy())
    return mesh


def export_mesh(mesh: trimesh.Trimesh, file_path: str) -> None:
    """
    Export the mesh to disk.

    Parameters
    ----------
    mesh : trimesh.Trimesh
        Mesh to export.
    file_path : str
        Output path (extension determines format, e.g., .ply).
    """
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    mesh.export(file_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create free-height pavement meshes per BGT instance.")
    parser.add_argument("--bgt_gpkg", type=str, required=True, help="Path to BGT pavements GeoPackage.")
    parser.add_argument(
        "--ground_laz",
        type=str,
        required=True,
        help="Path to point cloud (.las/.laz) with ground points and BGT attributes.",
    )
    parser.add_argument("--output_path", type=str, required=True, help="Directory to write meshes under 'pavements/'.")
    args = parser.parse_args()

    logger.info("Reading BGT GeoPackage.")
    gdf = gpd.read_file(args.bgt_gpkg)

    logger.info("Reading ground LAS/LAZ file.")
    input_laz = laspy.read(args.ground_laz)

    for class_id in [1, 2, 3]:
        logger.info("Processing all BGT geometries with class=%s.", class_id)
        for coordinates, polygon, instance in get_coordinates(gdf=gdf, classification=class_id):
            vertices = create_vertices(
                coordinates=coordinates, class_id=class_id, instance=instance, input_laz=input_laz
            )
            if vertices is None:
                logger.error("Insufficient points for class %s, instance %s; skipping.", class_id, instance)
                continue

            mesh = create_mesh(vertices)
            if mesh is None:
                logger.error("Invalid mesh for class %s, instance %s; skipping.", class_id, instance)
                continue

            mesh = update_mesh(mesh, polygon)
            outfname = os.path.join(args.output_path, "pavements", f"{class_id}_{instance}.ply")
            logger.info("Writing mesh %s to file", outfname)
            export_mesh(mesh, outfname)
