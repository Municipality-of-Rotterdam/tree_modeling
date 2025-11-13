# PointCloud_Tree_Modelling by Amsterdam Intelligence, GPL-3.0 license

"""
Open3D utility methods - Module (Python)
"""

import math
import trimesh
import numpy as np
import open3d as o3d
from typing import Optional, List

from tree_modeling.misc.quaternion import Quaternion


def surface_variation_filter(pcd: o3d.geometry.PointCloud, radius: float, threshold: float) -> np.ndarray:
    """
    Compute the surface variation of a point cloud and return a mask of valid points.

    Args:
        pcd (o3d.geometry.PointCloud): Input point cloud.
        radius (float): Search radius for covariance estimation.
        threshold (float): Threshold for surface variation filtering.

    Returns:
        np.ndarray: Boolean mask indicating points that satisfy the threshold condition.
    """
    pcd.estimate_covariances(search_param=o3d.geometry.KDTreeSearchParamRadius(radius=radius))
    eig_val, _ = np.linalg.eig(np.asarray(pcd.covariances))
    eig_val = np.sort(eig_val, axis=1)
    sv = eig_val[:, 0] / eig_val.sum(axis=1)
    mask = sv < threshold
    return mask


def curvature_filter(
    pcd: o3d.geometry.PointCloud,
    radius: float,
    min1: float = 0,
    max1: float = 100,
    min2: float = 0,
    max2: float = 100,
    min3: float = 0,
    max3: float = 100,
) -> np.ndarray:
    """
    Compute curvature-based filtering for a point cloud based on eigenvalues.

    Args:
        pcd (o3d.geometry.PointCloud): Input point cloud.
        radius (float): Search radius for curvature estimation.
        min1, max1, min2, max2, min3, max3 (float, optional): Min and max thresholds for eigenvalues.

    Returns:
        np.ndarray: Boolean mask indicating points that satisfy the filtering criteria.
    """

    # estimate eigenvalues
    pcd.estimate_covariances(search_param=o3d.geometry.KDTreeSearchParamRadius(radius=radius))
    eig_val, _ = np.linalg.eig(np.asarray(pcd.covariances))
    eig_val = np.sort(eig_val, axis=1)
    eig_val[eig_val[:, 2] == 1] = np.zeros(3)
    L1, L2, L3 = eig_val[:, 2], eig_val[:, 1], eig_val[:, 0]
    L1 = (L1 - L1.min()) / ((L1.max() - L1.min()) / 100)
    L2 = (L2 - L2.min()) / ((L2.max() - L2.min()) / 100)
    L3 = (L3 - L3.min()) / ((L3.max() - L3.min()) / 100)

    mask = (L1 > min1) & (L1 < max1) & (L2 > min2) & (L2 < max2) & (L3 > min3) & (L3 < max3)

    return mask


def project(pcd: o3d.geometry.PointCloud, axis: int, voxel_size: Optional[float] = None) -> np.ndarray:
    """
    Project a point cloud onto a 2D plane along a specified axis.

    Args:
        pcd (o3d.geometry.PointCloud): Input point cloud.
        axis (int): Axis along which projection is performed (0: x, 1: y, 2: z).
        voxel_size (Optional[float], optional): Voxel size for downsampling.

    Returns:
        np.ndarray: 2D projected points.
    """
    pts = np.array(pcd.points)
    pts[:, axis] = 0
    pcd_ = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(pts))
    if voxel_size:
        pcd_ = pcd_.voxel_down_sample(voxel_size)
    pts = np.asarray(pcd_.points)[:, :2]
    return pts


def cloud_height(cloud: o3d.geometry.PointCloud) -> float:
    """
    Compute the height of a point cloud.

    Args:
        cloud (o3d.geometry.PointCloud): Input point cloud.

    Returns:
        float: Height of the cloud.
    """
    height: float = cloud.get_max_bound()[2] - cloud.get_min_bound()[2]
    return height


def to_trimesh(mesh: o3d.geometry.TriangleMesh) -> trimesh.Trimesh:
    """
    Convert an Open3D mesh to a Trimesh object.

    Args:
        mesh (o3d.geometry.TriangleMesh): Input Open3D mesh.

    Returns:
        trimesh.Trimesh: Converted Trimesh object.
    """
    return trimesh.base.Trimesh(mesh.vertices, mesh.triangles)


def surface_mesh_creation(cloud: o3d.geometry.PointCloud) -> o3d.geometry.TriangleMesh:
    """
    Create a surface mesh from a point cloud using ball-pivoting.

    Args:
        cloud (o3d.geometry.PointCloud): Input point cloud.

    Returns:
        o3d.geometry.TriangleMesh: Generated surface mesh.
    """

    ground_cloud = cloud.voxel_down_sample(0.25)
    _, ind = ground_cloud.remove_statistical_outlier(nb_neighbors=16, std_ratio=2.0)
    # o3d_utils.display_inlier_outlier(ground_cloud, ind)
    ground_cloud = ground_cloud.select_by_index(ind)
    ground_cloud.estimate_normals()

    surface_mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(
        ground_cloud, o3d.utility.DoubleVector([0.75])
    )
    surface_mesh.compute_vertex_normals()
    return surface_mesh


def mesh_from_cylinders(
    cyl_array: np.ndarray, color: List[float] = [0.7, 0.7, 0.7], resolution: int = 15
) -> o3d.geometry.TriangleMesh:
    """
    Construct a mesh from an array of cylindrical sections.

    Args:
        cyl_array (np.ndarray): Array defining cylindrical sections with centers and radii.
        color (List[float], optional): RGB color for the mesh. Defaults to [0.7, 0.7, 0.7].
        resolution (int, optional): Resolution for circular segments. Defaults to 15.

    Returns:
        o3d.geometry.TriangleMesh: Constructed mesh object.
    """
    circle_fits = [(rim[:3], rim[3]) for rim in cyl_array]
    num_slices = len(circle_fits)

    # compute directional axis
    Z = np.array([0, 0, 1], dtype=float)
    centers = np.array([c for c, r in circle_fits])
    axis = centers[-1] - centers[0]
    axis_length = np.linalg.norm(axis)
    if axis_length <= 1e-12:
        axis = Z

    quat = Quaternion()
    rot = quat.fromData(Z, axis).to_matrix()

    # create vertices
    angles = [2 * math.pi * i / float(resolution) for i in range(resolution)]
    rim = np.array([[math.cos(theta), math.sin(theta), 0.0] for theta in angles])
    rim = np.dot(rot, rim.T).T

    rims = np.array([rim * r + c for c, r in circle_fits], dtype=float)
    rims = rims.reshape((-1, 3))

    vertices = np.vstack([[centers[0], centers[-1]], rims])

    # create faces
    bottom_fan = np.array([[0, (i + 1) % resolution + 2, i + 2] for i in range(resolution)], dtype=int)

    top_fan = np.array(
        [
            [1, i + 2 + resolution * (num_slices - 1), (i + 1) % resolution + 2 + resolution * (num_slices - 1)]
            for i in range(resolution)
        ],
        dtype=int,
    )

    slice_fan = np.array(
        [
            [
                [2 + i, (i + 1) % resolution + 2, i + resolution + 2],
                [i + resolution + 2, (i + 1) % resolution + 2, (i + 1) % resolution + resolution + 2],
            ]
            for i in range(resolution)
        ],
        dtype=int,
    )
    slice_fan = slice_fan.reshape((-1, 3), order="C")

    side_fan = np.array([slice_fan + resolution * i for i in range(num_slices - 1)], dtype=int)
    side_fan = side_fan.reshape((-1, 3), order="C")

    faces = np.vstack([bottom_fan, top_fan, side_fan])

    # create mesh
    mesh = trimesh.base.Trimesh(vertices, faces).as_open3d
    mesh.paint_uniform_color(color)

    return mesh
