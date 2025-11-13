import numpy as np
import open3d as o3d
from tree_modeling.core.crown import find_lowest_crown_points_by_quadrant

def test_find_lowest_crown_points_by_quadrant():
    # Create sample point cloud
    points = np.array([
        [1, 1, 5],  # NE
        [2, 2, 3],  # NE (lowest)
        [1, -1, 4],  # SE
        [2, -2, 2],  # SE (lowest)
        [-1, -1, 6],  # SW
        [-2, -2, 1],  # SW (lowest)
        [-1, 1, 7],  # NW
        [-2, 2, 4]   # NW (lowest)
    ])
    
    crown_cloud = o3d.geometry.PointCloud()
    crown_cloud.points = o3d.utility.Vector3dVector(points)
    
    stem_basepoint = np.array([0, 0, 0])
    
    expected_results = {
        'NE': (2, 2, 3),
        'SE': (2, -2, 2),
        'SW': (-2, -2, 1),
        'NW': (-2, 2, 4)
    }
    
    result = find_lowest_crown_points_by_quadrant(crown_cloud, stem_basepoint)
    
    assert result == expected_results, f"Expected {expected_results}, but got {result}"