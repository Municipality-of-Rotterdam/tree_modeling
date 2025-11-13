# tree_modeling/core/constants.py
from __future__ import annotations

# Geometry / metrics
BREAST_HEIGHT_M: float = 1.3  # m

# Colors (Open3D expects floats 0..1)
TREE_COLORS: dict[str, list[float]] = {
    "stem": [0.36, 0.25, 0.20],
    "foliage": [0.00, 0.48, 0.00],
    "wood": [0.45, 0.23, 0.07],
}

# Stem endpoint inference defaults
STEM_ROI_BUFFER_M: float = 1.5
CYL_ADDITIONAL_RADIUS_M: float = 1.0
EDGE_POINTS_DEFAULT: int = 20
MIN_GROUND_POINTS: int = 10
MIN_Z_AXIS_ALIGNMENT: float = 0.5

# Skeleton filtering constants
STEM_RADIUS_TH: float = 0.5  # meters (XY radius around base)
REF_STEM_LENGTH: float = 1.5  # meters (Z window above base)
ARTEFACT_STEM_SKELETON_TH: int = 10
