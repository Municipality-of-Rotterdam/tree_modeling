from __future__ import annotations

from typing import TypedDict
import numpy as np
import networkx as nx


CrownStats = TypedDict(
    "CrownStats",
    {
        "crown_center": tuple[float, float],
        "crown_basepoint": tuple[float, float, float],
        "crown_toppoint": tuple[float, float, float],
        "crown_lowest_points": dict[str, tuple[float, float, float] | None],
        "crown_size": float | None,
        "crown_base_height": float | None,
        "crown_top_height": float | None,
        "crown_diameter": float | None,
        "crown_shape": str | None,
        "crown_volume": float | None,
    },
    total=False,
)

SkeletonData = TypedDict(
    "SkeletonData",
    {
        "graph": nx.DiGraph | None,
        "vertices": np.ndarray | None,
        "edges": np.ndarray | None,
    },
)


class Labels:
    """Convenience class for label codes used in classification and shape typing."""

    LEAF = 0
    WOOD = 1
    STEM = 2
    CONICAL = 11
    INVERSE_CONICAL = 12
    CYLINDRICAL = 13
    SPHERICAL = 14

    STR_DICT: dict[int, str] = {
        LEAF: "Leaf",
        WOOD: "Wood",
        STEM: "Stem",
        CONICAL: "Conical",
        INVERSE_CONICAL: "Inverse Conical",
        CYLINDRICAL: "Cylindrical",
        SPHERICAL: "Spherical",
    }

    @staticmethod
    def get_str(label: int) -> str:
        """Return the human-readable string for a given numeric label code."""
        return Labels.STR_DICT.get(label, f"Unknown({label})")
