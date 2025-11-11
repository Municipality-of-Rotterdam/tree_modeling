# PointCloud_Tree_Modelling by Amsterdam Intelligence, GPL-3.0 license

"""
Smallest enclosing circle - Library (Python)

The Library is adapted from:
https://www.nayuki.io/page/smallest-enclosing-circle
"""

import math
import random
from typing import List, Tuple, Optional


# Data conventions: A point is a pair of floats (x, y). A circle is a triple of floats (center x, center y, radius).


# Returns the smallest circle that encloses all the given points. Runs in expected O(n) time, randomized.
# Input: A sequence of pairs of floats or ints, e.g. [(0,5), (3.1,-2.7)].
# Output: A triple of floats representing a circle.
# Note: If 0 points are given, None is returned. If 1 point is given, a circle of radius 0 is returned.
#
# Initially: No boundary points known
def make_circle(points: List[Tuple[float, float]]) -> Optional[Tuple[float, float, float]]:
    """
    Returns the smallest circle that encloses all the given points.
    Args:
            points: A sequence of pairs of floats or ints, e.g. [(0, 5), (3.1, -2.7)].
    Returns:
            A triple of floats (center x, center y, radius) representing the circle.
            If no points are given, None is returned. If one point is given, a circle with radius 0 is returned.
    """
    # Convert to float and randomize order
    shuffled = [(float(x), float(y)) for (x, y) in points]
    random.shuffle(shuffled)

    # Progressively add points to circle or recompute circle
    c = None
    for i, p in enumerate(shuffled):
        if c is None or not is_in_circle(c, p):
            c = _make_circle_one_point(shuffled[: i + 1], p)
    return c


def _make_circle_one_point(points: List[Tuple[float, float]], p: Tuple[float, float]) -> Tuple[float, float, float]:
    """
    Returns the smallest circle that encloses all points given one boundary point.
    Args:
            points: List of points in the form (x, y).
            p: A boundary point (x, y).
    Returns:
            A tuple (center x, center y, radius) representing the circle.
    """
    c = (p[0], p[1], 0.0)
    for i, q in enumerate(points):
        if not is_in_circle(c, q):
            if c[2] == 0.0:
                c = make_diameter(p, q)
            else:
                c = _make_circle_two_points(points[: i + 1], p, q)
    return c


def _make_circle_two_points(
    points: List[Tuple[float, float]], p: Tuple[float, float], q: Tuple[float, float]
) -> Tuple[float, float, float]:
    """
    Returns the smallest circle that encloses all points given two boundary points.
    Args:
            points: List of points in the form (x, y).
            p: First boundary point (x, y).
            q: Second boundary point (x, y).
    Returns:
            A tuple (center x, center y, radius) representing the circle.
    """
    circ = make_diameter(p, q)
    left = None
    right = None
    px, py = p
    qx, qy = q

    # For each point not in the two-point circle
    for r in points:
        if is_in_circle(circ, r):
            continue

        # Form a circumcircle and classify it on left or right side
        cross = _cross_product(px, py, qx, qy, r[0], r[1])
        c = make_circumcircle(p, q, r)
        if c is None:
            continue
        elif cross > 0.0 and (
            left is None
            or _cross_product(px, py, qx, qy, c[0], c[1]) > _cross_product(px, py, qx, qy, left[0], left[1])
        ):
            left = c
        elif cross < 0.0 and (
            right is None
            or _cross_product(px, py, qx, qy, c[0], c[1]) < _cross_product(px, py, qx, qy, right[0], right[1])
        ):
            right = c

    # Select which circle to return
    if left is None and right is None:
        return circ
    elif left is None:
        return right
    elif right is None:
        return left
    else:
        return left if (left[2] <= right[2]) else right


def make_diameter(a: Tuple[float, float], b: Tuple[float, float]) -> Tuple[float, float, float]:
    """
    Returns the circle defined by the diameter formed by points a and b.
    Args:
            a: First point (x, y).
            b: Second point (x, y).
    Returns:
            A tuple (center x, center y, radius) representing the circle.
    """
    cx = (a[0] + b[0]) / 2
    cy = (a[1] + b[1]) / 2
    r0 = math.hypot(cx - a[0], cy - a[1])
    r1 = math.hypot(cx - b[0], cy - b[1])
    return (cx, cy, max(r0, r1))


def make_circumcircle(
    a: Tuple[float, float], b: Tuple[float, float], c: Tuple[float, float]
) -> Optional[Tuple[float, float, float]]:
    """
    Returns the circumcircle of the triangle formed by points a, b, and c.
    Args:
            a: First point (x, y).
            b: Second point (x, y).
            c: Third point (x, y).
    Returns:
            A tuple (center x, center y, radius) representing the circumcircle.
            If the points are collinear, None is returned.
    """
    # Mathematical algorithm from Wikipedia: Circumscribed circle
    ox = (min(a[0], b[0], c[0]) + max(a[0], b[0], c[0])) / 2
    oy = (min(a[1], b[1], c[1]) + max(a[1], b[1], c[1])) / 2
    ax = a[0] - ox
    ay = a[1] - oy
    bx = b[0] - ox
    by = b[1] - oy
    cx = c[0] - ox
    cy = c[1] - oy
    d = (ax * (by - cy) + bx * (cy - ay) + cx * (ay - by)) * 2.0
    if d == 0.0:
        return None
    x = ox + ((ax * ax + ay * ay) * (by - cy) + (bx * bx + by * by) * (cy - ay) + (cx * cx + cy * cy) * (ay - by)) / d
    y = oy + ((ax * ax + ay * ay) * (cx - bx) + (bx * bx + by * by) * (ax - cx) + (cx * cx + cy * cy) * (bx - ax)) / d
    ra = math.hypot(x - a[0], y - a[1])
    rb = math.hypot(x - b[0], y - b[1])
    rc = math.hypot(x - c[0], y - c[1])
    return (x, y, max(ra, rb, rc))


_MULTIPLICATIVE_EPSILON = 1 + 1e-14


def is_in_circle(c: Optional[Tuple[float, float, float]], p: Tuple[float, float]) -> bool:
    """
    Checks if a point is inside or on the boundary of a circle.
    Args:
            c: A circle defined by (center x, center y, radius), or None if no circle.
            p: A point (x, y).
    Returns:
            True if the point is inside or on the boundary of the circle, False otherwise.
    """
    return c is not None and math.hypot(p[0] - c[0], p[1] - c[1]) <= c[2] * _MULTIPLICATIVE_EPSILON


def _cross_product(x0: float, y0: float, x1: float, y1: float, x2: float, y2: float) -> float:
    """
    Returns twice the signed area of the triangle defined by points (x0, y0), (x1, y1), (x2, y2).
    Args:
            x0, y0: Coordinates of the first point.
            x1, y1: Coordinates of the second point.
            x2, y2: Coordinates of the third point.
    Returns:
            Twice the signed area of the triangle formed by the points.
    """
    return (x1 - x0) * (y2 - y0) - (y1 - y0) * (x2 - x0)
