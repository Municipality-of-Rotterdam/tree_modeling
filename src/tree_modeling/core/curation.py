from __future__ import annotations

import numpy as np
import open3d as o3d
from shapely.geometry import Polygon

from tree_modeling.logger import logger


def find_longest_non_nan_sequence(arr: list[float]) -> list[bool]:
    """
    Mark the longest contiguous run of non-NaN values in a 1D sequence.

    Parameters
    ----------
    arr : list of float
        Sequence that can contain ``np.nan`` gaps.

    Returns
    -------
    list[bool]
        Boolean mask of the same length as ``arr`` where ``True`` marks the
        longest contiguous non-NaN run. If there are no NaNs, all values are ``True``.
    """
    if not np.any(np.isnan(arr)):
        return [True] * len(arr)

    max_len = 0
    max_start_index = 0
    current_len = 0
    current_start_index = 0

    for i, val in enumerate(arr):
        if not np.isnan(val):
            if current_len == 0:  # start new run
                current_start_index = i
            current_len += 1
        else:
            if current_len > max_len:
                max_len = current_len
                max_start_index = current_start_index
            current_len = 0

    if current_len > max_len:  # tail run
        max_len = current_len
        max_start_index = current_start_index

    result = [False] * len(arr)
    for i in range(max_start_index, max_start_index + max_len):
        result[i] = True
    return result


def extract_area_and_points(slices: list[tuple[np.ndarray, Polygon | None, float]]) -> tuple[list[float], list[int]]:
    """
    Extract bounding-box areas and point counts for each slice.

    Parameters
    ----------
    slices : list of tuple[np.ndarray, Polygon | None, float]
        Slices as ``(points, bbox, z_min)`` where ``bbox`` may be ``None``.

    Returns
    -------
    tuple[list[float], list[int]]
        - ``area_list`` where ``np.nan`` is used when ``bbox`` is ``None``.
        - ``points_list`` with the number of points per slice.
    """
    area_list = [bbox.area if bbox is not None else np.nan for _, bbox, _ in slices]
    points_list = [len(pts) for pts, _, _ in slices]
    return area_list, points_list


def validate_point_density(points_list: list[int], height_segment: float, min_points_per_m_slice: int) -> bool:
    """
    Check if any slice meets a minimum point density.

    Parameters
    ----------
    points_list : list of int
        Number of points per slice.
    height_segment : float
        Vertical thickness of each slice (meters).
    min_points_per_m_slice : int
        Minimum required points per meter of slice height.

    Returns
    -------
    bool
        ``True`` if at least one slice satisfies the density threshold, else ``False``.
    """
    return bool(sum(np.array(points_list) > min_points_per_m_slice * height_segment) > 0)


def find_bounds_valid_segments(
    points_list: list[int], height_segment: float, min_points_per_m_slice: int
) -> tuple[int, int]:
    """
    Find the first and last indices of slices that satisfy a point-density threshold.

    Parameters
    ----------
    points_list : list of int
        Number of points per slice.
    height_segment : float
        Vertical thickness of each slice (meters).
    min_points_per_m_slice : int
        Minimum required points per meter of slice height.

    Returns
    -------
    tuple[int, int]
        ``(first_valid_segment, last_valid_segment)`` indices in ``points_list``.
    """
    mask = np.array(points_list) > min_points_per_m_slice * height_segment
    first_valid_segment = np.where(mask)[0][0]
    last_valid_segment = np.where(mask)[0][-1]
    return first_valid_segment, last_valid_segment


def calculate_valid_bbox_area(
    stem_segments: list[tuple[np.ndarray, Polygon | None, float]],
    max_magnif_factor: float,
    min_valid_area: float = 0.01,
    ref_perc: float = 15,
) -> float | None:
    """
    Derive a valid bounding-box area threshold from near-base stem segments.

    The threshold is computed as
    ``nanpercentile(stem_bbox_area, ref_perc) * max_magnif_factor``,
    with empty/invalid bboxes treated as ``np.nan`` and clamped by ``min_valid_area``.

    Parameters
    ----------
    stem_segments : list of tuple[np.ndarray, Polygon | None, float]
        Consecutive near-base stem slices.
    max_magnif_factor : float
        Multiplier applied to the reference percentile area.
    min_valid_area : float, default=0.01
        Minimum area used in place of 0 to avoid degenerate thresholds.
    ref_perc : float, default=15
        Percentile (0–100) used to compute the reference area.

    Returns
    -------
    float or None
        Valid area threshold, or ``None`` if it cannot be computed.
    """
    stem_bbox_area = [
        max(stem_bbox.area, min_valid_area) if stem_bbox is not None else np.nan for _, stem_bbox, _ in stem_segments
    ]
    valid_bbox_area: float = np.nanpercentile(stem_bbox_area, ref_perc) * max_magnif_factor
    return valid_bbox_area if not np.isnan(valid_bbox_area) else None


def determine_valid_stem_slices(
    stem_segments: list[tuple[np.ndarray, Polygon | None, float]], valid_bbox_area: float
) -> list[bool]:
    """
    Mark stem slices as valid if their bbox area is below a threshold.

    Parameters
    ----------
    stem_segments : list of tuple[np.ndarray, Polygon | None, float]
        Stem slices as ``(points, bbox, z_min)`` where ``bbox`` may be ``None``.
    valid_bbox_area : float
        Area threshold; slices with ``bbox.area < valid_bbox_area`` are marked valid.

    Returns
    -------
    list[bool]
        Per-slice validity mask. ``False`` if ``bbox`` is ``None``.
    """
    return [stem_bbox.area < valid_bbox_area if stem_bbox is not None else False for _, stem_bbox, _ in stem_segments]


def determine_valid_crown_slices(crown_segments: list[tuple[np.ndarray, Polygon | None, float]]) -> list[bool]:
    """
    Select the longest contiguous run of slices with non-NaN bounding-box areas.

    Parameters
    ----------
    crown_segments : list of tuple[np.ndarray, Polygon | None, float]
        Crown slices as ``(points, bbox, z_min)`` where ``bbox`` may be ``None``.

    Returns
    -------
    list[bool]
        Boolean mask over slices where ``True`` marks slices belonging to the
        longest non-NaN area run.
    """
    area_list, _ = extract_area_and_points(crown_segments)
    return find_longest_non_nan_sequence(arr=area_list)


def mark_invalid_crown_points(
    pcd: o3d.geometry.PointCloud,
    slices: list[tuple[np.ndarray, Polygon | None, float]],
    valid_slices: list[bool],
    height_segment: float,
) -> np.ndarray:
    """
    Mark crown points as invalid if they fall outside the vertical range of valid slices.

    The valid crown band is determined from the first and last indices where
    ``valid_slices`` is ``True``; points outside that Z-interval are flagged invalid.
    """
    points = np.asarray(pcd.points)
    if len(valid_slices) > 0 and any(valid_slices):
        first_valid_ind = np.where(valid_slices)[0][0]
        last_valid_ind = np.where(valid_slices)[0][-1]
        invalid_pc_points_mask = (points[:, 2] < slices[first_valid_ind][2]) | (
            points[:, 2] > slices[last_valid_ind][2] + height_segment
        )
    else:
        invalid_pc_points_mask = np.ones(len(points), dtype=bool)
    return invalid_pc_points_mask


def get_valid_slices(
    slices: list[tuple[np.ndarray, Polygon | None, float]],
    first_valid_segment: int,
    valid_stem_bbox_list: list[bool],
) -> list[bool]:
    """
    Build a per-slice validity mask for stem slices.

    Returns
    -------
    list[bool]
        Per-slice validity mask aligned with ``slices`` length.
    """
    valid_slice_list = (
        [False] * first_valid_segment
        + valid_stem_bbox_list
        + [True] * (len(slices) - len(valid_stem_bbox_list) - first_valid_segment)
    )
    return [valid_slice_list[i] for i, _ in enumerate(slices)]


def mark_invalid_stem_points(
    pcd: o3d.geometry.PointCloud,
    slices: list[tuple[np.ndarray, Polygon | None, float]],
    valid_slices: list[bool],
    last_stem_segment: int,
    buff_size: float,
    height_segment: float,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Flag stem points outside buffered valid slice bounds as invalid.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        ``(invalid_mask, stem_mask)``, both boolean arrays over all points in ``pcd``.
    """
    points = np.asarray(pcd.points)
    invalid_mask_list: list[np.ndarray] = []
    stem_mask_list: list[np.ndarray] = []
    valid_slice_inds = [ind for ind in np.where(list(map(int, valid_slices)))[0] if ind <= last_stem_segment]

    for i, valid_slice in enumerate(valid_slices):
        if i > last_stem_segment:
            break
        if not valid_slice:
            closest_valid_ind = valid_slice_inds[np.argmin(np.abs(i - np.array(valid_slice_inds)))]
            slice_distance = np.abs(i - closest_valid_ind)
            valid_bnds = slices[closest_valid_ind][1].buffer(buff_size * slice_distance * height_segment).bounds
            invalid_pc_points_mask = (
                (
                    (points[:, 0] < valid_bnds[0])
                    | (points[:, 0] > valid_bnds[2])
                    | (points[:, 1] < valid_bnds[1])
                    | (points[:, 1] > valid_bnds[3])
                )
                & (points[:, 2] <= slices[i][2] + height_segment)
                & (points[:, 2] >= slices[i][2])
            )
            invalid_mask_list.append(invalid_pc_points_mask)
        else:
            valid_bnds = slices[i][1].bounds

        stem_points_mask = (
            (
                (points[:, 0] >= valid_bnds[0])
                & (points[:, 0] <= valid_bnds[2])
                & (points[:, 1] >= valid_bnds[1])
                & (points[:, 1] <= valid_bnds[3])
            )
            & (points[:, 2] <= slices[i][2] + height_segment)
            & (points[:, 2] >= slices[i][2])
        )
        stem_mask_list.append(stem_points_mask)

    if len(invalid_mask_list) == 0:
        return np.array([False] * len(points[:, 0])), np.any(stem_mask_list, axis=0)

    return np.any(invalid_mask_list, axis=0), np.any(stem_mask_list, axis=0)


def curate_stem_cloud(
    pcd: o3d.geometry.PointCloud,
    slices: list[tuple[np.ndarray, Polygon | None, float]],
    height_segment: float,
    max_magnif_factor: float = 1.75,
) -> tuple[np.ndarray, list[bool]] | tuple[None, None]:
    """
    Curate a stem point cloud by validating height slices.

    Each horizontal slice is evaluated by point density and bounding-box area.
    Invalid slices are removed and a boolean mask for valid points is produced.

    Returns
    -------
    tuple[np.ndarray, list[bool]] or tuple[None, None]
        ``(valid_points_mask, valid_slice_flags)`` or ``(None, None)`` if curation fails.
    """
    BUFF_SIZE = 0.1
    MIN_POINTS_PER_M_SLICE = 100
    REFERENCE_PERCENTILE = 25

    _, points_list = extract_area_and_points(slices)

    if not validate_point_density(points_list, height_segment, MIN_POINTS_PER_M_SLICE):
        logger.error("Point density of stem slices too low. Skipping")
        return None, None

    first_valid_segment, last_valid_segment = find_bounds_valid_segments(
        points_list, height_segment, MIN_POINTS_PER_M_SLICE
    )

    stem_segments = slices[first_valid_segment : last_valid_segment + 1]
    valid_bbox_area = calculate_valid_bbox_area(stem_segments, max_magnif_factor, ref_perc=REFERENCE_PERCENTILE)
    if valid_bbox_area is None:
        return None, None

    valid_stem_list = determine_valid_stem_slices(stem_segments, valid_bbox_area)
    valid_slices = get_valid_slices(slices, first_valid_segment, valid_stem_list)
    invalid_pc_points, _ = mark_invalid_stem_points(
        pcd=pcd,
        slices=slices,
        valid_slices=valid_slices,
        last_stem_segment=last_valid_segment,
        buff_size=BUFF_SIZE,
        height_segment=height_segment,
    )
    return ~invalid_pc_points, valid_slices


def curate_crown_cloud(
    pcd: o3d.geometry.PointCloud, slices: list[tuple[np.ndarray, Polygon | None, float]], height_segment: float
) -> tuple[np.ndarray, list[bool]]:
    """
    Validate crown slices and produce a per-point validity mask.

    Returns
    -------
    tuple[np.ndarray, list[bool]]
        - Boolean mask over ``pcd`` where ``True`` denotes a valid crown point.
        - Per-slice boolean list indicating valid crown slices.
    """
    valid_crown_slices = determine_valid_crown_slices(crown_segments=slices)
    invalid_pc_points = mark_invalid_crown_points(
        pcd=pcd, slices=slices, valid_slices=valid_crown_slices, height_segment=height_segment
    )
    return ~invalid_pc_points, valid_crown_slices
