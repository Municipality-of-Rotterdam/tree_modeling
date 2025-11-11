"""
Graph utility methods - Module (Python)
"""

from __future__ import annotations

from collections import deque
from typing import Iterable

import networkx as nx
import numpy as np
from plyfile import PlyData
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Line3DCollection
from mpl_toolkits.mplot3d.axes3d import Axes3D

from tree_modeling.helper_functions import max_successor_distance
from tree_modeling.logger import logger
from .types import SkeletonData
from .constants import (
    STEM_RADIUS_TH,
    REF_STEM_LENGTH,
    ARTEFACT_STEM_SKELETON_TH,
)


def read_ply(ply_file: str) -> tuple[nx.DiGraph, np.ndarray, np.ndarray]:
    """
    Read a PLY skeleton and construct a directed graph from its vertices and edges.

    Parameters
    ----------
    ply_file : str
        Path to the PLY file containing ``vertex`` (x,y,z) and ``edge`` (vertex1, vertex2).

    Returns
    -------
    tuple[nx.DiGraph, np.ndarray, np.ndarray]
        graph
            Directed graph with nodes carrying ``x``, ``y``, ``z`` attributes.
        vertices
            Unique vertex array of shape ``(N, 3)``.
        edges
            Edge index array of shape ``(M, 2)`` with 0-based indices into ``vertices``.
    """
    plydata = PlyData.read(ply_file)

    # vertices
    vertices = np.array([[c for c in p] for p in plydata["vertex"].data])
    vertices, reverse_ = np.unique(vertices, axis=0, return_inverse=True)

    # edges
    edges = np.array([reverse_[edge[0]] for edge in plydata["edge"].data])

    # construct graph
    graph = nx.DiGraph()
    for i, vertex in enumerate(vertices):
        graph.add_node(i, x=vertex[0], y=vertex[1], z=vertex[2])
    for i, j in edges:
        graph.add_edge(j, i)

    return graph, vertices, edges


def path_till_split(graph: nx.DiGraph, start_node: int) -> tuple[list[int], list[int]]:
    """
    Traverse from ``start_node`` and return the path until a branching/split is encountered.

    The traversal continues down single-successor chains. When multiple successors appear:
    - If *all* successors grow (have descendants) at least three levels deep, stop at the
      current node (but force-append the closest successor in XY if stopping at the first step).
    - Otherwise, append the shortest path toward a non-growing successor and stop.

    Parameters
    ----------
    graph : nx.DiGraph
        Directed skeleton graph with node attributes ``x``, ``y``, ``z``.
    start_node : int
        Node id to start traversal.

    Returns
    -------
    path : list[int]
        Node ids along the selected path.
    visited_nodes : list[int]
        Node ids visited during exploration (unique, in first-seen order).
    """
    if start_node not in graph:
        raise ValueError(f"Start node {start_node} is not in the graph.")

    path = [start_node]
    current_node = start_node
    visited_nodes: list[int] = [start_node]

    def xy(n: int) -> tuple[float, float]:
        nd = graph.nodes[n]
        if "x" in nd and "y" in nd:
            return float(nd["x"]), float(nd["y"])
        if "pos" in nd and isinstance(nd["pos"], (tuple, list)) and len(nd["pos"]) >= 2:
            return float(nd["pos"][0]), float(nd["pos"][1])
        raise ValueError("Nodes must have 'x' and 'y' (or 'pos') attributes.")

    base_x, base_y = xy(start_node)

    def grows_three_levels(node: int) -> bool:
        """Return True if there exists at least one successor at depth 3 from ``node``."""
        lvl1 = list(graph.successors(node))
        visited_nodes.extend(lvl1)
        if not lvl1:
            return False
        lvl2 = [s for n1 in lvl1 for s in graph.successors(n1)]
        visited_nodes.extend(lvl2)
        if not lvl2:
            return False
        lvl3 = [s for n2 in lvl2 for s in graph.successors(n2)]
        # Note: visited nodes list used only for logging/visualization
        visited_nodes.extend(lvl2)
        return len(lvl3) > 0

    def shortest_path_to_non_growing(node: int) -> list[int]:
        """
        Find the shortest path starting from the current node that leads to a successor
        which does not grow three levels deep.
        """
        queue = [(node, [node])]
        while queue:
            current, current_path = queue.pop(0)
            successors = list(graph.successors(current))
            visited_nodes.extend(successors)

            if not successors:  # No successors exist, stop at this node
                return current_path

            for succ in successors:
                if not grows_three_levels(succ):
                    return current_path + [succ]
                queue.append((succ, current_path + [succ]))
        return []  # Return an empty path if all successors grow sufficiently

    while True:
        successors = list(graph.successors(current_node))  # Get all successors of the current node
        visited_nodes.extend(successors)

        if len(successors) > 1:  # Multiple successors found
            # Check if all successors grow at least 3 levels deep
            if all(grows_three_levels(succ) for succ in successors):
                if len(path) == 1:
                    # Force-add the closest successor to the stem base in (x,y), then continue
                    def sqdist_to_base(n: int) -> float:
                        x, y = xy(n)
                        return (x - base_x) ** 2 + (y - base_y) ** 2

                    next_node = min(successors, key=sqdist_to_base)
                    path.append(next_node)

                break  # Stop at the current node

            # Otherwise, find the shortest path among non-growing successors
            shortest_non_growing_path: list[int] = []
            for succ in successors:
                if not grows_three_levels(succ):
                    candidate_path = shortest_path_to_non_growing(succ)
                    if not shortest_non_growing_path or len(candidate_path) < len(shortest_non_growing_path):
                        shortest_non_growing_path = candidate_path

            # Add the shortest non-growing path to the traversal
            path.extend(shortest_non_growing_path)
            break  # Stop after adding non-growing nodes

        elif len(successors) == 0:  # No more successors
            break

        # If a single successor exists, continue traversal
        current_node = successors[0]
        path.append(current_node)  # Add the successor to the path

    return path, list(dict.fromkeys(visited_nodes))


def plot_graph_nodes(
    ax: Axes3D,
    graph: nx.Graph,
    subgraph: nx.Graph,
    path: list[int],
    visited_nodes: list[int],
    plot_node_key: bool = False,
    non_visited_sample_size: int = 50,
) -> None:
    """
    Plot graph nodes efficiently using NumPy vectorization.

    Parameters
    ----------
    ax : Axes3D
        Matplotlib 3D axis.
    graph : nx.Graph
        Full graph (used to sample some non-visited nodes).
    subgraph : nx.Graph
        Subgraph restricted to visited nodes (must contain node coordinates).
    path : list[int]
        Path node ids to highlight.
    visited_nodes : list[int]
        Nodes considered visited.
    plot_node_key : bool, default False
        If True, annotate node ids.
    non_visited_sample_size : int, default 50
        Sample up to this many non-visited nodes for context.
    """
    node_ids = np.array(list(subgraph.nodes))
    node_positions = np.array([list(subgraph.nodes[n].values()) for n in node_ids], dtype=float)

    node_ids_all = np.array(list(graph.nodes))
    node_positions_all = np.array([list(graph.nodes[n].values()) for n in node_ids_all], dtype=float)

    # ---- Step 2: Create masks for node states ----
    selected_mask = np.isin(node_ids, path)
    visited_mask = np.isin(node_ids, visited_nodes)

    # ---- Step 3: Determine non-visited nodes and sample up to non_visited_sample_size ----
    nonvisited_positions = None
    if len(visited_nodes) > 0:
        # Candidate non-visited IDs by rule, restricted to nodes present in subgraph
        # (so we have positions for them)
        nonvisited_candidates = node_ids_all[(~np.isin(node_ids_all, visited_nodes))]
        if nonvisited_candidates.size > 0:
            rng = np.random.default_rng()
            k = min(non_visited_sample_size, nonvisited_candidates.size)
            sampled_ids = rng.choice(nonvisited_candidates, size=k, replace=False)
            nonvisited_mask = np.isin(node_ids_all, sampled_ids)
            nonvisited_positions = node_positions_all[nonvisited_mask]

    # ---- Step 4: Extract positions based on masks ----
    selected_positions = node_positions[selected_mask]
    visited_positions = node_positions[visited_mask]
    # Exclude selected from visited for plotting
    if visited_positions.size > 0:
        visited_not_selected_positions = visited_positions[~np.isin(node_ids[visited_mask], path)]
    else:
        visited_not_selected_positions = np.empty((0, 3))

    # ---- Step 5: Plot nodes in bulk ----
    # Selected/path nodes
    if selected_positions.size > 0:
        ax.scatter(
            selected_positions[:, 0],
            selected_positions[:, 1],
            selected_positions[:, 2],
            color="green",
            s=50,
            label="Selected",
        )

    # Visited (not selected) nodes
    if visited_not_selected_positions.size > 0:
        ax.scatter(
            visited_not_selected_positions[:, 0],
            visited_not_selected_positions[:, 1],
            visited_not_selected_positions[:, 2],
            color="blue",
            s=50,
            label="Visited",
        )

    # Non-visited sampled nodes
    if nonvisited_positions is not None and nonvisited_positions.size > 0:
        ax.scatter(
            nonvisited_positions[:, 0],
            nonvisited_positions[:, 1],
            nonvisited_positions[:, 2],
            color="lightgray",
            s=20,
            alpha=0.8,
            label="Non-visited (sample)",
        )

    # Optionally annotate node IDs
    if plot_node_key:
        offset = 0.05
        for idx, pos in zip(node_ids, node_positions):
            ax.text(
                pos[0] + offset,
                pos[1] + offset,
                pos[2] + offset,
                str(int(idx)),
                color="black",
                fontsize=8,
            )


def plot_graph_edges(ax: Axes3D, graph: nx.Graph, node_positions: dict[int, np.ndarray]) -> None:
    """
    Plot graph edges in 3D using ``Line3DCollection``.

    Parameters
    ----------
    ax : Axes3D
        Matplotlib 3D axis.
    graph : nx.Graph
        Graph whose edges will be drawn.
    node_positions : dict[int, np.ndarray]
        Mapping node id -> (x, y, z) positions.
    """
    edge_lines = [
        [node_positions[u], node_positions[v]] for u, v in graph.edges() if u in node_positions and v in node_positions
    ]
    if not edge_lines:
        return
    edge_lines_arr = np.asarray(edge_lines, dtype=float)  # (E, 2, 3)
    edge_collection = Line3DCollection(edge_lines_arr, colors="gray", linestyle="--", linewidth=0.5)
    ax.add_collection3d(edge_collection)


def set_equal_scale(ax: Axes3D) -> None:
    """
    Set equal scaling for x, y, z axes.

    Parameters
    ----------
    ax : Axes3D
        Matplotlib 3D axis.
    """
    # Get the current axis limits
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    zlim = ax.get_zlim()

    # Find the maximum range across all axes
    max_range = max(np.ptp(xlim), np.ptp(ylim), np.ptp(zlim))

    # Calculate the midpoints for each axis
    x_mid = (xlim[0] + xlim[1]) / 2
    y_mid = (ylim[0] + ylim[1]) / 2
    z_mid = (zlim[0] + zlim[1]) / 2

    # Set new limits to make all axes equal in scale
    ax.set_xlim([x_mid - max_range / 2, x_mid + max_range / 2])
    ax.set_ylim([y_mid - max_range / 2, y_mid + max_range / 2])
    ax.set_zlim([z_mid - max_range / 2, z_mid + max_range / 2])


def plot_graph_with_path(
    graph: nx.DiGraph, path: list[int], visited_nodes: list[int], output_file: str = "graph_path.png"
) -> None:
    """
    Plot the visited subgraph and highlight the selected ``path``.

    Parameters
    ----------
    graph : nx.DiGraph
        Input directed graph with node coordinates.
    path : list[int]
        Path node ids to draw.
    visited_nodes : list[int]
        Nodes considered visited (used to form the subgraph).
    output_file : str, default "graph_path.png"
        Output image file.
    """
    subgraph = graph.subgraph(visited_nodes)
    node_positions: dict[int, np.ndarray] = {
        n: np.asarray(list(subgraph.nodes[n].values()), dtype=float) for n in subgraph.nodes()
    }

    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection="3d")

    plot_graph_nodes(ax, graph, subgraph, path, visited_nodes)
    plot_graph_edges(ax, subgraph, node_positions)

    if path:
        path_in_sub = [n for n in path if n in node_positions]
        if path_in_sub:
            P = np.vstack([node_positions[n] for n in path_in_sub])
            ax.plot(P[:, 0], P[:, 1], P[:, 2], color="green", linewidth=2, label="Path")

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_title("Graph Path Visualization")
    plt.legend()
    ax.view_init(15, 45)
    set_equal_scale(ax)

    plt.savefig(output_file)
    plt.close()


def build_layered_traversal_levels(G: nx.DiGraph, source: int) -> tuple[dict[int, int], dict[int, list[int]]]:
    """
    Build outward 'levels' starting at `source` using a queue-based, breadth-first layered traversal.

    Parameters
    ----------
    G : nx.DiGraph
        Directed skeleton graph.
    source : int
        Node id to start the layered traversal from.

    Returns
    -------
    depth : dict[int, int]
        Shortest directed-edge distance from `source` to each discovered node.
    levels : dict[int, list[int]]
        Nodes grouped by their depth (distance in edges) from `source`.
    """
    depth: dict[int, int] = {source: 0}
    levels: dict[int, list[int]] = {0: [source]}
    q: deque[int] = deque([source])

    while q:
        u = q.popleft()
        du = depth[u]
        for v in G.successors(u):
            if v in depth:
                continue
            depth[v] = du + 1
            levels.setdefault(du + 1, []).append(v)
            q.append(v)

    return depth, levels


def max_depth_by_level_size(levels: dict[int, list[int]], cap: int) -> int:
    """
    Compute the maximum depth allowed when gating by *per-level size*.

    The traversal levels are produced by a layered (BFS-like) walk where
    ``levels[d]`` contains all nodes at distance ``d`` from the source.
    This function returns the deepest level ``D`` such that **every**
    level ``1..D`` is strictly smaller than ``cap``. As soon as a level
    with ``len(levels[d]) >= cap`` is encountered, the search stops and
    the previous depth is returned.

    Parameters
    ----------
    levels : dict[int, list[int]]
        Mapping depth → list of node IDs at that depth (e.g., from a BFS).
    cap : int
        Maximum allowed number of nodes per level.

    Returns
    -------
    int
        Maximum depth ``D`` satisfying the constraint. Returns ``0`` if
        level 1 already violates the cap or if no deeper level exists.
    """
    d, max_allowed = 1, 0
    while d in levels:
        if len(levels[d]) >= cap:
            break
        max_allowed = d
        d += 1
    return max_allowed


def max_depth_by_cumulative_nodes(levels: dict[int, list[int]], cap: int) -> int:
    """
    Compute the maximum depth allowed when gating by *cumulative node count*.

    The traversal levels are produced by a layered (BFS-like) walk where
    ``levels[d]`` contains all nodes at distance ``d`` from the source.
    This function returns the deepest level ``D`` such that the **total**
    number of nodes across levels ``1..D`` is strictly smaller than ``cap``.
    As soon as the cumulative count reaches or exceeds ``cap``, the search
    stops and the previous depth is returned.

    Parameters
    ----------
    levels : dict[int, list[int]]
        Mapping depth → list of node IDs at that depth (e.g., from a BFS).
    cap : int
        Maximum allowed *cumulative* number of nodes across the considered levels.

    Returns
    -------
    int
        Maximum depth ``D`` satisfying the cumulative constraint. Returns ``0`` if
        level 1 already causes the cumulative count to reach/exceed ``cap``.
    """
    d, max_allowed = 1, 0
    cumulative: list[int] = []
    while d in levels:
        cumulative.extend(levels[d])
        if len(cumulative) >= cap:
            break
        max_allowed = d
        d += 1
    return max_allowed


def bypass_and_remove_nodes(G: nx.DiGraph, nodes_to_remove: Iterable[int]) -> None:
    """
    Bypass and remove nodes from a directed graph.

    For each node ``u`` in ``nodes_to_remove``, this function connects every
    predecessor ``p`` of ``u`` to every successor ``v`` of ``u`` (if such an
    edge does not already exist and ``p != v``), effectively bypassing ``u``.
    After adding the bypass edges, the node ``u`` is removed.

    Parameters
    ----------
    G : nx.DiGraph
        Directed graph to modify in place.
    nodes_to_remove : Iterable[int]
        Iterable of node IDs to bypass and remove.

    Returns
    -------
    None
        Operates in place; no return value.
    """
    for u in nodes_to_remove:
        preds = list(G.predecessors(u))
        succs = list(G.successors(u))
        for p in preds:
            for v in succs:
                if p != v and not G.has_edge(p, v):
                    G.add_edge(p, v)
        if G.has_node(u):
            G.remove_node(u)


def estimate_stem_density(
    G: nx.DiGraph,
    base: int,
    x_attr: dict[int, float],
    y_attr: dict[int, float],
    z_attr: dict[int, float],
    radius_xy: float,
    z_window: float,
) -> int:
    """
    Count nodes in a cylindrical neighborhood around the stem base.

    The neighborhood is defined by:
    - a horizontal (XY) radius ``radius_xy`` centered at the base node, and
    - a vertical extent of ``z_window`` above the base node's ``z``.

    Parameters
    ----------
    G : nx.DiGraph
        Skeleton graph whose nodes have ``x``, ``y``, ``z`` attributes.
    base : int
        Node ID of the stem base (typically the minimum-``z`` node).
    x_attr, y_attr, z_attr : dict[int, float]
        Per-node coordinate mappings (as returned by ``nx.get_node_attributes``).
    radius_xy : float
        Horizontal radius (meters) used for the neighborhood.
    z_window : float
        Vertical window (meters) above the base ``z`` to include.

    Returns
    -------
    int
        Number of nodes inside the cylindrical neighborhood.
    """
    x0, y0, z0 = float(x_attr[base]), float(y_attr[base]), float(z_attr[base])
    count = 0
    for n in G.nodes:
        zn = float(z_attr[n])
        if zn - z0 >= z_window:
            continue
        dx = float(x_attr[n]) - x0
        dy = float(y_attr[n]) - y0
        if (dx * dx + dy * dy) ** 0.5 <= radius_xy:
            count += 1
    return count


def remove_isolated_nodes_mode(
    G: nx.DiGraph,
    depth: dict[int, int],
    levels: dict[int, list[int]],
    z_attr: dict[int, float],
    max_branch_size: int,
    max_succ_xy_dist: float,
    z_threshold: float,
) -> set[int]:
    """
    Select near-base nodes for removal under *isolated-nodes* rules.

    Strategy
    --------
    1. Determine a depth limit ``D`` via :func:`max_depth_by_level_size`, ensuring
       each level ``1..D`` has fewer than ``max_branch_size`` nodes.
    2. Consider nodes with ``1 <= depth <= D`` as candidates.
    3. Keep candidates that:
       - are leaves (no successors), or
       - have strongly diverging successors
         (``max_successor_distance(G, successors) > max_succ_xy_dist``), or
       - have at least one successor with small vertical gap
         (``|z(successor) - z(node)| <= z_threshold``).
       All other candidates are marked for removal.

    Parameters
    ----------
    G : nx.DiGraph
        Skeleton graph.
    depth : dict[int, int]
        Node → depth mapping (directed edge distance from base).
    levels : dict[int, list[int]]
        Depth → node list mapping from a layered traversal.
    z_attr : dict[int, float]
        Node → z coordinate mapping.
    max_branch_size : int
        Maximum allowed number of nodes per level (gating parameter).
    max_succ_xy_dist : float
        Threshold for “strong divergence” among a node’s successors (meters).
    z_threshold : float
        Maximum allowed vertical gap (meters) to treat a successor as “close in z”.

    Returns
    -------
    set[int]
        IDs of nodes selected for removal.
    """
    max_depth_allowed = max_depth_by_level_size(levels, int(max_branch_size))
    candidates = [n for n, d in depth.items() if 1 <= d <= max_depth_allowed]

    removed: set[int] = set()
    for n in candidates:
        subs = list(G.successors(n))
        if not subs:
            continue  # leaf: keep

        # Keep strongly diverging nodes.
        if len(subs) > 1 and max_successor_distance(G, subs) > float(max_succ_xy_dist):
            continue

        zn = float(z_attr[n])
        if any(abs(float(z_attr[m]) - zn) <= float(z_threshold) for m in subs):
            continue  # at least one close child in Z -> keep

        removed.add(n)

    return removed


def remove_artefact_nodes_mode(
    G: nx.DiGraph,
    depth: dict[int, int],
    levels: dict[int, list[int]],
    z_attr: dict[int, float],
    base: int,
    max_branch_size: int,
    max_succ_xy_dist: float,
    min_stem_height: float,
) -> set[int]:
    """
    Select near-base nodes for removal under *artefact-nodes* rules.

    Strategy
    --------
    1. Determine a depth limit ``D`` via :func:`max_depth_by_cumulative_nodes`, ensuring
       the cumulative number of nodes across levels ``1..D`` stays below ``max_branch_size``.
    2. Consider nodes with ``1 <= depth <= D`` as candidates.
    3. Keep candidates that:
       - are leaves **and** lie above ``min_stem_height`` from the base, or
       - have strongly diverging successors **and** lie above ``min_stem_height``.
       All other candidates are marked for removal.

    Parameters
    ----------
    G : nx.DiGraph
        Skeleton graph.
    depth : dict[int, int]
        Node → depth mapping (directed edge distance from base).
    levels : dict[int, list[int]]
        Depth → node list mapping from a layered traversal.
    z_attr : dict[int, float]
        Node → z coordinate mapping.
    base : int
        Base node ID (used to compute vertical offsets).
    max_branch_size : int
        Maximum allowed cumulative node count (gating parameter).
    max_succ_xy_dist : float
        Threshold for “strong divergence” among a node’s successors (meters).
    min_stem_height : float
        Minimum vertical offset from base (meters) for keeping endpoints/diverging nodes.

    Returns
    -------
    set[int]
        IDs of nodes selected for removal.
    """
    max_depth_allowed = max_depth_by_cumulative_nodes(levels, int(max_branch_size))
    candidates = [n for n, d in depth.items() if 1 <= d <= max_depth_allowed]

    removed: set[int] = set()
    z0 = float(z_attr[base])

    for n in candidates:
        subs = list(G.successors(n))
        dz = float(z_attr[n]) - z0

        # Keep single, sufficiently high endpoints.
        if not subs and dz > float(min_stem_height):
            continue

        # Keep strongly diverging nodes if sufficiently high.
        if len(subs) > 1 and max_successor_distance(G, subs) > float(max_succ_xy_dist) and dz > float(min_stem_height):
            continue

        removed.add(n)

    return removed


def filter_skeleton_nodes(
    skeleton: SkeletonData,
    out_file: str,
    z_threshold: float = 0.25,
    max_branch_size: int = 5,
    max_succ_xy_dist: float = 1.0,
    debug: bool = False,
) -> tuple[SkeletonData, set[int]]:
    """
    Filter nodes near the stem base using isolated-node or artefact-node rules.

    This function inspects the skeleton graph near the stem base and removes
    either isolated artefacts or dense artefactual regions depending on
    local node density and branching.

    Thresholds
    ----------
    STEM_RADIUS_TH : float = {STEM_RADIUS_TH}
        XY radius (in meters) around the stem base to consider for density.
    REF_STEM_LENGTH : float = {REF_STEM_LENGTH}
        Vertical window (in meters) above the base node used to assess stem region density.
    ARTEFACT_STEM_SKELETON_TH : int = {ARTEFACT_STEM_SKELETON_TH}
        Node-count threshold separating normal stems from artefactual ones.

    Parameters
    ----------
    skeleton : SkeletonData
        Input skeleton dictionary containing ``graph``, ``vertices``, and ``edges``.
    out_file : str
        Output file path to save filtered skeleton as PLY if ``debug=True``.
    z_threshold : float, default=0.25
        Z-difference threshold for detecting isolated nodes.
    max_branch_size : int, default=5
        Maximum allowed branch size before removal.
    max_succ_xy_dist : float, default=1.0
        Maximum XY distance between connected successors.
    debug : bool, default=False
        If True, writes an ASCII PLY with filtered nodes for inspection.

    Returns
    -------
    tuple[SkeletonData, set[int]]
        Filtered skeleton and a set of removed node IDs.
    """

    G: nx.DiGraph = skeleton["graph"]
    if G is None:
        raise ValueError("Skeleton dictionary must contain a 'graph' key with nx.DiGraph.")

    # Required node attributes
    z_attr = nx.get_node_attributes(G, "z")
    x_attr = nx.get_node_attributes(G, "x")
    y_attr = nx.get_node_attributes(G, "y")
    if not z_attr or not x_attr or not y_attr:
        raise ValueError("Graph nodes must have 'x', 'y', and 'z' attributes.")

    # Base node (minimal z)
    base = min(z_attr, key=z_attr.get)

    # Decide which mode to use
    stem_like_count = estimate_stem_density(G, base, x_attr, y_attr, z_attr, STEM_RADIUS_TH, REF_STEM_LENGTH)
    logger.debug("Estimated stem-like nodes near base: %d", stem_like_count)

    # Build layered traversal once
    depth, levels = build_layered_traversal_levels(G, base)

    if stem_like_count > ARTEFACT_STEM_SKELETON_TH:
        logger.info(
            "Dense/artefactual stem detected (count=%d > %d). Using artefact-nodes mode.",
            stem_like_count,
            ARTEFACT_STEM_SKELETON_TH,
        )
        removed = remove_artefact_nodes_mode(
            G=G,
            depth=depth,
            levels=levels,
            z_attr=z_attr,
            base=base,
            max_branch_size=stem_like_count,
            max_succ_xy_dist=max_succ_xy_dist,
            min_stem_height=REF_STEM_LENGTH,
        )
    else:
        logger.info(
            "Normal stem (count=%d ≤ %d). Using isolated-nodes mode.", stem_like_count, ARTEFACT_STEM_SKELETON_TH
        )
        removed = remove_isolated_nodes_mode(
            G=G,
            depth=depth,
            levels=levels,
            z_attr=z_attr,
            max_branch_size=max_branch_size,
            max_succ_xy_dist=max_succ_xy_dist,
            z_threshold=z_threshold,
        )

    # Apply removals with bypass
    bypass_and_remove_nodes(G, removed)
    logger.info("Removed %d nodes.", len(removed))

    # Optional debug snapshot with compact (reindexed) V/E
    if debug:
        old_to_new, V = _build_vertex_map_and_array(G, skeleton.get("vertices"))
        E = np.array([(old_to_new[u], old_to_new[v]) for u, v in G.edges()], dtype=int)
        outfname = out_file.replace(".ply", "_filtered.ply")
        _write_ply_ascii(outfname, V, E)
        logger.info("Filtered skeleton written to %s", outfname)

    filtered: SkeletonData = {
        "graph": G,
        "vertices": skeleton.get("vertices"),
        "edges": np.array(list(G.edges()), dtype=int),
    }
    return filtered, removed


def _build_vertex_map_and_array(
    H: nx.DiGraph,
    vertices_array: np.ndarray | None,
) -> tuple[dict[int, int], np.ndarray]:
    """
    Build a compact vertex map and coordinate array from a skeleton graph.

    Parameters
    ----------
    H : nx.DiGraph
        Directed skeleton graph whose nodes carry numeric ``x``, ``y``, ``z`` attributes.
    vertices_array : np.ndarray or None
        Optional array of shape ``(N, >=3)`` whose rows correspond to node indices and
        whose first three columns are ``x, y, z`` coordinates.

    Returns
    -------
    tuple[dict[int, int], np.ndarray]
        - A dictionary mapping original node IDs to new consecutive indices (0..N-1).
        - A float array of shape ``(N, 3)`` containing the vertex coordinates
          ordered by the new indices.

    Raises
    ------
    ValueError
        If any node is missing coordinates or if ``vertices_array`` has an invalid shape.
    """
    nodes = list(H.nodes())
    nodes_sorted = sorted(nodes)  # deterministic
    old_to_new = {old: i for i, old in enumerate(nodes_sorted)}

    # Build coordinate array
    coords: list[list[float]] = []
    if vertices_array is not None:
        if vertices_array.ndim != 2 or vertices_array.shape[1] < 3:
            raise ValueError("vertices array must be of shape (N,>=3) with xyz in first 3 columns.")
        nmax = vertices_array.shape[0]
        for nid in nodes_sorted:
            if 0 <= nid < nmax:
                xyz = vertices_array[nid]
                coords.append([float(xyz[0]), float(xyz[1]), float(xyz[2])])
            else:
                x = H.nodes[nid].get("x")
                y = H.nodes[nid].get("y")
                z = H.nodes[nid].get("z")
                if x is None or y is None or z is None:
                    raise ValueError(f"Missing coordinates for node {nid}.")
                coords.append([float(x), float(y), float(z)])
    else:
        for nid in nodes_sorted:
            x = H.nodes[nid].get("x")
            y = H.nodes[nid].get("y")
            z = H.nodes[nid].get("z")
            if x is None or y is None or z is None:
                raise ValueError(f"Missing 'x','y','z' attributes for node {nid}.")
            coords.append([float(x), float(y), float(z)])

    return old_to_new, np.asarray(coords, dtype=float)


def _write_ply_ascii(path: str, V: np.ndarray, E: np.ndarray) -> None:
    """Write a simple ASCII PLY with vertex and edge elements.

    Args:
        path: Output file path.
        V: ``(N,3)`` array of xyz vertex coordinates.
        E: ``(M,2)`` int array of 0-based vertex index pairs (edges).
    """
    N = V.shape[0]
    M = E.shape[0]
    lines = []
    lines.append("ply")
    lines.append("format ascii 1.0")
    lines.append(f"element vertex {N}")
    lines.append("property float x")
    lines.append("property float y")
    lines.append("property float z")
    lines.append(f"element edge {M}")
    lines.append("property int vertex1")
    lines.append("property int vertex2")
    lines.append("end_header")
    for x, y, z in V:
        lines.append(f"{x:.6f} {y:.6f} {z:.6f}")
    for a, b in E:
        lines.append(f"{int(a)} {int(b)}")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
