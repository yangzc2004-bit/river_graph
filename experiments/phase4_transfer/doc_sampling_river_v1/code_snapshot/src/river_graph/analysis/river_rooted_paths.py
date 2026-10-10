"""Unique geometric paths to a known mapped outlet, independent of code decoding."""

from __future__ import annotations

from itertools import pairwise

import networkx as nx
import numpy as np


def rooted_paths(reaches, outlet):
    """Require an exactly connected tree; do not bridge gaps or invent reaches.

    ``reaches`` is a list of (stable ID, LineString). The known outlet is an
    endpoint in the graph. Directions are defined towards that root, and source
    digitization direction is checked separately rather than presumed.
    """
    graph = nx.Graph()
    for edge_id, line in reaches:
        a, b = tuple(line.coords[0]), tuple(line.coords[-1])
        if graph.has_edge(a, b) or a == b:
            raise ValueError("Repeated or closed reach needs separate inspection")
        graph.add_edge(a, b, edge_id=edge_id, length_m=float(line.length), start=a, end=b)
    if not nx.is_tree(graph) or outlet not in graph:
        raise ValueError("Unique rooted paths require one connected tree and a known outlet")
    distance = nx.single_source_dijkstra_path_length(graph, outlet, weight="length_m")
    edges = []
    for a, b, data in graph.edges(data=True):
        up, down = (a, b) if distance[a] > distance[b] else (b, a)
        edges.append({**data, "upstream": up, "downstream": down,
                      "digitized_towards_outlet": data["start"] == up})
    leaves = sorted(n for n, degree in graph.degree if degree == 1 and n != outlet)
    paths = []
    for number, leaf in enumerate(leaves, start=1):
        nodes = nx.shortest_path(graph, leaf, outlet)
        ids = [graph[a][b]["edge_id"] for a, b in pairwise(nodes)]
        paths.append({"path_id": f"P{number}", "headwater_xy": leaf,
                      "path_length_m": distance[leaf], "edge_ids": ids})
    common = set.intersection(*(set(p["edge_ids"]) for p in paths)) if paths else set()
    lengths = np.array([p["path_length_m"] for p in paths])
    summary = {"n_reaches": len(edges), "n_nodes": len(graph), "n_mapped_headwaters": len(paths),
        "n_confluences": sum(degree >= 3 for _, degree in graph.degree),
        "total_mapped_length_m": sum(e["length_m"] for e in edges),
        "all_digitized_towards_known_outlet": all(e["digitized_towards_outlet"] for e in edges),
        "shared_terminal_length_m": sum(e["length_m"] for e in edges if e["edge_id"] in common),
        "path_min_m": float(lengths.min()), "path_max_m": float(lengths.max()),
        "path_range_m": float(np.ptp(lengths)),
        "path_cv": float(lengths.std()/lengths.mean())}
    return edges, paths, summary
