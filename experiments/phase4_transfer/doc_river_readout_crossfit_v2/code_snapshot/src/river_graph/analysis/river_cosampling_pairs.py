"""Check whether a downstream-gauge frontier conceals co-sampled branches."""

from __future__ import annotations

from river_graph.analysis.river_cosampling_geometry import pair_sampling_inventory


def disjoint_pair_inventory(gauges, station_dates, receiver_dates, *, receiver_area):
    """All non-nested candidate pairs; receiving and DOC responses are unused."""
    if gauges.comid.duplicated().any():
        raise ValueError("unique candidate gauge COMIDs required")
    parent = dict(zip(gauges.comid, gauges.downstream_candidate_comid, strict=True))
    ancestors = {}
    for comid, downstream in parent.items():
        seen, successor = set(), downstream
        while successor >= 0:
            if successor in seen or successor == comid:
                raise ValueError("candidate routing must be acyclic")
            seen.add(successor)
            successor = parent[successor]
        ancestors[comid] = seen
    pairs = pair_sampling_inventory(gauges.assign(frontier=True), station_dates, receiver_dates,
        receiver_area=receiver_area)
    if pairs.empty:
        return pairs
    non_nested = [b not in ancestors[a] and a not in ancestors[b]
        for a, b in zip(pairs.source_a_comid, pairs.source_b_comid, strict=True)]
    result = pairs[non_nested].copy()
    frontier = gauges.set_index("station").frontier
    result["both_nearest_frontier"] = result.source_a.map(frontier) & result.source_b.map(frontier)
    return result.reset_index(drop=True)
