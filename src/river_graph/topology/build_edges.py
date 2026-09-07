"""Build the directed station graph from NLDI downstream flowline walks.

Edge rule: station A -> station B iff B is the first *other* station
encountered walking downstream from A along the mainstem (DM navigation).
Pure functions here; network access lives in nldi.py.
"""

from __future__ import annotations

import pandas as pd


def first_station_downstream(
    flowline_comids: list[str],
    comid_to_sites: dict[str, list[str]],
    own_site: str,
) -> str | None:
    """First station found on the downstream walk, skipping own-only reaches."""
    for comid in flowline_comids:
        for site in comid_to_sites.get(comid, []):
            if site != own_site:
                return site
    return None


def build_edges(
    stations: pd.DataFrame,
    flowlines: dict[str, list[str]],
) -> pd.DataFrame:
    """Build the edge list.

    stations: DataFrame with columns site_no, comid (comid may be NaN/None;
              unmatched stations cannot take part in edges and are skipped)
    flowlines: site_no -> ordered downstream COMIDs (from NLDI DM navigation)

    Returns DataFrame(source, target) with duplicate edges removed.
    """
    matched = stations.dropna(subset=["comid"])
    comid_to_sites: dict[str, list[str]] = (
        matched.groupby("comid")["site_no"].agg(list).to_dict()
    )
    edges = []
    for site_no in matched["site_no"]:
        target = first_station_downstream(
            flowlines.get(site_no, []), comid_to_sites, site_no
        )
        if target is not None:
            edges.append({"source": site_no, "target": target})
    return pd.DataFrame(edges, columns=["source", "target"]).drop_duplicates()
