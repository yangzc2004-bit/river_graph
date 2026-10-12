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


def first_station_on_later_reach(
    flowline_comids: list[str],
    comid_to_sites: dict[str, list[str]],
    own_comid: str,
) -> str | None:
    """Return the first station on a different downstream reach.

    A DM walk starts with the source station's own reach.  All stations on
    that reach are skipped here because COMID navigation does not establish
    their within-reach order.
    """
    for comid in flowline_comids:
        if str(comid) == str(own_comid):
            continue
        sites = comid_to_sites.get(comid, [])
        if sites:
            return sites[0]
    return None


def _ordered_same_reach_sites(group: pd.DataFrame) -> list[str] | None:
    """Order a same-COMID group when every site has a unique NLDI measure.

    NLDI measures decrease in the downstream direction (for example, 100 at
    a headwater location and about 20 farther downstream).  If a station was
    coordinate-snapped, or if measures tie, we return ``None`` rather than
    inventing an upstream/downstream ordering.
    """
    if "measure" not in group.columns or len(group) < 2:
        return None
    measure = pd.to_numeric(group["measure"], errors="coerce")
    if measure.isna().any() or measure.duplicated().any():
        return None
    return (
        group.assign(_measure=measure)
        .sort_values(["_measure", "site_no"], ascending=[False, True])
        ["site_no"].astype(str).tolist()
    )


def build_edges(
    stations: pd.DataFrame,
    flowlines: dict[str, list[str]],
) -> pd.DataFrame:
    """Build the edge list.

    stations: DataFrame with columns site_no, comid (and optionally measure;
              comid may be NaN/None; unmatched stations cannot take part in
              edges and are skipped)
    flowlines: site_no -> ordered downstream COMIDs (from NLDI DM navigation)

    Returns DataFrame(source, target) with duplicate edges removed.
    """
    matched = stations.dropna(subset=["comid"]).copy()
    matched["site_no"] = matched["site_no"].astype(str)
    matched["comid"] = matched["comid"].astype(str)
    comid_to_sites: dict[str, list[str]] = (
        matched.groupby("comid", sort=False)["site_no"].agg(list).to_dict()
    )
    site_to_comid = dict(zip(matched["site_no"], matched["comid"], strict=True))
    ordered_groups = {
        comid: _ordered_same_reach_sites(group)
        for comid, group in matched.groupby("comid", sort=False)
        if len(group) > 1
    }
    edges = []
    for site_no in matched["site_no"]:
        own_comid = site_to_comid[site_no]
        same_reach_order = ordered_groups.get(own_comid)
        if same_reach_order is not None:
            pos = same_reach_order.index(site_no)
            if pos + 1 < len(same_reach_order):
                # The next station on the same reach is downstream when
                # ordered by the NLDI measure.
                target = same_reach_order[pos + 1]
            else:
                target = first_station_on_later_reach(
                    flowlines.get(site_no, []), comid_to_sites, own_comid
                )
        elif own_comid in ordered_groups:
            # Same reach, but position cannot be resolved reliably.  Skip the
            # whole same-COMID group and connect directly to the next known
            # downstream reach rather than creating an arbitrary cycle.
            target = first_station_on_later_reach(
                flowlines.get(site_no, []), comid_to_sites, own_comid
            )
        else:
            target = first_station_downstream(
                flowlines.get(site_no, []), comid_to_sites, site_no
            )
        if target is not None:
            edges.append({"source": site_no, "target": target})
    return pd.DataFrame(edges, columns=["source", "target"]).drop_duplicates()
