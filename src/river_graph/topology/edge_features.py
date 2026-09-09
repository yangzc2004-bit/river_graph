"""Edge features from the cached NLDI downstream flowline walks.

For an edge A -> B, the hop distance is the position of B's COMID in the
downstream walk from A's COMID (0 = same reach). Combined with geographic
distance this is the v1 "transport attribute" for HydroGRN (design_m4.md);
NHDPlus physical attributes per reach are a follow-up.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


def _load_walk(cache_dir: Path, comid: str) -> list[str]:
    """Downstream COMID walk for a comid, longest cached range first."""
    for dist in (9999, 500):
        f = cache_dir / f"comid_{comid}_dm{dist}.json"
        if f.exists():
            data = json.loads(f.read_text(encoding="utf-8"))
            return [
                str(ft["properties"]["nhdplus_comid"])
                for ft in data.get("features", [])
                if ft.get("properties", {}).get("nhdplus_comid") is not None
            ]
    return []


def compute_edge_features(
    nodes: pd.DataFrame, edges: pd.DataFrame, cache_dir: str | Path
) -> pd.DataFrame:
    """Return edges + hop_dist + geo_dist_deg. Edges whose hop distance
    cannot be recovered from the cache get hop_dist = NaN."""
    cache_dir = Path(cache_dir)
    comid_of = nodes.set_index("site_no")["comid"].to_dict()
    xy = nodes.set_index("site_no")[["dec_long_va", "dec_lat_va"]].astype(float)
    walks: dict[str, list[str]] = {}

    hops, geos = [], []
    for e in edges.itertuples(index=False):
        cs, ct = comid_of.get(e.source), comid_of.get(e.target)
        hop = np.nan
        if cs is not None and ct is not None:
            if cs not in walks:
                walks[cs] = _load_walk(cache_dir, cs)
            walk = walks[cs]
            if ct in walk:
                hop = float(walk.index(ct))
        hops.append(hop)
        if e.source in xy.index and e.target in xy.index:
            d = xy.loc[e.source].values - xy.loc[e.target].values
            geos.append(float(np.hypot(*d)))
        else:
            geos.append(np.nan)
    out = edges.copy()
    out["hop_dist"] = hops
    out["geo_dist_deg"] = geos
    return out
