"""Build edge features for the station graph -> data/processed/edge_features.csv

v1 (topology): hop distance + geographic distance from cached NLDI walks.
v2 (H2): merge the target reach's NHDPlus attributes (length, drainage
area, slope, stream order) — the transport path's physical properties.

Usage: python scripts/build_edge_features.py
"""

import pickle
from pathlib import Path

import pandas as pd

from river_graph.topology.edge_features import compute_edge_features

PROCESSED = Path("data/processed")
REACH_ATTRS = ["lengthkm", "totdasqkm", "slope", "streamorde"]


def main() -> None:
    with open(PROCESSED / "mississippi_graph.pkl", "rb") as fh:
        graph = pickle.load(fh)
    out = compute_edge_features(graph["nodes"], graph["edges"],
                                Path("data/raw/nldi/flowlines"))
    # transport happens toward the target reach: attach its attributes
    reach = pd.read_csv(PROCESSED / "reach_attributes.csv", dtype={"site_no": str})
    out = out.merge(
        reach.rename(columns={c: f"target_{c}" for c in REACH_ATTRS}),
        left_on="target", right_on="site_no", how="left",
    ).drop(columns=["site_no", "comid"])
    out.to_csv(PROCESSED / "edge_features.csv", index=False)
    print(f"edges: {len(out)}; hop distance recovered: {out['hop_dist'].notna().sum()}")
    print(out[["hop_dist", "geo_dist_deg",
               *[f"target_{c}" for c in REACH_ATTRS]]].describe().round(3).to_string())


if __name__ == "__main__":
    main()
