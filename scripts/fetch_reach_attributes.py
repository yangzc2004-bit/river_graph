"""Extract NHDPlus V2 reach attributes for graph node COMIDs.

Downloads the VAA (value-added attributes) table once via pynhd (cached),
keeps the five H2 attributes (design_m4.md): stream order, drainage area,
slope, flowline length (+ catchment area for reference).

Usage: python scripts/fetch_reach_attributes.py
"""

import pickle

import pandas as pd
from pynhd import nhdplus_vaa

KEEP = ["streamorde", "totdasqkm", "slope", "lengthkm", "areasqkm"]


def main() -> None:
    with open("data/processed/mississippi_graph.pkl", "rb") as fh:
        nodes = pickle.load(fh)["nodes"]
    comids = pd.to_numeric(nodes["comid"])

    vaa = nhdplus_vaa()
    sub = vaa[vaa["comid"].isin(comids)][["comid", *KEEP]].copy()
    print(f"matched {len(sub)}/{len(comids)} comids")
    missing = set(comids) - set(sub["comid"])
    if missing:
        print(f"missing comids: {sorted(missing)[:10]}")

    out = nodes[["site_no", "comid"]].copy()
    out["comid"] = pd.to_numeric(out["comid"])
    out = out.merge(sub, on="comid", how="left")
    out.to_csv("data/processed/reach_attributes.csv", index=False)
    print(out[KEEP].describe().round(3).to_string())
    print("saved data/processed/reach_attributes.csv")


if __name__ == "__main__":
    main()
