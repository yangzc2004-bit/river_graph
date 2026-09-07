"""Build Mississippi DOC Graph v0.1 (Milestone 1).

Pipeline: candidate DOC stations -> NLDI COMID match -> downstream flowline
walk -> edge list -> pickle + CSV + sanity-check map.

Resumable: NLDI responses are cached under data/raw/nldi/.

Usage: python scripts/build_graph.py
"""

import pickle
from pathlib import Path

import pandas as pd

from river_graph.config import load_config
from river_graph.data.nwis import fetch_doc_sites
from river_graph.topology.build_edges import build_edges, first_station_downstream
from river_graph.topology.nldi import get_downstream_flowline_comids, match_stations
from river_graph.topology.visualize import plot_graph

RAW = Path("data/raw")
PROCESSED = Path("data/processed")


def main() -> None:
    cfg = load_config()
    flt = cfg["station_filter"]

    sites = fetch_doc_sites(cfg["region"]["huc2"], RAW)
    inv = pd.read_csv(PROCESSED / "doc_site_inventory.csv", dtype={"site_no": str})
    candidates = inv[
        (inv["count_nu"] >= flt["min_samples"])
        & (inv["span_years"] >= flt["min_span_years"])
    ]
    nodes = sites.merge(candidates[["site_no"]], on="site_no")
    nodes = nodes.dropna(subset=["dec_lat_va", "dec_long_va"])
    print(f"candidate stations: {len(nodes)}")

    # Step 2.1: station -> COMID
    match = match_stations(
        sorted(nodes["site_no"]),
        RAW / "nldi" / "comid",
        coords=nodes[["site_no", "dec_lat_va", "dec_long_va"]],
    )
    nodes = nodes.merge(match, on="site_no", how="left")
    n_unmatched = nodes["comid"].isna().sum()
    print(f"NLDI matched: {len(nodes) - n_unmatched}/{len(nodes)} "
          f"({n_unmatched} unmatched)")

    # Step 2.2: downstream edges (navigate by COMID so coordinate-snapped
    # stations, which are absent from the NLDI nwissite index, also work)
    matched = nodes.dropna(subset=["comid"])
    unique_comids = sorted(matched["comid"].unique())
    comid_flowlines: dict[str, list[str]] = {}
    for i, comid in enumerate(unique_comids):
        try:
            comid_flowlines[comid] = get_downstream_flowline_comids(
                comid, RAW / "nldi" / "flowlines", distance_km=500
            )
        except Exception as e:  # noqa: BLE001 - transient network errors
            print(f"  flowline fetch failed for comid {comid}: {e}")
            continue
        if (i + 1) % 50 == 0:
            print(f"  flowlines {i + 1}/{len(unique_comids)}")
    flowlines = {
        r.site_no: comid_flowlines.get(r.comid, []) for r in matched.itertuples()
    }
    edges = build_edges(nodes, flowlines)
    print(f"edges: {len(edges)}")

    # second pass with max distance for stations that found nothing
    comid_to_sites = matched.groupby("comid")["site_no"].agg(list).to_dict()
    lonely = [
        s for s in sorted(matched["site_no"])
        if first_station_downstream(flowlines.get(s, []), comid_to_sites, s) is None
    ]
    print(f"stations with no downstream station within 500km: {len(lonely)}")
    lonely_comids = sorted(matched.set_index("site_no").loc[lonely, "comid"].unique())
    for comid in lonely_comids:
        try:
            comid_flowlines[comid] = get_downstream_flowline_comids(
                comid, RAW / "nldi" / "flowlines", distance_km=9999
            )
        except Exception as e:  # noqa: BLE001
            print(f"  long-range fetch failed for comid {comid}: {e}")
    flowlines = {
        r.site_no: comid_flowlines.get(r.comid, []) for r in matched.itertuples()
    }
    edges = build_edges(nodes, flowlines)
    print(f"edges after long-range pass: {len(edges)}")

    # outputs: pure graph first, DOC values merged later
    PROCESSED.mkdir(exist_ok=True)
    edges.to_csv(PROCESSED / "graph_edges.csv", index=False)
    nodes.to_csv(PROCESSED / "graph_nodes.csv", index=False)
    with open(PROCESSED / "mississippi_graph.pkl", "wb") as fh:
        pickle.dump({"nodes": nodes, "edges": edges}, fh)
    plot_graph(nodes, edges, PROCESSED / "mississippi_graph.png")
    print(f"saved graph: {len(nodes)} nodes, {len(edges)} edges "
          f"-> data/processed/mississippi_graph.pkl + .png")


if __name__ == "__main__":
    main()
