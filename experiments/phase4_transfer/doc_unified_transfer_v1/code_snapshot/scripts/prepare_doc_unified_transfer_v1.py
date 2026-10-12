"""Construct receiving station topology from cached VAA, without DOC labels."""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from river_graph.experiments.provenance import sha256_file
from river_graph.models.river_architecture_comparison import graph_view
from river_graph.topology.build_edges import build_edges

ROOT = Path("experiments/phase4_transfer/doc_unified_transfer_v1")
EXTERNAL = Path("experiments/phase4_transfer/doc_unmonitored_tasks_v1/external_inputs/02040104")


def prepare():
    destination = ROOT/"external"
    if (destination/"complete.json").exists():
        report = json.loads((destination/"complete.json").read_text())
        for name, expected in report["files"].items():
            if sha256_file(destination/name) != expected:
                raise ValueError("receiving topology artifact changed")
        return report
    destination.mkdir(parents=True, exist_ok=True)
    with np.load(EXTERNAL/"inputs.npz") as archive:
        inputs = {k: archive[k].copy() for k in ("site_no", "months", "x", "x_mask", "regime", "static")}
    nodes = pd.read_csv(EXTERNAL/"hydrolocations.csv", dtype={"site_no": str, "comid": str}).set_index("site_no").loc[inputs["site_no"]].reset_index()
    vaa_path = Path("cache/nldplus_vaa.parquet")
    vaa = pd.read_parquet(vaa_path, columns=["comid", "hydroseq", "dnhydroseq", "streamorde"])
    if vaa.hydroseq.duplicated().any() or vaa.comid.duplicated().any():
        raise ValueError("VAA keys not unique")
    vaa = vaa.sort_values("hydroseq")
    hydro, downstream, comid = (vaa[k].to_numpy(np.int64) for k in ("hydroseq", "dnhydroseq", "comid"))
    comid_order = np.argsort(comid)
    sorted_comid = comid[comid_order]
    walks = {}
    for row in nodes.itertuples(index=False):
        if pd.isna(row.comid):
            raise ValueError("unmatched external station; prospectively revise eligibility, not stream order")
        p = np.searchsorted(sorted_comid, int(row.comid))
        if p == len(comid) or sorted_comid[p] != int(row.comid):
            raise ValueError("station absent from VAA")
        i, seen, path = int(comid_order[p]), set(), []
        while i not in seen:
            seen.add(i)
            path.append(str(comid[i]))
            dn = downstream[i]
            j = int(np.searchsorted(hydro, dn))
            if dn <= 0 or j == len(hydro) or hydro[j] != dn:
                break
            i = j
        else:
            raise ValueError("cycle in cached receiving reach topology")
        walks[row.site_no] = path
    edges = build_edges(nodes, walks)
    lookup = {s: i for i, s in enumerate(inputs["site_no"])}
    edge_index = np.asarray([(lookup[a], lookup[b]) for a, b in edges.itertuples(index=False, name=None)], dtype=np.int64).reshape(-1, 2).T
    order = inputs["regime"][:, 0]
    graph_view(edge_index, order, np.arange(len(nodes)))
    # Source schema defines this as zero in-degree in the observed station
    # graph, not as a physical reach or watershed headwater flag.
    incoming = np.bincount(edge_index[1], minlength=len(nodes))
    previous_headwater = inputs["regime"][:, 3].copy()
    inputs["regime"][:, 3] = (incoming == 0).astype(np.float32)
    inputs["edge_index"] = edge_index
    source = torch.load("data/processed/mississippi_graph_graphfix_st357.pt", weights_only=False, map_location="cpu")
    if np.intersect1d(np.asarray(source["site_no"], str), inputs["site_no"]).size:
        raise ValueError("receiving stations overlap source")
    np.savez_compressed(destination/"inputs.npz", **inputs)
    edges.to_csv(destination/"edges.csv", index=False)
    nodes.to_csv(destination/"nodes.csv", index=False)
    report = {"huc8": "02040104", "name": "Middle Delaware-Mongaup-Brodhead",
        "name_source": "https://www.usgs.gov/data/delaware-river-basin-catchments-based-strahler-stream-order-1-national-hydrography-dataset-nhd",
        "stations": len(nodes), "months": len(inputs["months"]), "edges": len(edges), "station_overlap_source": 0,
        "positive_order_stations": int(np.sum(order > 0)), "previous_unknown_headwater_flags": int(np.sum(previous_headwater == -1)),
        "headwater_semantics": "source-compatible zero incoming station edges; recomputed before model fitting using topology only",
        "topology_policy": "cached NHDPlus VAA primary downstream chain; first downstream observed station; distinct same-reach positions resolved by NLDI measure; unresolved ties not ordered",
        "receiving_doc_accessed": False, "history": "previously inspected by other architecture releases; new replication for this unified model, not a virgin external confirmation cohort",
        "inputs_original_sha256": sha256_file(EXTERNAL/"inputs.npz"), "vaa_sha256": sha256_file(vaa_path),
        "hydrolocations_sha256": sha256_file(EXTERNAL/"hydrolocations.csv"),
        "files": {name: sha256_file(destination/name) for name in ("inputs.npz", "edges.csv", "nodes.csv")}}
    (destination/"complete.json").write_text(json.dumps(report, indent=2)+"\n")
    return report


if __name__ == "__main__":
    print(json.dumps(prepare(), indent=2))
