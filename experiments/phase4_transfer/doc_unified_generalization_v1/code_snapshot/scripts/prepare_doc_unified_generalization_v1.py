"""Prepare independent river-unit source roles and label-free confirmation topology."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from river_graph.experiments.provenance import sha256_file
from river_graph.models.river_architecture_comparison import graph_view
from river_graph.models.transfer_robust_doc import huc_labels
from river_graph.topology.build_edges import build_edges

ROOT = Path("experiments/phase4_transfer/doc_unified_generalization_v1")
SOURCE = Path("data/processed/mississippi_graph_graphfix_st357.pt")
NODES = Path("data/processed/graph_nodes_graphfix_st357.csv")
PAIR_ROLES = (("05", "10"), ("10", "07"), ("07", "05"))


def source_roles():
    data = torch.load(SOURCE, weights_only=False, map_location="cpu")
    nodes = pd.read_csv(NODES, dtype={"site_no": str, "huc_cd": str})
    groups = huc_labels(nodes, data["site_no"])
    if (groups["huc2"] == "unknown").any():
        raise ValueError("source river-unit metadata incomplete; do not derive from station prefixes")
    table = pd.DataFrame({"site_no": data["site_no"], **groups})
    table.to_csv(ROOT/"source_domains.csv", index=False)
    t = data["y"].shape[1]
    observed = np.flatnonzero(np.asarray(data["y_mask"]).ravel())
    report = {}
    for val, test in PAIR_ROLES:
        task = f"val{val}_test{test}"
        roles = np.where(groups["huc2"] == val, "val", np.where(groups["huc2"] == test, "test", "train"))
        cells = {role: observed[roles[observed//t] == role] for role in ("train", "val", "test")}
        path = ROOT/"masks"/f"{task}.npz"
        path.parent.mkdir(exist_ok=True)
        if path.exists():
            with np.load(path) as old:
                if any(not np.array_equal(old[role], cells[role]) for role in cells):
                    raise ValueError("source role artifact changed")
        else:
            np.savez_compressed(path, **cells)
        report[task] = {role: {"stations": int(np.sum(roles == role)), "doc_cells": len(cells[role]),
            "domains": np.unique(groups["huc2"][roles == role]).tolist()} for role in cells}
    return report


def confirmation_graph():
    raw = ROOT/"external_build/external_inputs/02030103"
    destination = ROOT/"confirmation"
    destination.mkdir(exist_ok=True)
    if (destination/"complete.json").exists():
        report = json.loads((destination/"complete.json").read_text())
        for name, expected in report["files"].items():
            if sha256_file(destination/name) != expected:
                raise ValueError("confirmation artifact changed")
        return report
    with np.load(raw/"inputs.npz") as archive:
        inputs = {k: archive[k].copy() for k in ("site_no", "months", "x", "x_mask", "regime", "static")}
    # Feature-only eligibility, applied identically to all models and fixed
    # before any prediction: stream orders must satisfy the source schema.
    order = inputs["regime"][:, 0]
    eligible = np.flatnonzero(np.isfinite(order) & (order > 0) & (order == np.round(order)))
    if len(eligible) < 20:
        raise ValueError("confirmation case lacks sufficient source-compatible river covariates")
    originals = inputs["site_no"].copy()
    for name in ("site_no", "x", "x_mask", "regime", "static"):
        inputs[name] = inputs[name][eligible]
    nodes = pd.read_csv(raw/"hydrolocations.csv", dtype={"site_no": str, "comid": str}).set_index("site_no").loc[inputs["site_no"]].reset_index()
    vaa_path = Path("cache/nldplus_vaa.parquet")
    vaa = pd.read_parquet(vaa_path, columns=["comid", "hydroseq", "dnhydroseq"])
    if vaa.comid.duplicated().any() or vaa.hydroseq.duplicated().any():
        raise ValueError("VAA topology keys not unique")
    vaa = vaa.sort_values("hydroseq")
    hydro, downstream, comid = (vaa[k].to_numpy(np.int64) for k in ("hydroseq", "dnhydroseq", "comid"))
    comid_order = np.argsort(comid)
    sorted_comid, walks = comid[comid_order], {}
    for row in nodes.itertuples(index=False):
        position = np.searchsorted(sorted_comid, int(row.comid))
        if position == len(comid) or sorted_comid[position] != int(row.comid):
            raise ValueError("confirmation reach absent from VAA")
        i, seen, path = int(comid_order[position]), set(), []
        while i not in seen:
            seen.add(i)
            path.append(str(comid[i]))
            dn = downstream[i]
            j = int(np.searchsorted(hydro, dn))
            if dn <= 0 or j == len(hydro) or hydro[j] != dn:
                break
            i = j
        else:
            raise ValueError("receiving reach cycle")
        walks[row.site_no] = path
    edges = build_edges(nodes, walks)
    mapping = {site: i for i, site in enumerate(inputs["site_no"])}
    edge_index = np.asarray([(mapping[a], mapping[b]) for a, b in edges.itertuples(index=False, name=None)], dtype=np.int64).reshape(-1, 2).T
    graph_view(edge_index, inputs["regime"][:, 0], np.arange(len(nodes)))
    inputs["regime"][:, 3] = (np.bincount(edge_index[1], minlength=len(nodes)) == 0).astype(np.float32)
    inputs["edge_index"] = edge_index
    source_nodes = pd.read_csv(NODES, dtype={"site_no": str})
    source_comids = set(pd.to_numeric(source_nodes.comid, errors="coerce").dropna().astype(np.int64))
    development_nodes = pd.read_csv("experiments/phase4_transfer/doc_unified_transfer_v2/external/nodes.csv", dtype={"site_no": str})
    if set(inputs["site_no"]) & (set(source_nodes.site_no) | set(development_nodes.site_no)):
        raise ValueError("confirmation stations overlap development/source")
    if source_comids & set(nodes.comid.astype(np.int64)):
        raise ValueError("confirmation/source reaches overlap")
    np.savez_compressed(destination/"inputs.npz", **inputs)
    np.savez_compressed(destination/"eligibility.npz", original_rows=eligible, original_site_no=originals)
    nodes.to_csv(destination/"nodes.csv", index=False)
    edges.to_csv(destination/"edges.csv", index=False)
    report = {"huc8": "02030103", "name": "Hackensack-Passaic", "original_stations": len(originals),
        "eligible_stations": len(eligible), "months": len(inputs["months"]), "edges": len(edges),
        "source_station_and_reach_overlap": 0, "development_station_overlap": 0,
        "eligibility": "positive integer source-compatible stream order, based on covariates only; no error-dependent filtering",
        "receiving_doc_accessed_by_topology_preparer": False,
        "labels": str(raw/"dataset_scoring_only.pt"), "labels_sha256": sha256_file(raw/"dataset_scoring_only.pt"),
        "original_inputs_sha256": sha256_file(raw/"inputs.npz"), "vaa_sha256": sha256_file(vaa_path),
        "files": {name: sha256_file(destination/name) for name in ("inputs.npz", "eligibility.npz", "nodes.csv", "edges.csv")}}
    (destination/"complete.json").write_text(json.dumps(report, indent=2)+"\n")
    return report


if __name__ == "__main__":
    ROOT.mkdir(parents=True, exist_ok=True)
    result = {"source_roles": source_roles(), "confirmation": confirmation_graph()}
    (ROOT/"preparation.json").write_text(json.dumps(result, indent=2)+"\n")
    print(json.dumps(result, indent=2))
