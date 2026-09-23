"""Build the ST-only core dataset from an audited graphfix dataset.

This is a cohort restriction, not a new data download: all monthly labels and
covariates retain their original values, while non-river station types are
removed and the induced station graph is re-indexed.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
import torch


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--dataset",
        default="data/processed/mississippi_graph_graphfix_cached370.pt",
    )
    ap.add_argument(
        "--out",
        default="data/processed/mississippi_graph_graphfix_st357.pt",
    )
    ap.add_argument(
        "--nodes-out",
        default="data/processed/graph_nodes_graphfix_st357.csv",
    )
    ap.add_argument(
        "--edges-out",
        default="data/processed/graph_edges_graphfix_st357.csv",
    )
    args = ap.parse_args()

    source = Path(args.dataset)
    out = Path(args.out)
    dataset = torch.load(source, weights_only=False)
    sites = [str(s) for s in dataset["site_no"]]
    nodes = pd.read_csv("data/processed/graph_nodes.csv", dtype={"site_no": str})
    nodes = nodes[nodes.site_no.isin(sites)].copy()
    nodes = nodes.set_index("site_no").loc[sites].reset_index()
    keep = nodes["site_tp_cd"].eq("ST").to_numpy()
    keep_idx = torch.as_tensor(keep.copy(), dtype=torch.bool)
    kept_sites = [s for s, ok in zip(sites, keep, strict=True) if ok]
    old_to_new = {
        old: new for new, old in enumerate(torch.nonzero(keep_idx).flatten().tolist())
    }

    edge_index = dataset["edge_index"]
    edge_keep = keep_idx[edge_index[0]] & keep_idx[edge_index[1]]
    old_edges = edge_index[:, edge_keep]
    new_edges = torch.tensor(
        [[old_to_new[int(v)] for v in old_edges[row]] for row in range(2)],
        dtype=torch.long,
    )
    reduced = {}
    for key, value in dataset.items():
        if key in {"site_no", "edge_index", "edge_attr"}:
            continue
        if isinstance(value, torch.Tensor) and value.shape[0] == len(sites):
            reduced[key] = value[keep_idx]
        else:
            reduced[key] = value
    reduced["site_no"] = kept_sites
    reduced["edge_index"] = new_edges
    if "edge_attr" in dataset:
        reduced["edge_attr"] = dataset["edge_attr"][edge_keep]

    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(reduced, out)
    node_out = nodes[keep].copy()
    edge_rows = pd.DataFrame({
        "source": [kept_sites[int(i)] for i in new_edges[0]],
        "target": [kept_sites[int(i)] for i in new_edges[1]],
    })
    Path(args.nodes_out).parent.mkdir(parents=True, exist_ok=True)
    node_out.to_csv(args.nodes_out, index=False)
    edge_rows.to_csv(args.edges_out, index=False)

    record = {
        "source_dataset": str(source),
        "output_dataset": str(out),
        "source_nodes": len(sites),
        "output_nodes": len(kept_sites),
        "source_edges": int(edge_index.shape[1]),
        "output_edges": int(new_edges.shape[1]),
        "kept_site_type": "ST",
        "excluded_station_types": nodes.loc[~keep, "site_tp_cd"].value_counts().to_dict(),
        "excluded_sites": nodes.loc[~keep, "site_no"].tolist(),
        "months": len(reduced["months"]),
        "observed_doc_cells": int(reduced["y_mask"].sum()),
    }
    Path(str(out).replace(".pt", ".provenance.json")).write_text(
        json.dumps(record, indent=2), encoding="utf-8"
    )
    print(json.dumps(record, indent=2))


if __name__ == "__main__":
    main()
