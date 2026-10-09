"""Audit cached detailed ecology before its source-only DOC comparison."""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from run_unified_doc_spatial import write_json

from river_graph.experiments.provenance import sha256_file
from river_graph.models.ecological_composition import FIELDS, composition_inputs

ROOT = Path("experiments/phase4_transfer/doc_ecological_composition_v1")
DATASET = Path("data/processed/mississippi_graph_graphfix_st357.pt")
NODES = Path("data/processed/graph_nodes_graphfix_st357.csv")
ATTRIBUTES = Path("data/processed/streamcat_attributes.csv")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    args = parser.parse_args()
    args.root.mkdir(parents=True, exist_ok=True)
    data = torch.load(DATASET, weights_only=False, map_location="cpu")
    nodes = pd.read_csv(NODES, dtype={"site_no": str, "comid": str})
    attrs = pd.read_csv(ATTRIBUTES, dtype={"comid": str})
    inputs = composition_inputs(data["site_no"], nodes, attrs)
    valid = inputs["valid"]
    profile = pd.DataFrame({"field": FIELDS, "available_stations": valid.sum(axis=0),
        "availability_fraction": valid.mean(axis=0), "valid_zero_stations": ((inputs["detailed"][:, :11] == 0) & valid).sum(axis=0)})
    profile.to_csv(args.root/"field_availability.csv", index=False)
    regime = np.asarray(data["regime"], float)[:, 4:8]
    families_valid = np.column_stack([valid[:, indices].all(axis=1) for indices in ((0, 1, 2), (3, 4), (5, 6, 7, 8), (9, 10))])
    comparison = families_valid & np.isfinite(regime) & (regime >= 0)
    difference = np.abs(inputs["family_totals"]*100-regime)
    maximum = float(difference[comparison].max()) if comparison.any() else None
    station = pd.DataFrame({"station": inputs["site_no"], "comid": inputs["comid"],
        "matched_comid": inputs["matched_comid"], "n_available_fields": valid.sum(axis=1),
        "n_reconciled_groups": comparison.sum(axis=1),
        "max_aggregate_difference_pct": np.max(np.where(comparison, difference, 0.), axis=1)})
    station.to_csv(args.root/"station_alignment.csv", index=False)
    record = {"station_count": len(valid), "streamcat_rows": len(attrs),
        "matched_stations": int(inputs["matched_comid"].sum()), "all_fields_available": int(valid.all(axis=1).sum()),
        "n_aggregate_comparisons": int(comparison.sum()), "max_aggregate_difference_pct": maximum,
        "reconciliation_tolerance_pct": 1e-3, "aggregate_alignment_ok": maximum is not None and maximum <= 1e-3,
        "uses_doc_labels": False, "temporal_note": "2019 static land-cover proxy, as in the retained regime; not historical land-cover reconstruction",
        "sources": {str(path): sha256_file(path) for path in (DATASET, NODES, ATTRIBUTES)},
        "audit_sha256": sha256_file(__file__), "feature_names": inputs["feature_names"]}
    write_json(args.root/"data_audit.json", record)
    np.savez_compressed(args.root/"composition_inputs.npz", detailed=inputs["detailed"],
        aggregate_control=inputs["aggregate_control"], site_no=inputs["site_no"], valid=valid)
    print(record)
    if not record["aggregate_alignment_ok"]:
        raise ValueError("resolve physical catchment alignment before source fitting")


if __name__ == "__main__":
    main()
