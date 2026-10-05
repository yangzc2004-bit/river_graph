"""Prepare value-blind HUC4 roles and screen existing external DOC inventories."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from run_unified_doc_spatial import write_json

from river_graph.data.wqp import extract_doc_obs, load_station_results
from river_graph.experiments.provenance import sha256_file
from river_graph.experiments.spatial_fewshot import support_schedule
from river_graph.experiments.transfer_data import _external_station_ids, _wqp_complete
from river_graph.experiments.unmonitored_doc import HUC4_BLOCKS, geographical_split

ROOT = Path("experiments/phase4_transfer/doc_unmonitored_tasks_v1")
DATASET = Path("data/processed/mississippi_graph_graphfix_st357.pt")
NODES = Path("data/processed/graph_nodes_graphfix_st357.csv")


def prepare_geography(root, dataset, nodes):
    names = np.asarray(dataset["site_no"], dtype=str)
    table = pd.read_csv(nodes, dtype=str).set_index("site_no")
    if not table.index.is_unique or not set(names) <= set(table.index):
        raise ValueError("graph-node site identities do not match the dataset")
    codes = table.loc[names, "huc_cd"].tolist()
    mask = np.asarray(dataset["y_mask"], dtype=bool)
    months = mask.shape[1]
    counts = mask.sum(1)
    rows = []
    for target in HUC4_BLOCKS:
        split, metadata = geographical_split(mask, codes, target)
        path = root / "geographical_masks" / f"huc4_{target}.npz"
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.exists():
            with np.load(path, allow_pickle=False) as previous:
                for role, cells in split.items():
                    np.testing.assert_array_equal(previous[role], cells)
        else:
            np.savez_compressed(path, **split)
        record = {"target_huc4": target, "validation_huc4": metadata["validation_huc4"],
                  "mask_path": str(path), "mask_hash": sha256_file(path)}
        for role in ("train", "val", "test"):
            cells = split[role]
            record[f"{role}_stations"] = len(np.unique(cells//months))
            record[f"{role}_cells"] = len(cells)
        eligible = split["test"][counts[split["test"]//months] >= 6]
        schedule, query = support_schedule(eligible, months)
        record["curve_stations"], record["curve_query_cells"] = len(schedule), len(query)
        record["test_sparse_stations"] = int((counts[np.unique(split["test"]//months)] < 6).sum())
        write_json(path.with_suffix(".json"), {**record, **metadata,
            "dataset_hash": sha256_file(DATASET), "nodes_hash": sha256_file(nodes),
            "huc_source": "graph_nodes.huc_cd, canonical zero-padded8-digit codes",
            "information_condition": "all target-station water-quality inputs hidden"})
        rows.append(record)
    pd.DataFrame(rows).to_csv(root / "geographical_tasks.csv", index=False)
    return rows


def external_screen(huc8, st_names, root):
    raw = Path("data/raw/transfer_external_inventory") / f"candidate_{huc8}"
    stations, results = raw / "doc_stations.csv", raw / "doc_results.csv"
    allowed = _external_station_ids(stations, huc8)-set(st_names)
    if not _wqp_complete(results):
        return {"huc8": huc8, "availability_pass": False, "reason": "incomplete DOC raw response"}
    obs = extract_doc_obs(load_station_results(results))
    obs = obs[obs.site_no.isin(allowed) & np.isfinite(obs.doc) & (obs.doc >= 0)].copy()
    obs["month"] = obs.date.dt.to_period("M")
    # Match the project's monthly arithmetic-mean DOC aggregation.
    labels = obs.groupby(["site_no", "month"], as_index=False).doc.mean()
    count = labels.groupby("site_no").size()
    eligible = count[count >= 6].index
    labels = labels[labels.site_no.isin(eligible)].copy()
    table = pd.read_csv(stations, dtype=str)
    table["site_no"] = table.Location_Identifier.str.removeprefix("USGS-")
    table = table[table.site_no.isin(eligible)].drop_duplicates("site_no")
    labels["month"] = labels.month.astype(str)
    span = (pd.Period(labels.month.max(), "M").ordinal-pd.Period(labels.month.min(), "M").ordinal+1) if len(labels) else 0
    record = {"huc8": huc8, "station_count_ge6": len(eligible), "station_months": len(labels),
              "calendar_span_months": int(span), "curve_query_cells": len(labels)-5*len(eligible),
              "st357_overlap_after_filter": 0, "min_month": labels.month.min() if len(labels) else None,
              "max_month": labels.month.max() if len(labels) else None,
              "station_file": str(stations), "station_file_hash": sha256_file(stations),
              "result_file": str(results), "result_file_hash": sha256_file(results),
              "selection_basis": "DOC availability only; no prediction outcomes",
              "feature_pipeline_status": "pending ecological/hydro feature construction",
              "provider_completeness": "raw response structurally complete; provider totals not independently reconciled"}
    record["availability_pass"] = bool(len(eligible) >= 50 and span >= 36 and record["curve_query_cells"] >= 1000)
    destination = root / "external_availability" / huc8
    destination.mkdir(parents=True, exist_ok=True)
    labels.to_parquet(destination / "monthly_doc.parquet", index=False)
    table.to_csv(destination / "stations.csv", index=False)
    write_json(destination / "screen.json", record)
    return record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    args = parser.parse_args()
    args.root.mkdir(parents=True, exist_ok=True)
    dataset = torch.load(DATASET, map_location="cpu", weights_only=False)
    geography = prepare_geography(args.root, dataset, NODES)
    candidates = [external_screen(code, dataset["site_no"], args.root) for code in ("02040104", "02030103")]
    first = next((candidate["huc8"] for candidate in candidates if candidate["availability_pass"]), None)
    protocol = {"version": 1, "date": "2026-10-05",
        "main_endpoint": "K0 native DOC MAE at entirely unmonitored stations",
        "input_conditions": ["no target water-quality inputs", "optional pH/conductance enhancement"],
        "dataset_hash": sha256_file(DATASET), "nodes_hash": sha256_file(NODES),
        "geographical_targets": list(HUC4_BLOCKS), "training_seeds_initial": [42, 43, 44],
        "training_seeds_confirmation": [42, 43, 44, 45, 46], "k_values": [0, 1, 3, 5],
        "k0_population": "all valid test DOC cells, without five-support reservation",
        "curve_population": "stations with>=6observations; same five candidates reserved at everyK",
        "external_doc_selection_order": ["02040104", "02030103"],
        "external_limits": {"stations_ge6": 50, "calendar_months": 36, "curve_queries": 1000},
        "external_priority_after_availability": first,
        "external_selection_complete": False, "external_remaining_requirement": "constructible common ecology/hydro inputs",
        "external_scope": "new DOC-only study; historical three-analyte protocol retained"}
    selected_path = args.root / "external_protocol_v1.json"
    if selected_path.exists():
        selected = json.loads(selected_path.read_text())
        protocol.update(external_selection_complete=True, external_selected_huc8=selected["selected_huc8"],
                        external_selection_manifest=str(selected_path))
        protocol.pop("external_remaining_requirement", None)
    write_json(args.root / "task_protocol.json", protocol)
    write_json(args.root / "availability_report.json", {"geography": geography, "external_candidates": candidates})
    print(pd.DataFrame(geography).to_string(index=False))
    print(pd.DataFrame(candidates)[["huc8", "station_count_ge6", "station_months", "curve_query_cells", "availability_pass"]].to_string(index=False))


if __name__ == "__main__":
    main()
