"""Measure changing source-water participation on fixed actual river arrangements."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from river_graph.analysis.river_active_structure import (
    HYDRO_METRICS,
    hydro_contrasts,
    observed_mixing,
    participation,
)
from river_graph.analysis.river_form_process import cluster_mean
from river_graph.analysis.river_mechanisms import permitted_doc
from river_graph.analysis.river_signal_timescale import source_geometry
from river_graph.experiments.provenance import sha256_file

BASE = Path("experiments/phase4_transfer")
ROOT = BASE/"doc_river_active_structure_v1"
PATHS = BASE/"doc_river_pathway_context_v1/analysis"
MONTHLY = BASE/"doc_river_monitored_arrivals_v1/analysis/monthly_series.parquet"
DATA = Path("data/processed/mississippi_graph_graphfix_st357.pt")
CELLS = BASE/"doc_river_mechanisms_v1/analysis/source_cells.npy"
INPUTS = (DATA, CELLS, MONTHLY, PATHS/"source_corridor_reaches.csv",
          PATHS/"receiver_pathway_panel.csv", ROOT/"study_plan.md")
CODE = (Path("scripts/analyze_doc_river_active_structure_v1.py"),
        Path("src/river_graph/analysis/river_active_structure.py"),
        Path("src/river_graph/analysis/river_joint_campaigns.py"),
        Path("src/river_graph/analysis/river_signal_timescale.py"),
        Path("src/river_graph/analysis/river_form_process.py"),
        Path("src/river_graph/analysis/river_mechanisms.py"))
TYPES = {"target": str, "source_station": str, "huc4": str}
DOC_METRICS = ("mixing_potential_change_pp", "weights_contribution_pp", "correlation_contribution_pp",
               "sd_contribution_pp", "log_ratio_change")


def build_tables():
    data = torch.load(DATA, weights_only=False, map_location="cpu")
    cells = np.load(CELLS)
    visible = permitted_doc(data, cells)
    lookup = {str(s): i for i, s in enumerate(data["site_no"])}
    dates = pd.DatetimeIndex(data["months"])
    hydro = list(data["feature_channels"]).index("discharge")
    q, qm = np.asarray(data["x"])[..., hydro].astype(float), np.asarray(data["x_mask"])[..., hydro].astype(bool)
    panel = pd.read_csv(INPUTS[4], dtype=TYPES)
    paths = pd.read_csv(INPUTS[3], dtype=TYPES)
    monthly = pd.read_parquet(MONTHLY)
    rows, source_rows, availability, hydro_rows, doc_rows, reference_rows = [], [], [], [], [], []
    for r in panel.itertuples():
        source, geometry = source_geometry(paths.loc[paths.target.eq(r.target)])
        source_ids = [lookup[s] for s in source.source_station]
        source_paths = source.path_km.to_numpy(float)
        reference = geometry["path_mean_km"]
        original = monthly.loc[monthly.target.eq(r.target)].sort_values("date")
        if original.empty:
            raise ValueError(f"Fixed monitored frontier has no original calendar: {r.target}")
        span = (dates >= original.date.min()) & (dates <= original.date.max())
        slots = np.flatnonzero(span)
        ids = [*source_ids, lookup[r.target]]
        measured = qm[ids][:, slots].T & np.isfinite(q[ids][:, slots].T) & (q[ids][:, slots].T > 0)
        raw_q = q[ids][:, slots].T
        receiver_valid = measured[:, -1]
        complete = measured.all(axis=1)
        meta = {"target": r.target, "cluster": int(r.cluster), "huc4": r.huc4,
                "component": int(r.component), "n_sources": len(source),
                "covered_area_fraction": r.covered_area_fraction}
        lo, hi = np.quantile(raw_q[receiver_valid, -1], [1/3, 2/3]) if receiver_valid.any() else (np.nan, np.nan)
        states = np.select([raw_q[:, -1] <= lo, raw_q[:, -1] >= hi], ["low", "high"], default="middle")
        reference_rows.append({**meta, "q_low": lo, "q_high": hi,
            "n_receiver_flow_reference_months": int(receiver_valid.sum()),
            "first_month": str(dates[slots].min().date()), "last_month": str(dates[slots].max().date()),
            "reference_mean_path_km": reference, "common_km": geometry["common_km"]})
        frame = participation(source_paths, raw_q[complete, :-1], reference_mean=reference,
                              common_km=geometry["common_km"])
        frame["date"] = dates[slots[complete]]
        frame["month_index"] = slots[complete]
        frame["flow_state"] = states[complete]
        frame["source_receiver_flow_share"] = raw_q[complete, :-1].sum(axis=1)/raw_q[complete, -1]
        rows.append(frame.assign(**meta))
        weights = raw_q[complete, :-1]/raw_q[complete, :-1].sum(axis=1, keepdims=True)
        for j, site in enumerate(source.source_station):
            source_rows.append(pd.DataFrame({**meta, "date": dates[slots[complete]],
                "source_station": site, "source_order": j, "discharge_cfs": raw_q[complete, j],
                "flow_weight": weights[:, j], "area_weight": source.area_weight.iloc[j],
                "fixed_path_km": source_paths[j]}))
        ledger = {**meta, "n_span_months": len(slots), "n_complete_positive_flow_months": int(complete.sum()),
            "n_incomplete_source_flow_months": int((~measured[:, :-1].all(axis=1)).sum()),
            "n_incomplete_receiver_flow_months": int((~receiver_valid).sum()),
            "hydro_eligible": False, "doc_eligible": False}
        try:
            contrast = hydro_contrasts(frame)
            hydro_rows.append({**meta, **contrast})
            ledger["hydro_eligible"] = True
            ledger["hydro_status"] = "eligible"
        except ValueError as error:
            ledger["hydro_status"] = str(error)
        doc_slots = original.month_index.to_numpy(int)
        eligible_slots = np.intersect1d(doc_slots, slots[complete])
        selected = frame.loc[frame.month_index.isin(eligible_slots)].sort_values("month_index")
        a = visible[source_ids][:, selected.month_index].T
        y = visible[lookup[r.target], selected.month_index]
        if not np.isfinite(np.r_[a.ravel(), y]).all():
            raise ValueError("Original common-DOC dates must be permitted source-training cells")
        ledger["n_common_doc_positive_flow_months"] = len(selected)
        try:
            doc_weights = q[source_ids][:, selected.month_index].T
            result = observed_mixing(a, y, doc_weights, selected.date, selected.flow_state)
            doc_rows.append({**meta, **result})
            ledger["doc_eligible"] = True
            ledger["doc_status"] = "eligible"
        except ValueError as error:
            ledger["doc_status"] = str(error)
        availability.append(ledger)
    return {"monthly_participation": pd.concat(rows, ignore_index=True),
        "source_flow_weights": pd.concat(source_rows, ignore_index=True),
        "availability": pd.DataFrame(availability), "flow_references": pd.DataFrame(reference_rows),
        "network_hydro_changes": pd.DataFrame(hydro_rows), "network_doc_changes": pd.DataFrame(doc_rows)}


def group_summaries(table, metrics, *, draws, population):
    rows = []
    for group, selected in (("all", table), *((f"class_{i}", table.loc[table.cluster.eq(i)]) for i in (1, 2, 3)),
                             ("broad_minus_elongated", table.loc[table.cluster.isin([1, 3])])):
        contrast = group == "broad_minus_elongated"
        for metric in metrics:
            result = cluster_mean(selected, metric, "component", draws=draws, contrast=contrast)
            if result is None:
                continue
            enough = (min(selected.loc[selected.cluster.eq(i), "component"].nunique() for i in (1, 3)) >= 3
                      if contrast else selected.component.nunique() >= 3)
            enough &= result["valid_bootstrap_draws"] >= .9*draws
            if not enough:
                result.update(ci_low=np.nan, ci_high=np.nan, interval_status="insufficient_independent_system_support")
            rows.append({"population": population, "group": group, **result})
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bootstrap-draws", type=int, default=5000)
    args = parser.parse_args()
    tables = build_tables()
    hmetrics = [m+suffix for m in HYDRO_METRICS for suffix in ("_raw_change", "_adjusted_change")]
    tables["group_summary"] = pd.concat([
        group_summaries(tables["network_hydro_changes"], hmetrics, draws=args.bootstrap_draws, population="hydro_only"),
        group_summaries(tables["network_doc_changes"], DOC_METRICS, draws=args.bootstrap_draws, population="doc_supported")], ignore_index=True)
    out = ROOT/"analysis"
    out.mkdir(parents=True, exist_ok=True)
    for name, table in tables.items():
        if name in ("monthly_participation", "source_flow_weights"):
            table.to_parquet(out/f"{name}.parquet", index=False)
        else:
            table.to_csv(out/f"{name}.csv", index=False)
    summary = {"n_fixed_networks": len(tables["availability"]),
        "n_hydro_eligible_networks": len(tables["network_hydro_changes"]),
        "n_doc_eligible_networks": len(tables["network_doc_changes"]),
        "n_complete_flow_network_months": len(tables["monthly_participation"]),
        "n_hydro_overlap_systems": int(tables["network_hydro_changes"].component.nunique()),
        "bootstrap_draws": args.bootstrap_draws,
        "new_prediction_training": False, "whole_network_activation_measured": False,
        "scope": "Measured monthly flow participation of fixed monitored frontiers; DOC restricted to source roles"}
    (out/"summary.json").write_text(json.dumps(summary, indent=2)+"\n")
    (ROOT/"analysis_sources.json").write_text(json.dumps({
        "input_hashes": {str(p): sha256_file(p) for p in INPUTS},
        "code_hashes": {str(p): sha256_file(p) for p in CODE},
        "output_hashes": {str(p): sha256_file(p) for p in sorted(out.iterdir())},
        "unit": "fixed receiving network; bootstrap complete overlapping systems",
        "discharge_unit": "dataset monthly cfs; normalized shares are unit independent"}, indent=2)+"\n")
    print(json.dumps(summary, indent=2), flush=True)
    print(tables["availability"].groupby("cluster")[["hydro_eligible", "doc_eligible"]].sum().to_string(), flush=True)


if __name__ == "__main__":
    main()
