"""Compare fixed river forms on shared observed months and held-region backgrounds."""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from river_graph.analysis.river_form_monthly import (
    ARMS,
    THRESHOLD,
    crossfit_backgrounds,
    information_scores,
    paired_records,
    response_evidence,
    response_summary,
)
from river_graph.analysis.river_mechanisms import permitted_doc
from river_graph.experiments.provenance import sha256_file

ROOT = Path("experiments/phase4_transfer/doc_river_form_monthly_comparison_v1")
PREVIOUS = Path("experiments/phase4_transfer/doc_river_morphology_effect_v1/analysis")
DATASET = Path("data/processed/mississippi_graph_graphfix_st357.pt")
CELLS = Path("experiments/phase4_transfer/doc_river_mechanisms_v1/analysis/source_cells.npy")
PANEL = PREVIOUS/"station_morphology_doc_panel.csv"
PAIRS = PREVIOUS/"covariate_selected_pairs.csv"
CODE = (Path("scripts/analyze_doc_river_form_monthly_comparison_v1.py"),
        Path("src/river_graph/analysis/river_form_monthly.py"),
        Path("src/river_graph/analysis/river_mechanisms.py"),
        Path("src/river_graph/analysis/river_morphology_effect.py"))


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def build_monthly_records(data=None):
    data = torch.load(DATASET, map_location="cpu", weights_only=False) if data is None else data
    panel = pd.read_csv(PANEL, dtype={"station": str, "huc4": str, "huc2": str})
    if panel.station.duplicated().any():
        raise ValueError("one fixed morphology per station required")
    pairs = pd.read_csv(PAIRS, dtype={"station_a": str, "station_b": str, "huc4": str})
    if not set(pairs.station_a).union(pairs.station_b).issubset(set(panel.station)):
        raise ValueError("all matched stations must belong to the fixed panel")
    visible = permitted_doc(data, np.load(CELLS))
    index = {str(s): i for i, s in enumerate(data["site_no"])}
    dates = pd.DatetimeIndex(data["months"])
    if not dates.to_period("M").equals(pd.period_range(dates[0], dates[-1], freq="M")):
        raise ValueError("continuous unique monthly calendar required")
    x, mask = np.asarray(data["x"], float), np.asarray(data["x_mask"], bool)
    qi, ti = (list(data["feature_channels"]).index(c) for c in ("discharge", "temperature"))
    rows = []
    static = ["log_polygon_area", "wetland", "forest", "agriculture", "urban", "precipitation", "climate_temperature",
              "latitude", "longitude", "log_basin_aspect", "log_network_axis_ratio", "log_drainage_density",
              "mainstem_share", "mainstem_sinuosity", "log_route_mean_scaled", "route_distance_cv"]
    for r in panel.itertuples():
        site = index[r.station]
        months = np.flatnonzero(np.isfinite(visible[site]))
        time = dates[months]
        angle = 2*np.pi*time.month.to_numpy()/12
        q, temperature = x[site, months, qi], x[site, months, ti]
        q_valid = mask[site, months, qi] & np.isfinite(q) & (q > 0)
        t_valid = mask[site, months, ti] & np.isfinite(temperature)
        f = pd.DataFrame({"station": r.station, "comid": int(r.comid), "huc4": r.huc4, "huc2": r.huc2,
                          "cluster": int(r.cluster), "month_index": months, "date": time,
                          "calendar_month": time.month.to_numpy(), "calendar_year": time.year.to_numpy()+(time.month.to_numpy()-1)/12,
                          "month_sin": np.sin(angle), "month_cos": np.cos(angle), "y_true": visible[site, months],
                          "log_discharge": np.log1p(np.where(q_valid, q, np.nan)),
                          "temperature": np.where(t_valid, temperature, np.nan),
                          "discharge_valid": q_valid, "temperature_valid": t_valid,
                          "form_2": float(r.cluster == 2), "form_3": float(r.cluster == 3)})
        for name in static:
            f[name] = getattr(r, name)
        rows.append(f)
    frame = pd.concat(rows, ignore_index=True)
    if frame.station.nunique() != len(panel) or frame.duplicated(["station", "month_index"]).any():
        raise ValueError("all fixed stations require unique observed source cells")
    return frame, pairs


def hydro_balance(records):
    rows = []
    for pair, f in records.groupby("pair_id"):
        r = f.iloc[0]
        for population in ("common_doc", "complete_hydro", "at_least_24_months"):
            s = f[f.complete_hydro] if population == "complete_hydro" else f
            if len(s) < (24 if population == "at_least_24_months" else 12) or pd.DatetimeIndex(s.date).month.nunique() < 6:
                continue
            row = {"pair_id": pair, "class_a": r.class_a, "class_b": r.class_b, "huc4": r.huc4,
                   "population": population, "n_months": len(s), "first_month": s.date.min(), "last_month": s.date.max()}
            for name in ("log_discharge", "temperature"):
                joint = np.isfinite(s[name+"_a"]) & np.isfinite(s[name+"_b"])
                row["n_joint_"+name] = int(joint.sum())
                row[name+"_difference"] = (s.loc[joint, name+"_b"]-s.loc[joint, name+"_a"]).mean()
            for name in ("discharge_valid", "temperature_valid"):
                row[name+"_fraction_a"] = s[name+"_a"].mean()
                row[name+"_fraction_b"] = s[name+"_b"].mean()
            rows.append(row)
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bootstrap-draws", type=int, default=5000)
    args = parser.parse_args()
    out = ROOT/"analysis"
    out.mkdir(parents=True, exist_ok=True)
    frame, pairs = build_monthly_records()
    predictions, states = crossfit_backgrounds(frame)
    records, ledger = paired_records(frame, pairs, predictions)
    evidence = response_evidence(records)
    contrasts, sensitivity = response_summary(evidence, args.bootstrap_draws)
    station, scores, gains = information_scores(predictions, args.bootstrap_draws)
    tables = {"pair_inclusion_ledger": ledger, "paired_response_evidence": evidence,
              "paired_response_contrasts": contrasts, "omitted_huc4_sensitivity": sensitivity,
              "hydro_balance": hydro_balance(records), "station_model_errors": station,
              "information_scores": scores, "information_gains": gains}
    for name, f in tables.items():
        f.to_csv(out/f"{name}.csv", index=False)
    frame.to_parquet(out/"station_month_inputs.parquet", index=False)
    predictions.to_parquet(out/"monthly_predictions.parquet", index=False)
    records.to_parquet(out/"paired_month_records.parquet", index=False)
    (out/"fitted_states.json").write_text(json.dumps(states, indent=2)+"\n")
    config = {"arms": ARMS, "high_doc_threshold_mg_l": THRESHOLD, "ridge_alpha": 10., "logistic_C": 1.,
              "logistic_max_iter": 1000, "cv": "five station-defined HUC4 folds", "training_weight": "one per station",
              "evaluation_weight": "station equal for information; pair equal after same-month responses",
              "pair_min_months": 12, "pair_min_months_of_year": 6, "longer_record_months": 24,
              "bootstrap_draws": args.bootstrap_draws, "bootstrap_seed": 42, "bootstrap_unit": "HUC4",
              "target_transform": "log1p", "source_roles": "union142/143/144", "matching": "unchanged preceding pairs"}
    (ROOT/"config.json").write_text(json.dumps(config, indent=2)+"\n")
    for path in CODE:
        target = ROOT/"code_snapshot"/path
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, target)
    runtime = {str(path): sha256_file(path) for path in CODE}
    sidecar = {"dataset_hash": sha256_file(DATASET), "config_hash": digest(config),
               "mask_hash": hashlib.sha256(frame[["station", "month_index", "huc4"]].to_csv(index=False).encode()).hexdigest(),
               "runtime_snapshot_hash": digest(runtime), "runtime_sources": runtime,
               "prediction_sha256": sha256_file(out/"monthly_predictions.parquet"),
               "source_cells_sha256": sha256_file(CELLS), "study_plan_sha256": sha256_file(ROOT/"study_plan.md"),
               "prediction_grain": "permitted observed station-month x fixed information arm",
               "visibility": "held HUC4 receiving DOC only scored; background uses other source HUC4 labels",
               "interpretation": "monthly morphology diagnostic; no released model retraining or external validation"}
    (out/"monthly_predictions.provenance.json").write_text(json.dumps(sidecar, indent=2)+"\n")
    sources = [DATASET, CELLS, PANEL, PAIRS, ROOT/"study_plan.md", *CODE]
    record = {"source_hashes": {str(p): sha256_file(p) for p in sources}, "bootstrap_draws": args.bootstrap_draws,
              "n_stations": frame.station.nunique(), "n_huc4": frame.huc4.nunique(), "n_source_months": len(frame),
              "n_pairs": ledger.included.sum().item(), "existing_neural_models_retrained": False,
              "current_date": "2026-10-07", "previous_results_seen": True}
    (ROOT/"analysis_sources.json").write_text(json.dumps(record, indent=2)+"\n")
    print(json.dumps({k: v for k, v in record.items() if k != "source_hashes"}, indent=2))
    print(contrasts[contrasts.class_a.eq(1) & contrasts.class_b.eq(3) & contrasts.population.eq("common_doc")].to_string(index=False))
    print(gains.to_string(index=False))


if __name__ == "__main__":
    main()
