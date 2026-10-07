"""Test terrestrial source amount versus placement on source-cohort DOC."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from river_graph.analysis.river_doc_structure import (
    bootstrap_ols,
    source_station_response,
)
from river_graph.analysis.river_planform_doc import CONTROLS, SHAPE
from river_graph.analysis.river_source_placement import PLACEMENT
from river_graph.experiments.provenance import sha256_file

ROOT = Path("experiments/phase4_transfer/doc_river_source_placement_v1")
PANEL = Path("experiments/phase4_transfer/doc_river_planform_doc_v1/analysis/station_doc_response.csv")
CELLS = Path("experiments/phase4_transfer/doc_river_mechanisms_v1/analysis/source_cells.npy")
DATASET = Path("data/processed/mississippi_graph_graphfix_st357.pt")
ARMS = {
    "environment_area": list(CONTROLS),
    "environment_area_shape": [*CONTROLS, *SHAPE],
    "environment_area_placement": [*CONTROLS, *PLACEMENT],
    "environment_area_shape_placement": [*CONTROLS, *SHAPE, *PLACEMENT],
}


def blocked_predictions(panel, population):
    s = panel[panel.included].copy().reset_index(drop=True)
    if s.huc4.nunique() < 5:
        raise ValueError("insufficient HUC4 groups for blocked comparison")
    y = np.log1p(s.doc_median.to_numpy(float))
    rows = []
    for fold, (train, test) in enumerate(GroupKFold(n_splits=5).split(s, groups=s.huc4)):
        for arm, numeric in ARMS.items():
            prep = ColumnTransformer([
                ("num", make_pipeline(SimpleImputer(strategy="median", add_indicator=True), StandardScaler()), numeric),
                ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), ["huc2"])])
            model = make_pipeline(prep, Ridge(alpha=10))
            model.fit(s.iloc[train], y[train])
            p = model.predict(s.iloc[test])
            for i, estimate in zip(test, p, strict=True):
                rows.append({"population": population, "station": s.station.iloc[i], "huc4": s.huc4.iloc[i],
                    "fold": fold, "model": arm, "y_log1p": y[i], "y_pred_log1p": estimate,
                    "y_true": s.doc_median.iloc[i], "y_pred": max(0., np.expm1(estimate))})
    return pd.DataFrame(rows)


def comparisons(predictions, draws):
    rows = []
    pairs = [("environment_area_shape", "environment_area"),
             ("environment_area_placement", "environment_area"),
             ("environment_area_shape_placement", "environment_area_shape"),
             ("environment_area_shape_placement", "environment_area_placement")]
    for population, p in predictions.groupby("population"):
        for space, truth, pred in (("native", "y_true", "y_pred"), ("log1p", "y_log1p", "y_pred_log1p")):
            p = p.copy()
            p["error"] = abs(p[truth]-p[pred])
            table = p.pivot(index=["station", "huc4", "fold"], columns="model", values="error").reset_index()
            folds = [table.fold.to_numpy() == fold for fold in sorted(table.fold.unique())]
            for candidate, reference in pairs:
                c, b = table[candidate].to_numpy(), table[reference].to_numpy()
                point_b, point_c = np.mean([b[f].mean() for f in folds]), np.mean([c[f].mean() for f in folds])
                for unit in ("station", "huc4"):
                    _, indices = np.unique(table[unit], return_inverse=True)
                    n = indices.max()+1
                    rng = np.random.default_rng(42)
                    boot = []
                    while len(boot) < draws:
                        w = rng.multinomial(n, np.full(n, 1/n))[indices]
                        if any(w[f].sum() == 0 for f in folds):
                            continue
                        mb = np.mean([np.average(b[f], weights=w[f]) for f in folds])
                        mc = np.mean([np.average(c[f], weights=w[f]) for f in folds])
                        boot.append(100*(mb-mc)/mb)
                    lo, hi = np.quantile(boot, [.025, .975])
                    rows.append({"population": population, "space": space, "candidate": candidate,
                        "reference": reference, "resampling_unit": unit, "reference_mae": point_b,
                        "candidate_mae": point_c, "gain_pct": 100*(point_b-point_c)/point_b,
                        "ci_low_pct": lo, "ci_high_pct": hi, "n_stations": len(table),
                        "n_huc4": table.huc4.nunique(), "positive_folds": sum(c[f].mean() < b[f].mean() for f in folds)})
    return pd.DataFrame(rows)


def associations(panel, draws):
    rows, diagnostics = [], []
    for outcome, transform in (("doc_median", "log1p"), ("doc_cv", "log1p"), ("harmonic_amplitude", "log1p")):
        s = panel[panel.included & panel[outcome].notna()].copy()
        numeric = [*CONTROLS, *SHAPE, *PLACEMENT]
        imputer = SimpleImputer(strategy="median", add_indicator=True)
        x = imputer.fit_transform(s[numeric])
        names = imputer.get_feature_names_out(numeric)
        variable = x.std(0) > 1e-9
        region = pd.get_dummies(s.huc2, prefix="huc2", drop_first=True, dtype=float)
        labels = ["intercept", *names[variable], *region.columns]
        x = np.column_stack([np.ones(len(s)), StandardScaler().fit_transform(x[:, variable]), region.to_numpy()])
        y = np.log1p(s[outcome].to_numpy(float)) if transform == "log1p" else s[outcome].to_numpy(float)
        point, boot, diag = bootstrap_ols(x, y, draws=draws, groups=s.huc4)
        diagnostics.append({"outcome": outcome, **diag})
        for i, term in enumerate(labels):
            if i in diag["dropped_columns"] or term not in (*PLACEMENT, *SHAPE):
                continue
            lo, hi = np.quantile(boot[:, i], [.025, .975])
            rows.append({"outcome": outcome, "term": term, "coefficient_per_station_sd": point[i],
                "huc4_ci_low": lo, "huc4_ci_high": hi, "n_stations": len(s), "n_huc4": s.huc4.nunique()})
    return pd.DataFrame(rows), diagnostics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bootstrap-draws", type=int, default=5000)
    args = parser.parse_args()
    source = ROOT/"analysis/station_source_placement.csv"
    placement = pd.read_csv(source, dtype={"station": str, "huc_cd": str})
    panel = pd.read_csv(PANEL, dtype={"station": str, "huc_cd": str, "huc2": str, "huc4": str})
    panel = panel.merge(placement.drop(columns=["comid", "huc_cd"]), on="station", how="left", validate="one_to_one")
    panel["included"] = panel.eligible & panel.inclusion_status.eq("included_landscape")
    panel.to_csv(ROOT/"analysis/station_doc_source_panel.csv", index=False)
    print(f"Source analysis: {panel.included.sum()}/{len(panel)} classified source stations", flush=True)
    predictions = [blocked_predictions(panel, "all_source_months")]
    dataset = torch.load(DATASET, map_location="cpu", weights_only=False)
    cells = np.load(CELLS)
    dates = pd.DatetimeIndex(dataset["months"])
    recent = cells[dates[cells % len(dates)].year >= 2009]
    recent_response, _, _ = source_station_response(dataset, recent)
    fixed = panel.drop(columns=recent_response.columns.intersection(panel.columns).difference(["station"]))
    recent_panel = fixed.merge(recent_response, on="station", how="left", validate="one_to_one")
    # Refresh all response-derived hydro, coverage and sampling controls.
    recent_panel["log_flow"] = np.log1p(recent_panel.median_flow)
    recent_panel["log_n_doc"] = np.log1p(recent_panel.n_doc)
    recent_panel["included"] = recent_panel.eligible.eq(True) & recent_panel.inclusion_status.eq("included_landscape")
    recent_panel.to_csv(ROOT/"analysis/recent_station_doc_source_panel.csv", index=False)
    predictions.append(blocked_predictions(recent_panel, "source_months_since_2009"))
    predicted = pd.concat(predictions, ignore_index=True)
    predicted.to_csv(ROOT/"analysis/huc4_blocked_predictions.csv", index=False)
    gains = comparisons(predicted, args.bootstrap_draws)
    gains.to_csv(ROOT/"analysis/huc4_blocked_gains.csv", index=False)
    coefficients, diagnostics = associations(panel, args.bootstrap_draws)
    coefficients.to_csv(ROOT/"analysis/source_placement_associations.csv", index=False)
    numeric = [*PLACEMENT, "wetland_pct", "forest_pct", "drainage_mean_distance_km"]
    panel[panel.included].groupby("cluster")[numeric].agg(["median", "count"]).to_csv(ROOT/"analysis/class_source_placement.csv")
    paths = [PANEL, CELLS, DATASET, source, Path(__file__), ROOT/"study_plan.md",
             Path("src/river_graph/analysis/river_source_placement.py"),
             Path("src/river_graph/analysis/river_doc_structure.py"), Path("src/river_graph/analysis/river_planform_doc.py")]
    (ROOT/"analysis_sources.json").write_text(json.dumps({"source_hashes": {str(p): sha256_file(p) for p in paths},
        "draws": args.bootstrap_draws, "ridge_alpha": 10, "n_primary_stations": int(panel.included.sum()),
        "n_recent_stations": int(recent_panel.included.sum()), "diagnostics": diagnostics,
        "new_neural_training": False, "geographic_external_model_results_read": False}, indent=2)+"\n")
    print(gains[gains.resampling_unit.eq("huc4") & gains.space.eq("native")].to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
