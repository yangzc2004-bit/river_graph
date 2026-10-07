"""Study monthly flow activation of terrestrial DOC sources on real networks."""
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

from river_graph.analysis.river_hydrologic_activation import (
    AMOUNT_CONTROLS,
    clustered_interaction_fit,
    monthly_source_panel,
    station_flow_responses,
)
from river_graph.analysis.river_planform_doc import SHAPE
from river_graph.analysis.river_source_placement import PLACEMENT
from river_graph.experiments.provenance import sha256_file

ROOT = Path("experiments/phase4_transfer/doc_river_hydrologic_activation_v1")
DATASET = Path("data/processed/mississippi_graph_graphfix_st357.pt")
CELLS = Path("experiments/phase4_transfer/doc_river_mechanisms_v1/analysis/source_cells.npy")
STATIONS = Path("experiments/phase4_transfer/doc_river_source_placement_v1/analysis/station_doc_source_panel.csv")
ARMS = {"environment": list(AMOUNT_CONTROLS),
        "environment_shape": [*AMOUNT_CONTROLS, *SHAPE],
        "environment_placement": [*AMOUNT_CONTROLS, *PLACEMENT],
        "environment_shape_placement": [*AMOUNT_CONTROLS, *SHAPE, *PLACEMENT]}
PAIRS = [("environment_shape", "environment"), ("environment_placement", "environment"),
         ("environment_shape_placement", "environment_shape"),
         ("environment_shape_placement", "environment_placement")]


def geography_predictions(stations, population):
    rows = []
    y = stations.cq_linear.to_numpy()
    for fold, (train, test) in enumerate(GroupKFold(5).split(stations, groups=stations.huc4)):
        for arm, features in ARMS.items():
            prep = ColumnTransformer([
                ("num", make_pipeline(SimpleImputer(strategy="median", add_indicator=True), StandardScaler()), features),
                ("region", OneHotEncoder(handle_unknown="ignore", sparse_output=False), ["huc2"])])
            model = make_pipeline(prep, Ridge(alpha=10)).fit(stations.iloc[train], y[train])
            for i, prediction in zip(test, model.predict(stations.iloc[test]), strict=True):
                r = stations.iloc[i]
                rows.append({"population": population, "station": r.station, "huc4": r.huc4,
                             "fold": fold, "model": arm, "cq_linear_true": y[i],
                             "cq_linear_pred": prediction})
    return pd.DataFrame(rows)


def geography_gains(predictions, draws):
    rows = []
    for population, sub in predictions.groupby("population"):
        sub = sub.copy()
        sub["error"] = abs(sub.cq_linear_true-sub.cq_linear_pred)
        p = sub.pivot(index=["station", "huc4", "fold"], columns="model", values="error").reset_index()
        folds = [p.fold.to_numpy() == fold for fold in sorted(p.fold.unique())]
        for candidate, reference in PAIRS:
            a, b = p[candidate].to_numpy(), p[reference].to_numpy()
            mb, ma = np.mean([b[f].mean() for f in folds]), np.mean([a[f].mean() for f in folds])
            for unit in ("station", "huc4"):
                _, idx = np.unique(p[unit], return_inverse=True)
                n = idx.max()+1
                rng = np.random.default_rng(42)
                boot = []
                while len(boot) < draws:
                    weights = rng.multinomial(n, np.full(n, 1/n))[idx]
                    if any(weights[f].sum() == 0 for f in folds):
                        continue
                    ba = np.mean([np.average(a[f], weights=weights[f]) for f in folds])
                    bb = np.mean([np.average(b[f], weights=weights[f]) for f in folds])
                    boot.append(100*(bb-ba)/bb)
                low, high = np.quantile(boot, [.025, .975])
                rows.append({"population": population, "candidate": candidate, "reference": reference,
                             "resampling_unit": unit, "reference_mae": mb, "candidate_mae": ma,
                             "gain_pct": 100*(mb-ma)/mb, "ci_low_pct": low, "ci_high_pct": high,
                             "positive_folds": sum(a[f].mean() < b[f].mean() for f in folds),
                             "n_stations": len(p), "n_huc4": p.huc4.nunique()})
    return pd.DataFrame(rows)


def describe_classes(stations, population, draws):
    rows = []
    for group, sub in stations.groupby("cluster"):
        _, idx = np.unique(sub.huc4, return_inverse=True)
        n = idx.max()+1
        rng = np.random.default_rng(42)
        samples = []
        for _ in range(draws):
            copies = rng.multinomial(n, np.full(n, 1/n))
            sample = np.repeat(np.arange(len(sub)), copies[idx])
            samples.append(sample)
        for metric in ("cq_linear", "interquartile_response", "cq_quadratic"):
            values = sub[metric].to_numpy()
            bootstrap = [np.median(values[sample]) for sample in samples]
            low, high = np.quantile(bootstrap, [.025, .975])
            rows.append({"population": population, "cluster": int(group), "metric": metric,
                         "median": np.median(values), "ci_low": low, "ci_high": high,
                         "n_stations": len(sub), "n_huc4": n,
                         "positive_station_fraction": np.mean(values > 0)})
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bootstrap-draws", type=int, default=5000)
    args = parser.parse_args()
    out = ROOT/"analysis"
    out.mkdir(parents=True, exist_ok=True)
    dataset = torch.load(DATASET, weights_only=False, map_location="cpu")
    cells = np.load(CELLS)
    metadata = pd.read_csv(STATIONS, dtype={"station": str, "huc2": str, "huc4": str, "huc_cd": str})
    monthly = monthly_source_panel(dataset, cells, metadata)
    monthly.to_parquet(out/"monthly_source_panel.parquet", index=False)
    statistics, all_residuals, coefficients, class_responses, class_contrasts = [], [], [], [], []
    predictions, summaries, diagnostic = [], [], []
    populations = [("all_source_months", monthly, False),
                   ("observed_temperature", monthly, True),
                   ("source_months_since_2009", monthly[monthly.month.dt.year.ge(2009)], False)]
    for population, input_panel, temperature in populations:
        residual, fits = station_flow_responses(input_panel, temperature=temperature)
        fits["population"] = population
        statistics.append(fits)
        if residual.empty:
            raise ValueError(f"no identifiable station C-Q response: {population}")
        residual["population"] = population
        all_residuals.append(residual)
        good = fits[fits.response_status.eq("included")].copy()
        columns = metadata.columns.intersection(good.columns).difference(["station"])
        stations = metadata.drop(columns=columns).merge(good, on="station", validate="one_to_one")
        stations["log_flow"] = np.log1p(stations.median_flow)
        print(f"{population}: {len(stations)} stations, {len(residual)} monthly pairs, {stations.huc4.nunique()} HUC4", flush=True)
        for name, numeric, classes in (("source_shape_moderation", ARMS["environment_shape_placement"], False),
                                        ("adjusted_morphology_class", AMOUNT_CONTROLS, True)):
            coef, boot, basis, info = clustered_interaction_fit(residual, stations, numeric,
                classes=classes, draws=args.bootstrap_draws)
            coef["population"], coef["model"] = population, name
            coefficients.append(coef)
            diagnostic.append({"population": population, "model": name, **info})
            if classes:
                # Common mean environment/region; only the class indicators change.
                for group in (1, 2, 3):
                    vector = np.r_[basis.mean(axis=0), 0.]
                    labels = coef.term.to_list()
                    for g in (2, 3):
                        vector[labels.index(f"class_{g}_vs_1")] = float(group == g)
                    estimate = vector @ coef.coefficient
                    low, high = np.quantile(boot @ vector, [.025, .975])
                    class_responses.append({"population": population, "cluster": group,
                        "adjusted_cq_linear": estimate, "ci_low": low, "ci_high": high,
                        "n_stations": len(stations), "n_huc4": stations.huc4.nunique(),
                        "class_n_stations": int(stations.cluster.eq(group).sum())})
                for group, reference in ((2, 1), (3, 1), (3, 2)):
                    vector = np.zeros(len(coef))
                    for g in (2, 3):
                        vector[labels.index(f"class_{g}_vs_1")] = float(group == g)-float(reference == g)
                    low, high = np.quantile(boot @ vector, [.025, .975])
                    class_contrasts.append({"population": population, "cluster": group,
                        "reference_cluster": reference, "cq_linear_difference": vector @ coef.coefficient,
                        "ci_low": low, "ci_high": high})
        predictions.append(geography_predictions(stations, population))
        summaries.extend(describe_classes(stations, population, args.bootstrap_draws))
    pd.concat(statistics).to_csv(out/"station_response_fits.csv", index=False)
    pd.concat(all_residuals).to_parquet(out/"within_station_residuals.parquet", index=False)
    pd.concat(coefficients).to_csv(out/"hydrologic_moderation.csv", index=False)
    pd.DataFrame(class_responses).to_csv(out/"adjusted_class_responses.csv", index=False)
    pd.DataFrame(class_contrasts).to_csv(out/"adjusted_class_contrasts.csv", index=False)
    pd.DataFrame(summaries).to_csv(out/"class_response_distributions.csv", index=False)
    predictions = pd.concat(predictions, ignore_index=True)
    predictions.to_csv(out/"huc4_response_predictions.csv", index=False)
    gains = geography_gains(predictions, args.bootstrap_draws)
    gains.to_csv(out/"huc4_response_gains.csv", index=False)
    paths = [DATASET, CELLS, STATIONS, Path(__file__), ROOT/"study_plan.md",
             Path("src/river_graph/analysis/river_hydrologic_activation.py"),
             Path("src/river_graph/analysis/river_planform_doc.py"),
             Path("src/river_graph/analysis/river_source_placement.py")]
    receipt = {"source_hashes": {str(p): sha256_file(p) for p in paths},
               "bootstrap_draws": args.bootstrap_draws, "seed": 42, "diagnostics": diagnostic,
               "unique_source_cells": len(cells), "flow_unit": "NWIS daily mean monthly aggregate, cfs",
               "negative_or_reverse_flow_cells": int(monthly.flow_reason.eq("negative_or_reverse_flow").sum()),
               "DOC_reconstruction_benchmark": False, "neural_training": False,
               "non_source_DOC_read": False}
    (ROOT/"analysis_sources.json").write_text(json.dumps(receipt, indent=2)+"\n")
    print(gains[gains.resampling_unit.eq("huc4")].to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
