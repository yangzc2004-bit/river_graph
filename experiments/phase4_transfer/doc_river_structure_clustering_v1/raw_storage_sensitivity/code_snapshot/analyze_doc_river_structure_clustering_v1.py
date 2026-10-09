"""Classify real river organization, then quantify source-only DOC response."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, SplineTransformer, StandardScaler

from river_graph.analysis.river_doc_structure import (
    BLOCKS,
    bootstrap_ols,
    classify_rivers,
    huc_prefix,
    physical_features,
    source_station_response,
)
from river_graph.experiments.provenance import sha256_file

ROOT = Path("experiments/phase4_transfer/doc_river_structure_clustering_v1")
ATLAS = Path("experiments/phase4_transfer/doc_river_structure_atlas_v1")
DATASET = Path("data/processed/mississippi_graph_graphfix_st357.pt")
STREAMCAT = Path("data/processed/streamcat_attributes.csv")
MASKS = Path("experiments/phase4_transfer/unified_doc_spatial_v1/confirmation/masks")
CONTROLS = ("wetland", "forest", "agriculture", "urban", "precipitation", "climate_temperature",
            "log_flow", "median_temperature", "flow_available", "temperature_available",
            "median_year", "record_span", "n_calendar_months", "log_n_doc", "latitude", "longitude")
HYDRO = ("log_flow", "median_temperature", "flow_available", "temperature_available")
OUTCOMES = {"doc_median": "log1p", "doc_cv": "log1p", "harmonic_amplitude": "log1p",
            "q90_fraction": "raw", "cq_slope": "raw"}


def dump(path, obj):
    path.write_text(json.dumps(obj, indent=2, allow_nan=False)+"\n")


def controls(frame, names):
    imputer = SimpleImputer(strategy="median", add_indicator=True)
    raw = imputer.fit_transform(frame[list(names)])
    labels = list(imputer.get_feature_names_out(names))
    varied = np.std(raw, axis=0) > 1e-9
    raw = StandardScaler().fit_transform(raw[:, varied])
    return raw, [n for n, keep in zip(labels, varied, strict=True) if keep]


def design(frame, physical, names, *, mode, no_hydro=False):
    ctrl = [n for n in CONTROLS if not no_hydro or n not in HYDRO]
    x, labels = controls(frame, ctrl)
    regions = pd.get_dummies(frame.huc2, prefix="huc2", drop_first=True, dtype=float)
    x = np.column_stack([np.ones(len(frame)), x, regions.to_numpy()])
    labels = ["intercept", *labels, *regions.columns]
    if mode == "class":
        cats = pd.get_dummies(frame.cluster.astype(str), prefix="cluster", drop_first=True, dtype=float)
        return np.column_stack([x, cats.to_numpy()]), [*labels, *cats.columns], None
    sx = SimpleImputer(strategy="median").fit_transform(physical.loc[frame.station, names])
    sx = StandardScaler().fit_transform(sx)
    return np.column_stack([x, sx]), [*labels, *names], sx


def summarize_classes(panel, *, draws):
    rows = []
    for group, sub in panel[panel.eligible].groupby("cluster"):
        for metric in (*OUTCOMES, "doc_iqr"):
            a = sub[metric].dropna().to_numpy()
            if not len(a):
                continue
            rng = np.random.default_rng(42)
            sample = np.median(a[rng.integers(len(a), size=(draws, len(a)))], axis=1)
            rows.append({"cluster": group, "metric": metric, "n_stations": len(a), "median": np.median(a),
                         "q25": np.quantile(a, .25), "q75": np.quantile(a, .75),
                         "ci_low": np.quantile(sample, .025), "ci_high": np.quantile(sample, .975)})
    return pd.DataFrame(rows)


def regression_tables(panel, physical, names, *, draws):
    results, fits, curve_rows = [], [], []
    for metric, transform in OUTCOMES.items():
        sub = panel[panel.eligible & panel[metric].notna()].copy()
        y = sub[metric].to_numpy()
        if transform == "log1p":
            y = np.log1p(y)
        for mode, sensitivity in (("class", False), ("class", True), ("continuous", False)):
            x, labels, _ = design(sub, physical, names, mode=mode, no_hydro=sensitivity)
            point, boot, diag = bootstrap_ols(x, y, draws=draws)
            fit_name = f"{mode}_{'no_local_hydro' if sensitivity else 'hydro_adjusted'}"
            fits.append({"metric": metric, "model": fit_name, "transform": transform, **diag,
                         "in_sample_r2": 1-np.sum((y-x@point)**2)/np.sum((y-y.mean())**2)})
            for idx, term in enumerate(labels):
                if term.startswith("cluster_") or term in names:
                    low, high = np.quantile(boot[:, idx], [.025, .975])
                    results.append({"metric": metric, "model": fit_name, "term": term, "estimate": point[idx],
                                    "ci_low": low, "ci_high": high, "transform": transform,
                                    "n_stations": len(sub), "reference_class": str(sub.cluster.min()) if mode == "class" else "",
                                    "unit": "class contrast" if mode == "class" else "one transformed-feature SD"})
        print(f"Adjusted {metric}: {len(sub)} stations", flush=True)
    # Additive structure splines for concentration only. The other retained
    # structure variables remain linear controls, without selecting by DOC.
    sub = panel[panel.eligible].copy()
    y = np.log1p(sub.doc_median.to_numpy())
    cx, clabels, _ = design(sub, physical, names, mode="continuous")
    curves = [n for n in ("log_area", "junction_density_20", "tributary_balance_5", "storage_fraction_20km") if n in names]
    spline_parts, transformers, indices = [], {}, {}
    x = cx[:, :len(clabels)-len(names)]
    other = [n for n in names if n not in curves]
    if other:
        ox = SimpleImputer(strategy="median").fit_transform(physical.loc[sub.station, other])
        x = np.column_stack([x, StandardScaler().fit_transform(ox)])
    for name in curves:
        raw = physical.loc[sub.station, name].to_numpy().reshape(-1, 1)
        raw = np.where(np.isfinite(raw), raw, np.nanmedian(raw))
        tr = SplineTransformer(n_knots=5, degree=2, include_bias=False, knots="quantile")
        part = tr.fit_transform(raw)
        indices[name] = slice(x.shape[1], x.shape[1]+part.shape[1])
        transformers[name] = tr
        spline_parts.append(part)
        x = np.column_stack([x, part])
    point, boot, diag = bootstrap_ols(x, y, draws=draws)
    fits.append({"metric": "doc_median", "model": "additive_structure_splines", "transform": "log1p", **diag,
                 "in_sample_r2": 1-np.sum((y-x@point)**2)/np.sum((y-y.mean())**2)})
    for name in curves:
        grid = np.linspace(*physical.loc[sub.station, name].quantile([.1, .9]), 40)
        ref = np.tile(x.mean(axis=0), (len(grid), 1))
        ref[:, indices[name]] = transformers[name].transform(grid.reshape(-1, 1))
        pred = ref@point
        lower, upper = np.quantile(boot@ref.T, [.025, .975], axis=0)
        for g, p, lo, hi in zip(grid, pred, lower, upper, strict=True):
            curve_rows.append({"feature": name, "x_transformed": g, "adjusted_log1p_doc": p,
                               "ci_low": lo, "ci_high": hi, "n_stations": len(sub)})
    return pd.DataFrame(results), pd.DataFrame(fits), pd.DataFrame(curve_rows)


def blocked_comparison(panel, physical, names):
    sub = panel[panel.eligible].copy().reset_index(drop=True)
    frame = sub[list(CONTROLS)].copy()
    frame["huc2"], frame["river_class"] = sub.huc2, sub.cluster.astype(str)
    for name in names:
        frame[name] = physical.loc[sub.station, name].to_numpy()
    y = np.log1p(sub.doc_median)
    rows, predictions = [], []
    for fold, (train, val) in enumerate(GroupKFold(n_splits=5).split(frame, groups=sub.huc4)):
        for arm in ("environment", "environment_classes", "environment_structure"):
            numeric = [*CONTROLS, *(names if arm == "environment_structure" else [])]
            categorical = ["huc2", *(["river_class"] if arm == "environment_classes" else [])]
            prep = ColumnTransformer([
                ("num", make_pipeline(SimpleImputer(strategy="median", add_indicator=True), StandardScaler()), numeric),
                ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), categorical)])
            model = make_pipeline(prep, Ridge(alpha=10))
            model.fit(frame.iloc[train], y.iloc[train])
            p = model.predict(frame.iloc[val])
            rows.append({"fold": fold, "model": arm, "n_stations": len(val),
                         "mae_log1p": mean_absolute_error(y.iloc[val], p),
                         "mae_native": mean_absolute_error(sub.doc_median.iloc[val], np.maximum(0, np.expm1(p)))})
            for i, value in zip(val, p, strict=True):
                predictions.append({"station": sub.station.iloc[i], "huc4": sub.huc4.iloc[i], "fold": fold,
                                    "model": arm, "y_log1p": y.iloc[i], "y_pred_log1p": value})
    return pd.DataFrame(rows), pd.DataFrame(predictions)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--bootstrap-draws", type=int, default=2000)
    parser.add_argument("--stability-draws", type=int, default=100)
    args = parser.parse_args()
    out = args.root/"analysis"
    out.mkdir(parents=True, exist_ok=True)
    source_paths = [ATLAS/"analysis/station_structure.csv", STREAMCAT, DATASET, args.root/"study_plan.md",
                    Path(__file__), Path("src/river_graph/analysis/river_doc_structure.py"),
                    ATLAS/"analysis/balanced_source_upstream_doc_associations.csv"]
    s = pd.read_csv(source_paths[0], dtype={"station": str, "huc_cd": str})
    if len(s) != 357 or s.station.duplicated().any() or s.upstream_search_capped.any():
        raise ValueError("physical atlas must have 357 unique complete-search stations")
    f = physical_features(s)
    labels, candidates, stability, alternatives, names, blocks, x, imputer, scaler = classify_rivers(
        f, stability_draws=args.stability_draws)
    s["cluster"] = labels
    s["huc2"], s["huc4"] = s.huc_cd.map(lambda h: huc_prefix(h, 2)), s.huc_cd.map(huc_prefix)
    coords = PCA(n_components=2).fit_transform(x)
    s["pc1"], s["pc2"] = coords[:, 0], coords[:, 1]
    for k, alt in alternatives.items():
        s[f"candidate_k{k}"] = alt
    s.to_csv(out/"station_classification.csv", index=False)
    candidates.to_csv(out/"cluster_candidates.csv", index=False)
    stability.to_csv(out/"cluster_stability.csv", index=False)
    f.insert(0, "station", s.station)
    f.to_csv(out/"physical_features.csv", index=False)
    f = f.set_index("station")
    weighted = pd.DataFrame(x, columns=names).assign(cluster=labels)
    weighted.groupby("cluster").mean().to_csv(out/"weighted_centroids.csv")
    medians = s.groupby("cluster").agg(n_stations=("station", "size"), order=("stream_order", "median"),
        area_km2=("drainage_area_km2", "median"), slope=("slope", "median"), storage=("storage_fraction_20km", "median"),
        major_confluence_share=("confluence", "mean"), mainstem_share=("mainstem", "mean"), tributary_share=("low_order", "mean"))
    medians.to_csv(out/"class_physical_summary.csv")
    ablations = []
    from sklearn.cluster import AgglomerativeClustering
    from sklearn.metrics import adjusted_rand_score
    for omit, cols in blocks.items():
        keep = [i for i, name in enumerate(names) if name not in cols]
        alt = AgglomerativeClustering(n_clusters=int(labels.max()), linkage="ward").fit_predict(x[:, keep])
        ablations.append({"omitted_block": omit, "ari_vs_full": adjusted_rand_score(labels, alt)})
    pd.DataFrame(ablations).to_csv(out/"feature_block_sensitivity.csv", index=False)
    dump(args.root/"classification.json", {"selected_k": int(labels.max()), "features": names, "blocks": blocks,
        "imputation_medians": imputer.statistics_.tolist(), "scaler_mean": scaler.mean_.tolist(),
        "scaler_scale": scaler.scale_.tolist(), "selection_uses_DOC": False,
        "silhouette_selection_minimum_class": 20, "classification_n": len(s)})
    print("Structure classification fixed before DOC analysis:", candidates.to_string(index=False), flush=True)
    # Only now read water quality, exclusively in source training roles.
    dataset = torch.load(DATASET, map_location="cpu", weights_only=False)
    np.testing.assert_array_equal(s.station.to_numpy(), np.asarray(dataset["site_no"], str))
    cells = []
    for split in (142, 143, 144):
        p = MASKS/f"split{split}.npz"
        source_paths.append(p)
        with np.load(p) as mask:
            cells.extend(mask["train"].tolist())
    cells = np.unique(cells)
    np.save(out/"source_cells.npy", cells)
    responses, season, q90 = source_station_response(dataset, cells)
    cat = pd.read_csv(STREAMCAT)
    cat = cat.assign(wetland=cat.pcthbwet2019ws+cat.pctwdwet2019ws,
                     forest=cat.pctconif2019ws+cat.pctdecid2019ws+cat.pctmxfst2019ws,
                     agriculture=cat.pctcrop2019ws+cat.pcthay2019ws,
                     urban=cat.pcturbhi2019ws+cat.pcturblo2019ws+cat.pcturbmd2019ws+cat.pcturbop2019ws,
                     precipitation=cat.precip9120ws, climate_temperature=cat.tmean9120ws)
    ecology = ["wetland", "forest", "agriculture", "urban", "precipitation", "climate_temperature"]
    panel = responses.merge(s, on="station", validate="one_to_one").merge(cat[["comid", *ecology]], on="comid", how="left", validate="many_to_one")
    panel["log_flow"], panel["log_n_doc"] = np.log1p(panel.median_flow), np.log1p(panel.n_doc)
    panel.to_csv(out/"station_doc_response.csv", index=False)
    summarize_classes(panel, draws=args.bootstrap_draws).to_csv(out/"class_doc_response.csv", index=False)
    season = season.merge(panel.loc[panel.eligible, ["station", "cluster"]], on="station", validate="many_to_one")
    season.to_csv(out/"station_seasonal_response.csv", index=False)
    seasonal = []
    for (cluster, month), sub in season.groupby(["cluster", "month"]):
        a = sub.doc_median.to_numpy()
        rng = np.random.default_rng(42)
        boot = np.mean(a[rng.integers(len(a), size=(args.bootstrap_draws, len(a)))], axis=1)
        seasonal.append({"cluster": cluster, "month": month, "doc_station_mean": a.mean(),
                         "ci_low": np.quantile(boot, .025), "ci_high": np.quantile(boot, .975), "n_stations": len(a)})
    pd.DataFrame(seasonal).to_csv(out/"class_seasonal_response.csv", index=False)
    contrasts, diagnostics, curves = regression_tables(panel, f, names, draws=args.bootstrap_draws)
    contrasts.to_csv(out/"adjusted_associations.csv", index=False)
    diagnostics.to_csv(out/"regression_diagnostics.csv", index=False)
    curves.to_csv(out/"adjusted_structure_curves.csv", index=False)
    cv, cvpred = blocked_comparison(panel, f, names)
    cv.to_csv(out/"huc4_blocked_comparison.csv", index=False)
    cvpred.to_csv(out/"huc4_blocked_predictions.csv", index=False)
    pair = pd.read_csv(source_paths[6], dtype={"source": str, "target": str})
    pair = pair.groupby(["source", "target", "lag_months"], as_index=False).agg(
        rho=("rho_seasonal_anomaly", "mean"), path_length_km=("path_length_km", "first"))
    pair = pair.merge(s[["station", "cluster"]], left_on="target", right_on="station", validate="many_to_one")
    pair.to_csv(out/"fixed_edge_associations_by_class.csv", index=False)
    dump(args.root/"sources.json", {"source_hashes": {str(p): sha256_file(p) for p in source_paths},
        "bootstrap_draws": args.bootstrap_draws, "stability_draws": args.stability_draws,
        "n_unique_source_cells": len(cells), "n_source_stations": len(panel), "n_eligible_stations": int(panel.eligible.sum()),
        "q90_source_threshold_mg_L": q90, "DOC_roles": "deduplicated union of source-training roles 142/143/144",
        "no_model_training": True, "new_geographical_external_test_read": False})
    print("Complete DOC structure analysis", flush=True)


if __name__ == "__main__":
    main()
