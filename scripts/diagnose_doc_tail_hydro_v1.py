"""Describe hydro inputs and remaining DOC errors on source validation only.

No model is fitted and no outer-test prediction or DOC label is read. All
hydrological anomalies use masked covariates from current/past months only.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from river_graph.dataset import FEATURE_CHANNELS
from river_graph.experiments.provenance import sha256_file
from river_graph.experiments.unified_spatial_protocol import support_query_cells

ROOT = Path("experiments/phase4_transfer/unified_doc_spatial_v4")
OUTPUT = Path("experiments/phase4_transfer/doc_tail_residual_v1/hydro_diagnostics")
MODEL = "fusion_gru_tuned_anchor_k5"
SPLITS, SEEDS = (142, 143, 144), (42, 43, 44)
LAGS = (0, 1, 3, 12)


def shifted(values, lag):
    result = np.full_like(values, np.nan, dtype=float)
    if lag == 0:
        result[:] = values
    elif lag < values.shape[1]:
        result[:, lag:] = values[:, :-lag]
    return result


def hydro_features(dataset):
    """Covariate-only causal lags/anomalies; invalid hydro is never a zero."""
    if list(dataset.get("feature_channels", FEATURE_CHANNELS)) != ["temperature", "discharge"]:
        raise ValueError("Unexpected hydro feature order")
    x = np.asarray(dataset["x"], dtype=float)
    mask = np.asarray(dataset["x_mask"], dtype=bool)
    if x.ndim != 3 or x.shape[-1] != 2 or mask.shape != x.shape:
        raise ValueError("Expected aligned station/month/two-channel hydro arrays")
    if not np.isfinite(x[mask]).all():
        raise ValueError("Observed hydro values must be finite")
    observed = np.where(mask, x, np.nan)
    n, months, _ = x.shape
    features = {}
    for channel, name in enumerate(FEATURE_CHANNELS):
        values = observed[..., channel]
        for lag in LAGS:
            features[f"{name}_lag{lag}"] = shifted(values, lag)
        for lag in LAGS[1:]:
            if name == "temperature":
                delta = values - shifted(values, lag)
            else:
                signed_log = np.sign(values) * np.log1p(np.abs(values))
                delta = signed_log - shifted(signed_log, lag)
            features[f"{name}_change{lag}"] = delta
        mean, mean_absolute = np.full((n, months), np.nan), np.full((n, months), np.nan)
        counts = np.zeros((n, months), dtype=int)
        age = np.full((n, months), np.nan)
        last = np.full(n, -1)
        for month in range(months):
            historical = values[:, max(0, month - 12):month]
            count = np.isfinite(historical).sum(axis=1)
            counts[:, month] = count
            enough = count >= 3
            mean[enough, month] = np.nansum(historical[enough], axis=1) / count[enough]
            mean_absolute[enough, month] = np.nansum(np.abs(historical[enough]), axis=1) / count[enough]
            last[np.isfinite(values[:, month])] = month
            known = last >= 0
            age[known, month] = month - last[known]
        anomaly = values - mean
        if name == "discharge":
            anomaly = np.divide(anomaly, mean_absolute, out=np.full_like(anomaly, np.nan),
                                where=mean_absolute > 0)
        features[f"{name}_past12_anomaly"] = anomaly
        features[f"{name}_past12_count"] = counts
        features[f"{name}_observation_age"] = age
    return features, {
        "shape": list(x.shape), "channels": list(FEATURE_CHANNELS),
        "negative_observed_discharge_cells": int((observed[..., 1] < 0).sum()),
        "temperature_unit": "degrees Celsius", "discharge_unit": "cubic feet per second",
        "lag_definition": "calendar lag, not most recent observed value",
        "change_definition": "temperature difference; discharge difference of signed log1p(cfs)",
        "past12_anomaly": "current minus mean of prior 12 calendar months, requiring >=3 observed months",
        "flow_anomaly_scaling": "divide by prior-12 mean absolute discharge; unit invariant; preserve signed flow",
        "observation_age": "months since latest observed hydro at or before current month; NaN if never observed",
        "hydro_invalid_policy": "x_mask false is missing, never zero or carried-forward",
    }


def read_validation(root):
    diagnostic = root / "diagnostics"
    prediction_path = diagnostic / "validation_queries.parquet"
    meta_path = diagnostic / "validation_queries.meta.json"
    metadata = json.loads(meta_path.read_text())
    if metadata["prediction_sha256"] != sha256_file(prediction_path):
        raise ValueError("Changed source-validation prediction artifact")
    if metadata["config"]["visibility_role"] != "source_validation_query":
        raise ValueError("Only source-validation queries may be diagnosed")
    frame = pd.read_parquet(prediction_path)
    frame = frame[frame.model_name.eq(MODEL) & frame.k.eq(5)].copy()
    if frame.empty or not frame.visibility_role.eq("source_validation_query").all():
        raise ValueError("Missing source-validation updated-GRU queries")
    if frame.duplicated(["split_seed", "seed", "cell"]).any():
        raise ValueError("Duplicate validation prediction rows")
    if set(map(tuple, frame[["split_seed", "seed"]].drop_duplicates().to_numpy())) != {
            (split, seed) for split in SPLITS for seed in SEEDS}:
        raise ValueError("Expected all nine source-validation runs")
    sources = [prediction_path, meta_path]
    configs, splits = {}, {}
    for split in SPLITS:
        config_path = root / "runs" / f"split{split}_seed42" / "config.json"
        config = json.loads(config_path.read_text())
        mask_path = Path(config["mask_path"])
        if sha256_file(mask_path) != metadata["mask_hashes_by_split"][str(split)]:
            raise ValueError("Validation mask differs from diagnostic provenance")
        with np.load(mask_path, allow_pickle=False) as saved:
            splits[split] = {key: saved[key].copy() for key in saved.files}
        configs[split] = config
        sources.extend((config_path, mask_path))
    dataset_paths = {config["dataset_path"] for config in configs.values()}
    if len(dataset_paths) != 1:
        raise ValueError("Expected a common hydro dataset")
    dataset_path = Path(next(iter(dataset_paths)))
    if sha256_file(dataset_path) != metadata["dataset_hash"]:
        raise ValueError("Hydro dataset differs from diagnostic provenance")
    dataset = torch.load(dataset_path, weights_only=False, map_location="cpu")
    # Only covariates, grid dates and station IDs are used from the dataset;
    # DOC truth is taken exclusively from the validation-only prediction table.
    months = np.asarray(dataset["months"], dtype="datetime64[M]")
    if not np.all(np.diff(months.astype(np.int64)) == 1):
        raise ValueError("Hydro calendar lags require consecutive monthly grid columns")
    n_months = len(months)
    for split in SPLITS:
        _, allowed = support_query_cells(splits[split], target_role="val", k=5, n_months=n_months)
        for seed in SEEDS:
            part = frame[frame.split_seed.eq(split) & frame.seed.eq(seed)].sort_values("cell")
            np.testing.assert_array_equal(part.cell.to_numpy(), np.sort(allowed))
        if np.intersect1d(allowed, splits[split]["test"]).size:
            raise ValueError("Validation queries overlap outer-test cells")
    identity = frame.groupby(["split_seed", "cell"])[
        ["station", "month", "y_true", "is_q90", "q90_threshold_train"]].nunique()
    if (identity != 1).any().any() or not frame.groupby(["split_seed", "cell"]).size().eq(3).all():
        raise ValueError("Seed predictions must share validation query identities")
    frame["absolute_error"] = frame.residual.abs()
    frame["squared_error"] = frame.residual ** 2
    frame["absolute_log_error"] = frame.log_residual.abs()
    panel = frame.groupby(["split_seed", "cell"], as_index=False).agg(
        station=("station", "first"), month=("month", "first"), y_true=("y_true", "first"),
        y_pred=("y_pred", "mean"), y_pred_sd_seed=("y_pred", "std"), residual=("residual", "mean"),
        log_residual=("log_residual", "mean"), absolute_error=("absolute_error", "mean"),
        squared_error=("squared_error", "mean"), absolute_log_error=("absolute_log_error", "mean"),
        is_q90=("is_q90", "first"), q90_threshold_train=("q90_threshold_train", "first"))
    panel["station_index"] = panel.cell // n_months
    panel["month_index"] = panel.cell % n_months
    np.testing.assert_array_equal(np.asarray(panel.month, dtype="datetime64[M]"), months[panel.month_index])
    sites = np.asarray(dataset["site_no"], dtype=str)
    np.testing.assert_array_equal(panel.station.astype(str), sites[panel.station_index])
    sources.append(dataset_path)
    return panel, dataset, sources


def populations(frame):
    return (("all", frame), ("q90", frame[frame.is_q90]), ("non_tail", frame[~frame.is_q90]))


def describe(panel, feature_names):
    distributions, correlations, errors, coverage = [], [], [], []
    for split, part in panel.groupby("split_seed"):
        for population, group in populations(part):
            errors.append({"split_seed": split, "population": population, "n_query_cells": len(group),
                           "n_stations": group.station.nunique(), "mae": group.absolute_error.mean(),
                           "mean_residual": group.residual.mean(), "log_mae": group.absolute_log_error.mean(),
                           "mse": group.squared_error.mean()})
            for feature in feature_names:
                valid = group[feature].notna()
                values = group.loc[valid, feature]
                distributions.append({"split_seed": split, "population": population, "feature": feature,
                                      "n_cells": len(group), "n_observed": len(values),
                                      "coverage": float(valid.mean()) if len(group) else np.nan,
                                      "mean": values.mean(), "median": values.median(),
                                      "p10": values.quantile(.1), "p90": values.quantile(.9)})
                for metric in ("residual", "absolute_error"):
                    selected = group.loc[valid, ["station", feature, metric]].copy()
                    for adjustment in ("unadjusted", "station_centered"):
                        paired = selected.copy()
                        if adjustment == "station_centered":
                            counts = paired.groupby("station").station.transform("size")
                            paired = paired[counts >= 3].copy()
                            for column in (feature, metric):
                                paired[column] -= paired.groupby("station")[column].transform("mean")
                        varying = paired[feature].nunique() > 1 and paired[metric].nunique() > 1
                        rho = paired[feature].rank().corr(paired[metric].rank()) if varying else np.nan
                        correlations.append({"split_seed": split, "population": population,
                                             "feature": feature, "metric": metric, "adjustment": adjustment,
                                             "spearman_rho": rho, "n_pairs": len(paired),
                                             "n_stations": paired.station.nunique(), "status": "descriptive"})
            for availability, cells in group.groupby("current_hydro_availability"):
                coverage.append({"split_seed": split, "population": population, "availability": availability,
                                 "n_cells": len(cells), "fraction": len(cells) / len(group),
                                 "mae": cells.absolute_error.mean(), "mean_residual": cells.residual.mean()})
    return tuple(pd.DataFrame(rows) for rows in (distributions, correlations, errors, coverage))


def input_inventory():
    return pd.DataFrame([
        ("Current temperature/discharge and their masks", "yes", "yes, each month in the window",
         "Already present; simply adding current hydro again is not a new information source."),
        ("Target DOC lags 1/3/6/12", "explicit", "through visible DOC and memory",
         "RF lag columns refer to DOC, not lagged hydro."),
        ("Hydro lags 1 and 3", "not explicit", "available implicitly in 12-month GRU window",
         "Explicit changes could simplify their use by a small residual head."),
        ("Hydro lag 12 / same month last year", "not explicit", "outside current t-11...t window",
         "Add with separate availability flags if testing seasonal hydro changes."),
        ("Within-station flow relative anomaly", "not explicit", "not explicitly normalized within station",
         "A causal previous-12-month reference removes large between-station flow scale differences."),
        ("Hydro observation age and trailing support", "current masks only", "current masks in each month",
         "M1 age/count features describe DOC observations; add hydro freshness only if needed."),
    ], columns=["feature_group", "environmental_context_forest", "M1_GRU", "implication"])


def station_error_mass(panel):
    rows = panel.groupby(["split_seed", "station"], as_index=False).agg(
        n_query_cells=("cell", "size"), absolute_error_mass=("absolute_error", "sum"),
        squared_error_mass=("squared_error", "sum"), max_doc=("y_true", "max"))
    sizes = panel.groupby("split_seed").size()
    denominator = rows.split_seed.map(sizes) * len(SPLITS)
    rows["mae_contribution"] = rows.absolute_error_mass / denominator
    rows["mse_contribution"] = rows.squared_error_mass / denominator
    stations = rows.groupby("station", as_index=False).agg(
        n_partitions=("split_seed", "size"), n_query_occurrences=("n_query_cells", "sum"),
        mae_contribution=("mae_contribution", "sum"), mse_contribution=("mse_contribution", "sum"),
        max_doc=("max_doc", "max"))
    stations["absolute_error_share"] = stations.mae_contribution / stations.mae_contribution.sum()
    stations["squared_error_share"] = stations.mse_contribution / stations.mse_contribution.sum()
    return stations.sort_values("squared_error_share", ascending=False)


def report(panel, distributions, correlations, errors, top, station_mass, metadata, audit):
    n_unique = panel[["station", "month"]].drop_duplicates().shape[0]
    lines = ["# Hydro context of remaining DOC tail errors", "",
             "This is a descriptive diagnostic of reused source-validation queries for the updated-GRU",
             "fusion adapter at K=5. These labels previously supported model/checkpoint selection.",
             "No model is trained and no outer-test DOC label or prediction table is used.", "",
             f"The panel has {len(panel):,} partition-cell occurrences, {n_unique:,} distinct station-months",
             f"and {panel.station.nunique()} distinct stations. Predictions/errors are averaged over three",
             "seeds per partition. Tables retain individual partitions; summary means weight partitions equally.", "",
             "## Tail errors and hydro coverage", "",
             "Residual = prediction minus observation; negative values indicate underprediction.", "",
             "| Population | Mean partition MAE | Mean signed residual | Current flow coverage | Current temperature coverage |",
             "|---|---:|---:|---:|---:|"]
    for name in ("all", "q90", "non_tail"):
        e = errors[errors.population.eq(name)]
        d = distributions[distributions.population.eq(name)]
        q = d[d.feature.eq("discharge_lag0")].coverage.mean()
        t = d[d.feature.eq("temperature_lag0")].coverage.mean()
        lines.append(f"| {name} | {e.mae.mean():.3f} | {e.mean_residual.mean():.3f} | {q:.1%} | {t:.1%} |")
    lines += ["", "## Descriptive hydro shifts", "",
              "Flow anomaly is (current flow − prior-12-month mean) / prior mean absolute flow,",
              "requiring at least three earlier observed flow months. It preserves flow sign and is",
              "unit invariant. Temperature anomaly is a difference in degrees Celsius. No DOC values",
              "or future hydro months enter these features. Missing hydro remains missing.", "",
              "| Feature | Non-tail mean of partition medians | Q90 mean of partition medians | Q90 coverage |",
              "|---|---:|---:|---:|"]
    selected_features = ("discharge_past12_anomaly", "discharge_change1", "discharge_change3",
                         "discharge_change12", "temperature_past12_anomaly", "temperature_change1",
                         "temperature_change12")
    for feature in selected_features:
        d = distributions[distributions.feature.eq(feature)]
        low, high = d[d.population.eq("non_tail")], d[d.population.eq("q90")]
        lines.append(f"| {feature} | {low['median'].mean():.3f} | {high['median'].mean():.3f} | {high.coverage.mean():.1%} |")
    lines += ["", "Station-centered descriptive Spearman correlations with absolute error follow.",
              "Each variable is centered within station on available paired validation cells; stations",
              "need at least three pairs. This removes mean offsets, not all ecological confounding.", "",
              "| Feature | Split 142 | Split 143 | Split 144 |",
              "|---|---:|---:|---:|"]
    chosen = correlations[correlations.population.eq("all") & correlations.metric.eq("absolute_error")
                          & correlations.adjustment.eq("station_centered")]
    for feature in selected_features:
        group = chosen[chosen.feature.eq(feature)].set_index("split_seed")
        lines.append(f"| {feature} | " + " | ".join(f"{group.loc[s, 'spearman_rho']:.3f}" for s in SPLITS) + " |")
    cases = top.drop_duplicates("cell")
    q_valid = cases.discharge_past12_anomaly.notna()
    positive = int((cases.loc[q_valid, "discharge_past12_anomaly"] > 0).sum())
    dominant = station_mass.iloc[0]
    largest = cases.iloc[0]
    lines += ["", "## Largest-error station-month cases", "",
              f"`top_error_cases.csv` retains the top 20 cells per partition ({len(cases)} unique cells).",
              f"Among those unique cases, {int(cases.is_q90.sum())} meet their listed source-training Q90;",
              (f"{int(cases.discharge_lag0.isna().sum())} lack current flow and "
               f"{int(cases.temperature_lag0.isna().sum())} lack current temperature."),
              f"Only {int(q_valid.sum())} have a usable relative-flow anomaly; {positive} of these are positive.",
              "`top_error_months.csv` separately ranks calendar months by their total absolute-error mass",
              "within each partition, retaining query counts so an isolated extreme is not mistaken for",
              "a spatially widespread event. These are review cases, not independent event validation.", "",
              (f"Station {dominant.station} contributes {dominant.squared_error_share:.1%} of equal-partition "
               f"squared error and {dominant.absolute_error_share:.1%} of absolute error; its maximum validation "
               f"DOC is {dominant.max_doc:.1f} mg/L."),
              (f"The largest-error case is {largest.station}, {largest.month}: "
               f"observed {largest.y_true:.1f}, predicted {largest.y_pred:.2f} mg/L."),
              (f"The completed [source audit](../data_audit/audit.md) verifies {audit['verified']}/{audit['n']} "
               f"largest-error months as provider DOC records and reproduces the frozen aggregation in "
               f"{audit['matching']}/{audit['n']}. The files identify filtered USGS 00681 in mg/L; no "
               "pipeline defect or data deletion is supported."),
              (f"Laboratory method metadata are absent for {audit['missing_methods']}/{audit['n_raw']} retained "
               "point results. Verified archived records do not independently establish the physical "
               "mechanism of these historical events. This report reuses that audit rather than rerunning it."),
              "Station contributions under this report's equal-partition weighting are retained in",
              "`station_error_mass.csv`; the audit also reports unique-cell-weighted contributions.", "",
              "## What this diagnostic supports", "",
              "The tail problem is strong underprediction despite mostly available monthly hydro inputs.",
              "Short flow changes and relative flow anomalies show weak positive within-station associations",
              "with error across all queries; tail-only associations vary across partitions. Most large-error",
              "cases with a usable flow anomaly are not above their preceding flow baseline. A single",
              "high-flow explanation therefore does not describe the observed tail failures. Temperature",
              "changes and the annual-lag changes show weaker or inconsistent error associations.", "",
              "## Minimal additions to the current residual model", "",
              "The immediate native-MAE versus tail-weighted scalar-head experiment keeps inputs unchanged.",
              "The following compact additions are candidates for a subsequent input comparison:", "",
              "1. Keep the spatial encoder and GRU. Add a compact covariate block to the existing residual",
              "   branch: relative flow anomaly and signed-log flow changes at 1/3 months, with their",
              "   validity flags. The GRU already receives monthly raw hydro; these",
              "   are explicit transformations and missingness cues rather than new measurements.",
              "2. Supply prior-12 hydro counts or observation age alongside anomaly validity. M1's existing",
              "   age and support counts refer to DOC, so they do not directly describe hydro freshness.", "",
              "Temperature changes and same-month-last-year hydro differences are lower-priority additions",
              "given this diagnostic. The latter would provide t−12 information outside the current window,",
              "but the present associations do not make it the first change to implement.", "",
              "The tables show which transformations have usable coverage and consistent descriptive",
              "associations. They do not establish that any proposed input improves prediction. Preserve",
              "missingness indicators and avoid converting absent covariates into apparent low-flow events.",
              "Monthly flow/temperature and DOC may be aggregated from different observation dates; their",
              "co-occurrence cannot establish storm timing, source flushing or a causal event mechanism.",
              "A high monthly DOC value without a hydro shift can reflect unmeasured forcing, timing",
              "mismatch or observation variability; this diagnostic cannot distinguish those explanations.", "",
              "## Reproduction and definitions", "",
              "Run `uv run python scripts/diagnose_doc_tail_hydro_v1.py`. All derived quantities are",
              "covariate-only and causal in calendar time; the supplied DOC predictions themselves are",
              "retrospective K-shot reconstructions. Covariate order/units come from the dataset builder.",
              f"The source grid contains {metadata['negative_observed_discharge_cells']} observed negative",
              "flow cells; their signs are retained. Full bounded validation features are saved as parquet,",
              "with partition/population distributions, missingness, correlations, source hashes and an",
              "inventory of features already available to RF-context and M1. No significance tests are used.", ""]
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    panel, dataset, sources = read_validation(args.root)
    features, metadata = hydro_features(dataset)
    for name, array in features.items():
        panel[name] = array.ravel()[panel.cell.to_numpy()]
    q, t = panel.discharge_lag0.notna(), panel.temperature_lag0.notna()
    panel["current_hydro_availability"] = np.select(
        [q & t, q, t], ["both", "flow_only", "temperature_only"], default="neither")
    distributions, correlations, errors, coverage = describe(panel, list(features))
    top = panel.sort_values("absolute_error", ascending=False).groupby("split_seed").head(20)
    station_mass = station_error_mass(panel)
    audit_root = args.output.parent / "data_audit"
    audit_table = pd.read_csv(audit_root / "top20_trace.csv", dtype={"station": str})
    audit = {"n": len(audit_table), "verified": int(audit_table.classification.eq("verified_source_doc").sum()),
             "matching": int(audit_table.aggregation_matches_float32_dataset.sum()),
             "missing_methods": int(audit_table.n_method_missing.sum()),
             "n_raw": int(audit_table.n_accepted_raw_records.sum())}
    sources.extend((audit_root / "top20_trace.csv", audit_root / "audit.md"))
    by_month = panel.groupby(["split_seed", "month"], as_index=False).agg(
        n_query_cells=("cell", "size"), n_stations=("station", "nunique"), q90_n=("is_q90", "sum"),
        absolute_error_mass=("absolute_error", "sum"), mae=("absolute_error", "mean"),
        mean_residual=("residual", "mean"), max_doc=("y_true", "max"),
        flow_coverage=("discharge_lag0", lambda values: values.notna().mean()),
        temperature_coverage=("temperature_lag0", lambda values: values.notna().mean()))
    top_months = by_month.sort_values("absolute_error_mass", ascending=False).groupby("split_seed").head(10)
    output = args.output
    output.mkdir(parents=True, exist_ok=True)
    panel.to_parquet(output / "validation_hydro_cells.parquet", index=False)
    for name, table in (("hydro_distributions", distributions), ("hydro_residual_correlations", correlations),
                        ("population_errors", errors), ("coverage_condition_errors", coverage),
                        ("top_error_cases", top), ("top_error_months", top_months),
                        ("station_error_mass", station_mass),
                        ("current_feature_inventory", input_inventory())):
        table.to_csv(output / f"{name}.csv", index=False)
    (output / "diagnostic.md").write_text(report(
        panel, distributions, correlations, errors, top, station_mass, metadata, audit))
    code_sources = [Path(__file__), Path("src/river_graph/dataset.py"), Path("scripts/build_dataset.py"),
                    Path("src/river_graph/models/kgml_local_transport.py"),
                    Path("src/river_graph/experiments/temporal_h2x.py"),
                    Path("src/river_graph/experiments/graph_upgrade_v2.py")]
    (output / "sources.json").write_text(json.dumps({
        "role": "descriptive reused source-validation hydro diagnostic; no model fitting or outer-test labels",
        "model": MODEL, "residual_sign": "prediction_minus_observation",
        "seed_aggregation": "mean predictions, signed errors and absolute losses separately per partition-cell",
        "summary_weighting": "equal partitions; feature quantiles are reported per partition",
        "hydro_definitions": metadata,
        "files": [{"path": str(path), "sha256": sha256_file(path)} for path in sorted(set(sources + code_sources))],
    }, indent=2) + "\n")
    print(output / "diagnostic.md")


if __name__ == "__main__":
    main()
