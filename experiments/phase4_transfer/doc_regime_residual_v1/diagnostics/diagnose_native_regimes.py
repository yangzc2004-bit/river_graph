"""Source/validation-only stratification of native DOC context residuals."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

REPO = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPO / "scripts"))

from analyze_unified_doc_spatial import sha256
from diagnose_doc_residual_shift import read_cells

from river_graph.models.causal_flow_features import (
    build_causal_flow_features,
)
from river_graph.models.ecological_residual_transfer import ECOLOGY_NAMES

OUT = Path(__file__).resolve().parent
PRIOR = Path("experiments/phase4_transfer/doc_flow_interaction_v1")


def station_weights(frame):
    counts = frame.groupby("station").station.transform("size").to_numpy()
    return 1.0 / counts


def weighted_tertiles(values, weights):
    order = np.argsort(values, kind="stable")
    cumulative = np.cumsum(weights[order])
    return np.interp(np.array([1 / 3, 2 / 3]) * cumulative[-1], cumulative, values[order])


def enrich(cells):
    frames, cut_records, sources = [], [], []
    for (partition, seed), group in cells.groupby(["split_seed", "seed"]):
        config_path = PRIOR / "runs" / f"split{partition}_seed{seed}" / "config.json"
        config = json.loads(config_path.read_text())
        dataset_path = Path(config["dataset_path"])
        dataset = torch.load(dataset_path, weights_only=False, map_location="cpu")
        if sha256(dataset_path) != config["dataset_hash"]:
            raise ValueError("Dataset changed")
        n_months = dataset["y"].shape[1]
        frame = group.copy()
        station = frame.cell.to_numpy() // n_months
        regime = np.asarray(dataset["regime"], dtype=float)[:, 4:13]
        regime[(regime == -1) | ~np.isfinite(regime)] = np.nan
        for index, name in enumerate(ECOLOGY_NAMES):
            frame[name] = regime[station, index]
        flow = build_causal_flow_features(dataset)["full"].reshape(-1, 10)
        frame["flow_anomaly"] = flow[frame.cell, 0]
        frame.loc[flow[frame.cell, 1] == 0, "flow_anomaly"] = np.nan
        source = frame[frame.role.eq("source_oof")]
        source_stations = source.drop_duplicates("station")
        for variable in (*ECOLOGY_NAMES, "base_pred", "flow_anomaly"):
            fit = source_stations if variable in ECOLOGY_NAMES else source
            valid = fit[variable].notna()
            reference = fit.loc[valid]
            weights = np.ones(len(reference)) if variable in ECOLOGY_NAMES else station_weights(reference)
            cuts = weighted_tertiles(reference[variable].to_numpy(), weights)
            values = frame[variable].to_numpy()
            bins = np.searchsorted(cuts, np.nan_to_num(values, nan=0.0), side="right")
            frame[f"bin_{variable}"] = np.where(np.isnan(values), "missing", np.array(["low", "middle", "high"])[bins])
            cut_records.append({"split_seed": partition, "seed": seed, "variable": variable,
                                "lower_cut": cuts[0], "upper_cut": cuts[1],
                                "source_reference_cells": len(reference),
                                "source_reference_stations": reference.station.nunique(),
                                "degenerate": bool(cuts[0] == cuts[1])})
        frame["bin_flow_observed"] = np.where(frame.flow_observed, "observed", "missing")
        frame["bin_upstream_support"] = np.where(frame.upstream_fraction > 0, "present", "absent")
        frame["bin_predicted_q90"] = np.where(frame.base_pred >= frame.q90_threshold_train, "above", "below")
        frames.append(frame)
        sources.extend({"path": str(path), "sha256": sha256(path)} for path in (config_path, dataset_path))
    return pd.concat(frames, ignore_index=True), pd.DataFrame(cut_records), sources


def summarize(frame):
    rows = []
    variables = ("all", *ECOLOGY_NAMES, "base_pred", "flow_anomaly", "flow_observed", "upstream_support", "predicted_q90")
    for (partition, seed, role), group in frame.groupby(["split_seed", "seed", "role"]):
        for variable in variables:
            strata = [("all", group)] if variable == "all" else list(group.groupby(f"bin_{variable}"))
            for level, part in strata:
                for region in ("overall", "q90", "nontail"):
                    sub = part if region == "overall" else part[part.is_q90.eq(region == "q90")]
                    if not len(sub):
                        continue
                    for weighting in ("pooled_cells", "equal_stations"):
                        weights = np.ones(len(sub)) if weighting == "pooled_cells" else station_weights(sub)
                        weights /= weights.sum()
                        y, base, residual = sub.y_true.to_numpy(), sub.base_pred.to_numpy(), sub.residual.to_numpy()
                        rows.append({"split_seed": partition, "seed": seed, "role": role,
                                     "variable": variable, "level": level, "region": region, "weighting": weighting,
                                     "n_cells": len(sub), "n_stations": sub.station.nunique(),
                                     "small_group": len(sub) < 20 or sub.station.nunique() < 5,
                                     "y_mean": np.dot(weights, y), "base_mean": np.dot(weights, base),
                                     "residual_mean": np.dot(weights, residual),
                                     "mae": np.dot(weights, np.abs(residual)),
                                     "underprediction_fraction": np.dot(weights, residual > 0),
                                     "q90_fraction": np.dot(weights, sub.is_q90.to_numpy())})
    runs = pd.DataFrame(rows)
    groups = ["role", "variable", "level", "region", "weighting"]
    metrics = ["y_mean", "base_mean", "residual_mean", "mae", "underprediction_fraction", "q90_fraction"]
    parts = runs.groupby(["split_seed", *groups], as_index=False).agg(
        **{name: (name, "mean") for name in metrics}, n_seeds=("seed", "nunique"),
        n_cells=("n_cells", "mean"), n_stations=("n_stations", "mean"), small_group=("small_group", "any"))
    summary = parts.groupby(groups, as_index=False).agg(
        **{name: (name, "mean") for name in metrics}, n_partitions=("split_seed", "nunique"),
        min_seeds=("n_seeds", "min"), n_cells=("n_cells", "mean"), n_stations=("n_stations", "mean"),
        small_group=("small_group", "any"))
    summary.loc[(summary.n_partitions < 3) | (summary.min_seeds < 3), metrics] = np.nan
    return runs, parts, summary


def write_report(summary, parts, cuts):
    lines = ["# Source/validation DOC residual regimes", "",
             "Positive native residual is observed DOC minus the context prediction (underprediction).",
             "Only source OOF cells and fixed source-validation queries are used. No outer-test query",
             "labels or outcomes are inspected. Source and validation differ in station composition and",
             "OOF versus full-source context fitting; their contrasts do not isolate a cause.", "",
             "All nine existing ecology fields, context prediction and valid causal relative-flow anomaly",
             "are divided at source-defined tertiles. Ecology uses unique source stations; dynamic cuts",
             "use equal source-station weights. Missing values remain separate. Equal-station summaries",
             "give each station with observations in the reported group the same total weight; pooled",
             "summaries weight every cell equally. Seeds are averaged within partition, then partitions",
             "are weighted equally. Counts are mean per-partition counts, not independent repeated samples.", "",
             "## Overall concentration regimes", "",
             "| Role | Region | Weighting | Mean DOC | Mean base | Residual | MAE | Stations |",
             "|---|---|---|---:|---:|---:|---:|---:|"]
    for row in summary[summary.variable.eq("all")].itertuples():
        lines.append(f"| {row.role} | {row.region} | {row.weighting} | {row.y_mean:.2f} | {row.base_mean:.2f} | "
                     f"{row.residual_mean:+.2f} | {row.mae:.2f} | {row.n_stations:.0f} |")
    lines += ["", "## High-DOC residual strata", "",
              "| Variable | Level | Source: pooled / station-balanced residual | Validation: pooled / station-balanced residual | Validation tail cells / stations | Small group |",
              "|---|---|---:|---:|---:|---|"]
    tail = summary[summary.region.eq("q90")]
    for (variable, level), group in tail.groupby(["variable", "level"], sort=False):
        if variable == "all":
            continue
        table = group.set_index(["role", "weighting"])
        keys = [(role, weight) for role in ("source_oof", "source_validation") for weight in ("pooled_cells", "equal_stations")]
        if not all(key in table.index for key in keys):
            continue
        a, b, c, d = (table.loc[key] for key in keys)
        lines.append(f"| {variable} | {level} | {a.residual_mean:+.2f} / {b.residual_mean:+.2f} | "
                     f"{c.residual_mean:+.2f} / {d.residual_mean:+.2f} | {d.n_cells:.0f} / {d.n_stations:.0f} | "
                     f"{bool(group.small_group.any())} |")
    lines += ["", "Small groups have fewer than 20 cells or five stations in at least one run. Groups",
              "missing any partition are retained in the per-run tables and not silently averaged.",
              "Tertile labels may describe unequal groups when source values tie; the saved cut table",
              "identifies degenerate boundaries. Strata are marginal descriptions, not adjusted effects",
              "or evidence that one covariate causes the error.", "",
              "## Partition-level validation tail", "",
              "| Partition | Weighting | Mean DOC | Mean base | Residual | MAE |",
              "|---|---|---:|---:|---:|---:|"]
    for row in parts[parts.role.eq("source_validation") & parts.variable.eq("all") & parts.region.eq("q90")].itertuples():
        lines.append(f"| {row.split_seed} | {row.weighting} | {row.y_mean:.2f} | {row.base_mean:.2f} | {row.residual_mean:+.2f} | {row.mae:.2f} |")
    lines += ["", "## Implication for the residual head", "",
              "Tail underprediction persists after equal-station weighting. It is not explained by",
              "missing discharge alone: large residuals also occur when discharge is observed, and",
              "the relative-flow strata do not show a consistent monotone trend. Static ecological",
              "strata show heterogeneous residual levels; several relations are nonmonotone and",
              "differ between source and validation. This supports testing joint regime conditioning,",
              "rather than choosing one ecological variable as a physical rule.", "",
              "The context base often predicts below its source Q90 threshold on true high-DOC cells.",
              "Those cells have larger positive residuals than tail cells already predicted above Q90.",
              "A correction activated only by a high context prediction would therefore miss an",
              "important part of the underprediction. Context concentration should be a continuous",
              "conditioning input, not an oracle tail indicator or a hard high-concentration gate.", "",
              "The minimal next test is a source-normalized regime-conditioned scalar residual head:",
              "retain the GRU and flow-interaction branch, expose the existing nine ecology variables",
              "and context prediction directly, and compare additive inputs, hidden-by-concentration",
              "interaction, and hidden-by-ecology interaction. Source training uses OOF context",
              "predictions; validation uses its full-source context predictor. Availability flags stay",
              "explicit. The tail loss weight need not increase for this test. These controls ask",
              "whether conditioning the correction helps beyond merely exposing the same features.", "",
              "The tables describe where residuals concentrate. They do not fit an alternative",
              "predictor, choose a threshold from validation, or estimate an error floor.", ""]
    (OUT / "native_regimes.md").write_text("\n".join(lines))


def main():
    cells, _, sources = read_cells(PRIOR)
    enriched, cuts, extra_sources = enrich(cells)
    runs, parts, summary = summarize(enriched)
    for name, frame in (("strata_by_run", runs), ("strata_by_partition", parts), ("strata_summary", summary), ("source_cuts", cuts)):
        frame.to_csv(OUT / f"{name}.csv", index=False)
    write_report(summary, parts, cuts)
    sources.extend(extra_sources)
    for path in (Path(__file__), Path("scripts/diagnose_doc_residual_shift.py"),
                 Path("src/river_graph/models/causal_flow_features.py")):
        sources.append({"path": str(path), "sha256": sha256(path)})
    (OUT / "sources.json").write_text(json.dumps(list({row["path"]: row for row in sources}.values()), indent=2) + "\n")
    print(json.dumps({"output": str(OUT), "source_and_validation_rows": len(cells),
                      "outer_test_labels_used": False, "models_fitted": 0}))


if __name__ == "__main__":
    main()
