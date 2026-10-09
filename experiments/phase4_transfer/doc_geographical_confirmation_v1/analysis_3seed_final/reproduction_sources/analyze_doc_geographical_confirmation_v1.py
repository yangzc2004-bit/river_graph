"""Verify and analyze the fixed whole-HUC4 DOC reconstruction experiment."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from analyze_doc_chemistry_confirmation_v1 import metric_summary
from analyze_doc_source_retrieval_v1 import paired
from analyze_unified_doc_spatial import joint_station_bootstrap
from run_doc_geographical_confirmation_v1 import MODELS, ROOT, TASKS
from run_unified_doc_spatial import (
    digest,
    verify_files,
    verify_runtime_snapshot,
    write_json,
)

from river_graph.experiments.provenance import run_identity_sha256, sha256_file
from river_graph.models.ecological_residual_transfer import EcologicalResidualTransfer
from river_graph.models.encoder_native_residual import EncoderNativeResidual


def verify(run, runtime):
    config = json.loads((run / "config.json").read_text())
    if config["runtime_snapshot_hash"] != runtime:
        raise ValueError("incorrect geographical execution identity")
    for kind in ("dataset", "mask"):
        if sha256_file(config[f"{kind}_path"]) != config[f"{kind}_hash"]:
            raise ValueError(f"changed geographical {kind}")
    for record in ("complete.json", "backbone_complete.json", "trees_complete.json", "current_complete.json"):
        verify_files(run, record, config)
    source = torch.load(config["dataset_path"], weights_only=False)
    truth = np.asarray(source["y"]).ravel()
    months = source["y"].shape[1]
    with np.load(config["mask_path"], allow_pickle=False) as saved:
        cells = saved["test"].copy()
    model = EncoderNativeResidual.from_payload(torch.load(run / "native.pt", weights_only=False))
    with np.load(run / "test_inputs.npz", allow_pickle=False) as saved:
        inputs = {key: saved[key].copy() for key in ("raw", "age", "support", "env", "extra")}
        context = saved["context"].copy()
        np.testing.assert_array_equal(saved["cell"], cells)
    native = np.maximum(0, context + model.selected_scale_ * model.predict_delta(inputs))
    ids = np.unique(cells//months)
    row, column = np.searchsorted(ids, cells//months), cells % months
    memory = EcologicalResidualTransfer.from_dict(json.loads((run / "memory.json").read_text()))
    with np.load(run / "components.npz", allow_pickle=False) as saved:
        np.testing.assert_allclose(native, saved["native"][ids], rtol=1e-6, atol=1e-6)
        complete_native = saved["native"].copy()
        environment = saved["environment"].copy()
        integrated = memory.predict(environment, complete_native)
        np.testing.assert_array_equal(integrated, saved["integrated"])
    for filename in ("predictions.parquet", "support_curves.parquet"):
        meta = json.loads((run / filename.replace(".parquet", ".meta.json")).read_text())
        if (meta["config_hash"] != digest(config) or meta["prediction_sha256"] != sha256_file(run / filename)
                or meta["run_identity_sha256"] != run_identity_sha256(digest(config), config["started_at"], runtime)):
            raise ValueError("changed geographical prediction sidecar")
        frame = pd.read_parquet(run / filename)
        if set(frame.model_name) != set(MODELS) or not frame.visibility_role.eq("test").all():
            raise ValueError("incomplete test panel")
        if frame.duplicated(["model_name", "k", "cell"]).any():
            raise ValueError("duplicate geographical query rows")
        query = None
        for (_, _), rows in frame.groupby(["model_name", "k"]):
            selected = rows.cell.to_numpy()
            if query is None:
                query = selected
            np.testing.assert_array_equal(selected, query)
            np.testing.assert_array_equal(rows.y_true, truth[selected])
            np.testing.assert_array_equal(rows.station, np.asarray(source["site_no"], str)[selected//months])
            np.testing.assert_array_equal(rows.month, np.asarray(source["months"], str)[selected % months])
            if not np.isfinite(rows.y_pred).all():
                raise ValueError("nonfinite geographical prediction")
        if filename == "predictions.parquet":
            np.testing.assert_array_equal(query, cells)
            np.testing.assert_allclose(frame[frame.model_name.eq("unmonitored_residual")].y_pred,
                                       native[row, column], rtol=1e-6, atol=1e-6)
            np.testing.assert_array_equal(frame[frame.model_name.eq("unmonitored_integrated")].y_pred,
                                          integrated.ravel()[cells])
    folds = json.loads((run / "oof_records.json").read_text())
    for scheme, records in folds.items():
        for record in records:
            a, b = ("held_station_ids", "training_station_ids") if scheme == "current" else ("hidden_stations", "fitted_stations")
            if set(record[a]) & set(record[b]):
                raise ValueError("station-OOF fit contains its held station")
    return {"run": run.name, "identity": True, "source_oof_exclusion": True,
            "native_replay": "rtol/atol1e-6; different inference batch composition",
            "integrated_replay": "bitwise", "test_cells": len(cells)}


def summaries(panel, thresholds):
    """Use the established metrics with the actual region count, not three."""
    runs, regions, summary = metric_summary(panel, thresholds)
    count = panel.split_seed.nunique()
    for metric in ("q90_mae", "r2", "log_r2"):
        values = regions.groupby(["model_name", "k"])[metric]
        means, counts = values.mean(), values.count()
        for index, row in summary.iterrows():
            key = (row.model_name, row.k)
            summary.loc[index, metric] = means[key] if counts[key] == count else np.nan
    station_runs = panel.assign(error=np.abs(panel.y_pred-panel.y_true)).groupby(
        ["split_seed", "seed", "model_name", "k", "station"], as_index=False).error.mean()
    station_regions = station_runs.groupby(["split_seed", "model_name", "k"], as_index=False).error.mean()
    station_summary = station_regions.groupby(["model_name", "k"]).error.mean()
    summary["station_equal_mae"] = [station_summary[(r.model_name, r.k)] for r in summary.itertuples()]
    return runs, regions, summary, station_runs


def additional_metrics(panel, thresholds):
    rows = []
    for (region, seed, model, k), group in panel.groupby(["split_seed", "seed", "model_name", "k"]):
        threshold = thresholds[(region, seed)]
        actual, detected = group.y_true.to_numpy() >= threshold, group.y_pred.to_numpy() >= threshold
        tp = int((actual & detected).sum())
        rows.append({"split_seed": region, "seed": seed, "model_name": model, "k": k,
            "q90_threshold_train": threshold, "q90_n": int(actual.sum()), "q90_tp": tp,
            "q90_recall": tp/int(actual.sum()) if actual.any() else np.nan,
            "q90_precision": tp/int(detected.sum()) if detected.any() else np.nan,
            "q90_bias": float(np.mean((group.y_pred-group.y_true).to_numpy()[actual])) if actual.any() else np.nan,
            "q90_unstable": int(actual.sum()) < 20})
    return pd.DataFrame(rows)


def improving_region_count(comparison):
    """Count directions for the evaluated population, including tail subsets."""
    per_seed = comparison.groupby(["split_seed", "seed"])[[
        "candidate_error", "reference_error"]].mean()
    per_region = per_seed.groupby("split_seed").mean()
    return int((per_region.candidate_error < per_region.reference_error).sum())


def strata_for_run(run, panel):
    config = json.loads((run / "config.json").read_text())
    dataset = torch.load(config["dataset_path"], weights_only=False)
    months = dataset["y"].shape[1]
    with np.load(config["mask_path"], allow_pickle=False) as saved:
        source_ids = np.unique(saved["train"]//months)
    memory = json.loads((run / "memory.json").read_text())
    ecology = np.asarray(dataset["regime"], float)[:, 4:13]
    norm = memory["ecology_scaler"]
    valid = np.isfinite(ecology) & (ecology != -1)
    standardized = (np.where(valid, ecology, norm["median"])-norm["median"]) / norm["iqr"]
    standardized[:, ~np.asarray(norm["active"], bool)] = 0
    # Novelty measures deviation from the source center. Nearest-source distance
    # measures the amount of comparable donor experience; both use source scales.
    novelty = np.sqrt(np.mean(standardized**2, axis=1))
    squared = np.mean((standardized[:, None]-standardized[source_ids][None])**2, axis=2)
    squared[source_ids, np.arange(len(source_ids))] = np.inf
    nearest = np.sqrt(squared.min(axis=1))
    labels = {"ecological_novelty": np.digitize(novelty,
                    np.quantile(novelty[source_ids], [1/3, 2/3])),
              "source_similarity_distance": np.digitize(nearest,
                    np.quantile(nearest[source_ids], [1/3, 2/3]))}
    cells = panel.cell.to_numpy()
    hydro = np.asarray(dataset["x_mask"], bool).reshape(-1, 2)[cells].sum(axis=1)
    records = []
    definitions = [(name, value[cells//months]) for name, value in labels.items()]
    definitions.append(("hydro_channels_available", hydro))
    error = np.abs(panel.y_pred.to_numpy()-panel.y_true.to_numpy())
    for name, group_labels in definitions:
        table = panel[["model_name", "k"]].copy()
        table["group"], table["error"] = group_labels, error
        grouped = table.groupby(["model_name", "k", "group"]).agg(mae=("error", "mean"), n_cells=("error", "size"))
        for (model, k, group), row in grouped.iterrows():
            records.append({"split_seed": int(config["split_seed"]), "seed": config["seed"],
                "model_name": model, "k": k, "stratum": name, "group": int(group),
                "mae": row.mae, "n_cells": int(row.n_cells)})
    return records


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44])
    parser.add_argument("--bootstrap-draws", type=int, default=5000)
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args()
    torch.set_num_threads(2)
    runtime = verify_runtime_snapshot(args.root)
    regions = json.loads((TASKS / "task_protocol.json").read_text())["geographical_targets"]
    paths = [args.root / "runs" / f"huc4_{region}_seed{seed}" for region in regions for seed in args.seeds]
    missing = [str(p) for p in paths if not (p / "complete.json").exists()]
    if missing:
        raise ValueError(f"geographical run packages missing: {missing}")
    write_json(args.root / "verification" / "replay.json", [verify(p, runtime) for p in paths])
    if args.verify_only:
        return
    frames, curves, thresholds, stratification = [], [], {}, []
    for path in paths:
        config = json.loads((path / "config.json").read_text())
        primary = pd.read_parquet(path / "predictions.parquet")
        frames.append(primary)
        curves.append(pd.read_parquet(path / "support_curves.parquet"))
        thresholds[(config["split_seed"], config["seed"])] = config["q90_threshold_train"]
        stratification.extend(strata_for_run(path, primary))
    output = args.root / "analysis"
    output.mkdir(exist_ok=True)
    primary, curves = pd.concat(frames, ignore_index=True), pd.concat(curves, ignore_index=True)
    for label, panel in (("primary", primary), ("curves", curves)):
        runs, parts, summary, station_runs = summaries(panel, thresholds)
        for name, table in (("run_metrics", runs), ("region_metrics", parts), ("summary", summary),
                             ("station_metrics", station_runs), ("q90_diagnostics", additional_metrics(panel, thresholds))):
            table.to_csv(output / f"{label}_{name}.csv", index=False)
    effects, station_effects = [], []
    for population, panel in (("all_observed_k0", primary), ("fixed_query_curve", curves)):
        for k in sorted(panel.k.unique()):
            for reference in ("current_model", "station_hidden_trees", "matched_daily_trees", "unmonitored_residual"):
                comparison = paired(panel[panel.k.eq(k)], "unmonitored_integrated", reference)
                for zone in ("overall", "q90"):
                    selected = comparison
                    if zone == "q90":
                        limit = np.array([thresholds[(s, t)] for s, t in zip(
                            comparison.split_seed, comparison.seed, strict=True)])
                        selected = comparison[comparison.y_true_candidate.to_numpy() >= limit]
                    if selected.split_seed.nunique() != len(regions):
                        effects.append({"candidate": "unmonitored_integrated", "reference": reference,
                            "population": population, "k": k, "zone": zone,
                            "relative_gain_pct": np.nan, "gain_ci_low_pct": np.nan,
                            "gain_ci_high_pct": np.nan, "improving_regions": np.nan,
                            "n_regions_usable": selected.split_seed.nunique(),
                            "status": "not estimable for five-region mean: an empty tail stratum"})
                        continue
                    effects.append({"candidate": "unmonitored_integrated", "reference": reference,
                        "population": population, "k": k, "zone": zone,
                        **joint_station_bootstrap(selected, draws=args.bootstrap_draws),
                        "improving_regions": improving_region_count(selected),
                        "n_regions_usable": len(regions), "status": "estimated"})
                stations = comparison.groupby(["split_seed", "station"], as_index=False).agg(
                    candidate_mae=("candidate_error", "mean"), reference_mae=("reference_error", "mean"), n_cells=("cell", "nunique"))
                stations["delta_mae"] = stations.candidate_mae-stations.reference_mae
                stations["population"], stations["k"], stations["reference"] = population, k, reference
                station_effects.append(stations)
    pd.DataFrame(effects).to_csv(output / "paired_effects.csv", index=False)
    pd.concat(station_effects, ignore_index=True).to_csv(output / "station_effects.csv", index=False)
    strata = pd.DataFrame(stratification)
    strata.to_csv(output / "strata_by_run.csv", index=False)
    region_strata = strata.groupby(["split_seed", "model_name", "k", "stratum", "group"], as_index=False).mae.mean()
    region_strata.to_csv(output / "strata_by_region.csv", index=False)
    region_strata.groupby(["model_name", "k", "stratum", "group"], as_index=False).agg(
        mae=("mae", "mean"), n_regions=("split_seed", "nunique")).to_csv(output / "strata_summary.csv", index=False)
    write_json(output / "sources.json", {"bootstrap_draws": args.bootstrap_draws,
        "seeds": args.seeds, "regions": regions, "estimand": "seed mean within region, equal regions",
        "scope": "ST357 retrospective geographical replication", "runtime": runtime,
        "completed_runs": {str(p / "complete.json"): sha256_file(p / "complete.json") for p in paths}})
    print(pd.read_csv(output / "primary_summary.csv")[["model_name", "mae", "station_equal_mae", "q90_mae"]].to_string(index=False))


if __name__ == "__main__":
    main()
