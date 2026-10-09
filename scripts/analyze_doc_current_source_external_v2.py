"""Evaluate the fixed current-source release, preserving earlier external scores."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from analyze_doc_source_retrieval_v1 import paired
from analyze_unified_doc_spatial import joint_station_bootstrap
from run_doc_current_source_external_v2 import NEW_MODELS, ROOT
from run_doc_external_replication_v1 import EXTERNAL_PROTOCOL
from run_unified_doc_spatial import write_json

from river_graph.experiments.provenance import sha256_file
from river_graph.models.ecological_residual_transfer import EcologicalResidualTransfer


def coverage_interval(group, *, draws=5000):
    """Cell-weighted coverage with station-clustered uncertainty."""
    covered = (group.y_true >= group.pi_lower) & (group.y_true <= group.pi_upper)
    table = group[["station"]].copy()
    table["covered"] = covered.to_numpy()
    station = table.groupby("station").covered.agg(["sum", "count"])
    rng = np.random.default_rng(42)
    weights = rng.multinomial(len(station), np.full(len(station), 1/len(station)), size=draws)
    samples = (weights@station["sum"].to_numpy())/(weights@station["count"].to_numpy())
    return {"coverage": float(covered.mean()), "coverage_ci_low": float(np.quantile(samples, .025)),
            "coverage_ci_high": float(np.quantile(samples, .975))}


def metrics(group, threshold, *, draws):
    truth, predicted = group.y_true.to_numpy(), group.y_pred.to_numpy()
    error, tail = predicted-truth, truth >= threshold
    detected = predicted >= threshold
    tp = int((tail & detected).sum())
    total = np.sum((truth-truth.mean())**2)
    result = {"mae": float(np.abs(error).mean()), "rmse": float(np.sqrt(np.mean(error**2))),
        "r2": float(1-np.sum(error**2)/total) if total else np.nan,
        "bias": float(error.mean()), "log_mae": float(np.abs(np.log1p(predicted)-np.log1p(truth)).mean()),
        "station_equal_mae": float(group.assign(error=np.abs(error)).groupby("station").error.mean().mean()),
        "n_cells": len(group), "n_stations": group.station.nunique(), "q90_threshold_source": threshold,
        "q90_n": int(tail.sum()), "q90_unstable": int(tail.sum()) < 20,
        "q90_mae": float(np.abs(error[tail]).mean()) if tail.any() else np.nan,
        "q90_bias": float(error[tail].mean()) if tail.any() else np.nan,
        "q90_recall": tp/int(tail.sum()) if tail.any() else np.nan,
        "q90_precision": tp/int(detected.sum()) if detected.any() else np.nan,
        "width_median": float((group.pi_upper-group.pi_lower).median()),
        "width_mean": float((group.pi_upper-group.pi_lower).mean()),
        **coverage_interval(group, draws=draws)}
    if tail.any():
        result.update({f"q90_{key}": value for key, value in coverage_interval(group[tail], draws=draws).items()})
    return result


def environmental_strata(root, config, cells):
    """Source-derived ecological bins; no outcome-dependent categorization."""
    first = next(iter(config["source_completions"]))
    run = Path(first).parent
    fit = json.loads((run/"config.json").read_text())
    dataset = torch.load(fit["dataset_path"], weights_only=False, map_location="cpu")
    with np.load(fit["mask_path"], allow_pickle=False) as saved:
        ids = np.unique(saved["train"]//dataset["y"].shape[1])
    memory = EcologicalResidualTransfer.from_dict(json.loads((run/"memory.json").read_text()))
    state = memory.to_dict()["ecology_scaler"]
    source = np.asarray(dataset["regime"], float)[ids, 4:13]
    external = json.loads(EXTERNAL_PROTOCOL.read_text())
    with np.load(external["label_free_inputs"], allow_pickle=False) as saved:
        receiving = saved["regime"][:, 4:13].copy()
        hydro, months = saved["x_mask"].copy(), len(saved["months"])

    def standardize(raw):
        valid = np.isfinite(raw) & (raw != -1)
        x = (np.where(valid, raw, state["median"])-state["median"])/state["iqr"]
        x[:, ~np.asarray(state["active"], bool)] = 0
        return x, valid

    sources, _ = standardize(source)
    targets, valid = standardize(receiving)
    source_novelty = np.sqrt(np.mean(sources**2, axis=1))
    target_novelty = np.sqrt(np.mean(targets**2, axis=1))
    distances = np.mean((sources[:, None]-sources[None])**2, axis=-1)
    np.fill_diagonal(distances, np.inf)
    source_nearest = np.sqrt(distances.min(1))
    target_nearest = np.sqrt(np.mean((targets[:, None]-sources[None])**2, axis=-1).min(1))
    table = pd.DataFrame({"cell": cells, "hydro_channels_available": hydro.reshape(-1, 2)[cells].sum(1),
        "ecology_features_available": valid.sum(1)[cells//months]})
    definitions = {}
    for name, source_values, values in (("ecological_novelty", source_novelty, target_novelty),
                                       ("source_similarity_distance", source_nearest, target_nearest)):
        thirds = np.quantile(source_values, [1/3, 2/3])
        table[name] = values[cells//months]
        table[name+"_group"] = np.digitize(values[cells//months], thirds)
        definitions[name] = {"thresholds_source": thirds.tolist(), "degenerate": bool(thirds[0] == thirds[1])}
    write_json(root/"analysis/strata_definition.json", definitions)
    return table


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--bootstrap-draws", type=int, default=5000)
    args = parser.parse_args()
    root, draws = args.root, args.bootstrap_draws
    meta = json.loads((root/"predictions.meta.json").read_text())
    if sha256_file(root/"predictions.parquet") != meta["prediction_sha256"]:
        raise ValueError("external prediction contents changed")
    if (sha256_file(root/"point_generation_complete.json") != meta["point_generation_sha256"]
            or sha256_file(root/"source_calibration.json") != meta["source_calibration_sha256"]):
        raise ValueError("external point source or calibration changed")
    panel = pd.read_parquet(root/"predictions.parquet")
    if panel.duplicated(["population", "k", "model_name", "cell"]).any():
        raise ValueError("duplicate external query identities")
    config = meta["config"]
    thresholds = []
    for completion in config["source_completions"]:
        thresholds.append(json.loads((Path(completion).parent/"config.json").read_text())["q90_threshold_train"])
    np.testing.assert_array_equal(thresholds, np.repeat(thresholds[0], len(thresholds)))
    threshold = thresholds[0]
    panel["split_seed"] = 2040104
    output = root/"analysis"
    output.mkdir(exist_ok=True)
    records = [{"population": population, "model_name": name, "k": k,
        **metrics(group, threshold, draws=draws)} for (population, name, k), group in
        panel.groupby(["population", "model_name", "k"])]
    summary = pd.DataFrame(records)
    summary.to_csv(output/"metrics.csv", index=False)
    effects, stations = [], []
    for (population, k), group in panel.groupby(["population", "k"]):
        for reference in ("unmonitored_integrated", "current_model", "station_hidden_trees", "matched_daily_trees", "available_real_native"):
            pair = paired(group, NEW_MODELS[1], reference)
            for zone in ("overall", "q90"):
                selected = pair if zone == "overall" else pair[pair.y_true_candidate >= threshold]
                if not len(selected):
                    continue
                effects.append({"population": population, "k": k, "candidate": NEW_MODELS[1],
                    "reference": reference, "zone": zone, **joint_station_bootstrap(selected, draws=draws)})
            station = pair.groupby("station", as_index=False).agg(candidate_mae=("candidate_error", "mean"),
                reference_mae=("reference_error", "mean"), n_cells=("cell", "size"))
            station["population"], station["k"], station["reference"] = population, k, reference
            station["delta_mae"] = station.candidate_mae-station.reference_mae
            stations.append(station)
    pd.DataFrame(effects).to_csv(output/"paired_effects.csv", index=False)
    pd.concat(stations, ignore_index=True).to_csv(output/"station_effects.csv", index=False)
    primary = panel[panel.population.eq("all_observed_k0")]
    strata = environmental_strata(root, config, np.sort(primary.cell.unique()))
    strata_rows = []
    for variable in ("hydro_channels_available", "ecological_novelty_group", "source_similarity_distance_group"):
        selected = primary.merge(strata[["cell", variable]], on="cell", validate="many_to_one")
        for (name, level), group in selected.groupby(["model_name", variable]):
            strata_rows.append({"model_name": name, "stratum": variable, "group": level,
                               **metrics(group, threshold, draws=draws)})
    pd.DataFrame(strata_rows).to_csv(output/"strata.csv", index=False)
    seed_rows = []
    first = next(iter(config["source_completions"]))
    fit = json.loads((Path(first).parent/"config.json").read_text())
    q = primary[primary.model_name.eq(NEW_MODELS[1])].sort_values("cell")
    with np.load(root/"seed_components.npz", allow_pickle=False) as saved:
        for name in config["procedures"]:
            for seed, values in zip(config["seeds"], saved[f"{name}__final_pred"], strict=True):
                pred = values.ravel()[q.cell]
                seed_rows.append({"model_name": name, "seed": seed,
                    "mae": float(np.abs(pred-q.y_true.to_numpy()).mean()),
                    "bias": float((pred-q.y_true.to_numpy()).mean()),
                    "q90_mae": float(np.abs(pred-q.y_true.to_numpy())[q.y_true >= threshold].mean())})
    pd.DataFrame(seed_rows).to_csv(output/"individual_seed_metrics.csv", index=False)
    # Source validation selected the fitted states. Describe it separately;
    # it does not add independent ecological replication to the external case.
    source_panel = pd.read_parquet(root/"source_validation.parquet")
    calibration = json.loads((root/"source_calibration.json").read_text())["procedures"]
    validation_rows, validation_effects = [], []
    ensemble_rows = []
    for name, group in source_panel.groupby("model_name"):
        center = group.groupby(["cell", "station", "month"], as_index=False).agg(
            y_pred=("y_pred", "mean"), y_true=("y_true", "first"), truths=("y_true", "nunique"))
        if not center.truths.eq(1).all() or not group.groupby("cell").size().eq(5).all():
            raise ValueError("source validation requires five aligned members and identical truth")
        from run_doc_external_replication_v1 import empirical_interval
        center["pi_lower"], center["pi_upper"] = empirical_interval(center.y_pred,
            calibration[name]["primary_k0_log_half_width"])
        validation_rows.append({"model_name": name, "role": "source_validation_used_for_selection",
            "seed_mean_mae": float(np.abs(group.y_pred-group.y_true).mean()),
            **metrics(center, threshold, draws=draws)})
        ensemble_rows.append(center.assign(model_name=name, seed=-1, split_seed=0))
    source_ensemble = pd.concat(ensemble_rows, ignore_index=True)
    for reference in ("unmonitored_integrated", "current_model", "station_hidden_trees", NEW_MODELS[0]):
        pair = paired(source_ensemble, NEW_MODELS[1], reference)
        validation_effects.append({"candidate": NEW_MODELS[1], "reference": reference,
            "role": "source_validation_used_for_selection", **joint_station_bootstrap(pair, draws=draws)})
    pd.DataFrame(validation_rows).to_csv(output/"source_validation_summary.csv", index=False)
    pd.DataFrame(validation_effects).to_csv(output/"source_validation_effects.csv", index=False)
    with np.load(root/"seed_components.npz", allow_pickle=False) as saved:
        counts = saved[f"{NEW_MODELS[1]}__source_support_count"][:, q.cell.to_numpy()//len(saved["month"]),
            q.cell.to_numpy() % len(saved["month"])]
        priors = saved[f"{NEW_MODELS[1]}__attention_prior_mass"][:, q.cell.to_numpy()//len(saved["month"]),
            q.cell.to_numpy() % len(saved["month"])]
        donor_group = np.digitize(counts.mean(0), [.5, 4.5])
        donor_rows = []
        for group_id, label in enumerate(("no_current_source", "one_to_four_sources", "five_or_more_sources")):
            selected = q[donor_group == group_id]
            if len(selected):
                donor_rows.append({"group": label, **metrics(selected, threshold, draws=draws)})
        pd.DataFrame(donor_rows).to_csv(output/"source_availability_strata.csv", index=False)
        write_json(output/"source_availability.json", {"n_cells": len(q),
            "mean_source_count": float(counts.mean()), "supported_cell_fraction": float((counts > 0).mean()),
            "mean_zero_prior_mass": float(priors.mean()), "definition": "source-seed mean over all observed external cells"})
    write_json(output/"sources.json", {"config": config, "prediction_hash": meta["prediction_sha256"],
        "analyzer_sha256": sha256_file(__file__), "bootstrap_draws": draws,
        "q90_source_training_threshold": fit["q90_threshold_train"],
        "scope": config["external_exposure"]})
    print(summary[summary.population.eq("all_observed_k0")][[
        "model_name", "mae", "q90_mae", "coverage", "width_median"]].to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
