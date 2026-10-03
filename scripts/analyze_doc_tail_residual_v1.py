"""Evaluate all saved native-DOC residual arms on fixed station holdouts.

This is a development comparison on previously examined station partitions.
The script neither selects a model nor changes predictions, checkpoints or
support adapters. Positive relative reduction favors the named candidate.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from analyze_unified_doc_spatial import (
    config_digest,
    joint_station_bootstrap,
    paired_cells,
    sha256,
)
from analyze_unified_doc_spatial_v2 import (
    KS,
    SEEDS,
    SPLITS,
    canonical_panel,
    validate_panel,
)
from analyze_unified_doc_spatial_v3 import (
    add_directions,
    read_bound_json,
    summarize_metrics,
)

from river_graph.experiments.provenance import run_identity_sha256

ROOT = Path("experiments/phase4_transfer/doc_tail_residual_v1")
ARMS = ("native_mae", "native_tail")
SHAPES = ("constant", "gru_tuned_anchor")
MODELS = tuple(f"{base}_{shape}" for base in ("context", *ARMS) for shape in SHAPES)
REFERENCES = ("v4_fusion_gru_tuned_anchor", "v4_fusion_tree_episodic")
REGIONS = ("overall", "q90", "nontail")
PROFILE_METRICS = ("mae", "log_mae", "signed_bias", "log_signed_bias", "q90_false_positive_rate")


def comparison_definitions():
    rows = []
    for arm in ARMS:
        for shape in SHAPES:
            for k in (0, 5):
                rows.append((f"{arm}_vs_context_{shape}_k{k}", f"{arm}_{shape}", k,
                             f"context_{shape}", k, "matched_environmental_control"))
            for reference in REFERENCES:
                rows.append((f"{arm}_{shape}_vs_{reference}_k5", f"{arm}_{shape}", 5,
                             reference, 5, "prior_fusion_reference"))
    for shape in SHAPES:
        for k in (0, 5):
            rows.append((f"tail_vs_ordinary_{shape}_k{k}", f"native_tail_{shape}", k,
                         f"native_mae_{shape}", k, "training_objective_control"))
    return rows


def read_predictions(run, sources):
    """Check the completed product, its sidecar, and its named model files."""
    completion_path = run / "complete.json"
    if not completion_path.is_file():
        raise ValueError(f"All nine completed runs are required; missing {completion_path}")
    completion = json.loads(completion_path.read_text())
    config = read_bound_json(run, "config.json", completion, sources)
    digest = config_digest(config)
    if completion["config_hash"] != digest:
        raise ValueError(f"Changed configuration identity: {run}")
    metadata = read_bound_json(run, "predictions.meta.json", completion, sources)
    prediction_path = run / "predictions.parquet"
    content_hash = sha256(prediction_path)
    expected = {
        "config_hash": digest, "runtime_snapshot_hash": config["runtime_snapshot_hash"],
        "dataset_sha256": config["dataset_hash"], "mask_sha256": config["mask_hash"],
        "run_identity_sha256": run_identity_sha256(digest, config["started_at"],
                                                   config["runtime_snapshot_hash"]),
        "prediction_sha256": content_hash, "selection_role": "source_validation",
    }
    if (config_digest(metadata["config"]) != digest
            or any(metadata.get(key) != value for key, value in expected.items())
            or completion["files"].get(prediction_path.name) != content_hash):
        raise ValueError(f"Prediction sidecar identity differs: {run}")
    if not metadata.get("model_files"):
        raise ValueError(f"Prediction has no bound model files: {run}")
    for name, recorded in metadata["model_files"].items():
        actual = sha256(run / name)
        if recorded != actual or completion["files"].get(name) != actual:
            raise ValueError(f"Changed prediction model dependency: {run / name}")
        sources.append({"path": str(run / name), "sha256": actual})
    raw = pd.read_parquet(prediction_path)
    if len(raw) != metadata["rows"]:
        raise ValueError(f"Sidecar row count differs: {run}")
    frame = canonical_panel(raw)
    if (not frame.split_seed.eq(config["split_seed"]).all()
            or not frame.seed.eq(config["seed"]).all()
            or frame.cell.nunique() != int(config["query_cells"])):
        raise ValueError(f"Prediction run/query identity differs: {run}")
    sources.extend({"path": str(path), "sha256": sha256(path)}
                   for path in (prediction_path, completion_path))
    return frame, config, completion


def load_panel(root):
    frames, thresholds, sources, states = [], {}, [], []
    for split in SPLITS:
        for seed in SEEDS:
            run = root / "runs" / f"split{split}_seed{seed}"
            frame, config, completion = read_predictions(run, sources)
            if (config["split_seed"], config["seed"]) != (split, seed):
                raise ValueError(f"Directory/configuration identity differs: {run}")
            if (set(config["models"]) != set(MODELS) or tuple(config["k_values"]) != KS
                    or config["inference_roles"] != ["train"]):
                raise ValueError(f"Unexpected model/K/visibility configuration: {run}")
            threshold = float(config["q90_threshold_train"])
            if not np.isfinite(threshold) or threshold < 0:
                raise ValueError(f"Invalid source-training Q90: {run}")
            thresholds[(split, seed)] = threshold
            prior = Path(config["prior_run"])
            if sha256(prior / "complete.json") != config["prior_completion_hash"]:
                raise ValueError(f"Prior reference package changed: {prior}")
            old, old_config, _ = read_predictions(prior, sources)
            for key in ("split_seed", "seed", "dataset_hash", "mask_hash", "q90_threshold_train"):
                if old_config[key] != config[key]:
                    raise ValueError(f"Reference configuration differs in {key}: {run}")
            for shape in SHAPES:
                name = f"context_{shape}"
                current = frame[frame.model_name.eq(name)].sort_values(["k", "cell"])
                previous = old[old.model_name.eq(name)].sort_values(["k", "cell"])
                np.testing.assert_array_equal(current.y_pred, previous.y_pred)
            reference = old[old.model_name.isin(tuple(name.removeprefix("v4_")
                                                     for name in REFERENCES))].copy()
            reference["model_name"] = "v4_" + reference.model_name
            frames.extend([frame, reference])
            states.append({"split_seed": split, "seed": seed, "config": config,
                           "adapters": read_bound_json(run, "adapters.json", completion, sources),
                           "training": {arm: read_bound_json(run, f"{arm}.json", completion, sources)
                                        for arm in ARMS}})
        if len({thresholds[(split, seed)] for seed in SEEDS}) != 1:
            raise ValueError(f"Q90 threshold changes across training seeds: {split}")
    panel = pd.concat(frames, ignore_index=True)
    validate_panel(panel, expected_models=MODELS + REFERENCES)
    for _, run in panel[panel.k.eq(0)].groupby(["split_seed", "seed"]):
        zero = run.pivot(index="cell", columns="model_name", values="y_pred")
        for base in ("context", *ARMS):
            np.testing.assert_array_equal(zero[f"{base}_constant"], zero[f"{base}_gru_tuned_anchor"])
    return panel, thresholds, sources, states


def error_profiles(panel, thresholds):
    rows = []
    for (split, seed, model, k), group in panel.groupby(["split_seed", "seed", "model_name", "k"]):
        y, pred = group.y_true.to_numpy(), group.y_pred.to_numpy()
        threshold = thresholds[(int(split), int(seed))]
        tail = y >= threshold
        error, log_error = pred - y, np.log1p(pred) - np.log1p(y)
        for region, selected in (("overall", np.ones(len(y), dtype=bool)), ("q90", tail), ("nontail", ~tail)):
            negatives = selected & ~tail
            rows.append({"split_seed": split, "seed": seed, "model_name": model, "k": k,
                         "region": region, "n_query_cells": int(selected.sum()),
                         "n_stations": int(group.loc[selected, "station"].nunique()),
                         "q90_threshold_train": threshold, "unstable": int(selected.sum()) < 20,
                         "mae": float(np.abs(error[selected]).mean()) if selected.any() else np.nan,
                         "log_mae": float(np.abs(log_error[selected]).mean()) if selected.any() else np.nan,
                         "signed_bias": float(error[selected].mean()) if selected.any() else np.nan,
                         "log_signed_bias": float(log_error[selected].mean()) if selected.any() else np.nan,
                         "n_actual_nontail": int(negatives.sum()),
                         "n_false_q90": int(((pred >= threshold) & negatives).sum()),
                         "q90_false_positive_rate": float((pred[negatives] >= threshold).mean())
                         if negatives.any() else np.nan})
    runs = pd.DataFrame(rows)
    splits = runs.groupby(["split_seed", "model_name", "k", "region"], as_index=False).agg(
        **{metric: (metric, "mean") for metric in PROFILE_METRICS},
        n_query_cells=("n_query_cells", "first"), n_stations=("n_stations", "first"),
        unstable=("unstable", "any"), n_seeds=("seed", "size"))
    profiles = splits.groupby(["model_name", "k", "region"], as_index=False).agg(
        **{metric: (metric, "mean") for metric in PROFILE_METRICS},
        n_nonempty_splits=("n_query_cells", lambda x: int((x > 0).sum())),
        unstable_any_split=("unstable", "any"), n_splits=("split_seed", "size"))
    # No mean over fewer than three partitions is silently presented as the common estimand.
    profiles.loc[profiles.n_nonempty_splits.ne(len(SPLITS)), list(PROFILE_METRICS)] = np.nan
    return runs, splits, profiles


def station_effects(pairs, name, role):
    cells = pairs.groupby(["split_seed", "station", "cell"], as_index=False).agg(
        candidate_error=("candidate_error", "mean"), reference_error=("reference_error", "mean"),
        ecological_novelty=("ecological_novelty", "mean"), upstream_support=("upstream_support", "mean"))
    counts = cells.groupby("split_seed").cell.transform("size")
    cells["weighted_gain"] = (cells.reference_error - cells.candidate_error) / counts / len(SPLITS)
    cells["weighted_reference_error"] = cells.reference_error / counts / len(SPLITS)
    stations = cells.groupby(["split_seed", "station"], as_index=False).agg(
        candidate_mae=("candidate_error", "mean"), reference_mae=("reference_error", "mean"),
        n_query_cells=("cell", "size"), weighted_gain=("weighted_gain", "sum"),
        weighted_reference_error=("weighted_reference_error", "sum"),
        ecological_novelty=("ecological_novelty", "mean"), upstream_support=("upstream_support", "mean"))
    stations["comparison"], stations["comparison_role"] = name, role
    add_directions(stations)
    # Combine repeated station identities after applying the equal-partition weights.
    global_station = stations.groupby("station", as_index=False).agg(
        weighted_gain=("weighted_gain", "sum"), weighted_reference_error=("weighted_reference_error", "sum"),
        n_partitions=("split_seed", "size"), n_split_cell_occurrences=("n_query_cells", "sum"))
    global_station["comparison"] = name
    gain = global_station.weighted_gain.clip(lower=0)
    harm = (-global_station.weighted_gain).clip(lower=0)
    concentration = {"comparison": name, "comparison_role": role,
                     "net_mae_reduction": float(gain.sum() - harm.sum()),
                     "positive_station_gain_mass": float(gain.sum()), "station_harm_mass": float(harm.sum()),
                     "stations_improved": int((gain > 0).sum()), "stations_worsened": int((harm > 0).sum()),
                     "n_unique_stations": len(global_station),
                     "top5_share_of_positive_station_gain": float(gain.nlargest(5).sum() / gain.sum())
                     if gain.sum() > 0 else np.nan,
                     "top5_share_of_station_harm": float(harm.nlargest(5).sum() / harm.sum())
                     if harm.sum() > 0 else np.nan}
    return stations, global_station, concentration


def compare(panel, thresholds, draws):
    effects, directions, station_rows, global_rows, concentrations = [], [], [], [], []
    for name, candidate, ck, reference, rk, role in comparison_definitions():
        pairs = paired_cells(panel, candidate, ck, reference, rk)
        threshold = np.asarray([thresholds[(int(split), int(seed))]
                                for split, seed in zip(pairs.split_seed, pairs.seed, strict=True)])
        tail = pairs.y_true_candidate.to_numpy() >= threshold
        for region, mask in (("overall", np.ones(len(pairs), bool)), ("q90", tail), ("nontail", ~tail)):
            selected = pairs.loc[mask].copy()
            per_seed = selected.groupby(["split_seed", "seed"], as_index=False).agg(
                candidate_mae=("candidate_error", "mean"), reference_mae=("reference_error", "mean"))
            per_seed["comparison"], per_seed["comparison_role"], per_seed["region"] = name, role, region
            add_directions(per_seed)
            directions.append(per_seed)
            split_delta = per_seed.groupby("split_seed").delta_mae.mean()
            counts = selected.drop_duplicates(["split_seed", "cell"]).groupby("split_seed").size()
            identity = {"comparison": name, "comparison_role": role, "candidate": candidate,
                        "candidate_k": ck, "reference": reference, "reference_k": rk, "region": region,
                        "improved_split_seed_pairs": int((per_seed.delta_mae < 0).sum()),
                        "n_split_seed_pairs": len(per_seed), "improved_splits": int((split_delta < 0).sum()),
                        "equal_splits": int((split_delta == 0).sum()),
                        "n_nonempty_splits": len(counts), "unstable_any_split": bool((counts < 20).any())}
            for metric in ("mae", "log_mae", "signed_bias", "log_signed_bias"):
                evaluation = selected.copy()
                if metric != "mae":
                    for side in ("candidate", "reference"):
                        error = evaluation[f"y_pred_{side}"] - evaluation[f"y_true_{side}"]
                        if metric.startswith("log_"):
                            error = (np.log1p(evaluation[f"y_pred_{side}"])
                                     - np.log1p(evaluation[f"y_true_{side}"]))
                        evaluation[f"{side}_error"] = np.abs(error) if metric == "log_mae" else error
                metric_identity = dict(identity)
                if metric != "mae":
                    metric_identity.update(improved_split_seed_pairs=np.nan, improved_splits=np.nan,
                                           equal_splits=np.nan)
                if len(counts) == len(SPLITS):
                    result = joint_station_bootstrap(evaluation, draws=draws)
                    result = {key.replace("_mae", "_value"): value for key, value in result.items()}
                    if metric.endswith("bias"):
                        for key in ("relative_gain_pct", "gain_ci_low_pct", "gain_ci_high_pct"):
                            result[key] = np.nan
                    effects.append({**metric_identity, "metric": metric, "status": "estimated", **result})
                else:
                    effects.append({**metric_identity, "metric": metric, "status": "missing_partition"})
            if region == "overall":
                stations, global_station, concentration = station_effects(pairs, name, role)
                station_rows.append(stations)
                global_rows.append(global_station)
                concentrations.append(concentration)
        # False high-DOC alarms are conditional on actual non-tail cells.
        negatives = pairs.loc[~tail].copy()
        for side in ("candidate", "reference"):
            negatives[f"{side}_error"] = (negatives[f"y_pred_{side}"].to_numpy() >= threshold[~tail]).astype(float)
        if negatives.split_seed.nunique() == len(SPLITS):
            result = joint_station_bootstrap(negatives, draws=draws)
            result = {key.replace("_mae", "_value"): value for key, value in result.items()}
            # Relative rate reductions are undefined for some zero-alarm bootstrap draws.
            for key in ("relative_gain_pct", "gain_ci_low_pct", "gain_ci_high_pct"):
                result[key] = np.nan
            effects.append({"comparison": name, "comparison_role": role, "candidate": candidate,
                            "candidate_k": ck, "reference": reference, "reference_k": rk,
                            "region": "nontail", "metric": "q90_false_positive_rate",
                            "status": "estimated", **result})
    directions = pd.concat(directions, ignore_index=True)
    splits = directions.groupby(["comparison", "comparison_role", "region", "split_seed"], as_index=False).agg(
        candidate_mae=("candidate_mae", "mean"), reference_mae=("reference_mae", "mean"), n_seeds=("seed", "size"))
    add_directions(splits)
    return (pd.DataFrame(effects), directions, splits, pd.concat(station_rows, ignore_index=True),
            pd.concat(global_rows, ignore_index=True), pd.DataFrame(concentrations))


def selection_records(states):
    training, choices = [], []
    for run in states:
        identity = {"split_seed": run["split_seed"], "seed": run["seed"]}
        if set(run["adapters"]) != set(MODELS):
            raise ValueError("Saved adapter models differ from the six specified arms")
        for model, state in run["adapters"].items():
            if set(map(int, state["selection_by_k"])) != set(KS) or state["selection_role"] != "source_validation":
                raise ValueError(f"Invalid saved adapter selection: {model}")
            for k, choice in state["selection_by_k"].items():
                choices.append({**identity, "model_name": model, "k": int(k), **choice,
                                "selection_role": state["selection_role"]})
        for arm, state in run["training"].items():
            trace = state["trace"]
            selected = [row for row in trace if row["epoch"] == state["best_epoch"]]
            if (not trace or trace[0]["epoch"] != 0 or len(selected) != 1
                    or state["protocol"]["selection_role"] != "source_validation"
                    or selected[0]["validation_scale"] != state["selected_scale"]
                    or selected[0]["validation_mae"] != state["validation_metrics"]["validation_mae"]):
                raise ValueError(f"Invalid native residual checkpoint selection: {identity}, {arm}")
            training.append({**identity, "arm": arm, "tail_weight": state["config"]["tail_weight"],
                             "best_epoch": state["best_epoch"], "epochs_run": state["epochs_run"],
                             "selected_scale": state["selected_scale"],
                             "context_fallback": state["selected_scale"] == 0,
                             "initial_validation_mae": trace[0]["validation_mae"],
                             "selected_validation_mae": selected[0]["validation_mae"],
                             **{key: state[key] for key in ("trainable_parameter_count", "n_source_cells",
                                                           "n_source_tail_cells", "n_validation_query",
                                                           "temporal_parameter_distance", "decay_parameter_distance",
                                                           "head_parameter_norm")}})
    return pd.DataFrame(training), pd.DataFrame(choices)


def write_report(out, curves, profiles, comparisons, training, concentration, draws):
    lines = ["# Native-DOC temporal residual: development results", "",
             "Nine completed partition–seed packages (18 neural fits) compare ordinary native-scale MAE",
             "training with doubled weight on",
             "source-training Q90 observations. The environmental context predictor is fixed; the",
             "existing GRU and decay plus a zero-initialized scalar head learn the native-DOC correction.",
             "Each base has constant and frozen-v4-GRU support adapters. All saved arms are reported.", "",
             "These are previously evaluated station partitions 142–144 within the same cohort.",
             "This development analysis does not select or automatically promote a model.",
             "Source-validation chooses epochs, residual scale, and support adaptation; outer query labels",
             "are used here only for evaluation. Target support is retrospective and identical across arms.", "",
             f"Intervals use {draws:,} paired whole-station bootstrap draws with station IDs resampled",
             "jointly across partitions. Individual-seed losses are averaged within each partition,",
             "then partitions receive equal weight. Bias is prediction minus observation; negative bias",
             "means underprediction. Q90 is determined only from each partition's source training labels.", "",
             "## Predictions at K = 0 and K = 5", "",
             "| Saved model | K | MAE | RMSE | R² | Log MAE | Q90 MAE |",
             "|---|---:|---:|---:|---:|---:|---:|"]
    for row in curves[curves.k.isin((0, 5))].itertuples():
        lines.append(f"| {row.model_name} | {row.k} | {row.mae:.4f} | {row.rmse:.3f} | {row.r2:.3f} | "
                     f"{row.log_mae:.4f} | {row.q90_mae:.3f} |")
    lines += ["", "At K = 0, the two support adapters for a given base are exactly identical.",
              "Their repeated rows describe the same prediction, not independent evidence.", ""]
    overall = comparisons[comparisons.region.eq("overall") & comparisons.metric.eq("mae")]
    for role, title in (("matched_environmental_control", "Correction beyond the environmental predictor"),
                        ("training_objective_control", "Tail-weighted versus ordinary training"),
                        ("prior_fusion_reference", "Comparison with the existing fusion references")):
        lines += [f"## {title}", "", "| Comparison | ΔMAE [95% CI] | MAE reduction [95% CI] | Improved partitions; fits |",
                  "|---|---:|---:|---:|"]
        for row in overall[overall.comparison_role.eq(role)].itertuples():
            lines.append(f"| {row.comparison} | {row.delta_value:.4f} [{row.delta_ci_low:.4f}, {row.delta_ci_high:.4f}] | "
                         f"{row.relative_gain_pct:.2f}% [{row.gain_ci_low_pct:.2f}, {row.gain_ci_high_pct:.2f}] | "
                         f"{int(row.improved_splits)}/3; {int(row.improved_split_seed_pairs)}/9 |")
        lines.append("")
    lines += ["## High-DOC improvement and ordinary-concentration cost", "",
              "| Saved model | K | Q90 MAE | Q90 signed bias | Non-tail MAE | Non-tail log MAE | False Q90 rate |",
              "|---|---:|---:|---:|---:|---:|---:|"]
    for k in (0, 5):
        # K0 constant and shape arms are identical; display each base once here.
        models = tuple(name for name in MODELS + REFERENCES if k == 5 or "constant" not in name)
        for model in models:
            selected = profiles[profiles.model_name.eq(model) & profiles.k.eq(k)].set_index("region")
            tail, rest = selected.loc["q90"], selected.loc["nontail"]
            lines.append(f"| {model} | {k} | {tail.mae:.4f} | {tail.signed_bias:.4f} | {rest.mae:.4f} | "
                         f"{rest.log_mae:.4f} | {100 * rest.q90_false_positive_rate:.2f}% |")
    lines += ["", "The false Q90 rate is P(prediction ≥ source Q90 | observation < source Q90).",
              "It is not the false-discovery fraction among predicted high values. Overall, Q90 and",
              "non-tail raw/log MAE, signed-bias intervals and absolute false-positive-rate differences are retained",
              "in `comparisons.csv`. All K curves and signed biases are retained in `error_profiles.csv`.", "",
              "## Where gains and losses occur", "",
              "| Native correction vs matched context | Improved stations | Worsened stations | Top-five share of positive gain | Top-five share of harm |",
              "|---|---:|---:|---:|---:|"]
    for row in concentration[concentration.comparison_role.eq("matched_environmental_control")
                             & (concentration.comparison.str.endswith("k5")
                                | concentration.comparison.str.contains("gru_tuned_anchor_k0"))].itertuples():
        lines.append(f"| {row.comparison} | {row.stations_improved} | {row.stations_worsened} | "
                     f"{100 * row.top5_share_of_positive_station_gain:.1f}% | {100 * row.top5_share_of_station_harm:.1f}% |")
    lines += ["", "Station contributions use the same equal-partition weighting as overall MAE.",
              "Repeated station identities are combined. Gain and harm shares have separate positive",
              "denominators, avoiding unstable percentages of a nearly zero net improvement.", "",
              "## What source-validation selected", "",
              "| Training objective | Active residual fits | Context fallbacks | Selected scale counts | Selected epochs |",
              "|---|---:|---:|---|---|"]
    for arm, group in training.groupby("arm"):
        scales = ", ".join(f"{scale:g}: {count}" for scale, count in group.selected_scale.value_counts().sort_index().items())
        epochs = ", ".join(str(value) for value in group.best_epoch)
        lines.append(f"| {arm} | {int((~group.context_fallback).sum())}/9 | {int(group.context_fallback.sum())}/9 | {scales} | {epochs} |")
    lines += ["", "A selected residual scale of zero is the environmental context fallback and is not",
              "evidence of a neural correction. Training choices and every support-adapter selection",
              "are available in `training_choices.csv` and `adapter_choices.csv`.", "",
              "## Interpretation", ""]
    for arm in ARMS:
        name = f"{arm}_vs_context_gru_tuned_anchor_k0"
        row = overall[overall.comparison.eq(name)].iloc[0]
        tail = comparisons[comparisons.comparison.eq(name) & comparisons.region.eq("q90")
                           & comparisons.metric.eq("mae")].iloc[0]
        rest = comparisons[comparisons.comparison.eq(name) & comparisons.region.eq("nontail")
                           & comparisons.metric.eq("mae")].iloc[0]
        lines.append(f"- **{arm}, no target support (K = 0):** MAE changes from "
                     f"{row.reference_value:.4f} to {row.candidate_value:.4f} mg/L "
                     f"({row.relative_gain_pct:.2f}% reduction, 95% CI "
                     f"[{row.gain_ci_low_pct:.2f}, {row.gain_ci_high_pct:.2f}]%). "
                     f"{int(row.improved_splits)}/3 partitions and {int(row.improved_split_seed_pairs)}/9 "
                     "partition–seed pairs improve. "
                     f"Q90 MAE changes by {tail.delta_value:+.4f} mg/L (95% CI "
                     f"[{tail.delta_ci_low:+.4f}, {tail.delta_ci_high:+.4f}]), equivalent to "
                     f"{tail.relative_gain_pct:+.2f}% reduction (95% CI "
                     f"[{tail.gain_ci_low_pct:+.2f}, {tail.gain_ci_high_pct:+.2f}]%). "
                     f"Non-tail MAE changes by {rest.delta_value:+.4f} mg/L (95% CI "
                     f"[{rest.delta_ci_low:+.4f}, {rest.delta_ci_high:+.4f}]).")
    for arm in ARMS:
        name = f"{arm}_vs_context_gru_tuned_anchor_k5"
        row = overall[overall.comparison.eq(name)].iloc[0]
        tail = comparisons[comparisons.comparison.eq(name) & comparisons.region.eq("q90")
                           & comparisons.metric.eq("mae")].iloc[0]
        rest = comparisons[comparisons.comparison.eq(name) & comparisons.region.eq("nontail")
                           & comparisons.metric.eq("mae")].iloc[0]
        wording = "lower" if row.delta_value < 0 else "higher" if row.delta_value > 0 else "unchanged"
        strength = ("the paired interval excludes zero" if row.delta_ci_low > 0 or row.delta_ci_high < 0
                    else "the paired interval includes zero")
        lines.append(f"- **{arm}, K = 5 with frozen GRU support adapter:** overall MAE is {wording} "
                     f"({row.relative_gain_pct:+.2f}% reduction; {strength}). Q90 and non-tail "
                     f"MAE changes are {tail.delta_value:+.4f} and {rest.delta_value:+.4f} mg/L, respectively.")
    for k in (0, 5):
        contrast = comparisons[comparisons.comparison.eq(f"tail_vs_ordinary_gru_tuned_anchor_k{k}")]
        tail = contrast[contrast.region.eq("q90") & contrast.metric.eq("mae")].iloc[0]
        rest = contrast[contrast.region.eq("nontail") & contrast.metric.eq("mae")].iloc[0]
        fpr = contrast[contrast.metric.eq("q90_false_positive_rate")].iloc[0]
        lines.append(f"- **Tail weighting versus ordinary training, K = {k}:** Q90 MAE changes by "
                     f"{tail.delta_value:+.4f} mg/L (95% CI [{tail.delta_ci_low:+.4f}, "
                     f"{tail.delta_ci_high:+.4f}]); non-tail MAE changes by {rest.delta_value:+.4f} mg/L "
                     f"(95% CI [{rest.delta_ci_low:+.4f}, {rest.delta_ci_high:+.4f}]). "
                     f"The false Q90 rate changes by {100 * fpr.delta_value:+.3f} percentage points "
                     f"(95% CI [{100 * fpr.delta_ci_low:+.3f}, {100 * fpr.delta_ci_high:+.3f}]).")
    lines += ["", "The clearest gain occurs before target-station support is available. With five",
              "support observations, the native correction adds little beyond the existing station",
              "adapter. Tail weighting changes the allocation of error between high and ordinary DOC",
              "rather than establishing an additional overall-MAE advantage over ordinary training."]
    lines += ["", "The named comparisons evaluate both training objectives and both support adapters;",
              "they do not define a winner from this test panel. The existing v4 GRU and tree fusion",
              "references remain explicit performance comparators. Error concentration is descriptive",
              "and does not identify an irreducible error floor or a physical mechanism.", ""]
    (out / "findings.md").write_text("\n".join(lines))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--bootstrap-draws", type=int, default=5000)
    args = parser.parse_args()
    if args.bootstrap_draws < 1:
        raise ValueError("Bootstrap draws must be positive")
    panel, thresholds, sources, states = load_panel(args.root)
    runs, splits, curves = summarize_metrics(panel, thresholds)
    profile_runs, profile_splits, profiles = error_profiles(panel, thresholds)
    comparisons, directions, partitions, stations, global_stations, concentration = compare(
        panel, thresholds, args.bootstrap_draws)
    training, choices = selection_records(states)
    out = args.root / "analysis"
    out.mkdir(parents=True, exist_ok=True)
    for name, frame in (("metrics_by_run", runs), ("metrics_by_partition", splits), ("k_curves", curves),
                        ("error_profiles_by_run", profile_runs), ("error_profiles_by_partition", profile_splits),
                        ("error_profiles", profiles), ("comparisons", comparisons),
                        ("directions_by_seed", directions), ("directions_by_partition", partitions),
                        ("station_responses", stations), ("global_station_contributions", global_stations),
                        ("gain_loss_concentration", concentration), ("training_choices", training),
                        ("adapter_choices", choices)):
        frame.to_csv(out / f"{name}.csv", index=False)
    write_report(out, curves, profiles, comparisons, training, concentration, args.bootstrap_draws)
    sources.extend({"path": str(path), "sha256": sha256(path)} for path in (
        Path(__file__), Path("scripts/analyze_unified_doc_spatial.py"),
        Path("scripts/analyze_unified_doc_spatial_v2.py"), Path("scripts/analyze_unified_doc_spatial_v3.py")))
    unique_sources = {row["path"]: row for row in sources}
    (out / "sources.json").write_text(json.dumps(list(unique_sources.values()), indent=2) + "\n")
    status = {"complete": True, "n_runs": len(states), "n_models": len(MODELS + REFERENCES),
              "bootstrap_draws": args.bootstrap_draws, "bootstrap_unit": "joint whole-station identity",
              "study_role": "development on previously evaluated station partitions", "model_selected": False,
              "sidecars_verified": True, "context_predictions_identical_to_v4": True,
              "analysis_inputs": len(unique_sources)}
    (out / "status.json").write_text(json.dumps(status, indent=2) + "\n")
    print(json.dumps({**status, "output": str(out)}, indent=2))


if __name__ == "__main__":
    main()
