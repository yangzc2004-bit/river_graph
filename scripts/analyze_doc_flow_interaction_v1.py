"""Evaluate recurrent-state by flow interactions in the existing DOC head.

All completed arms and fixed historical references are retained. This analysis
uses previously evaluated station holdouts and performs no model selection.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from analyze_doc_tail_residual_v1 import (
    error_profiles,
    read_predictions,
    station_effects,
)
from analyze_unified_doc_spatial import joint_station_bootstrap, paired_cells, sha256
from analyze_unified_doc_spatial_v2 import KS, SEEDS, SPLITS, validate_panel
from analyze_unified_doc_spatial_v3 import (
    add_directions,
    read_bound_json,
    summarize_metrics,
)

ROOT = Path("experiments/phase4_transfer/doc_flow_interaction_v1")
ARMS = ("interaction_frozen", "interaction_tuned")
SHAPES = ("constant", "gru_tuned_anchor")
MODELS = tuple(f"{base}_{shape}" for base in ("context", *ARMS) for shape in SHAPES)
PRIOR_ADDITIVE = tuple(f"prior_flow_values_{shape}" for shape in SHAPES)
PRIOR_FUSION = ("v4_fusion_gru_tuned_anchor", "v4_fusion_tree_episodic")
ALL_MODELS = MODELS + PRIOR_ADDITIVE + PRIOR_FUSION


def comparison_definitions():
    rows = []
    for shape in SHAPES:
        for k in (0, 5):
            rows.append((f"tuned_vs_frozen_{shape}_k{k}", f"interaction_tuned_{shape}", k,
                         f"interaction_frozen_{shape}", k, "recurrent_update_in_interaction"))
            for arm in ARMS:
                rows.append((f"{arm}_vs_additive_{shape}_k{k}", f"{arm}_{shape}", k,
                             f"prior_flow_values_{shape}", k, "prior_additive_readout"))
                rows.append((f"{arm}_vs_context_{shape}_k{k}", f"{arm}_{shape}", k,
                             f"context_{shape}", k, "environmental_reference"))
        for arm in ARMS:
            for reference in PRIOR_FUSION:
                rows.append((f"{arm}_{shape}_vs_{reference}_k5", f"{arm}_{shape}", 5,
                             reference, 5, "prior_fusion_reference"))
    return rows


def check_source_identity(config, reference, keys):
    for key in keys:
        if config[key] != reference[key]:
            raise ValueError(f"Current/reference identity differs in {key}")


def load_panel(root):
    frames, thresholds, sources, states = [], {}, [], []
    for split in SPLITS:
        for seed in SEEDS:
            run = root / "runs" / f"split{split}_seed{seed}"
            frame, config, completion = read_predictions(run, sources)
            if (config["split_seed"], config["seed"]) != (split, seed):
                raise ValueError(f"Directory identity differs: {run}")
            if (set(config["models"]) != set(MODELS) or tuple(config["k_values"]) != KS
                    or config["inference_roles"] != ["train"]
                    or config["extra_dim"] != 10 or config["tail_weight"] != 2
                    or config["epochs"] != 30 or config["patience"] != 5
                    or config["interaction_indices"] != [0, 2, 4]
                    or config["train_memory_by_arm"] != {"interaction_frozen": False, "interaction_tuned": True}):
                raise ValueError(f"Unexpected complete-production configuration: {run}")
            threshold = float(config["q90_threshold_train"])
            if not np.isfinite(threshold) or threshold < 0:
                raise ValueError(f"Invalid source-training Q90: {run}")
            thresholds[(split, seed)] = threshold
            flow = read_bound_json(run, "flow_features.json", completion, sources)
            if (flow != config["flow_definition"] or len(flow["feature_names"]) != 10
                    or flow["value_feature_indices"] != [0, 2, 4]
                    or flow["freshness_feature_indices"] != [1, 3, 5, 6, 7, 8, 9]):
                raise ValueError(f"Changed flow readout definition: {run}")
            references = {}
            for prefix in ("prior", "basis"):
                old_run = Path(config[f"{prefix}_run"])
                if sha256(old_run / "complete.json") != config[f"{prefix}_completion_hash"]:
                    raise ValueError(f"Reference package changed: {old_run}")
                old, old_config, _ = read_predictions(old_run, sources)
                check_source_identity(config, old_config,
                                      ("split_seed", "seed", "dataset_hash", "mask_hash", "q90_threshold_train"))
                references[prefix] = old
            for shape in SHAPES:
                name = f"context_{shape}"
                current = frame[frame.model_name.eq(name)].sort_values(["k", "cell"])
                for old in references.values():
                    previous = old[old.model_name.eq(name)].sort_values(["k", "cell"])
                    np.testing.assert_array_equal(current.y_pred, previous.y_pred)
            old_additive = references["prior"]
            old_additive = old_additive[old_additive.model_name.isin(
                tuple(f"flow_values_{shape}" for shape in SHAPES))].copy()
            old_additive["model_name"] = "prior_" + old_additive.model_name
            old_fusion = references["basis"]
            old_fusion = old_fusion[old_fusion.model_name.isin(tuple(name.removeprefix("v4_")
                                                                   for name in PRIOR_FUSION))].copy()
            old_fusion["model_name"] = "v4_" + old_fusion.model_name
            frames.extend([frame, old_additive, old_fusion])
            states.append({"split_seed": split, "seed": seed, "config": config,
                           "adapters": read_bound_json(run, "adapters.json", completion, sources),
                           "training": {arm: read_bound_json(run, f"{arm}.json", completion, sources)
                                        for arm in ARMS}})
        if len({thresholds[(split, seed)] for seed in SEEDS}) != 1:
            raise ValueError(f"Q90 threshold changes across seeds: {split}")
    panel = pd.concat(frames, ignore_index=True)
    validate_panel(panel, expected_models=ALL_MODELS)
    for _, group in panel[panel.k.eq(0)].groupby(["split_seed", "seed"]):
        zero = group.pivot(index="cell", columns="model_name", values="y_pred")
        for base in ("context", *ARMS, "prior_flow_values"):
            np.testing.assert_array_equal(zero[f"{base}_constant"], zero[f"{base}_gru_tuned_anchor"])
    return panel, thresholds, sources, states


def bootstrap_metric(pairs, metric, draws):
    evaluation = pairs.copy()
    if metric != "mae":
        for side in ("candidate", "reference"):
            error = evaluation[f"y_pred_{side}"] - evaluation[f"y_true_{side}"]
            if metric.startswith("log_"):
                error = np.log1p(evaluation[f"y_pred_{side}"]) - np.log1p(evaluation[f"y_true_{side}"])
            evaluation[f"{side}_error"] = np.abs(error) if metric == "log_mae" else error
    result = joint_station_bootstrap(evaluation, draws=draws)
    result = {key.replace("_mae", "_value"): value for key, value in result.items()}
    if metric.endswith("bias"):
        for key in ("relative_gain_pct", "gain_ci_low_pct", "gain_ci_high_pct"):
            result[key] = np.nan
    return result


def compare(panel, thresholds, draws):
    effects, directions, station_rows, global_rows, concentrations = [], [], [], [], []
    for name, candidate, ck, reference, rk, role in comparison_definitions():
        pairs = paired_cells(panel, candidate, ck, reference, rk)
        threshold = np.asarray([thresholds[(int(split), int(seed))]
                                for split, seed in zip(pairs.split_seed, pairs.seed, strict=True)])
        tail = pairs.y_true_candidate.to_numpy() >= threshold
        identity = {"comparison": name, "comparison_role": role, "candidate": candidate,
                    "candidate_k": ck, "reference": reference, "reference_k": rk}
        for region, mask in (("overall", np.ones(len(pairs), bool)), ("q90", tail), ("nontail", ~tail)):
            selected = pairs.loc[mask].copy()
            per_seed = selected.groupby(["split_seed", "seed"], as_index=False).agg(
                candidate_mae=("candidate_error", "mean"), reference_mae=("reference_error", "mean"))
            per_seed["comparison"], per_seed["comparison_role"], per_seed["region"] = name, role, region
            add_directions(per_seed)
            directions.append(per_seed)
            split_delta = per_seed.groupby("split_seed").delta_mae.mean()
            counts = selected.drop_duplicates(["split_seed", "cell"]).groupby("split_seed").size()
            regional = {**identity, "region": region, "n_nonempty_splits": len(counts),
                        "unstable_any_split": len(counts) < len(SPLITS) or bool((counts < 20).any())}
            for metric in ("mae", "log_mae", "signed_bias", "log_signed_bias"):
                directional = {
                    "improved_split_seed_pairs": int((per_seed.delta_mae < 0).sum()) if metric == "mae" else np.nan,
                    "improved_splits": int((split_delta < 0).sum()) if metric == "mae" else np.nan,
                    "equal_splits": int((split_delta == 0).sum()) if metric == "mae" else np.nan,
                    "n_split_seed_pairs": len(per_seed)}
                if len(counts) == len(SPLITS):
                    effects.append({**regional, **directional, "metric": metric, "status": "estimated",
                                    **bootstrap_metric(selected, metric, draws)})
                else:
                    effects.append({**regional, **directional, "metric": metric, "status": "missing_partition"})
        stations, global_station, concentration = station_effects(pairs, name, role)
        station_rows.append(stations)
        global_rows.append(global_station)
        concentrations.append(concentration)
        negatives = pairs.loc[~tail].copy()
        for side in ("candidate", "reference"):
            negatives[f"{side}_error"] = (negatives[f"y_pred_{side}"].to_numpy() >= threshold[~tail]).astype(float)
        if negatives.split_seed.nunique() == len(SPLITS):
            result = joint_station_bootstrap(negatives, draws=draws)
            result = {key.replace("_mae", "_value"): value for key, value in result.items()}
            for key in ("relative_gain_pct", "gain_ci_low_pct", "gain_ci_high_pct"):
                result[key] = np.nan
            effects.append({**identity, "region": "nontail", "metric": "q90_false_positive_rate",
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
            raise ValueError("Saved adapters differ from the six specified arms")
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
                    or state["config"]["tail_weight"] != 2 or state["config"]["extra_dim"] != 10
                    or state["config"]["interaction_indices"] != [0, 2, 4]
                    or state["config"].get("train_memory", True) != (arm == "interaction_tuned")
                    or state["protocol"]["selection_role"] != "source_validation"
                    or selected[0]["validation_scale"] != state["selected_scale"]
                    or selected[0]["validation_mae"] != state["validation_metrics"]["validation_mae"]):
                raise ValueError(f"Invalid selected flow residual checkpoint: {identity}, {arm}")
            head_parameters = state["hidden_size"] * 4 + 10 + 1
            if arm == "interaction_frozen" and (state["trainable_parameter_count"] != head_parameters
                    or state["temporal_parameter_distance"] != 0 or state["decay_parameter_distance"] != 0):
                raise ValueError(f"Frozen-memory arm changed recurrence or its parameter budget: {identity}")
            training.append({**identity, "arm": arm, "tail_weight": 2,
                             "extra_dim": 10, "interaction_dim": state["hidden_size"] * 3,
                             "head_parameters": head_parameters, "train_memory": arm == "interaction_tuned",
                             "best_epoch": state["best_epoch"], "epochs_run": state["epochs_run"],
                             "selected_scale": state["selected_scale"], "context_fallback": state["selected_scale"] == 0,
                             "initial_validation_mae": trace[0]["validation_mae"],
                             "selected_validation_mae": selected[0]["validation_mae"],
                             **{key: state[key] for key in ("trainable_parameter_count", "n_source_cells",
                                                           "n_source_tail_cells", "n_validation_query",
                                                           "temporal_parameter_distance", "decay_parameter_distance",
                                                           "head_parameter_norm")}})
    return pd.DataFrame(training), pd.DataFrame(choices)


def write_report(out, curves, profiles, comparisons, concentration, training, draws):
    overall = comparisons[comparisons.region.eq("overall") & comparisons.metric.eq("mae")]
    lines = ["# State-dependent discharge corrections for DOC: development results", "",
             "Nine partition–seed packages contain 18 neural fits. Both new arms use tail weight 2,",
             "a 30-epoch ceiling, patience 5, and the original GRU/decay initialization. The same ten",
             "causal discharge inputs enter a zero-initialized native-DOC head. An outer product of",
             "the 64-dimensional recurrent state and three numerical anomaly/change features adds",
             "192 interaction weights: readout = [h, f, flatten(h outer f_numeric)].", "",
             "`interaction_tuned` trains recurrence, decay and the expanded head; `interaction_frozen`",
             "trains the identical head while retaining original recurrence and decay exactly.",
             "The primary architecture contrast is tuned interaction versus the saved additive",
             "full-flow predictor, which also trained recurrence under the same objective and budget.",
             "Tuned versus frozen interaction tests recurrent updating within the expanded readout.",
             "Frozen interaction versus additive changes both readout and trainability and is a",
             "performance comparison. Trainable parameter counts therefore differ between new arms.", "",
             "The context forest, spatial encoder and frozen-v4 support basis remain unchanged.",
             "The recurrent state is a learned representation; its interactions describe predictive",
             "conditioning rather than graph transport, physical coefficients or causal effects.", "",
             "Source-validation chooses checkpoint, scale and support-adapter parameters. Partitions",
             "142–144 were previously evaluated, making this a development comparison. All saved",
             "models remain visible and no winner or promotion is selected from this analysis.",
             "K-shot support is retrospective; the complete adapted model is not prospective forecasting.", "",
             f"Intervals use {draws:,} paired whole-station bootstrap draws, with shared station identities",
             "resampled jointly across partitions. Seed losses are averaged within each partition and",
             "partitions receive equal weight. Positive relative reduction and negative ΔMAE favor",
             "the candidate. Q90 uses source-training labels; signed bias is prediction minus observation.", "",
             "## Overall performance", "",
             "| Saved model | K | MAE | RMSE | R² | Log MAE | Q90 MAE |",
             "|---|---:|---:|---:|---:|---:|---:|"]
    for row in curves[curves.k.isin((0, 5))].itertuples():
        lines.append(f"| {row.model_name} | {row.k} | {row.mae:.4f} | {row.rmse:.3f} | {row.r2:.3f} | "
                     f"{row.log_mae:.4f} | {row.q90_mae:.4f} |")
    lines += ["", "At K0 the constant and shape adapters for each base produce exactly identical",
              "predictions; duplicated K0 table rows are not independent evidence.", ""]
    for role, title in (("prior_additive_readout", "Interaction readout versus the existing additive model"),
                        ("recurrent_update_in_interaction", "Recurrent updating within the interaction model"),
                        ("environmental_reference", "Comparison with the fixed environmental predictor"),
                        ("prior_fusion_reference", "Existing K5 fusion references")):
        lines += [f"## {title}", "", "| Comparison | ΔMAE [95% CI] | Reduction [95% CI] | Improved partitions; fits |",
                  "|---|---:|---:|---:|"]
        for row in overall[overall.comparison_role.eq(role)].itertuples():
            lines.append(f"| {row.comparison} | {row.delta_value:+.4f} [{row.delta_ci_low:+.4f}, {row.delta_ci_high:+.4f}] | "
                         f"{row.relative_gain_pct:+.2f}% [{row.gain_ci_low_pct:+.2f}, {row.gain_ci_high_pct:+.2f}] | "
                         f"{int(row.improved_splits)}/3; {int(row.improved_split_seed_pairs)}/9 |")
        lines.append("")
    lines += ["## Tail, ordinary concentrations and false alarms", "",
              "| Saved model | K | Q90 MAE | Q90 bias | Non-tail MAE | Non-tail log MAE | False Q90 rate |",
              "|---|---:|---:|---:|---:|---:|---:|"]
    for k in (0, 5):
        for model in ALL_MODELS:
            if k == 0 and model.endswith("constant"):
                continue
            selected = profiles[profiles.model_name.eq(model) & profiles.k.eq(k)].set_index("region")
            tail, rest = selected.loc["q90"], selected.loc["nontail"]
            lines.append(f"| {model} | {k} | {tail.mae:.4f} | {tail.signed_bias:.4f} | {rest.mae:.4f} | "
                         f"{rest.log_mae:.4f} | {100 * rest.q90_false_positive_rate:.2f}% |")
    lines += ["", "False Q90 rate = P(prediction ≥ source Q90 | observation < source Q90).",
              "Raw/log MAE, signed-bias intervals and false-alarm rate differences appear in",
              "`comparisons.csv`; regional sample counts and all K curves are saved separately.", "",
              "## Station gain and loss concentration", "",
              "| Contrast, GRU support basis | Stations improved / worsened | Top-five share of positive gain | Top-five share of harm |",
              "|---|---:|---:|---:|"]
    selection = concentration.comparison.str.contains("gru_tuned_anchor") & concentration.comparison_role.isin(
        ("recurrent_update_in_interaction", "prior_additive_readout"))
    for row in concentration[selection].itertuples():
        lines.append(f"| {row.comparison} | {row.stations_improved} / {row.stations_worsened} | "
                     f"{100 * row.top5_share_of_positive_station_gain:.1f}% | {100 * row.top5_share_of_station_harm:.1f}% |")
    lines += ["", "Station contributions retain equal partition weighting and combine repeated station",
              "identities. Positive gain and harm have separate denominators; they do not imply",
              "a physical process or an irreducible error floor.", "",
              "## Source-validation selections", "",
              "| Arm | Trainable parameters | Active fits | Context fallbacks | Selected scales (count) | Selected epochs |",
              "|---|---:|---:|---:|---|---|"]
    for arm, group in training.groupby("arm"):
        scales = ", ".join(f"{scale:g}: {count}" for scale, count in group.selected_scale.value_counts().sort_index().items())
        epochs = ", ".join(str(value) for value in group.best_epoch)
        parameter_counts = ", ".join(str(value) for value in sorted(group.trainable_parameter_count.unique()))
        lines.append(f"| {arm} | {parameter_counts} | {int((~group.context_fallback).sum())}/9 | "
                     f"{int(group.context_fallback.sum())}/9 | {scales} | {epochs} |")
    lines += ["", "Scale zero is an exact context fallback, not a learned neural gain.", "",
              "## Interpretation of the fixed contrasts", ""]
    for k in (0, 5):
        for prefix in ("tuned_vs_frozen", "interaction_tuned_vs_additive", "interaction_frozen_vs_additive"):
            name = f"{prefix}_gru_tuned_anchor_k{k}"
            contrast = comparisons[comparisons.comparison.eq(name)]
            row = contrast[contrast.region.eq("overall") & contrast.metric.eq("mae")].iloc[0]
            tail = contrast[contrast.region.eq("q90") & contrast.metric.eq("mae")].iloc[0]
            rest = contrast[contrast.region.eq("nontail") & contrast.metric.eq("mae")].iloc[0]
            false = contrast[contrast.metric.eq("q90_false_positive_rate")].iloc[0]
            lines.append(f"- **{name}:** overall MAE reduction {row.relative_gain_pct:+.2f}% "
                         f"(95% CI [{row.gain_ci_low_pct:+.2f}, {row.gain_ci_high_pct:+.2f}]); "
                         f"{int(row.improved_splits)}/3 partitions, {int(row.improved_split_seed_pairs)}/9 fits improve. "
                         f"Q90 ΔMAE {tail.delta_value:+.4f} mg/L (95% CI [{tail.delta_ci_low:+.4f}, "
                         f"{tail.delta_ci_high:+.4f}]); Q90 improves in {int(tail.improved_splits)}/3 "
                         f"partitions and {int(tail.improved_split_seed_pairs)}/9 fits. "
                         f"Non-tail ΔMAE {rest.delta_value:+.4f} mg/L "
                         f"(95% CI [{rest.delta_ci_low:+.4f}, {rest.delta_ci_high:+.4f}]). "
                         f"False Q90 rate changes by {100 * false.delta_value:+.3f} percentage points "
                         f"(95% CI [{100 * false.delta_ci_low:+.3f}, {100 * false.delta_ci_high:+.3f}]).")
    lines += ["", "All comparisons are descriptive development evidence from this fixed candidate set.",
              "Tuned interaction versus additive tests the added state-dependent readout; tuned",
              "versus frozen interaction tests recurrent updating. Read tail and non-tail changes",
              "alongside overall error before interpreting a concentration-dependent benefit.", ""]
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
    write_report(out, curves, profiles, comparisons, concentration, training, args.bootstrap_draws)
    sources.extend({"path": str(path), "sha256": sha256(path)} for path in (
        Path(__file__), Path("scripts/analyze_doc_tail_residual_v1.py"),
        Path("scripts/analyze_unified_doc_spatial.py"), Path("scripts/analyze_unified_doc_spatial_v2.py"),
        Path("scripts/analyze_unified_doc_spatial_v3.py")))
    unique = {row["path"]: row for row in sources}
    (out / "sources.json").write_text(json.dumps(list(unique.values()), indent=2) + "\n")
    status = {"complete": True, "n_runs": len(states), "n_neural_fits": len(training),
              "n_models": len(ALL_MODELS), "n_comparisons": len(comparison_definitions()),
              "bootstrap_draws": args.bootstrap_draws, "bootstrap_unit": "joint whole-station identity",
              "study_role": "development on previously evaluated station partitions", "model_selected": False,
              "sidecars_verified": True, "context_predictions_identical_to_references": True,
              "interaction_dim": 192, "frozen_memory_unchanged": True,
              "same_head_dimension": True, "trainable_parameter_counts_differ": True,
              "analysis_inputs": len(unique)}
    (out / "status.json").write_text(json.dumps(status, indent=2) + "\n")
    print(json.dumps({**status, "output": str(out)}, indent=2))


if __name__ == "__main__":
    main()
