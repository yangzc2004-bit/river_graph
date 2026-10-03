"""Evaluate source-station ecological residual transfer on frozen DOC experts.

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

ROOT = Path("experiments/phase4_transfer/doc_ecological_transfer_v1")
ARMS = ("global_bias", "ecological_bias", "global_affine", "ecological_affine")
SHAPES = ("constant", "gru_tuned_anchor")
MODELS = tuple(f"{base}_{shape}" for base in ("interaction", *ARMS) for shape in SHAPES)


def comparison_definitions():
    rows = []
    for shape in SHAPES:
        for k in (0, 5):
            for form in ("bias", "affine"):
                rows.append((f"ecological_vs_global_{form}_{shape}_k{k}", f"ecological_{form}_{shape}", k,
                             f"global_{form}_{shape}", k, "ecological_donor_information"))
            for scope in ("global", "ecological"):
                rows.append((f"{scope}_affine_vs_bias_{shape}_k{k}", f"{scope}_affine_{shape}", k,
                             f"{scope}_bias_{shape}", k, "concentration_conditioning"))
            for arm in ARMS:
                rows.append((f"{arm}_vs_interaction_{shape}_k{k}", f"{arm}_{shape}", k,
                             f"interaction_{shape}", k, "existing_interaction_reference"))
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
                    or config["ecological_columns"] != list(range(4, 13))
                    or config["neighbor_grid"] != [20, 40, 80]
                    or config["memory_ridge_grid"] != [.1, 1]
                    or config["gamma_grid"] != [0, .25, .5, 1]):
                raise ValueError(f"Unexpected complete-production configuration: {run}")
            threshold = float(config["q90_threshold_train"])
            if not np.isfinite(threshold) or threshold < 0:
                raise ValueError(f"Invalid source-training Q90: {run}")
            thresholds[(split, seed)] = threshold
            prior = Path(config["prior_run"])
            if sha256(prior / "complete.json") != config["prior_completion_hash"]:
                raise ValueError(f"Reference package changed: {prior}")
            old, old_config, _ = read_predictions(prior, sources)
            check_source_identity(config, old_config,
                                  ("split_seed", "seed", "dataset_hash", "mask_hash", "q90_threshold_train"))
            for shape in SHAPES:
                current = frame[frame.model_name.eq(f"interaction_{shape}")].sort_values(["k", "cell"])
                previous = old[old.model_name.eq(f"interaction_tuned_{shape}")].sort_values(["k", "cell"])
                np.testing.assert_array_equal(current.y_pred, previous.y_pred)
            memories = {arm: read_bound_json(run, f"{arm}.json", completion, sources) for arm in ARMS}
            for arm, state in memories.items():
                if state["selected"]["gamma"] == 0:
                    for shape in SHAPES:
                        current = frame[frame.model_name.eq(f"{arm}_{shape}")].sort_values(["k", "cell"])
                        baseline = frame[frame.model_name.eq(f"interaction_{shape}")].sort_values(["k", "cell"])
                        np.testing.assert_array_equal(current.y_pred, baseline.y_pred)
            frames.append(frame)
            states.append({"split_seed": split, "seed": seed, "config": config,
                           "adapters": read_bound_json(run, "adapters.json", completion, sources),
                           "memories": memories})
        if len({thresholds[(split, seed)] for seed in SEEDS}) != 1:
            raise ValueError(f"Q90 threshold changes across seeds: {split}")
    panel = pd.concat(frames, ignore_index=True)
    validate_panel(panel, expected_models=MODELS)
    for _, group in panel[panel.k.eq(0)].groupby(["split_seed", "seed"]):
        zero = group.pivot(index="cell", columns="model_name", values="y_pred")
        for base in ("interaction", *ARMS):
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


def selection_records(states, panel):
    memories, choices, donors = [], [], []
    for run in states:
        identity = {"split_seed": run["split_seed"], "seed": run["seed"]}
        if set(run["adapters"]) != set(MODELS):
            raise ValueError("Saved adapters differ from the ten specified models")
        for model, state in run["adapters"].items():
            if set(map(int, state["selection_by_k"])) != set(KS) or state["selection_role"] != "source_validation":
                raise ValueError(f"Invalid saved adapter selection: {model}")
            for k, choice in state["selection_by_k"].items():
                choices.append({**identity, "model_name": model, "k": int(k), **choice,
                                "selection_role": state["selection_role"]})
        current = panel[panel.split_seed.eq(run["split_seed"]) & panel.seed.eq(run["seed"]) & panel.k.eq(0)]
        baseline = current[current.model_name.eq("interaction_constant")].sort_values("cell")
        for mode, state in run["memories"].items():
            selected = state["selected"]
            if state["mode"] != mode or state["selection_role"] != "source_validation":
                raise ValueError("Saved memory identity differs")
            expected = min(state["validation_trace"], key=lambda row: (
                row["validation_mae"], row["gamma"] != 0, -row["ridge"], -(row["k"] or 0), row["gamma"]))
            if selected != expected:
                raise ValueError("Saved memory does not match its recorded validation selection")
            coef = np.asarray(state["coefficients"], dtype=float)
            norm = state["normalization"]
            if (coef.shape != (state["n_nodes"], 2) or not np.isfinite(coef).all()
                    or norm["residual_scale"] <= 0 or norm["log_context_sd"] <= 0
                    or len(state["station_diagnostics"]) != state["n_nodes"]):
                raise ValueError("Invalid memory coefficient/diagnostic dimensions")
            if mode.endswith("bias") and (coef[:, 1] != 0).any():
                raise ValueError("Bias memory must have exactly zero slope")
            gamma = float(selected["gamma"])
            memories.append({**identity, "mode": mode, "neighbors": selected["k"],
                             "ridge": selected["ridge"], "gamma": gamma,
                             "unchanged_interaction": gamma == 0, "memory_replaces_interaction": gamma == 1,
                             "mixed_residuals": 0 < gamma < 1,
                             "selected_validation_mae": selected["validation_mae"],
                             "baseline_validation_mae": state["baseline_validation_mae"],
                             "validation_mae_change": selected["validation_mae"] - state["baseline_validation_mae"],
                             "n_source_stations": len(state["source_station_ids"]),
                             "n_source_cells": len(state["source_cells"]),
                             "n_validation_query": len(state["validation_cells"]), **norm})
            candidate = current[current.model_name.eq(f"{mode}_constant")].sort_values("cell")
            np.testing.assert_array_equal(candidate.cell, baseline.cell)
            change = pd.DataFrame({"station_index": candidate.cell.to_numpy() // state["n_months"],
                                   "station": candidate.station.to_numpy(),
                                   "change": candidate.y_pred.to_numpy() - baseline.y_pred.to_numpy()})
            change["absolute_change"] = change.change.abs()
            change = change.groupby("station_index").agg(station=("station", "first"),
                query_mean_change=("change", "mean"), query_mean_absolute_change=("absolute_change", "mean"),
                query_max_absolute_change=("absolute_change", "max")).to_dict("index")
            source_ids, val_ids = set(state["source_station_ids"]), set(state["validation_station_ids"])
            for diagnostic in state["station_diagnostics"]:
                station = int(diagnostic["station"])
                donor_ids = diagnostic["donor_ids"]
                weights = np.asarray(diagnostic["donor_weights"], dtype=float)
                if (station in donor_ids or not set(donor_ids).issubset(source_ids)
                        or len(donor_ids) != diagnostic["donor_count"]
                        or len(weights) != len(donor_ids) or not np.isfinite(weights).all()
                        or not np.allclose(weights, 1 / len(weights))
                        or not diagnostic["optimizer"]["success"]):
                    raise ValueError("Invalid source-station donor weighting or optimization")
                role = "source" if station in source_ids else "source_validation" if station in val_ids else "outer_target"
                donor_counts = diagnostic["donor_source_counts"]
                donors.append({**identity, "mode": mode, "station_index": station, "role": role,
                               "gamma": gamma, "memory_active": gamma > 0,
                               **{key: diagnostic[key] for key in (
                                   "donor_count", "effective_count", "max_weight", "nearest_distance", "radius_distance",
                                   "fallback", "ecology_valid_features", "source_self_excluded")},
                               "min_donor_observations": min(donor_counts),
                               "max_donor_observations": max(donor_counts),
                               "total_donor_observations": sum(donor_counts),
                               "memory_intercept_native": norm["residual_scale"] * coef[station, 0],
                               "memory_slope_per_log_context": norm["residual_scale"] * coef[station, 1] / norm["log_context_sd"],
                               **change.get(station, {})})
    return pd.DataFrame(memories), pd.DataFrame(choices), pd.DataFrame(donors)


def write_report(out, curves, profiles, comparisons, concentration, memories, donors, draws):
    overall = comparisons[comparisons.region.eq("overall") & comparisons.metric.eq("mae")]
    lines = ["# Ecological residual transfer on the DOC interaction model", "",
             "Nine frozen-expert packages contain four competing residual memories each. Source",
             "DOC minus station-blocked OOF context predictions provides the residual library.",
             "The current interaction model and the support-adaptation representation remain fixed.", "",
             "Each source station receives equal total weight. Global versus ecological donor pools",
             "are crossed with an intercept-only versus concentration-dependent affine residual.",
             "Nine ecology variables use source-station median/IQR scaling. Ecological candidates",
             "use 20/40/80 eligible neighbors; all arms share ridge 0.1/1 and smooth absolute error.",
             "Affine conditioning uses source-standardized log1p(context prediction), not a physical",
             "state. No extra tail weighting is introduced in the memory objective.", "",
             "The memory competes with the temporal correction:", "",
             "    prediction = max(0, context + (1-gamma)*(interaction-context) + gamma*memory)", "",
             "Gamma=0 exactly preserves the interaction model; gamma=1 replaces its correction.",
             "Intermediate gamma mixes them. Memory improvement is not automatically an improvement",
             "of the neural representation. Source-validation K0 MAE selects gamma, ridge and donors;",
             "the usual support adapters are then fitted on the same validation protocol.", "",
             "These are previously evaluated same-cohort station partitions 142–144. This development",
             "analysis retains all four memories and does not select or promote a winner. Target",
             "support is retrospective and query populations are fixed across models and K.", "",
             f"Paired intervals use {draws:,} joint whole-station bootstrap draws, with repeated station",
             "identities resampled together. Seed losses are averaged within partitions, then partitions",
             "receive equal weight. Negative ΔMAE or positive relative reduction favors the candidate.",
             "Q90 is source-training-derived. Bias is prediction minus observed DOC.", "",
             "## Overall performance", "",
             "| Saved model | K | MAE | RMSE | R² | Log MAE | Q90 MAE |",
             "|---|---:|---:|---:|---:|---:|---:|"]
    for row in curves[curves.k.isin((0, 5))].itertuples():
        lines.append(f"| {row.model_name} | {row.k} | {row.mae:.4f} | {row.rmse:.3f} | {row.r2:.3f} | "
                     f"{row.log_mae:.4f} | {row.q90_mae:.4f} |")
    lines += ["", "At K0 the two support adapters give identical predictions for a given base.",
              "Repeated K0 rows are not independent evidence.", ""]
    for role, title in (("ecological_donor_information", "Ecological similarity beyond global calibration"),
                        ("concentration_conditioning", "Affine conditioning beyond a constant correction"),
                        ("existing_interaction_reference", "Comparison with the current interaction model")):
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
        for model in MODELS:
            if k == 0 and model.endswith("constant"):
                continue
            selected = profiles[profiles.model_name.eq(model) & profiles.k.eq(k)].set_index("region")
            tail, rest = selected.loc["q90"], selected.loc["nontail"]
            lines.append(f"| {model} | {k} | {tail.mae:.4f} | {tail.signed_bias:.4f} | {rest.mae:.4f} | "
                         f"{rest.log_mae:.4f} | {100 * rest.q90_false_positive_rate:.2f}% |")
    lines += ["", "False Q90 rate = P(prediction ≥ source Q90 | observation < source Q90).",
              "All regional raw/log MAE and signed-bias intervals are retained in comparisons.csv.", "",
              "## Source-validation choices", "",
              "| Memory | Unchanged interaction | Mixed corrections | Memory replaces correction | Selected gamma counts | Neighbor counts | Ridge counts |",
              "|---|---:|---:|---:|---|---|---|"]
    for mode, group in memories.groupby("mode", sort=False):
        gamma = ", ".join(f"{value:g}: {count}" for value, count in group.gamma.value_counts().sort_index().items())
        neighbors = ", ".join(str(int(value)) for value in sorted(group.neighbors.dropna().unique())) or "all source stations"
        ridge = ", ".join(f"{value:g}: {count}" for value, count in group.ridge.value_counts().sort_index().items())
        lines.append(f"| {mode} | {int(group.unchanged_interaction.sum())}/9 | {int(group.mixed_residuals.sum())}/9 | "
                     f"{int(group.memory_replaces_interaction.sum())}/9 | {gamma} | {neighbors} | {ridge} |")
    lines += ["", "Donor profiles are still saved when gamma=0, but they do not contribute to prediction.", ""]
    if memories[memories["mode"].eq("global_affine")].unchanged_interaction.all():
        lines += ["Global affine selects gamma=0 in all nine packages. It therefore equals the existing",
                  "interaction model at every K. Ecological-affine comparisons against global affine",
                  "and against interaction are the same numerical contrast, not independent evidence.", ""]
    lines += [
              "## Donor support on outer target stations", "",
              "| Memory | Mean effective donors | Maximum donor weight | Global fallback station-fits | Mean nearest squared distance | Mean absolute K0 prediction change |",
              "|---|---:|---:|---:|---:|---:|"]
    for mode, group in donors[donors.role.eq("outer_target")].groupby("mode", sort=False):
        distance = group.nearest_distance.mean()
        distance_label = f"{distance:.4f}" if np.isfinite(distance) else "not applicable"
        lines.append(f"| {mode} | {group.effective_count.mean():.1f} | {group.max_weight.max():.3f} | "
                     f"{int(group.fallback.notna().sum())}/{len(group)} | {distance_label} | "
                     f"{group.query_mean_absolute_change.mean():.4f} |")
    lines += ["", "Distances are mean squared differences in source-scaled ecology, undefined for global",
              "donor pools. Donor summaries are station-fit summaries across seeds and partitions,",
              "not independent station counts. Each donor station has uniform total weight regardless",
              "of its number of observations. Saved diagnostics include donor observation counts,",
              "maximum weight, ecological fallback, coefficient scales and target correction magnitude.", "",
              "## Station gain and loss concentration", "",
              "| Contrast with GRU support basis | Improved / worsened stations | Top-five share of positive gain | Top-five share of harm |",
              "|---|---:|---:|---:|"]
    focus = ("ecological_vs_global_affine", "ecological_affine_vs_interaction", "ecological_affine_vs_bias")
    for row in concentration[concentration.comparison.str.contains("gru_tuned_anchor")
                             & concentration.comparison.str.startswith(focus)].itertuples():
        lines.append(f"| {row.comparison} | {row.stations_improved} / {row.stations_worsened} | "
                     f"{100 * row.top5_share_of_positive_station_gain:.1f}% | {100 * row.top5_share_of_station_harm:.1f}% |")
    lines += ["", "Positive gains and harms have separate denominators. Contributions use equal",
              "partition weights and combine repeated station identities.", "",
              "## Interpretation of the main fixed contrasts", ""]
    for k in (0, 5):
        for prefix in focus:
            name = f"{prefix}_gru_tuned_anchor_k{k}"
            contrast = comparisons[comparisons.comparison.eq(name)]
            row = contrast[contrast.region.eq("overall") & contrast.metric.eq("mae")].iloc[0]
            tail = contrast[contrast.region.eq("q90") & contrast.metric.eq("mae")].iloc[0]
            rest = contrast[contrast.region.eq("nontail") & contrast.metric.eq("mae")].iloc[0]
            false = contrast[contrast.metric.eq("q90_false_positive_rate")].iloc[0]
            lines.append(f"- **{name}:** overall MAE reduction {row.relative_gain_pct:+.2f}% "
                         f"(95% CI [{row.gain_ci_low_pct:+.2f}, {row.gain_ci_high_pct:+.2f}]); "
                         f"{int(row.improved_splits)}/3 partitions and {int(row.improved_split_seed_pairs)}/9 fits improve. "
                         f"Q90 ΔMAE {tail.delta_value:+.4f} mg/L (95% CI [{tail.delta_ci_low:+.4f}, "
                         f"{tail.delta_ci_high:+.4f}]); non-tail ΔMAE {rest.delta_value:+.4f} mg/L "
                         f"(95% CI [{rest.delta_ci_low:+.4f}, {rest.delta_ci_high:+.4f}]). "
                         f"False Q90 rate changes by {100 * false.delta_value:+.3f} percentage points "
                         f"(95% CI [{100 * false.delta_ci_low:+.3f}, {100 * false.delta_ci_high:+.3f}]).")
    lines += ["", "Ecological versus global affine tests the value of ecological donor selection within",
              "the same residual family. Ecological affine versus ecological bias tests concentration",
              "conditioning. Each arm selects its mixing weight on source validation, so these are",
              "comparisons of complete fitting-and-selection procedures, not coefficients held at a",
              "common mixing weight. Improvement over interaction alone cannot establish either",
              "component without these controls. No physical process or causal interpretation is assigned.", ""]
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
    memories, choices, donors = selection_records(states, panel)
    out = args.root / "analysis"
    out.mkdir(parents=True, exist_ok=True)
    for name, frame in (("metrics_by_run", runs), ("metrics_by_partition", splits), ("k_curves", curves),
                        ("error_profiles_by_run", profile_runs), ("error_profiles_by_partition", profile_splits),
                        ("error_profiles", profiles), ("comparisons", comparisons),
                        ("directions_by_seed", directions), ("directions_by_partition", partitions),
                        ("station_responses", stations), ("global_station_contributions", global_stations),
                        ("gain_loss_concentration", concentration), ("memory_choices", memories),
                        ("adapter_choices", choices), ("donor_diagnostics", donors)):
        frame.to_csv(out / f"{name}.csv", index=False)
    write_report(out, curves, profiles, comparisons, concentration, memories, donors, args.bootstrap_draws)
    sources.extend({"path": str(path), "sha256": sha256(path)} for path in (
        Path(__file__), Path("scripts/analyze_doc_tail_residual_v1.py"),
        Path("scripts/analyze_unified_doc_spatial.py"), Path("scripts/analyze_unified_doc_spatial_v2.py"),
        Path("scripts/analyze_unified_doc_spatial_v3.py")))
    unique = {row["path"]: row for row in sources}
    (out / "sources.json").write_text(json.dumps(list(unique.values()), indent=2) + "\n")
    status = {"complete": True, "n_runs": len(states), "n_memory_fits": len(memories), "n_new_neural_fits": 0,
              "n_models": len(MODELS), "n_comparisons": len(comparison_definitions()),
              "bootstrap_draws": args.bootstrap_draws, "bootstrap_unit": "joint whole-station identity",
              "study_role": "development on previously evaluated station partitions", "model_selected": False,
              "sidecars_verified": True, "interaction_predictions_identical_to_prior": True,
              "gamma_zero_exact_prior": True, "source_station_donor_weights_checked": True,
              "analysis_inputs": len(unique)}
    (out / "status.json").write_text(json.dumps(status, indent=2) + "\n")
    print(json.dumps({**status, "output": str(out)}, indent=2))


if __name__ == "__main__":
    main()
