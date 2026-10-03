"""Evaluate support-dependent mixing of saved DOC ecological residual memories.

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

ROOT = Path("experiments/phase4_transfer/doc_ecological_transfer_v2")
ARMS = ("global_bias", "ecological_bias", "global_affine", "ecological_affine")
SHAPES = ("constant", "gru_tuned_anchor")
MODELS = tuple(f"{base}_{shape}" for base in ("interaction", *ARMS) for shape in SHAPES)
REFERENCES = tuple(f"fixed_{base}_{shape}" for base in ARMS for shape in SHAPES)


def comparison_definitions():
    rows = []
    for shape in SHAPES:
        for arm in ARMS:
            rows.append((f"{arm}_dynamic_vs_fixed_{shape}_k5", f"{arm}_{shape}", 5,
                         f"fixed_{arm}_{shape}", 5, "support_dependent_mixing"))
        for k in (1, 3, 5):
            for arm in ("ecological_bias", "ecological_affine"):
                rows.append((f"{arm}_vs_interaction_{shape}_k{k}", f"{arm}_{shape}", k,
                             f"interaction_{shape}", k, "existing_interaction_reference"))
        rows.append((f"ecological_affine_vs_bias_{shape}_k5", f"ecological_affine_{shape}", 5,
                     f"ecological_bias_{shape}", 5, "concentration_conditioning"))
        rows.append((f"ecological_vs_global_affine_{shape}_k5", f"ecological_affine_{shape}", 5,
                     f"global_affine_{shape}", 5, "ecological_donor_information"))
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
                    or config["gamma_grid"] != [0, .25, .5, 1]
                    or config["alpha_grid"] != [0, .25, .5, .75, 1]
                    or config["ridge_grid"] != [.1, 1, 10, "infinity"]):
                raise ValueError(f"Unexpected complete-production configuration: {run}")
            threshold = float(config["q90_threshold_train"])
            if not np.isfinite(threshold) or threshold < 0:
                raise ValueError(f"Invalid source-training Q90: {run}")
            thresholds[(split, seed)] = threshold
            prior = Path(config["prior_run"])
            if sha256(prior / "complete.json") != config["prior_completion_hash"]:
                raise ValueError(f"Prior ecological package changed: {prior}")
            old, old_config, old_completion = read_predictions(prior, sources)
            check_source_identity(config, old_config,
                                  ("split_seed", "seed", "dataset_hash", "mask_hash", "q90_threshold_train"))
            for model in MODELS:
                current = frame[frame.model_name.eq(model)]
                previous = old[old.model_name.eq(model)]
                if not model.startswith("interaction_"):
                    current, previous = current[current.k.eq(0)], previous[previous.k.eq(0)]
                current, previous = current.sort_values(["k", "cell"]), previous.sort_values(["k", "cell"])
                np.testing.assert_array_equal(current.cell, previous.cell)
                np.testing.assert_array_equal(current.y_true, previous.y_true)
                np.testing.assert_array_equal(current.y_pred, previous.y_pred)
            reference = old[~old.model_name.str.startswith("interaction_")].copy()
            reference["model_name"] = "fixed_" + reference.model_name
            frames.extend([frame, reference])
            memories = {arm: read_bound_json(prior, f"{arm}.json", old_completion, sources) for arm in ARMS}
            binding = read_bound_json(run, "profile_bindings.json", completion, sources)
            expected_profiles = {arm: sha256(prior / f"{arm}.json") for arm in ARMS}
            if (config["profile_files"] != expected_profiles
                    or binding != {"prior_run": str(prior), "prior_completion_hash": config["prior_completion_hash"],
                                   "profiles": expected_profiles}):
                raise ValueError("Current wrapper is not bound to the unchanged v1 memory profiles")
            adapters = read_bound_json(run, "mixers.json", completion, sources)
            states.append({"split_seed": split, "seed": seed, "config": config,
                           "adapters": adapters, "memories": memories})
        if len({thresholds[(split, seed)] for seed in SEEDS}) != 1:
            raise ValueError(f"Q90 threshold changes across seeds: {split}")
    panel = pd.concat(frames, ignore_index=True)
    validate_panel(panel, expected_models=MODELS + REFERENCES)
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
    choices, scores, profiles = [], [], []
    for run in states:
        identity = {"split_seed": run["split_seed"], "seed": run["seed"]}
        current = panel[panel.split_seed.eq(run["split_seed"]) & panel.seed.eq(run["seed"])]
        if set(run["adapters"]) != {f"{mode}_{shape}" for mode in ARMS for shape in SHAPES}:
            raise ValueError("Saved mixers differ from the eight specified memory/support combinations")
        for mode, memory in run["memories"].items():
            chosen = memory["selected"]
            profiles.append({**identity, "mode": mode, "prior_gamma": chosen["gamma"],
                             "neighbors": chosen["k"], "memory_ridge": chosen["ridge"],
                             "memory_validation_mae": chosen["validation_mae"],
                             "profile_refitted": False})
            for shape in SHAPES:
                model = f"{mode}_{shape}"
                state = run["adapters"][model]
                if (state["selection_role"] != "source_validation"
                        or set(map(int, state["selection_by_k"])) != set(KS)
                        or state["gamma_k0"] != chosen["gamma"]
                        or state["gamma_values"] != [0, .25, .5, 1]):
                    raise ValueError(f"Invalid support-aware selection: {model}")
                prior_gamma = state["gamma_k0"]
                for k in KS:
                    selection = state["selection_by_k"][str(k)]
                    candidates = [row for row in state["gamma_scores"] if row["k"] == k]
                    if len(candidates) != 4 or not all(row["valid"] for row in candidates):
                        raise ValueError("Each K requires four valid gamma candidates")
                    eligible = [row for row in candidates if k != 0 or row["gamma"] == prior_gamma]
                    best = min(eligible, key=lambda row: (row["mae"], row["gamma"] != 0,
                                                         row["gamma"] != prior_gamma, row["gamma"]))
                    if selection != {**best, "locked": k == 0}:
                        raise ValueError("Selection differs from source-validation trace or tie rule")
                    fixed = next(row for row in candidates if row["gamma"] == prior_gamma)
                    if k > 0 and selection["mae"] > fixed["mae"]:
                        raise ValueError("Joint selection is worse than its available fixed-gamma candidate")
                    if selection["gamma"] == 0:
                        proposed = current[current.model_name.eq(model) & current.k.eq(k)].sort_values("cell")
                        reference = current[current.model_name.eq(f"interaction_{shape}") & current.k.eq(k)].sort_values("cell")
                        np.testing.assert_array_equal(proposed.cell, reference.cell)
                        np.testing.assert_array_equal(proposed.y_pred, reference.y_pred)
                    choices.append({**identity, "model_name": model, "mode": mode, "shape": shape,
                                    "prior_gamma": prior_gamma, **selection,
                                    "changed_from_k0": selection["gamma"] != prior_gamma,
                                    "fixed_gamma_validation_mae": fixed["mae"],
                                    "validation_mae_change": selection["mae"] - fixed["mae"]})
                    scores.extend({**identity, "model_name": model, "mode": mode, "shape": shape,
                                   "prior_gamma": prior_gamma, **row} for row in candidates)
    return pd.DataFrame(choices), pd.DataFrame(scores), pd.DataFrame(profiles)


def write_report(out, curves, profiles, comparisons, concentration, choices, memory_profiles, draws):
    lines = ["# Support-aware integration of ecological and temporal DOC residuals", "",
             "This iteration reuses the four source-station memory profiles from v1 and the saved",
             "temporal expert. It changes how their residual corrections are mixed after target",
             "support becomes available. No ecological profile or neural weight is refitted.", "",
             "For each K > 0, source-validation query MAE jointly chooses the mixing weight gamma",
             "and the existing support adapter's alpha and ridge. Gamma=0 uses the temporal expert,",
             "gamma=1 uses the ecological/global memory correction, and intermediate values mix them.",
             "The K0 gamma is locked to v1. All K0 outputs are verified bitwise identical to v1;",
             "their previously established gains are the same evidence, not new confirmation.", "",
             "The comparison retains all ten current products and eight fixed-mixing references.",
             "All use the same target queries and nested retrospective support sets. Target support",
             "can follow the query in calendar time. Source validation and the three same-cohort",
             "station partitions have been used during model development; outer outcomes do not",
             "select a mode, K or hyperparameter in this analysis.", "",
             f"Intervals use {draws:,} joint whole-station bootstrap draws. Repeated station identities",
             "are resampled together across partitions; seed losses are averaged within partition,",
             "then the three partitions receive equal weight. Positive reduction and negative",
             "Delta MAE favor the candidate. Q90 thresholds come from source training observations.", "",
             "## K curves with the existing GRU support basis", "",
             "| Saved model | K0 MAE | K1 MAE | K3 MAE | K5 MAE | K5 RMSE | K5 R² | K5 log MAE |",
             "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for model in MODELS + REFERENCES:
        if not model.endswith("gru_tuned_anchor"):
            continue
        rows = curves[curves.model_name.eq(model)].set_index("k")
        row = rows.loc[5]
        lines.append(f"| {model} | " + " | ".join(f"{rows.loc[k, 'mae']:.4f}" for k in KS)
                     + f" | {row.rmse:.3f} | {row.r2:.3f} | {row.log_mae:.4f} |")
    lines += ["", "Full metrics, including constant-only adaptation, are in k_curves.csv.", ""]
    overall = comparisons[comparisons.region.eq("overall") & comparisons.metric.eq("mae")]
    for role, title in (("support_dependent_mixing", "Joint versus fixed mixing at K5"),
                        ("existing_interaction_reference", "Ecological integration versus the existing interaction expert"),
                        ("concentration_conditioning", "Affine versus bias memory"),
                        ("ecological_donor_information", "Ecological versus global affine memory")):
        lines += [f"## {title}", "", "| Comparison | Delta MAE [95% CI] | Reduction [95% CI] | Improved partitions; fits |",
                  "|---|---:|---:|---:|"]
        for row in overall[overall.comparison_role.eq(role)].itertuples():
            lines.append(f"| {row.comparison} | {row.delta_value:+.4f} [{row.delta_ci_low:+.4f}, {row.delta_ci_high:+.4f}] | "
                         f"{row.relative_gain_pct:+.2f}% [{row.gain_ci_low_pct:+.2f}, {row.gain_ci_high_pct:+.2f}] | "
                         f"{int(row.improved_splits)}/3; {int(row.improved_split_seed_pairs)}/9 |")
        lines.append("")
    lines += ["## Tail, ordinary concentrations and false alarms", "",
              "| Saved model | K | Q90 MAE | Q90 bias | Non-tail MAE | Non-tail log MAE | False Q90 rate |",
              "|---|---:|---:|---:|---:|---:|---:|"]
    for model in MODELS + REFERENCES:
        if not model.endswith("gru_tuned_anchor"):
            continue
        for k in (0, 5):
            selected = profiles[profiles.model_name.eq(model) & profiles.k.eq(k)].set_index("region")
            tail, rest = selected.loc["q90"], selected.loc["nontail"]
            lines.append(f"| {model} | {k} | {tail.mae:.4f} | {tail.signed_bias:+.4f} | {rest.mae:.4f} | "
                         f"{rest.log_mae:.4f} | {100 * rest.q90_false_positive_rate:.2f}% |")
    lines += ["", "Bias is predicted minus observed DOC. False Q90 rate is the fraction of non-tail",
              "observations predicted at or above the source Q90 threshold. Regional intervals for",
              "raw/log MAE, bias and false alarms are retained for every contrast in comparisons.csv.", "",
              "## Source-validation selections", "",
              "| Mode | Support adapter | K | Gamma counts | Changed from K0 | Mean validation MAE change vs fixed gamma |",
              "|---|---|---:|---|---:|---:|"]
    for (mode, shape, k), group in choices.groupby(["mode", "shape", "k"], sort=False):
        gamma = ", ".join(f"{value:g}: {count}" for value, count in group.gamma.value_counts().sort_index().items())
        lines.append(f"| {mode} | {shape} | {k} | {gamma} | {int(group.changed_from_k0.sum())}/9 | "
                     f"{group.validation_mae_change.mean():+.5f} |")
    lines += ["", "K0 selection is locked. At positive K, fixed gamma remains an available candidate.",
              "Validation MAE therefore cannot worsen when the candidate set expands, but this does",
              "not ensure improved target-station performance. Alpha/ridge choices and all gamma",
              "scores are saved separately. Gamma=0 target predictions are checked against the",
              "unchanged interaction support adapter.", "",
              "## Reused ecological profiles", "",
              "| Memory | Locked K0 gamma counts | Neighbor counts | Memory ridge counts |",
              "|---|---|---|---|"]
    for mode, group in memory_profiles.groupby("mode", sort=False):
        gamma = ", ".join(f"{value:g}: {count}" for value, count in group.prior_gamma.value_counts().sort_index().items())
        neighbors = ", ".join(str(int(value)) for value in sorted(group.neighbors.dropna().unique())) or "all source stations"
        ridge = ", ".join(f"{value:g}: {count}" for value, count in group.memory_ridge.value_counts().sort_index().items())
        lines.append(f"| {mode} | {gamma} | {neighbors} | {ridge} |")
    lines += ["", "The bound v1 memory states retain the original source-only ecological scaler, donor",
              "IDs, equal station weights, distances, missing-ecology fallback and coefficient arrays.",
              "These profiles are reused intact, including profiles whose locked K0 gamma was zero.",
              "Their source fitting and donor diagnostics remain documented in the v1 analysis.", "",
              "## Station gain and loss concentration", "",
              "| GRU-basis K5 contrast | Improved / worsened stations | Top-five positive gain | Top-five harm |",
              "|---|---:|---:|---:|"]
    focus = ("ecological_affine_dynamic_vs_fixed", "ecological_affine_vs_interaction", "ecological_bias_vs_interaction")
    for row in concentration[concentration.comparison.str.contains("gru_tuned_anchor_k5")
                             & concentration.comparison.str.startswith(focus)].itertuples():
        lines.append(f"| {row.comparison} | {row.stations_improved} / {row.stations_worsened} | "
                     f"{100 * row.top5_share_of_positive_station_gain:.1f}% | {100 * row.top5_share_of_station_harm:.1f}% |")
    lines += ["", "Positive gain and harm have separate denominators; station contributions use equal",
              "partition weights and combine repeated station identities.", "",
              "## Main fixed K5 contrasts", ""]
    for prefix in focus:
        name = f"{prefix}_gru_tuned_anchor_k5"
        contrast = comparisons[comparisons.comparison.eq(name)]
        row = contrast[contrast.region.eq("overall") & contrast.metric.eq("mae")].iloc[0]
        tail = contrast[contrast.region.eq("q90") & contrast.metric.eq("mae")].iloc[0]
        rest = contrast[contrast.region.eq("nontail") & contrast.metric.eq("mae")].iloc[0]
        false = contrast[contrast.metric.eq("q90_false_positive_rate")].iloc[0]
        lines.append(f"- **{name}:** MAE reduction {row.relative_gain_pct:+.2f}% "
                     f"(95% CI [{row.gain_ci_low_pct:+.2f}, {row.gain_ci_high_pct:+.2f}]); "
                     f"{int(row.improved_splits)}/3 partitions and {int(row.improved_split_seed_pairs)}/9 fits improve. "
                     f"Q90 Delta MAE {tail.delta_value:+.4f} mg/L [{tail.delta_ci_low:+.4f}, {tail.delta_ci_high:+.4f}]; "
                     f"non-tail Delta MAE {rest.delta_value:+.4f} mg/L [{rest.delta_ci_low:+.4f}, {rest.delta_ci_high:+.4f}]. "
                     f"False Q90 rate change {100 * false.delta_value:+.3f} percentage points "
                     f"[{100 * false.delta_ci_low:+.3f}, {100 * false.delta_ci_high:+.3f}].")
    lines += ["", "These comparisons assess how source-validation support selects between two saved",
              "residual explanations. Improvements do not imply newly learned neural features or a",
              "physical mechanism. The full fixed comparison set is reported without target-based",
              "selection or automatic model promotion.", ""]
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
    choices, gamma_scores, memory_profiles = selection_records(states, panel)
    out = args.root / "analysis"
    out.mkdir(parents=True, exist_ok=True)
    for name, frame in (("metrics_by_run", runs), ("metrics_by_partition", splits), ("k_curves", curves),
                        ("error_profiles_by_run", profile_runs), ("error_profiles_by_partition", profile_splits),
                        ("error_profiles", profiles), ("comparisons", comparisons),
                        ("directions_by_seed", directions), ("directions_by_partition", partitions),
                        ("station_responses", stations), ("global_station_contributions", global_stations),
                        ("gain_loss_concentration", concentration), ("memory_profiles", memory_profiles),
                        ("adapter_choices", choices), ("gamma_scores", gamma_scores)):
        frame.to_csv(out / f"{name}.csv", index=False)
    write_report(out, curves, profiles, comparisons, concentration, choices, memory_profiles, args.bootstrap_draws)
    sources.extend({"path": str(path), "sha256": sha256(path)} for path in (
        Path(__file__), Path("scripts/analyze_doc_tail_residual_v1.py"),
        Path("scripts/analyze_unified_doc_spatial.py"), Path("scripts/analyze_unified_doc_spatial_v2.py"),
        Path("scripts/analyze_unified_doc_spatial_v3.py")))
    unique = {row["path"]: row for row in sources}
    (out / "sources.json").write_text(json.dumps(list(unique.values()), indent=2) + "\n")
    status = {"complete": True, "n_runs": len(states), "n_reused_memory_profiles": len(memory_profiles), "n_new_memory_fits": 0, "n_new_neural_fits": 0,
              "n_models": len(MODELS + REFERENCES), "n_comparisons": len(comparison_definitions()),
              "bootstrap_draws": args.bootstrap_draws, "bootstrap_unit": "joint whole-station identity",
              "study_role": "development on previously evaluated station partitions", "model_selected": False,
              "sidecars_verified": True, "interaction_predictions_identical_to_prior": True,
              "gamma_zero_exact_interaction": True, "k0_identical_to_v1": True,
              "memory_profiles_reused": True, "joint_choices_match_source_validation": True,
              "analysis_inputs": len(unique)}
    (out / "status.json").write_text(json.dumps(status, indent=2) + "\n")
    print(json.dumps({**status, "output": str(out)}, indent=2))


if __name__ == "__main__":
    main()
