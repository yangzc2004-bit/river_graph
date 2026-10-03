"""Evaluate partial spatial/ecological encoder updates in the DOC residual model.

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

ROOT = Path("experiments/phase4_transfer/doc_encoder_residual_v1")
ARMS = ("frozen", "last_self", "last_self_ecology")
SHAPES = ("constant", "gru_tuned_anchor")
REFERENCES = tuple(f"prior_{base}_{shape}" for base in ("concentration", "ecological_affine") for shape in SHAPES)
DIRECT_MODELS = tuple(f"{base}_{shape}" for base in ("context", *ARMS) for shape in SHAPES)
MIXED_MODELS = tuple(f"{arm}_integrated_{shape}" for arm in ARMS for shape in SHAPES)
MODELS = DIRECT_MODELS + MIXED_MODELS + REFERENCES
INTERACTIONS = [0, 2, 4, 28]


def comparison_definitions():
    rows = []
    for shape in SHAPES:
        for k in ((5,) if shape == "constant" else (0, 5)):
            for arm in ARMS:
                rows.append((f"{arm}_vs_prior_concentration_{shape}_k{k}", f"{arm}_{shape}", k,
                             f"prior_concentration_{shape}", k, "previous_concentration_reference"))
                rows.append((f"{arm}_integrated_vs_prior_ecological_{shape}_k{k}", f"{arm}_integrated_{shape}", k,
                             f"prior_ecological_affine_{shape}", k, "previous_overall_reference"))
                rows.append((f"{arm}_integrated_vs_direct_{shape}_k{k}", f"{arm}_integrated_{shape}", k,
                             f"{arm}_{shape}", k, "ecological_integration"))
            for suffix in ("", "_integrated"):
                for candidate, reference in (("last_self", "frozen"), ("last_self_ecology", "frozen"),
                                             ("last_self_ecology", "last_self")):
                    rows.append((f"{candidate}_vs_{reference}{suffix}_{shape}_k{k}", f"{candidate}{suffix}_{shape}", k,
                                 f"{reference}{suffix}_{shape}", k, "matched_encoder_update"))
    return rows


def check_source_identity(config, reference, keys):
    for key in keys:
        if config[key] != reference[key]:
            raise ValueError(f"Current/reference identity differs in {key}")


def load_panel(root, expected_epochs=30):
    frames, thresholds, sources, states = [], {}, [], []
    for split in SPLITS:
        for seed in SEEDS:
            run = root / "runs" / f"split{split}_seed{seed}"
            frame, config, completion = read_predictions(run, sources)
            if (config["split_seed"], config["seed"]) != (split, seed):
                raise ValueError(f"Directory identity differs: {run}")
            if (set(config["models"]) != set(MODELS) or tuple(config["k_values"]) != KS
                    or config["inference_roles"] != ["train"] or config["extra_dim"] != 30
                    or config["epochs"] != expected_epochs or config["patience"] != 5 or config["tail_weight"] != 2
                    or config["interaction_indices"] != INTERACTIONS or config["train_memory"] is not True
                    or config["encoder_learning_rate"] != 1e-5 or tuple(config["encoder_modes"]) != ARMS
                    or config["gamma_grid"] != [0, .25, .5, 1]
                    or config["alpha_grid"] != [0, .25, .5, .75, 1]
                    or config["ridge_grid"] != [.1, 1, 10, "infinity"]):
                raise ValueError(f"Unexpected production settings: {run}")
            threshold = float(config["q90_threshold_train"])
            if not np.isfinite(threshold) or threshold < 0:
                raise ValueError(f"Invalid source Q90: {run}")
            thresholds[(split, seed)] = threshold
            prior, memory_run = Path(config["prior_run"]), Path(config["memory_run"])
            for name, path in (("prior", prior), ("memory", memory_run)):
                if sha256(path / "complete.json") != config[f"{name}_completion_hash"]:
                    raise ValueError(f"Changed bound {name} package")
            previous, prior_config, _ = read_predictions(prior, sources)
            check_source_identity(config, prior_config,
                                  ("split_seed", "seed", "dataset_hash", "mask_hash", "q90_threshold_train"))
            memory_completion = json.loads((memory_run / "complete.json").read_text())
            memory = read_bound_json(memory_run, "ecological_affine.json", memory_completion, sources)
            if (sha256(memory_run / "ecological_affine.json") != config["memory_file_hash"]
                    or memory["mode"] != "ecological_affine" or memory["selection_role"] != "source_validation"):
                raise ValueError("Changed frozen ecological-affine memory")
            for name in REFERENCES:
                a = frame[frame.model_name.eq(name)].sort_values(["k", "cell"])
                old_name = name.removeprefix("prior_") if name.startswith("prior_concentration_") else name
                b = previous[previous.model_name.eq(old_name)].sort_values(["k", "cell"])
                for field in ("cell", "y_true", "y_pred"):
                    np.testing.assert_array_equal(a[field], b[field])
            features = read_bound_json(run, "feature_definition.json", completion, sources)
            if (features != config["feature_definition"] or len(features["feature_names"]) != 30
                    or features["interaction_indices"]["concentration"] != INTERACTIONS
                    or features["normalization"]["ecology"]["columns"] != list(range(4, 13))):
                raise ValueError("Changed regime readout feature definition")
            frames.append(frame)
            states.append({"split_seed": split, "seed": seed, "config": config,
                           "adapters": read_bound_json(run, "adapters.json", completion, sources),
                           "mixers": read_bound_json(run, "mixers.json", completion, sources),
                           "training": {arm: read_bound_json(run, f"{arm}.json", completion, sources) for arm in ARMS}})
            sources.append({"path": str(memory_run / "complete.json"), "sha256": sha256(memory_run / "complete.json")})
        if len({thresholds[(split, seed)] for seed in SEEDS}) != 1:
            raise ValueError(f"Source Q90 differs across seeds: {split}")
    panel = pd.concat(frames, ignore_index=True)
    validate_panel(panel, expected_models=MODELS)
    for _, group in panel[panel.k.eq(0)].groupby(["split_seed", "seed"]):
        pivot = group.pivot(index="cell", columns="model_name", values="y_pred")
        for base in ("context", *ARMS, *(f"{arm}_integrated" for arm in ARMS), "prior_concentration", "prior_ecological_affine"):
            np.testing.assert_array_equal(pivot[f"{base}_constant"], pivot[f"{base}_gru_tuned_anchor"])
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


def compare(panel, thresholds, draws, definitions=None):
    effects, directions, station_rows, global_rows, concentrations = [], [], [], [], []
    for name, candidate, ck, reference, rk, role in comparison_definitions() if definitions is None else definitions:
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
    training, choices, mixing, scores = [], [], [], []
    for run in states:
        identity = {"split_seed": run["split_seed"], "seed": run["seed"]}
        current = panel[panel.split_seed.eq(run["split_seed"]) & panel.seed.eq(run["seed"])]
        if set(run["adapters"]) != set(DIRECT_MODELS) or set(run["mixers"]) != set(MIXED_MODELS):
            raise ValueError("Unexpected direct adapter or mixer models")
        for model, state in run["adapters"].items():
            if state["selection_role"] != "source_validation" or set(map(int, state["selection_by_k"])) != set(KS):
                raise ValueError("Direct support adapter has wrong selection role/K")
            choices.extend({**identity, "model_name": model, "k": int(k), **selected}
                           for k, selected in state["selection_by_k"].items())
        for arm, state in run["training"].items():
            cfg, trace = state["config"], state["trace"]
            chosen = [row for row in trace if row["epoch"] == state["best_epoch"]]
            if (cfg["extra_dim"] != 30 or cfg["interaction_indices"] != INTERACTIONS
                    or cfg.get("train_memory", True) is not True or cfg["tail_weight"] != 2
                    or cfg["epochs"] != run["config"]["epochs"] or cfg["patience"] != 5
                    or cfg["encoder_mode"] != arm or cfg["encoder_learning_rate"] != 1e-5
                    or state["protocol"]["selection_role"] != "source_validation"
                    or len(chosen) != 1 or chosen[0]["validation_scale"] != state["selected_scale"]):
                raise ValueError("Saved head training differs from the fixed experiment")
            if arm == "frozen" and (state["encoder_trainable_parameter_count"] != 0
                                    or state["spatial_parameter_distance"] != 0):
                raise ValueError("Frozen encoder changed or contains trainable spatial parameters")
            if arm != "last_self_ecology" and state["ecology_parameter_distance"] != 0:
                raise ValueError("Ecological encoder changed outside its designated arm")
            head_parameters = state["hidden_size"] * (1 + len(INTERACTIONS)) + 31
            training.append({**identity, "arm": arm, "head_parameters": head_parameters,
                             "extra_dim": 30, "interaction_dim": state["hidden_size"] * len(INTERACTIONS),
                             "epoch_ceiling": cfg["epochs"],
                             "best_epoch": state["best_epoch"], "epochs_run": state["epochs_run"],
                             "selected_scale": state["selected_scale"], "context_fallback": state["selected_scale"] == 0,
                             "initial_validation_mae": trace[0]["validation_mae"],
                             "selected_validation_mae": chosen[0]["validation_mae"],
                             **{key: state[key] for key in ("trainable_parameter_count", "n_source_cells", "n_source_tail_cells",
                                 "n_validation_query", "temporal_parameter_distance", "decay_parameter_distance", "head_parameter_norm",
                                 "encoder_trainable_parameter_count", "spatial_parameter_distance",
                                 "last_self_parameter_distance", "ecology_parameter_distance")}})
        for model, state in run["mixers"].items():
            if state["selection_role"] != "source_validation" or set(map(int, state["selection_by_k"])) != set(KS):
                raise ValueError("Invalid integrated support selection")
            gamma_k0 = state["gamma_k0"]
            initial = [row for row in state["gamma_scores"] if row["k"] == 0]
            if gamma_k0 != min(initial, key=lambda row: (row["mae"], row["gamma"]))["gamma"]:
                raise ValueError("Integrated K0 gamma differs from source-validation selection")
            for k in KS:
                selected = state["selection_by_k"][str(k)]
                candidates = [row for row in state["gamma_scores"] if row["k"] == k]
                eligible = [row for row in candidates if k != 0 or row["gamma"] == gamma_k0]
                chosen = min(eligible, key=lambda row: (row["mae"], row["gamma"] != 0,
                                                       row["gamma"] != gamma_k0, row["gamma"]))
                if selected != {**chosen, "locked": k == 0}:
                    raise ValueError("Integrated selection differs from recorded validation trace")
                if selected["gamma"] == 0:
                    direct = model.replace("_integrated_", "_")
                    a = current[current.model_name.eq(model) & current.k.eq(k)].sort_values("cell")
                    b = current[current.model_name.eq(direct) & current.k.eq(k)].sort_values("cell")
                    np.testing.assert_array_equal(a.y_pred, b.y_pred)
                mixing.append({**identity, "model_name": model, "gamma_k0": gamma_k0, **selected})
                scores.extend({**identity, "model_name": model, **row} for row in candidates)
    return pd.DataFrame(training), pd.DataFrame(choices), pd.DataFrame(mixing), pd.DataFrame(scores)


def processing_differences(panel):
    rows = []
    for (partition, seed), group in panel.groupby(["split_seed", "seed"]):
        for shape in SHAPES:
            for k in KS:
                a = group[group.model_name.eq(f"frozen_{shape}") & group.k.eq(k)].sort_values("cell")
                b = group[group.model_name.eq(f"prior_concentration_{shape}") & group.k.eq(k)].sort_values("cell")
                np.testing.assert_array_equal(a.cell, b.cell)
                delta = a.y_pred.to_numpy() - b.y_pred.to_numpy()
                rows.append({"split_seed": partition, "seed": seed, "shape": shape, "k": k,
                             "query_cells": len(a), "max_absolute_prediction_difference": np.abs(delta).max(),
                             "mean_absolute_prediction_difference": np.abs(delta).mean(),
                             "root_mean_square_prediction_difference": np.sqrt(np.mean(delta ** 2)),
                             "bitwise_equal": bool(np.array_equal(a.y_pred.to_numpy(), b.y_pred.to_numpy()))})
    return pd.DataFrame(rows)


def write_report(out, curves, profiles, comparisons, concentration, training, mixing, processing, draws,
                 expected_epochs=30):
    if expected_epochs == 30:
        predecessor_context = [
            "The new frozen arm is the matched control for partial updating. The previous saved",
            "concentration model used cached encodings. Batched raw encoding can introduce small",
            "floating-point changes that optimization amplifies, so final prediction equality is",
            "measured rather than assumed. Historical reference products themselves are copied exactly."]
        predecessor_title = "Frozen processing versus the cached predecessor"
        predecessor_interpretation = [
            "These compare final trained products and are empirical processing diagnostics.",
            "No equivalence threshold is fitted to the observed differences. Raw-versus-cached",
            "initial encoder features and saved-model replay are checked independently by the verifier."]
    else:
        predecessor_context = [
            "The new frozen arm is the matched control for partial updating under the same extended",
            f"{expected_epochs}-epoch ceiling. Its preceding saved concentration reference had a 30-epoch",
            "ceiling. Their final prediction differences therefore include the training-duration",
            "extension; they are not a raw-versus-cached processing-equivalence test. Initial encoder",
            "numeric parity and saved-model replay are checked separately. Historical predictions are copied exactly.",
            "The budget extension was motivated by source-validation learning traces reaching the",
            "earlier ceiling. The objective, tail weight, stopping patience and target endpoints are unchanged."]
        predecessor_title = "Frozen budget extension versus preceding 30-epoch concentration model"
        predecessor_interpretation = [
            f"These products use different epoch ceilings ({expected_epochs} versus 30). Differences must not be",
            "attributed solely to numerical processing. Initial raw-versus-cached encoder parity is",
            "verified independently; the matched raw-input 60-versus-30 comparisons are reported separately."
            if expected_epochs == 60 else "verified independently; the matched budget comparisons are reported separately."]
    lines = ["# Partial encoder adaptation for native DOC residual learning", "",
             "Nine station-partition/seed packages contain three neural fits each. All modes retain",
             "the concentration-interaction head: 30 direct regime/flow features and hidden-state",
             "interactions with three numerical flow features and context concentration. The source",
             f"OOF context forest, native DOC objective, tail weight 2, {expected_epochs}-epoch ceiling, patience 5,",
             "GRU initialization, support basis and ecological-memory profile remain fixed.", "",
             "Raw monthly inputs now pass through the existing spatial/ecological encoder during",
             "optimization. Its input normalization is unchanged and dropout is disabled for every",
             "mode. Empty edges are retained: this is a test of local representation adaptation,",
             "not renewed river-message training or evidence for graph transport.", "",
             "The matched trainability ladder is:", "",
             "- frozen: encoder weights remain fixed; GRU/decay and the concentration head train.",
             "- last_self: also update the encoder's final self-path layer.",
             "- last_self_ecology: also update its existing ecological encoder.", "",
             "Selected encoder parameters use learning rate 1e-5. GRU/decay and head learning rates",
             "remain 1e-4 and 1e-3. Every scalar head starts at zero, and source validation selects",
             "checkpoint and residual scale using overall native query MAE. Encoder parameter",
             "counts and selected-checkpoint movements are reported separately.", "",
             *predecessor_context, "",
             "Each new base is also integrated with the same frozen v1 ecological-affine residual profile.",
             "Source-validation selects the K0 residual mixing weight; positive-K selection jointly",
             "chooses mixing and the existing support adapter. Gamma=0 recovers its direct neural base,",
             "gamma=1 replaces the neural correction with ecological memory. Both paths use the same",
             "frozen-v4 GRU support basis. All models and old concentration/ecological-v2 references remain",
             "visible. This analysis does not select a winning arm or promote it from target outcomes.", "",
             "Station partitions 142–144 have been examined previously. This is model development on",
             "the same cohort. Source validation has also been reused. Target support is retrospective",
             "and all K values use a fixed query set, with the full five reserved supports excluded.", "",
             f"Intervals use {draws:,} paired whole-station bootstrap draws; repeated station identities",
             "are jointly resampled across partitions. Seed losses are averaged within partition, then",
             "the three partitions receive equal weight. Negative Delta MAE and positive relative",
             "reduction favor the candidate. Q90 is source-training-derived. Bias is prediction minus truth.", "",
             "## K curves with the existing GRU support basis", "",
             "| Saved model | K0 MAE | K1 MAE | K3 MAE | K5 MAE | K5 RMSE | K5 R² | K5 log MAE |",
             "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for model in MODELS:
        if not model.endswith("gru_tuned_anchor"):
            continue
        rows = curves[curves.model_name.eq(model)].set_index("k")
        row = rows.loc[5]
        lines.append(f"| {model} | " + " | ".join(f"{rows.loc[k, 'mae']:.4f}" for k in KS)
                     + f" | {row.rmse:.3f} | {row.r2:.3f} | {row.log_mae:.4f} |")
    lines += ["", "Constant-only adaptation is retained in all CSVs. At K0 it is bitwise identical to",
              "the GRU-support path, so its duplicate bootstrap contrasts are omitted.", ""]
    overall = comparisons[comparisons.region.eq("overall") & comparisons.metric.eq("mae")]
    for role, title in (("matched_encoder_update", "Matched partial encoder updates"),
                        ("previous_concentration_reference", "Comparison with the prior concentration model"),
                        ("previous_overall_reference", "Integrated models versus prior ecological-v2 model"),
                        ("ecological_integration", "Ecological integration versus each direct head")):
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
    for model in MODELS:
        if not model.endswith("gru_tuned_anchor"):
            continue
        for k in (0, 5):
            selected = profiles[profiles.model_name.eq(model) & profiles.k.eq(k)].set_index("region")
            tail, rest = selected.loc["q90"], selected.loc["nontail"]
            lines.append(f"| {model} | {k} | {tail.mae:.4f} | {tail.signed_bias:+.4f} | {rest.mae:.4f} | "
                         f"{rest.log_mae:.4f} | {100 * rest.q90_false_positive_rate:.2f}% |")
    lines += ["", "False Q90 rate conditions on an observed concentration below the source Q90 threshold.",
              "All regional raw/log MAE and bias intervals are in comparisons.csv.", "",
              "## Source-validation training choices", "",
              "| Mode | Total trainable parameters | Trainable encoder parameters | Selected epoch range | Scale counts | Mean validation MAE change |",
              "|---|---:|---:|---|---|---:|"]
    for arm, group in training.groupby("arm", sort=False):
        scales = ", ".join(f"{v:g}: {c}" for v, c in group.selected_scale.value_counts().sort_index().items())
        lines.append(f"| {arm} | {int(group.trainable_parameter_count.iloc[0])} | {int(group.encoder_trainable_parameter_count.iloc[0])} | "
                     f"{group.best_epoch.min()}–{group.best_epoch.max()} | {scales} | "
                     f"{(group.selected_validation_mae-group.initial_validation_mae).mean():+.4f} |")
    lines += ["", "| Mode | Mean spatial weight movement | Mean last-self movement | Mean ecology movement |",
              "|---|---:|---:|---:|"]
    for arm, group in training.groupby("arm", sort=False):
        lines.append(f"| {arm} | {group.spatial_parameter_distance.mean():.6f} | "
                     f"{group.last_self_parameter_distance.mean():.6f} | {group.ecology_parameter_distance.mean():.6f} |")
    lines += ["", "Movements are Euclidean parameter distances from initialization at the selected",
              "checkpoint. They establish which weights changed, not whether the representation improved.", "",
              f"## {predecessor_title}", "",
              "| Support path | K | Maximum query prediction difference | Mean run absolute difference | Bitwise-equal runs |",
              "|---|---:|---:|---:|---:|"]
    for (shape, k), group in processing.groupby(["shape", "k"], sort=False):
        lines.append(f"| {shape} | {k} | {group.max_absolute_prediction_difference.max():.9g} | "
                     f"{group.mean_absolute_prediction_difference.mean():.9g} | {int(group.bitwise_equal.sum())}/9 |")
    lines += ["", *predecessor_interpretation, ""]
    lines += ["", "The selected scale applies to the raw head output before adding it to context.",
              "Saved full-grid arm_delta columns are unscaled; realized corrections also include",
              "the selected scale and zero concentration floor. Raw head magnitudes are not directly",
              "comparable as applied prediction corrections.", "",
              "## Source-validation ecological mixing", "",
              "| Integrated model | K | Gamma counts |",
              "|---|---:|---|"]
    for (model, k), group in mixing.groupby(["model_name", "k"], sort=False):
        gamma = ", ".join(f"{v:g}: {c}" for v, c in group.gamma.value_counts().sort_index().items())
        lines.append(f"| {model} | {k} | {gamma} |")
    lines += ["", "Mixer and support-adapter choices are fitted independently for each direct head.",
              "At gamma=0 integrated predictions are checked against that head's direct support adapter.", "",
              "## Station gains and harms", "",
              "| GRU-basis contrast | Improved / worsened stations | Top-five positive gain | Top-five harm |",
              "|---|---:|---:|---:|"]
    for row in concentration[concentration.comparison.str.contains("gru_tuned_anchor")
                             & concentration.comparison.str.startswith(("last_self_vs_frozen", "last_self_ecology_vs_frozen", "last_self_ecology_vs_last_self", "last_self_ecology_integrated_vs_prior_ecological"))].itertuples():
        lines.append(f"| {row.comparison} | {row.stations_improved} / {row.stations_worsened} | "
                     f"{100 * row.top5_share_of_positive_station_gain:.1f}% | {100 * row.top5_share_of_station_harm:.1f}% |")
    lines += ["", "Positive gains and harms have separate denominators. Station contributions retain",
              "equal partition weights and combine repeated station identities.", "",
              "## Fixed primary contrasts with tail tradeoffs", ""]
    for k in (0, 5):
        for prefix in ("last_self_vs_frozen", "last_self_ecology_vs_frozen", "last_self_ecology_vs_last_self", "last_self_ecology_vs_prior_concentration",
                       "last_self_ecology_integrated_vs_prior_ecological"):
            name = f"{prefix}_gru_tuned_anchor_k{k}"
            contrast = comparisons[comparisons.comparison.eq(name)]
            row = contrast[contrast.region.eq("overall") & contrast.metric.eq("mae")].iloc[0]
            tail = contrast[contrast.region.eq("q90") & contrast.metric.eq("mae")].iloc[0]
            rest = contrast[contrast.region.eq("nontail") & contrast.metric.eq("mae")].iloc[0]
            false = contrast[contrast.metric.eq("q90_false_positive_rate")].iloc[0]
            lines.append(f"- **{name}:** MAE reduction {row.relative_gain_pct:+.2f}% "
                         f"(95% CI [{row.gain_ci_low_pct:+.2f}, {row.gain_ci_high_pct:+.2f}]); "
                         f"{int(row.improved_splits)}/3 partitions and {int(row.improved_split_seed_pairs)}/9 fits improve. "
                         f"Q90 Delta MAE {tail.delta_value:+.4f} [{tail.delta_ci_low:+.4f}, {tail.delta_ci_high:+.4f}]; "
                         f"non-tail Delta MAE {rest.delta_value:+.4f} [{rest.delta_ci_low:+.4f}, {rest.delta_ci_high:+.4f}]. "
                         f"False Q90 change {100 * false.delta_value:+.3f} percentage points "
                         f"[{100 * false.delta_ci_low:+.3f}, {100 * false.delta_ci_high:+.3f}].")
    lines += ["", "The matched controls test which existing representations benefit from the same residual",
              "objective. Any improvement of the ecology-integrated",
              "product can also involve a changed source-validation mixing weight. It does not alone",
              "establish a better neural representation or a physical ecological/transport mechanism.", ""]
    (out / "findings.md").write_text("\n".join(lines))


def budget_extension_analysis(out, panel, thresholds, states, training, reference_root, expected_epochs, draws):
    """Compare matched raw-input fits against the immutable 30-epoch study."""
    if expected_epochs <= 30:
        raise ValueError("A budget-extension comparison requires an epoch ceiling above 30")
    previous, old_thresholds, sources, old_states = load_panel(reference_root, expected_epochs=30)
    if thresholds != old_thresholds:
        raise ValueError("Source tail thresholds differ across training budgets")
    old_by_run = {(row["split_seed"], row["seed"]): row for row in old_states}
    for current in states:
        old = old_by_run[(current["split_seed"], current["seed"])]
        excluded = {"epochs", "started_at", "runtime_snapshot_hash", "experiment"}
        new_config = {k: v for k, v in current["config"].items() if k not in excluded}
        old_config = {k: v for k, v in old["config"].items() if k not in excluded}
        if new_config != old_config:
            changed = sorted(k for k in new_config.keys() | old_config.keys()
                             if new_config.get(k) != old_config.get(k))
            raise ValueError(f"Training budgets have unmatched settings: {changed}")
    old_training, _, _, _ = selection_records(old_states, previous)
    models = tuple(f"{arm}{suffix}_{shape}" for arm in ARMS for suffix in ("", "_integrated") for shape in SHAPES)
    old_panel = previous[previous.model_name.isin(models)].copy()
    old_panel["model_name"] = "budget30_" + old_panel.model_name
    combined = pd.concat([panel, old_panel], ignore_index=True)
    validate_panel(combined, expected_models=(*MODELS, *(f"budget30_{model}" for model in models)))
    definitions = [(f"{model}_budget{expected_epochs}_vs30_k{k}", model, k, f"budget30_{model}", k,
                    "matched_training_duration") for model in models for k in (0, 5)]
    effects, directions, partitions, stations, global_stations, concentration = compare(
        combined, thresholds, draws, definitions=definitions)
    choices = training.merge(old_training, on=["split_seed", "seed", "arm"], suffixes=("_extended", "_30"),
                             validate="one_to_one")
    for name, frame in (("comparisons", effects), ("directions_by_seed", directions),
                        ("directions_by_partition", partitions), ("station_responses", stations),
                        ("global_station_contributions", global_stations),
                        ("gain_loss_concentration", concentration), ("training_choices", choices)):
        frame.to_csv(out / f"budget_extension_{name}.csv", index=False)
    lines = [f"# Matched {expected_epochs}-versus-30 epoch budget comparison", "",
             "The source-validation traces at the earlier ceiling motivated a longer maximum budget.",
             "The objective, tail weight, learning rates, initialization, early-stopping patience, data,",
             "fixed target queries, source thresholds and candidate modes remain unchanged. Each budget",
             "uses its own source-validation-selected checkpoint, scale and support/ecological mixing.",
             "Thus integrated-product differences include changes in those validation choices as well as",
             "training duration. No model or target threshold is selected by this comparison.", "",
             f"The reference is `{reference_root}`. All nine packages are bound to their saved files.",
             f"The same {draws:,} joint whole-station bootstrap draws and equal-partition estimator are used.",
             "K0 constant and GRU-support predictions coincide; their repeated rows describe the same evidence.",
             "Negative Delta MAE or false-positive-rate change favors the extended budget.", "",
             "## Overall and concentration-specific paired effects", "",
             "| Mode / support path | K | Overall Delta MAE [95% CI] | Reduction [95% CI] | Q90 Delta MAE [95% CI] | Non-tail Delta MAE [95% CI] | False Q90 change, percentage points [95% CI] | Improved partitions; fits |",
             "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for name, model, k, _, _, _ in definitions:
        rows = effects[effects.comparison.eq(name)]
        def pick(region, metric="mae", rows=rows):
            return rows[rows.region.eq(region) & rows.metric.eq(metric)].iloc[0]
        def interval(row, scale=1):
            return f"{scale * row.delta_value:+.4f} [{scale * row.delta_ci_low:+.4f}, {scale * row.delta_ci_high:+.4f}]"
        overall, tail, rest, false = pick("overall"), pick("q90"), pick("nontail"), pick("nontail", "q90_false_positive_rate")
        lines.append(f"| {model} | {k} | {interval(overall)} | "
                     f"{overall.relative_gain_pct:+.2f}% [{overall.gain_ci_low_pct:+.2f}, {overall.gain_ci_high_pct:+.2f}] | "
                     f"{interval(tail)} | {interval(rest)} | {interval(false, 100)} | "
                     f"{int(overall.improved_splits)}/3; {int(overall.improved_split_seed_pairs)}/9 |")
    lines += ["", "The CSV retains raw/log MAE and signed bias for every region, plus partition/seed",
              "directions and station-level gain and harm contributions.", "",
              "## Validation-selected duration", "",
              "| Mode | Selected epoch: 30-budget range | Selected epoch: extended range | Fits selecting epoch >30 | Mean source-validation MAE change, extended minus 30 |",
              "|---|---:|---:|---:|---:|"]
    for arm, group in choices.groupby("arm", sort=False):
        delta = group.selected_validation_mae_extended - group.selected_validation_mae_30
        lines.append(f"| {arm} | {group.best_epoch_30.min()}–{group.best_epoch_30.max()} | "
                     f"{group.best_epoch_extended.min()}–{group.best_epoch_extended.max()} | "
                     f"{int((group.best_epoch_extended > 30).sum())}/9 | {delta.mean():+.5f} |")
    lines += ["", "A fit may stop before its maximum budget. Longer ceilings alone do not guarantee longer",
              "selected training. Same-cohort target results remain development evidence.", ""]
    (out / "budget_extension_findings.md").write_text("\n".join(lines))
    return sources, len(definitions)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--bootstrap-draws", type=int, default=5000)
    parser.add_argument("--expected-epochs", type=int, default=30)
    parser.add_argument("--budget-reference-root", type=Path,
                        help="Optional fixed 30-epoch encoder study for paired duration comparisons")
    args = parser.parse_args()
    if args.bootstrap_draws < 1 or args.expected_epochs < 1:
        raise ValueError("Bootstrap draws and expected epochs must be positive")
    panel, thresholds, sources, states = load_panel(args.root, expected_epochs=args.expected_epochs)
    runs, splits, curves = summarize_metrics(panel, thresholds)
    profile_runs, profile_splits, profiles = error_profiles(panel, thresholds)
    comparisons, directions, partitions, stations, global_stations, concentration = compare(
        panel, thresholds, args.bootstrap_draws)
    training, choices, mixing, gamma_scores = selection_records(states, panel)
    processing = processing_differences(panel)
    out = args.root / "analysis"
    out.mkdir(parents=True, exist_ok=True)
    for name, frame in (("metrics_by_run", runs), ("metrics_by_partition", splits), ("k_curves", curves),
                        ("error_profiles_by_run", profile_runs), ("error_profiles_by_partition", profile_splits),
                        ("error_profiles", profiles), ("comparisons", comparisons),
                        ("directions_by_seed", directions), ("directions_by_partition", partitions),
                        ("station_responses", stations), ("global_station_contributions", global_stations),
                        ("gain_loss_concentration", concentration), ("training_choices", training),
                        ("adapter_choices", choices), ("mixing_choices", mixing), ("gamma_scores", gamma_scores),
                        ("frozen_processing_differences", processing)):
        frame.to_csv(out / f"{name}.csv", index=False)
    write_report(out, curves, profiles, comparisons, concentration, training, mixing, processing,
                 args.bootstrap_draws, expected_epochs=args.expected_epochs)
    n_budget_comparisons = 0
    if args.budget_reference_root is not None:
        budget_sources, n_budget_comparisons = budget_extension_analysis(
            out, panel, thresholds, states, training, args.budget_reference_root,
            args.expected_epochs, args.bootstrap_draws)
        sources.extend(budget_sources)
    sources.extend({"path": str(path), "sha256": sha256(path)} for path in (
        Path(__file__), Path("scripts/analyze_doc_tail_residual_v1.py"),
        Path("scripts/analyze_unified_doc_spatial.py"), Path("scripts/analyze_unified_doc_spatial_v2.py"),
        Path("scripts/analyze_unified_doc_spatial_v3.py")))
    unique = {row["path"]: row for row in sources}
    (out / "sources.json").write_text(json.dumps(list(unique.values()), indent=2) + "\n")
    status = {"complete": True, "n_runs": len(states), "n_reused_memory_profiles": len(states), "n_new_memory_fits": 0, "n_new_neural_fits": len(training),
              "n_models": len(MODELS), "n_comparisons": len(comparison_definitions()),
              "epoch_ceiling": args.expected_epochs, "n_budget_comparisons": n_budget_comparisons,
              "budget_reference_root": str(args.budget_reference_root) if args.budget_reference_root is not None else None,
              "bootstrap_draws": args.bootstrap_draws, "bootstrap_unit": "joint whole-station identity",
              "study_role": "development on previously evaluated station partitions", "model_selected": False,
              "sidecars_verified": True, "historical_references_identical": True,
              "gamma_zero_exact_direct": True, "k0_support_shapes_identical": True,
              "memory_profiles_reused": True, "joint_choices_match_source_validation": True,
              "frozen_encoder_weights_unchanged": True, "frozen_vs_prior_processing_reported": True,
              "analysis_inputs": len(unique)}
    (out / "status.json").write_text(json.dumps(status, indent=2) + "\n")
    print(json.dumps({**status, "output": str(out)}, indent=2))


if __name__ == "__main__":
    main()
