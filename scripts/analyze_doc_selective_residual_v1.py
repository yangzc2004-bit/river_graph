"""Compare four fixed loss definitions for the existing DOC residual encoder.

All arms and references are reported. Analysis does not select a model, alter
predictions or use target outcomes to choose scales, support fits or mixtures.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from analyze_doc_encoder_residual_v1 import check_source_identity, compare
from analyze_doc_tail_residual_v1 import error_profiles, read_predictions
from analyze_unified_doc_spatial import joint_station_bootstrap, paired_cells, sha256
from analyze_unified_doc_spatial_v2 import KS, SEEDS, SPLITS, validate_panel
from analyze_unified_doc_spatial_v3 import read_bound_json, summarize_metrics

ROOT = Path("experiments/phase4_transfer/doc_selective_residual_v1")
LOSS_SPECS = {
    "tail2": {"tail_weight": 2, "source_weighting": "cell", "ordinary_overprediction_penalty": 0.0},
    "mae": {"tail_weight": 1, "source_weighting": "cell", "ordinary_overprediction_penalty": 0.0},
    "selective": {"tail_weight": 2, "source_weighting": "cell", "ordinary_overprediction_penalty": 0.5},
    "station": {"tail_weight": 2, "source_weighting": "station", "ordinary_overprediction_penalty": 0.0},
}
ARMS = tuple(LOSS_SPECS)
SHAPES = ("constant", "gru_tuned_anchor")
REFERENCE_BASES = ("prior_encoder", "prior_encoder_integrated", "prior_ecological_affine")
REFERENCES = tuple(f"{base}_{shape}" for base in REFERENCE_BASES for shape in SHAPES)
DIRECT_MODELS = tuple(f"{base}_{shape}" for base in ("context", *ARMS) for shape in SHAPES)
MIXED_MODELS = tuple(f"{arm}_integrated_{shape}" for arm in ARMS for shape in SHAPES)
MODELS = DIRECT_MODELS + MIXED_MODELS + REFERENCES
INTERACTIONS = [0, 2, 4, 28]


def comparison_definitions():
    rows = []
    for shape in SHAPES:
        for k in ((5,) if shape == "constant" else (0, 5)):
            for suffix in ("", "_integrated"):
                for candidate in ("mae", "selective", "station"):
                    rows.append((f"{candidate}_vs_tail2{suffix}_{shape}_k{k}", f"{candidate}{suffix}_{shape}", k,
                                 f"tail2{suffix}_{shape}", k, "matched_loss_definition"))
                rows.append((f"station_vs_selective{suffix}_{shape}_k{k}", f"station{suffix}_{shape}", k,
                             f"selective{suffix}_{shape}", k, "station_vs_ordinary_pressure"))
            for arm in ARMS:
                rows.append((f"{arm}_vs_context_{shape}_k{k}", f"{arm}_{shape}", k,
                             f"context_{shape}", k, "fixed_context_control"))
                rows.append((f"{arm}_integrated_vs_direct_{shape}_k{k}", f"{arm}_integrated_{shape}", k,
                             f"{arm}_{shape}", k, "ecological_integration"))
                rows.append((f"{arm}_integrated_vs_prior_ecological_{shape}_k{k}", f"{arm}_integrated_{shape}", k,
                             f"prior_ecological_affine_{shape}", k, "previous_overall_reference"))
    return rows


def check_training_prefix(current, previous):
    """Require exact shared optimization; completed early stops cannot change."""
    if current["config"]["epochs"] < previous["config"]["epochs"]:
        raise ValueError("Current maximum budget must cover the reference budget")
    old_trace = previous["trace"]
    if current["trace"][:len(old_trace)] != old_trace:
        raise ValueError("Shared training trace differs across matched budgets")
    early_stopped = old_trace[-1]["stale_epochs"] >= previous["config"]["patience"]
    require_identical = (current["config"]["epochs"] == previous["config"]["epochs"] or early_stopped)
    if require_identical and current["trace"] != old_trace:
        raise ValueError("An unchanged early-stopping decision unexpectedly continued training")
    return require_identical


def load_panel(root, expected_epochs=60):
    frames, thresholds, sources, states, controls = [], {}, [], [], []
    for split in SPLITS:
        for seed in SEEDS:
            run = root / "runs" / f"split{split}_seed{seed}"
            frame, config, completion = read_predictions(run, sources)
            if ((config["split_seed"], config["seed"]) != (split, seed)
                    or set(config["models"]) != set(MODELS) or tuple(config["arms"]) != ARMS
                    or tuple(config["k_values"]) != KS or config["inference_roles"] != ["train"]
                    or config["loss_specs"] != LOSS_SPECS or config["extra_dim"] != 30
                    or config["epochs"] != expected_epochs or config["patience"] != 5
                    or config["interaction_indices"] != INTERACTIONS or config["train_memory"] is not True
                    or config["encoder_learning_rate"] != 1e-5 or config["encoder_mode"] != "last_self_ecology"
                    or config["gamma_grid"] != [0, .25, .5, 1]
                    or config["alpha_grid"] != [0, .25, .5, .75, 1]
                    or config["ridge_grid"] != [.1, 1, 10, "infinity"]):
                raise ValueError(f"Unexpected selective experiment settings: {run}")
            threshold = float(config["q90_threshold_train"])
            if not np.isfinite(threshold) or threshold < 0:
                raise ValueError("Invalid source-training Q90")
            thresholds[(split, seed)] = threshold
            prior, memory_run = Path(config["prior_run"]), Path(config["memory_run"])
            for name, path in (("prior", prior), ("memory", memory_run)):
                if sha256(path / "complete.json") != config[f"{name}_completion_hash"]:
                    raise ValueError(f"Changed bound {name} package")
            previous, old_config, old_completion = read_predictions(prior, sources)
            check_source_identity(config, old_config, (
                "split_seed", "seed", "dataset_hash", "mask_hash", "q90_threshold_train",
                "source_run", "basis_run", "oof_run", "memory_run", "memory_file_hash",
                "feature_definition", "interaction_indices", "patience"))
            mapping = {f"prior_encoder_{shape}": f"last_self_ecology_{shape}" for shape in SHAPES}
            mapping.update({f"prior_encoder_integrated_{shape}": f"last_self_ecology_integrated_{shape}"
                            for shape in SHAPES})
            mapping.update({f"prior_ecological_affine_{shape}": f"prior_ecological_affine_{shape}"
                            for shape in SHAPES})
            mapping.update({f"context_{shape}": f"context_{shape}" for shape in SHAPES})
            for current_name, old_name in mapping.items():
                current = frame[frame.model_name.eq(current_name)].sort_values(["k", "cell"])
                old = previous[previous.model_name.eq(old_name)].sort_values(["k", "cell"])
                for field in ("cell", "y_true", "y_pred"):
                    np.testing.assert_array_equal(current[field], old[field])
            memory_completion = json.loads((memory_run / "complete.json").read_text())
            memory = read_bound_json(memory_run, "ecological_affine.json", memory_completion, sources)
            if (sha256(memory_run / "ecological_affine.json") != config["memory_file_hash"]
                    or memory["mode"] != "ecological_affine" or memory["selection_role"] != "source_validation"):
                raise ValueError("Changed ecological memory profile")
            features = read_bound_json(run, "feature_definition.json", completion, sources)
            if (features != config["feature_definition"] or len(features["feature_names"]) != 30
                    or features["interaction_indices"]["concentration"] != INTERACTIONS):
                raise ValueError("Changed readout features")
            training = {arm: read_bound_json(run, f"{arm}.json", completion, sources) for arm in ARMS}
            old_training = read_bound_json(prior, "last_self_ecology.json", old_completion, sources)
            require_identical = check_training_prefix(training["tail2"], old_training)
            for suffix in ("", "_integrated"):
                for shape in SHAPES:
                    for k in KS:
                        current = frame[frame.model_name.eq(f"tail2{suffix}_{shape}") & frame.k.eq(k)].sort_values("cell")
                        old = previous[previous.model_name.eq(f"last_self_ecology{suffix}_{shape}")
                                       & previous.k.eq(k)].sort_values("cell")
                        np.testing.assert_array_equal(current.cell, old.cell)
                        bitwise_equal = np.array_equal(current.y_pred, old.y_pred)
                        if require_identical:
                            np.testing.assert_array_equal(current.y_pred, old.y_pred)
                        controls.append({"split_seed": split, "seed": seed, "integrated": bool(suffix),
                                         "shape": shape, "k": k, "query_cells": len(current),
                                         "predictions_bitwise_equal": bitwise_equal,
                                         "prediction_identity_required": require_identical,
                                         "training_trace_exact": training["tail2"]["trace"] == old_training["trace"],
                                         "training_prefix_exact": True,
                                         "reference_epoch_cap": old_training["config"]["epochs"],
                                         "current_epoch_cap": expected_epochs,
                                         "max_absolute_prediction_difference": float(np.max(np.abs(
                                             current.y_pred.to_numpy()-old.y_pred.to_numpy())))})
            states.append({"split_seed": split, "seed": seed, "config": config, "training": training,
                           "adapters": read_bound_json(run, "adapters.json", completion, sources),
                           "mixers": read_bound_json(run, "mixers.json", completion, sources)})
            frames.append(frame)
        if len({thresholds[(split, seed)] for seed in SEEDS}) != 1:
            raise ValueError("Source Q90 differs across training seeds")
    panel = pd.concat(frames, ignore_index=True)
    validate_panel(panel, expected_models=MODELS)
    for _, group in panel[panel.k.eq(0)].groupby(["split_seed", "seed"]):
        pivot = group.pivot(index="cell", columns="model_name", values="y_pred")
        for base in ("context", *ARMS, *(f"{arm}_integrated" for arm in ARMS), *REFERENCE_BASES):
            np.testing.assert_array_equal(pivot[f"{base}_constant"], pivot[f"{base}_gru_tuned_anchor"])
    return panel, thresholds, sources, states, pd.DataFrame(controls)


def selection_records(states, panel):
    training, choices, mixing, scores = [], [], [], []
    for run in states:
        identity = {"split_seed": run["split_seed"], "seed": run["seed"]}
        current = panel[panel.split_seed.eq(run["split_seed"]) & panel.seed.eq(run["seed"])]
        if set(run["adapters"]) != set(DIRECT_MODELS) or set(run["mixers"]) != set(MIXED_MODELS):
            raise ValueError("Unexpected adapter/mixer names")
        for model, state in run["adapters"].items():
            if state["selection_role"] != "source_validation" or set(map(int, state["selection_by_k"])) != set(KS):
                raise ValueError("Wrong support selection role or K")
            choices.extend({**identity, "model_name": model, "k": int(k), **selection}
                           for k, selection in state["selection_by_k"].items())
        for arm, state in run["training"].items():
            cfg, trace = state["config"], state["trace"]
            chosen = [r for r in trace if r["epoch"] == state["best_epoch"]]
            if (any(cfg[key] != value for key, value in LOSS_SPECS[arm].items())
                    or cfg["encoder_mode"] != "last_self_ecology" or cfg["encoder_learning_rate"] != 1e-5
                    or cfg["extra_dim"] != 30 or cfg["interaction_indices"] != INTERACTIONS
                    or cfg["epochs"] != run["config"]["epochs"] or cfg["patience"] != 5 or not cfg.get("train_memory", True)
                    or state["protocol"]["selection_role"] != "source_validation"
                    or len(chosen) != 1 or chosen[0]["validation_scale"] != state["selected_scale"]):
                raise ValueError("Unexpected saved loss/encoder training definition")
            training.append({**identity, "arm": arm, **LOSS_SPECS[arm],
                             "epoch_ceiling": cfg["epochs"],
                             "best_epoch": state["best_epoch"], "epochs_run": state["epochs_run"],
                             "selected_scale": state["selected_scale"],
                             "initial_validation_mae": trace[0]["validation_mae"],
                             "selected_validation_mae": chosen[0]["validation_mae"],
                             **{key: state[key] for key in ("trainable_parameter_count", "n_source_cells",
                                 "n_source_tail_cells", "n_validation_query", "temporal_parameter_distance",
                                 "decay_parameter_distance", "head_parameter_norm", "encoder_trainable_parameter_count",
                                 "spatial_parameter_distance", "last_self_parameter_distance", "ecology_parameter_distance")}})
        for model, state in run["mixers"].items():
            if state["selection_role"] != "source_validation" or set(map(int, state["selection_by_k"])) != set(KS):
                raise ValueError("Wrong mixture selection role or K")
            gamma_k0 = state["gamma_k0"]
            initial = [r for r in state["gamma_scores"] if r["k"] == 0]
            if gamma_k0 != min(initial, key=lambda r: (r["mae"], r["gamma"]))["gamma"]:
                raise ValueError("K0 gamma differs from saved validation selection")
            for k in KS:
                selected = state["selection_by_k"][str(k)]
                candidates = [r for r in state["gamma_scores"] if r["k"] == k]
                eligible = [r for r in candidates if k != 0 or r["gamma"] == gamma_k0]
                chosen = min(eligible, key=lambda r: (r["mae"], r["gamma"] != 0,
                                                      r["gamma"] != gamma_k0, r["gamma"]))
                if selected != {**chosen, "locked": k == 0}:
                    raise ValueError("Mixture choice differs from source-validation trace")
                if selected["gamma"] == 0:
                    direct = model.replace("_integrated_", "_")
                    a = current[current.model_name.eq(model) & current.k.eq(k)].sort_values("cell")
                    b = current[current.model_name.eq(direct) & current.k.eq(k)].sort_values("cell")
                    np.testing.assert_array_equal(a.y_pred, b.y_pred)
                mixing.append({**identity, "model_name": model, "gamma_k0": gamma_k0, **selected})
                scores.extend({**identity, "model_name": model, **row} for row in candidates)
    return tuple(pd.DataFrame(rows) for rows in (training, choices, mixing, scores))


def classification_metrics(panel, thresholds):
    rows = []
    for (split, seed, model, k), group in panel.groupby(["split_seed", "seed", "model_name", "k"]):
        threshold = thresholds[(int(split), int(seed))]
        actual = group.y_true.to_numpy() >= threshold
        predicted = group.y_pred.to_numpy() >= threshold
        tp, fp = int((actual & predicted).sum()), int((~actual & predicted).sum())
        rows.append({"split_seed": split, "seed": seed, "model_name": model, "k": k,
                     "n_actual_q90": int(actual.sum()), "n_actual_nontail": int((~actual).sum()),
                     "n_predicted_q90": int(predicted.sum()), "true_q90": tp, "false_q90": fp,
                     "unstable": int(actual.sum()) < 20,
                     "q90_recall": tp / int(actual.sum()) if actual.any() else np.nan,
                     "q90_precision": tp / int(predicted.sum()) if predicted.any() else np.nan,
                     "q90_false_positive_rate": fp / int((~actual).sum()) if (~actual).any() else np.nan})
    runs = pd.DataFrame(rows)
    rates = ["q90_recall", "q90_precision", "q90_false_positive_rate"]
    splits = runs.groupby(["split_seed", "model_name", "k"], as_index=False).agg(
        **{name: (name, "mean") for name in rates}, n_actual_q90=("n_actual_q90", "first"),
        n_actual_nontail=("n_actual_nontail", "first"), unstable=("unstable", "any"),
        n_seeds_precision_defined=("q90_precision", "count"))
    summary = splits.groupby(["model_name", "k"], as_index=False).agg(
        **{name: (name, "mean") for name in rates}, unstable_any_split=("unstable", "any"),
        n_splits_recall_defined=("q90_recall", "count"), n_splits_precision_defined=("q90_precision", "count"))
    return runs, splits, summary


def recall_comparisons(panel, thresholds, draws, definitions):
    rows = []
    for name, candidate, ck, reference, rk, role in definitions:
        pairs = paired_cells(panel, candidate, ck, reference, rk)
        threshold = np.array([thresholds[(int(split), int(seed))]
                              for split, seed in zip(pairs.split_seed, pairs.seed, strict=True)])
        tail = pairs.y_true_candidate.to_numpy() >= threshold
        selected = pairs.loc[tail].copy()
        counts = selected.drop_duplicates(["split_seed", "cell"]).groupby("split_seed").size()
        identity = {"comparison": name, "comparison_role": role, "candidate": candidate,
                    "candidate_k": ck, "reference": reference, "reference_k": rk,
                    "region": "q90", "metric": "q90_recall", "n_nonempty_splits": len(counts),
                    "unstable_any_split": len(counts) < len(SPLITS) or bool((counts < 20).any())}
        if selected.split_seed.nunique() != len(SPLITS):
            rows.append({**identity, "status": "missing_partition"})
            continue
        for side in ("candidate", "reference"):
            selected[f"{side}_error"] = (selected[f"y_pred_{side}"].to_numpy() >= threshold[tail]).astype(float)
        result = joint_station_bootstrap(selected, draws=draws)
        result = {key.replace("_mae", "_value"): value for key, value in result.items()}
        for key in ("relative_gain_pct", "gain_ci_low_pct", "gain_ci_high_pct"):
            result[key] = np.nan
        rows.append({**identity, "status": "estimated", **result})
    return pd.DataFrame(rows)


def write_report(out, curves, profiles, classification, effects, training, mixing, controls, draws,
                 expected_epochs=60):
    lines = ["# Selectivity and station balance in the existing DOC residual", "",
             "Four source-training objectives share the same partially trainable encoder, original",
             f"initialization, concentration head, source OOF context base, {expected_epochs}-epoch cap and patience 5.",
             "All use the same uniform cell shuffle; only source loss weighting/asymmetry changes.", "",
             "- tail2: current cell-weighted MAE, with weight two at or above source Q90.",
             "- mae: ordinary cell-weighted MAE, with no extra tail weight.",
             "- selective: tail2 plus 0.5 times the positive prediction error on ordinary source cells.",
             "- station: tail weights normalized to unit total within each source station, then equal station averaging.", "",
             "The fixed full-source mean weight normalizes each minibatch. Checkpoint and residual scale",
             "still use unweighted pooled source-validation MAE. Support and ecological mixing are fitted",
             "separately for each arm using the same validation episodes, candidate grids and frozen support basis.",
             "No combined selective/station objective is fitted. The no-message self path remains in use.", "",
             "All 24 saved models and all K values are retained. Tail2 directly and after ecological",
             "integration reproduces the preceding encoder wherever the prior stopping decision",
             "was already complete. Shared training prefixes must always match exactly; a changed",
             "prediction is allowed only when extra epochs can follow a previously binding ceiling.",
             "Prior encoder and ecological-affine reference products themselves are copied exactly.",
             f"The control checks contain {len(controls)} support-path/K/run comparisons, with",
             f"{int(controls.predictions_bitwise_equal.sum())} bitwise-identical predictions and",
             f"{int(controls.prediction_identity_required.sum())} required identical comparisons.", "",
             f"Intervals use {draws:,} paired whole-station bootstrap draws, jointly resampling repeated",
             "station identities across partitions. Seed means are averaged within partition; partitions",
             "have equal weight. Negative error or false-positive deltas favor the candidate; positive",
             "recall deltas indicate improved detection. Q90 uses source-training labels and includes threshold ties.",
             "These are previously examined development partitions, and source validation has been reused.",
             "No model, threshold, support count or mixture is selected using these target comparisons.", "",
             "## Complete K curves", "",
             "| Model (GRU support basis) | K0 MAE | K1 MAE | K3 MAE | K5 MAE | K5 RMSE | K5 R² |",
             "|---|---:|---:|---:|---:|---:|---:|"]
    for model in MODELS:
        if not model.endswith("gru_tuned_anchor"):
            continue
        selected = curves[curves.model_name.eq(model)].set_index("k")
        lines.append(f"| {model} | " + " | ".join(f"{selected.loc[k, 'mae']:.4f}" for k in KS)
                     + f" | {selected.loc[5, 'rmse']:.4f} | {selected.loc[5, 'r2']:.4f} |")
    lines += ["", "Constant-only support adaptation is retained in all CSVs. Its K0 predictions duplicate",
              "the GRU-basis K0 predictions, so duplicate K0 bootstrap comparisons are omitted.", "",
              "## Tail accuracy, ordinary error and classification", "",
              "| Model | K | Q90 MAE | Q90 bias | Ordinary MAE | Ordinary bias | Q90 recall | False-Q90 rate |",
              "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for model in MODELS:
        if not model.endswith("gru_tuned_anchor"):
            continue
        for k in (0, 5):
            profile = profiles[profiles.model_name.eq(model) & profiles.k.eq(k)].set_index("region")
            rates = classification[classification.model_name.eq(model) & classification.k.eq(k)].iloc[0]
            tail, ordinary = profile.loc["q90"], profile.loc["nontail"]
            lines.append(f"| {model} | {k} | {tail.mae:.4f} | {tail.signed_bias:+.4f} | {ordinary.mae:.4f} | "
                         f"{ordinary.signed_bias:+.4f} | {100*rates.q90_recall:.2f}% | {100*rates.q90_false_positive_rate:.2f}% |")
    lines += ["", "Recall conditions on true high DOC; false-Q90 rate conditions on ordinary observations.",
              "A lower false-positive rate alone is not better selectivity if recall also collapses.", ""]
    for role, title in (("matched_loss_definition", "Matched loss contrasts"),
                        ("station_vs_ordinary_pressure", "Station balance versus ordinary overprediction pressure"),
                        ("previous_overall_reference", "Integrated models versus the fixed ecological reference"),
                        ("ecological_integration", "Ecological integration versus direct residuals"),
                        ("fixed_context_control", "Direct residuals versus context")):
        lines += [f"## {title}", "", "| Comparison | Overall Delta MAE [95% CI] | Q90 Delta MAE | Ordinary Delta MAE | Recall change [95% CI], pp | False-Q90 change [95% CI], pp |",
                  "|---|---:|---:|---:|---:|---:|"]
        selected = effects[effects.comparison_role.eq(role) & effects.region.eq("overall") & effects.metric.eq("mae")]
        for row in selected.itertuples():
            group = effects[effects.comparison.eq(row.comparison)]

            def pick(region, metric, group=group):
                return group[group.region.eq(region) & group.metric.eq(metric)].iloc[0]

            def interval(value, multiplier=1):
                if value.status != "estimated":
                    return "not estimable"
                return (f"{multiplier*value.delta_value:+.4f} "
                        f"[{multiplier*value.delta_ci_low:+.4f}, {multiplier*value.delta_ci_high:+.4f}]")

            tail, ordinary = pick("q90", "mae"), pick("nontail", "mae")
            recall, false = pick("q90", "q90_recall"), pick("nontail", "q90_false_positive_rate")
            lines.append(f"| {row.comparison} | {interval(row)} | {interval(tail)} | {interval(ordinary)} | "
                         f"{interval(recall, 100)} | {interval(false, 100)} |")
        lines.append("")
    lines += ["## Source-validation choices", "",
              "| Loss arm | Selected epoch range | Total epochs executed | Scale counts | Mean selected validation MAE |",
              "|---|---:|---:|---|---:|"]
    for arm, group in training.groupby("arm", sort=False):
        scales = ", ".join(f"{scale:g}: {count}" for scale, count in group.selected_scale.value_counts().sort_index().items())
        lines.append(f"| {arm} | {group.best_epoch.min()}–{group.best_epoch.max()} | {group.epochs_run.sum()} | {scales} | {group.selected_validation_mae.mean():.6f} |")
    lines += ["", "Training loss levels differ across objectives and should not be treated as directly",
              "comparable performance scores. Encoder/GRU parameter movements and support choices are",
              "saved in the training and adapter CSVs.", "",
              "| Integrated model | K | Selected gamma counts |", "|---|---:|---|"]
    for (model, k), group in mixing.groupby(["model_name", "k"], sort=False):
        counts = ", ".join(f"{value:g}: {count}" for value, count in group.gamma.value_counts().sort_index().items())
        lines.append(f"| {model} | {k} | {counts} |")
    lines += ["", "Interpret direct loss contrasts before integrated ones: the latter also include changes",
              "in source-validation-selected ecological mixing and positive-K support calibration.",
              "A narrow or sign-changing contrast does not establish equivalence. Station gain/harm",
              "concentration and partition/seed directions are retained alongside the overall means.", ""]
    (out / "findings.md").write_text("\n".join(lines))


def budget_comparison_definitions(extended_epochs, reference_epochs):
    return [(f"{arm}{suffix}_{shape}_budget{extended_epochs}_vs{reference_epochs}_k{k}",
             f"{arm}{suffix}_{shape}", k, f"budget{reference_epochs}_{arm}{suffix}_{shape}", k,
             "matched_training_duration")
            for arm in ARMS for suffix in ("", "_integrated") for shape in SHAPES
            for k in ((5,) if shape == "constant" else (0, 5))]


def budget_extension_analysis(out, panel, thresholds, states, training, reference_root,
                              expected_epochs, reference_epochs, draws):
    """Report all duration contrasts without selecting a model from outcomes."""
    if expected_epochs <= reference_epochs:
        raise ValueError("Extended budget must exceed the reference maximum")
    previous, old_thresholds, sources, old_states, _ = load_panel(reference_root, reference_epochs)
    if thresholds != old_thresholds:
        raise ValueError("Source-derived Q90 differs across budgets")
    old_by_run = {(state["split_seed"], state["seed"]): state for state in old_states}
    prefixes = []
    for current in states:
        old = old_by_run[(current["split_seed"], current["seed"])]
        excluded = {"epochs", "started_at", "runtime_snapshot_hash", "experiment"}
        new_config = {key: value for key, value in current["config"].items() if key not in excluded}
        old_config = {key: value for key, value in old["config"].items() if key not in excluded}
        if new_config != old_config:
            changed = sorted(key for key in new_config.keys() | old_config.keys()
                             if new_config.get(key) != old_config.get(key))
            raise ValueError(f"Unmatched duration-comparison settings: {changed}")
        for arm in ARMS:
            now, before = current["training"][arm], old["training"][arm]
            require_identical = check_training_prefix(now, before)
            prefixes.append({"split_seed": current["split_seed"], "seed": current["seed"], "arm": arm,
                             "prefix_exact": True, "reference_trace_epochs": before["epochs_run"],
                             "extended_epochs_run": now["epochs_run"],
                             "reference_already_stopped": require_identical,
                             "complete_trace_exact": now["trace"] == before["trace"]})
    old_training, _, _, _ = selection_records(old_states, previous)
    models = tuple(f"{arm}{suffix}_{shape}" for arm in ARMS for suffix in ("", "_integrated") for shape in SHAPES)
    old_panel = previous[previous.model_name.isin(models)].copy()
    old_panel["model_name"] = f"budget{reference_epochs}_" + old_panel.model_name
    combined = pd.concat([panel, old_panel], ignore_index=True)
    validate_panel(combined, expected_models=(*MODELS, *(f"budget{reference_epochs}_{model}" for model in models)))
    definitions = budget_comparison_definitions(expected_epochs, reference_epochs)
    effects, directions, partitions, stations, global_stations, concentration = compare(
        combined, thresholds, draws, definitions=definitions)
    effects = pd.concat([effects, recall_comparisons(combined, thresholds, draws, definitions)], ignore_index=True)
    choices = training.merge(old_training, on=["split_seed", "seed", "arm"], suffixes=("_extended", "_reference"),
                             validate="one_to_one")
    products = (("comparisons", effects), ("directions_by_seed", directions),
                ("directions_by_partition", partitions), ("station_responses", stations),
                ("global_station_contributions", global_stations), ("gain_loss_concentration", concentration),
                ("training_choices", choices), ("training_prefix_checks", pd.DataFrame(prefixes)))
    output_hashes = {}
    for name, frame in products:
        path = out / f"budget_extension_{name}.csv"
        frame.to_csv(path, index=False)
        output_hashes[path.name] = sha256(path)
    lines = [f"# Matched {expected_epochs}-versus-{reference_epochs} epoch budget comparison", "",
             "The duration extension was chosen from source-validation stopping trajectories before",
             "examining the target results of either loss panel. The four loss definitions, patience,",
             "initialization, learning rates, data, thresholds and fixed query populations are unchanged.",
             "Each budget retains its own validation-selected checkpoint, residual scale, ecological",
             "mixture and support adapter. Every arm and both positive and negative effects are shown.", "",
             f"All {len(prefixes)} shared training prefixes reproduce exactly. Prior runs whose patience",
             "was already exhausted must reproduce their entire trajectory. Reference encoder and",
             "ecological-affine products remain the same historical products under both budgets.", "",
             f"The reference root is `{reference_root}`. Intervals use the same {draws:,} joint station",
             "bootstrap draws and equal-partition estimator as the main analysis. Repeated seeds",
             "predict the same ecological samples. K0 constant and GRU-support predictions coincide,",
             "so the duplicate constant K0 contrast is omitted. Recall gains must be read with false",
             "alarms and ordinary/tail error rather than used to choose a preferred budget post hoc.", "",
             "## Fixed paired duration contrasts", "",
             "| Model / support path | K | Overall Delta MAE [95% CI] | Q90 Delta MAE [95% CI] | Ordinary Delta MAE [95% CI] | Recall change [95% CI], pp | False-Q90 change [95% CI], pp |",
             "|---|---:|---:|---:|---:|---:|---:|"]
    for name, model, k, _, _, _ in definitions:
        group = effects[effects.comparison.eq(name)]

        def pick(region, metric, group=group):
            return group[group.region.eq(region) & group.metric.eq(metric)].iloc[0]

        def interval(row, scale=1):
            if row.status != "estimated":
                return "not estimable"
            return f"{scale*row.delta_value:+.4f} [{scale*row.delta_ci_low:+.4f}, {scale*row.delta_ci_high:+.4f}]"

        overall, tail, ordinary = pick("overall", "mae"), pick("q90", "mae"), pick("nontail", "mae")
        recall, false = pick("q90", "q90_recall"), pick("nontail", "q90_false_positive_rate")
        lines.append(f"| {model} | {k} | {interval(overall)} | {interval(tail)} | {interval(ordinary)} | "
                     f"{interval(recall, 100)} | {interval(false, 100)} |")
    lines += ["", "Negative error/FPR deltas favor the extended budget; positive recall deltas mean",
              "more detected high observations. Raw/log error and bias comparisons, seed/partition",
              "directions, and station gain/harm concentrations are retained in the CSVs.", "",
              "## Source-validation duration and selection", "",
              "| Arm | Reference total epochs | Extended total epochs | Reference best epoch range | Extended best epoch range | Fits selecting beyond reference cap | Mean validation MAE change |",
              "|---|---:|---:|---:|---:|---:|---:|"]
    for arm, group in choices.groupby("arm", sort=False):
        change = group.selected_validation_mae_extended-group.selected_validation_mae_reference
        lines.append(f"| {arm} | {group.epochs_run_reference.sum()} | {group.epochs_run_extended.sum()} | "
                     f"{group.best_epoch_reference.min()}–{group.best_epoch_reference.max()} | "
                     f"{group.best_epoch_extended.min()}–{group.best_epoch_extended.max()} | "
                     f"{int((group.best_epoch_extended>reference_epochs).sum())}/9 | {change.mean():+.6f} |")
    lines += ["", "A higher cap does not force a longer fit or a later selected checkpoint. Improvements",
              "in integrated products can include changed validation mixture/adapter choices; they",
              "cannot be attributed entirely to additional neural optimization. This is a model",
              "development comparison on the same previously examined station partitions.", ""]
    report_path = out / "budget_extension_findings.md"
    report_path.write_text("\n".join(lines))
    output_hashes[report_path.name] = sha256(report_path)
    return sources, output_hashes, len(definitions)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--bootstrap-draws", type=int, default=5000)
    parser.add_argument("--expected-epochs", type=int, default=60)
    parser.add_argument("--budget-reference-root", type=Path,
                        help="Optional matched lower-budget loss panel for paired duration analysis")
    parser.add_argument("--reference-epochs", type=int, default=60)
    args = parser.parse_args()
    if args.bootstrap_draws < 1 or args.expected_epochs < 1 or args.reference_epochs < 1:
        raise ValueError("Bootstrap draws and epoch ceilings must be positive")
    panel, thresholds, sources, states, controls = load_panel(args.root, args.expected_epochs)
    runs, splits, curves = summarize_metrics(panel, thresholds)
    profile_runs, profile_splits, profiles = error_profiles(panel, thresholds)
    class_runs, class_splits, classification = classification_metrics(panel, thresholds)
    definitions = comparison_definitions()
    effects, directions, partitions, stations, global_stations, concentration = compare(
        panel, thresholds, args.bootstrap_draws, definitions=definitions)
    recalls = recall_comparisons(panel, thresholds, args.bootstrap_draws, definitions)
    effects = pd.concat([effects, recalls], ignore_index=True)
    training, choices, mixing, gamma_scores = selection_records(states, panel)
    out = args.root / "analysis"
    out.mkdir(parents=True, exist_ok=True)
    outputs = (("metrics_by_run", runs), ("metrics_by_partition", splits), ("k_curves", curves),
               ("error_profiles_by_run", profile_runs), ("error_profiles_by_partition", profile_splits),
               ("error_profiles", profiles), ("classification_by_run", class_runs),
               ("classification_by_partition", class_splits), ("classification", classification),
               ("comparisons", effects), ("directions_by_seed", directions),
               ("directions_by_partition", partitions), ("station_responses", stations),
               ("global_station_contributions", global_stations), ("gain_loss_concentration", concentration),
               ("training_choices", training), ("adapter_choices", choices),
               ("mixing_choices", mixing), ("gamma_scores", gamma_scores), ("tail2_control_replication", controls))
    for name, frame in outputs:
        frame.to_csv(out / f"{name}.csv", index=False)
    write_report(out, curves, profiles, classification, effects, training, mixing, controls,
                 args.bootstrap_draws, args.expected_epochs)
    output_hashes = {f"{name}.csv": sha256(out / f"{name}.csv") for name, _ in outputs}
    output_hashes["findings.md"] = sha256(out / "findings.md")
    budget_count = 0
    if args.budget_reference_root is not None:
        more_sources, budget_hashes, budget_count = budget_extension_analysis(
            out, panel, thresholds, states, training, args.budget_reference_root,
            args.expected_epochs, args.reference_epochs, args.bootstrap_draws)
        sources.extend(more_sources)
        output_hashes.update(budget_hashes)
    (out / "analysis_manifest.json").write_text(json.dumps({
        "analysis_script": str(Path(__file__)), "analysis_script_sha256": sha256(Path(__file__)),
        "bootstrap_draws": args.bootstrap_draws, "models": MODELS, "k_values": KS,
        "expected_epochs": args.expected_epochs,
        "budget_reference_root": str(args.budget_reference_root) if args.budget_reference_root else None,
        "reference_epochs": args.reference_epochs if args.budget_reference_root else None,
        "budget_comparison_count": budget_count,
        "comparison_count": len(definitions), "sources": sources,
        "outputs": output_hashes,
        "role": "same-cohort model development; no target-based model selection",
        "estimand": "cell-pooled within seed; seed mean within partition; equal partition mean",
    }, indent=2) + "\n")
    print(curves[curves.k.isin((0, 5))][["model_name", "k", "mae", "rmse"]].to_string(index=False))
    print(f"Saved all four loss arms and {len(definitions)} fixed contrasts to {out}")


if __name__ == "__main__":
    main()
