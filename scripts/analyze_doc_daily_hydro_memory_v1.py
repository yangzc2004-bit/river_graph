"""Evaluate daily-flow input to existing DOC memory, with matched tree controls.

All models use fixed queries. This entry point reports all planned comparisons;
it never fits experts, tunes support/mixture choices, or selects target outcomes.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from analyze_doc_daily_hydro_fallback_v1 import read_full_grid
from analyze_doc_encoder_residual_v1 import check_source_identity, compare
from analyze_doc_selective_residual_v1 import classification_metrics, recall_comparisons
from analyze_doc_tail_residual_v1 import error_profiles, read_predictions
from analyze_unified_doc_spatial import sha256
from analyze_unified_doc_spatial_v2 import KS, SEEDS, SPLITS, validate_panel
from analyze_unified_doc_spatial_v3 import read_bound_json, summarize_metrics

ROOT = Path("experiments/phase4_transfer/doc_daily_hydro_memory_v1")
ARMS = ("off", "current_only", "full_history")
TREE_ARMS = ("tree_current", "tree_history")
SHAPES = ("constant", "gru_tuned_anchor")
REFERENCE_BASES = ("prior_daily", "prior_daily_integrated", "prior_ecological_affine")
REFERENCES = tuple(f"{base}_{shape}" for base in REFERENCE_BASES for shape in SHAPES)
DIRECT_MODELS = tuple(f"{base}_{shape}" for base in ("context", *ARMS, *TREE_ARMS) for shape in SHAPES)
MIXED_MODELS = tuple(f"{arm}_integrated_{shape}" for arm in ARMS for shape in SHAPES)
MODELS = DIRECT_MODELS + MIXED_MODELS + REFERENCES
INTERACTIONS = [0, 2, 4, 28, 30, 31, 32]
STRATA = {
    "daily_current_validity": ("none_valid", "partially_valid", "all_three_valid"),
    "visible_target_age": ("never_visible", "age0", "age1_3", "age4_12", "age_over12"),
    "daily_history_support": ("history0", "history1_5", "history6_11", "history12"),
}


def comparison_definitions():
    rows = []
    for shape in SHAPES:
        for k in ((5,) if shape == "constant" else (0, 5)):
            for suffix in ("", "_integrated"):
                for candidate, reference in (("full_history", "current_only"), ("full_history", "off"),
                                             ("current_only", "off")):
                    rows.append((f"{candidate}_vs_{reference}{suffix}_{shape}_k{k}",
                                 f"{candidate}{suffix}_{shape}", k, f"{reference}{suffix}_{shape}", k,
                                 "matched_memory_input"))
                for candidate, reference in (("off", "tree_current"), ("current_only", "tree_current"),
                                             ("full_history", "tree_history")):
                    rows.append((f"{candidate}{suffix}_vs_{reference}_{shape}_k{k}",
                                 f"{candidate}{suffix}_{shape}", k, f"{reference}_{shape}", k,
                                 "matched_tree_information"))
            rows.append((f"tree_history_vs_tree_current_{shape}_k{k}", f"tree_history_{shape}", k,
                         f"tree_current_{shape}", k, "tree_history_information"))
            for arm in ARMS:
                rows.append((f"{arm}_integrated_vs_direct_{shape}_k{k}", f"{arm}_integrated_{shape}", k,
                             f"{arm}_{shape}", k, "ecological_integration"))
                rows.append((f"{arm}_integrated_vs_prior_ecological_{shape}_k{k}", f"{arm}_integrated_{shape}", k,
                             f"prior_ecological_affine_{shape}", k, "previous_overall_reference"))
            for arm in (*ARMS, *TREE_ARMS):
                rows.append((f"{arm}_vs_context_{shape}_k{k}", f"{arm}_{shape}", k,
                             f"context_{shape}", k, "fixed_context_control"))
    return rows


def validate_descriptors(full, metadata, config, sources):
    """Reconstruct fixed descriptive strata without reading DOC values."""
    full = full.sort_index()
    np.testing.assert_array_equal(full.index, np.arange(len(full)))
    months = full.month.nunique()
    if months < 1 or len(full) % months:
        raise ValueError("Full grid does not have complete station calendars")
    n = len(full) // months
    dates = full.month.to_numpy().reshape(n, months)
    if not (dates == dates[0]).all():
        raise ValueError("Station calendars differ")
    flags = full[metadata["feature_names"][5:8]].to_numpy()
    if not np.isin(flags, (0, 1)).all():
        raise ValueError("Daily numeric validity must be binary")
    count = flags.sum(axis=1).reshape(n, months).astype(int)
    usable = (count > 0).astype(int)
    csum = np.pad(usable.cumsum(axis=1), ((0, 0), (1, 0)))
    ends = np.arange(months)+1
    history = csum[:, ends]-csum[:, np.maximum(0, ends-12)]
    mask_path = Path(config["mask_path"])
    if sha256(mask_path) != config["mask_hash"]:
        raise ValueError("Changed mask used for visible observation age")
    sources.append({"path": str(mask_path), "sha256": config["mask_hash"]})
    with np.load(mask_path, allow_pickle=False) as archive:
        train = archive["train"]
    visible = np.zeros(len(full), bool)
    visible[train] = True
    visible = visible.reshape(n, months)
    timeline = np.broadcast_to(np.arange(months), (n, months))
    last = np.maximum.accumulate(np.where(visible, timeline, -1), axis=1)
    seen = last >= 0
    expected = {"daily_numeric_valid_count": count,
        "daily_history_valid_months": history,
        "daily_history_possible_months": np.broadcast_to(np.minimum(ends, 12), (n, months)),
        "observation_age_months": np.where(seen, timeline-last, -1), "visible_target_history": seen}
    for name, value in expected.items():
        np.testing.assert_array_equal(full[name], value.ravel())
    return tuple(expected)


def load_panel(root, expected_epochs=120, allow_unmatched_control=False):
    """Load all completed products, preserving the original query estimand."""
    frames, thresholds, sources, states, controls = [], {}, [], [], []
    daily_identity = None
    for split in SPLITS:
        for seed in SEEDS:
            run = root / "runs" / f"split{split}_seed{seed}"
            frame, config, completion = read_predictions(run, sources)
            if ((config["split_seed"], config["seed"]) != (split, seed)
                    or config["experiment"] != "doc_daily_hydro_memory_v1"
                    or set(config["models"]) != set(MODELS) or tuple(config["arms"]) != ARMS
                    or tuple(config["tree_arms"]) != TREE_ARMS
                    or tuple(config["k_values"]) != KS or config["inference_roles"] != ["train"]
                    or config["tail_weight"] != 2 or config["extra_dim"] != 38
                    or config["epochs"] != expected_epochs or config["patience"] != 5
                    or config["interaction_indices"] != INTERACTIONS or config["train_memory"] is not True
                    or config["encoder_learning_rate"] != 1e-5 or config["encoder_mode"] != "last_self_ecology"
                    or config["gamma_grid"] != [0, .25, .5, 1]
                    or config["alpha_grid"] != [0, .25, .5, .75, 1]
                    or config["ridge_grid"] != [.1, 1, 10, "infinity"]):
                raise ValueError(f"Unexpected daily-hydro experiment settings: {run}")
            threshold = float(config["q90_threshold_train"])
            if not np.isfinite(threshold) or threshold < 0:
                raise ValueError("Invalid source-training Q90")
            thresholds[(split, seed)] = threshold
            paths = (Path(config["daily_features_path"]), Path(config["daily_metadata_path"]))
            hashes = (config["daily_features_hash"], config["daily_metadata_hash"])
            if any(sha256(path) != value for path, value in zip(paths, hashes, strict=True)):
                raise ValueError("Daily hydrological artifact changed")
            if daily_identity is not None and hashes != daily_identity:
                raise ValueError("Daily hydrological artifact differs across runs")
            if daily_identity is None:
                sources.extend({"path": str(path), "sha256": value}
                               for path, value in zip(paths, hashes, strict=True))
                daily_identity = hashes
            metadata = json.loads(paths[1].read_text())
            if (metadata["dataset_hash"] != config["dataset_hash"]
                    or metadata["value_feature_indices"] != [0, 1, 2]
                    or metadata["availability_feature_indices"] != [3, 4, 5, 6, 7]
                    or metadata["validity_feature_indices"] != [5, 6, 7]):
                raise ValueError("Daily hydrological dataset or column identity differs")
            prior, memory_run = Path(config["prior_run"]), Path(config["memory_run"])
            for name, path in (("prior", prior), ("memory", memory_run)):
                if sha256(path / "complete.json") != config[f"{name}_completion_hash"]:
                    raise ValueError(f"Changed bound {name} package")
            previous, old_config, old_completion = read_predictions(prior, sources)
            check_source_identity(config, old_config, (
                "split_seed", "seed", "dataset_hash", "mask_hash", "q90_threshold_train",
                "source_run", "basis_run", "oof_run", "memory_run", "memory_file_hash", "patience",
                "source_completion_hash", "basis_completion_hash", "oof_completion_hash",
                "daily_features_hash", "daily_metadata_hash", "feature_definition"))
            matching_budget = old_config["epochs"] == config["epochs"]
            if not matching_budget and not allow_unmatched_control:
                raise ValueError("Off historical control requires the same epoch ceiling")
            for name in ("source", "basis", "oof"):
                path = Path(config[f"{name}_run"]) / "complete.json"
                actual = sha256(path)
                if actual != config[f"{name}_completion_hash"]:
                    raise ValueError(f"Changed frozen {name} expert dependency")
                sources.append({"path": str(path), "sha256": actual})
            mapping = {f"prior_daily_{shape}": f"daily_{shape}" for shape in SHAPES}
            mapping.update({f"prior_daily_integrated_{shape}": f"daily_integrated_{shape}"
                            for shape in SHAPES})
            mapping.update({f"prior_ecological_affine_{shape}": f"prior_ecological_affine_{shape}"
                            for shape in SHAPES})
            mapping.update({f"context_{shape}": f"context_{shape}" for shape in SHAPES})
            for current_name, old_name in mapping.items():
                current = frame[frame.model_name.eq(current_name)].sort_values(["k", "cell"])
                old = previous[previous.model_name.eq(old_name)].sort_values(["k", "cell"])
                for field in ("k", "cell", "y_true", "y_pred"):
                    np.testing.assert_array_equal(current[field], old[field])
                controls.append({"split_seed": split, "seed": seed, "model_name": current_name,
                                 "reference_model": old_name, "query_rows_all_k": len(current),
                                 "predictions_bitwise_equal": True, "control_role": "copied_reference",
                                 "prediction_max_abs_difference": 0.0})
            memory_completion = json.loads((memory_run / "complete.json").read_text())
            memory = read_bound_json(memory_run, "ecological_affine.json", memory_completion, sources)
            if (sha256(memory_run / "ecological_affine.json") != config["memory_file_hash"]
                    or memory["mode"] != "ecological_affine" or memory["selection_role"] != "source_validation"):
                raise ValueError("Changed ecological memory profile")
            sources.append({"path": str(memory_run / "complete.json"),
                            "sha256": sha256(memory_run / "complete.json")})
            features = read_bound_json(run, "feature_definition.json", completion, sources)
            if (features != config["feature_definition"] or len(features["feature_names"]) != 38
                    or features != old_config["feature_definition"]
                    or features["feature_names"][30:] != features["daily_feature_names"]
                    or features["normalization"] != old_config["feature_definition"]["normalization"]
                    or features["daily_value_feature_indices"] != [30, 31, 32]
                    or features["daily_availability_feature_indices"] != [33, 34, 35, 36, 37]
                    or features["combined_interaction_indices"] != INTERACTIONS
                    or features["daily_feature_names"] != metadata["feature_names"]
                    or features["daily_feature_policy"] != metadata["policy"]
                    or features["source_station_ids"] != old_config["feature_definition"]["source_station_ids"]):
                raise ValueError("Changed daily readout feature definition")
            full = read_full_grid(run, config, completion, sources)
            old_full = read_full_grid(prior, old_config, old_completion, sources)
            np.testing.assert_array_equal(full.index, old_full.index)
            for field in ("context_pred", "ecological_memory", *metadata["feature_names"]):
                np.testing.assert_array_equal(full[field], old_full[field])
            if matching_budget:
                for current_field, prior_field in (("off_pred", "daily_pred"), ("off_delta", "daily_delta"),
                                                    ("off_integrated_k0_pred", "daily_integrated_k0_pred")):
                    np.testing.assert_array_equal(full[current_field], old_full[prior_field])
            source_validation = run / "source_validation.csv"
            if sha256(source_validation) != completion["files"].get(source_validation.name):
                raise ValueError("Changed source-validation descriptive table")
            sources.append({"path": str(source_validation), "sha256": sha256(source_validation)})
            descriptors = validate_descriptors(full, metadata, config, sources)
            raw = pd.read_parquet(run / "predictions.parquet", columns=["cell", *descriptors])
            for field in descriptors:
                np.testing.assert_array_equal(raw[field], full.loc[raw.cell, field])
                frame[field] = frame.cell.map(full[field])
            # The head-only off arm is a retraining control; report its replication separately.
            for suffix in ("", "_integrated"):
                for shape in SHAPES:
                    a = frame[frame.model_name.eq(f"off{suffix}_{shape}")].sort_values(["k", "cell"])
                    b = previous[previous.model_name.eq(f"daily{suffix}_{shape}")].sort_values(["k", "cell"])
                    np.testing.assert_array_equal(a.cell, b.cell)
                    np.testing.assert_array_equal(a.y_true, b.y_true)
                    if matching_budget:
                        np.testing.assert_array_equal(a.y_pred, b.y_pred)
                    controls.append({"split_seed": split, "seed": seed,
                        "matched_budget": matching_budget, "model_name": f"off{suffix}_{shape}", "reference_model": f"daily{suffix}_{shape}",
                        "query_rows_all_k": len(a), "control_role": "off_retraining",
                        "predictions_bitwise_equal": np.array_equal(a.y_pred, b.y_pred),
                        "prediction_max_abs_difference": float(np.abs(a.y_pred.to_numpy()-b.y_pred.to_numpy()).max())})
            for arm in ("context", *ARMS, *TREE_ARMS):
                query = frame[frame.k.eq(0) & frame.model_name.eq(f"{arm}_constant")].sort_values("cell")
                np.testing.assert_array_equal(query.y_pred, full.loc[query.cell, f"{arm}_pred"])
            for arm in ARMS:
                query = frame[frame.k.eq(0) & frame.model_name.eq(f"{arm}_integrated_constant")].sort_values("cell")
                np.testing.assert_array_equal(query.y_pred, full.loc[query.cell, f"{arm}_integrated_k0_pred"])
            training = {arm: read_bound_json(run, f"{arm}.json", completion, sources) for arm in ARMS}
            if matching_budget:
                old_training = read_bound_json(prior, "daily.json", old_completion, sources)
                if training["off"] != old_training:
                    raise ValueError("Off control changed its historical model definition or trace")
            states.append({"split_seed": split, "seed": seed, "config": config, "training": training,
                           "trees": {arm: read_bound_json(run, f"{arm}.json", completion, sources) for arm in TREE_ARMS},
                           "adapters": read_bound_json(run, "adapters.json", completion, sources),
                           "mixers": read_bound_json(run, "mixers.json", completion, sources)})
            frames.append(frame)
        if len({thresholds[(split, seed)] for seed in SEEDS}) != 1:
            raise ValueError("Source Q90 differs across training seeds")
    panel = pd.concat(frames, ignore_index=True)
    validate_panel(panel, expected_models=MODELS)
    panel = attach_strata(panel)
    if (panel.groupby("cell")[["daily_numeric_valid_count", "daily_history_valid_months", "daily_history_possible_months"]].nunique() != 1).any().any():
        raise ValueError("Daily hydrological availability changes across seeds/partitions")
    for _, group in panel[panel.k.eq(0)].groupby(["split_seed", "seed"]):
        pivot = group.pivot(index="cell", columns="model_name", values="y_pred")
        for base in ("context", *ARMS, *TREE_ARMS, *(f"{arm}_integrated" for arm in ARMS), *REFERENCE_BASES):
            np.testing.assert_array_equal(pivot[f"{base}_constant"], pivot[f"{base}_gru_tuned_anchor"])
    return panel, thresholds, sources, states, pd.DataFrame(controls)


def selection_records(states, panel):
    training, choices, mixing, scores, trees = [], [], [], [], []
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
            if (state["model_class"] != "EncoderNativeResidual" or cfg["tail_weight"] != 2
                    or cfg["encoder_mode"] != "last_self_ecology" or cfg["encoder_learning_rate"] != 1e-5
                    or cfg["extra_dim"] != 38 or cfg["interaction_indices"] != INTERACTIONS
                    or cfg["learning_rate"] != 1e-4 or cfg["head_learning_rate"] != 1e-3
                    or cfg.get("hydro_sequence_mode", "off") != arm
                    or cfg["lookback"] != 12 or cfg["batch_size"] != 512
                    or cfg["scales"] != [0, .25, .5, 1] or cfg["seed"] != run["seed"]
                    or cfg["epochs"] != run["config"]["epochs"] or cfg["patience"] != 5 or not cfg.get("train_memory", True)
                    or state["protocol"]["selection_role"] != "source_validation"
                    or len(chosen) != 1 or chosen[0]["validation_scale"] != state["selected_scale"]):
                raise ValueError("Unexpected saved daily-hydro encoder training definition")
            projection_count = state.get("hydro_projection_trainable_parameter_count", 0)
            expected_count = 0 if arm == "off" else 8 * state["hidden_size"]
            if projection_count != expected_count:
                raise ValueError("Unexpected hydrological projection parameter count")
            for key in ("hydro_projection_parameter_distance", "hydro_projection_parameter_norm"):
                if not np.isfinite(state.get(key, 0)) or state.get(key, 0) < 0:
                    raise ValueError("Invalid projection movement")
            training.append({**identity, "arm": arm, "tail_weight": cfg["tail_weight"],
                             "epoch_ceiling": cfg["epochs"],
                             "hydro_projection_trainable_parameter_count": projection_count,
                             "hydro_projection_parameter_distance": state.get("hydro_projection_parameter_distance", 0),
                             "hydro_projection_parameter_norm": state.get("hydro_projection_parameter_norm", 0),
                             "best_epoch": state["best_epoch"], "epochs_run": state["epochs_run"],
                             "selected_scale": state["selected_scale"],
                             "initial_validation_mae": trace[0]["validation_mae"],
                             "selected_validation_mae": chosen[0]["validation_mae"],
                             **{key: state[key] for key in ("trainable_parameter_count", "n_source_cells",
                                 "n_source_tail_cells", "n_validation_query", "temporal_parameter_distance",
                                 "decay_parameter_distance", "head_parameter_norm", "encoder_trainable_parameter_count",
                                 "spatial_parameter_distance", "last_self_parameter_distance", "ecology_parameter_distance")}})
        for arm, state in run["trees"].items():
            expected_mode = arm.removeprefix("tree_")
            params, parent_params = state["forest_parameters"], state["selected_parent_parameters"]
            if (state["mode"] != expected_mode or state["lookback"] != 12
                    or state["n_features"] != 147 or len(state["feature_names"]) != 147
                    or state["visible_roles"] != ["train"] or state["inference_roles"] != ["train"]
                    or state["target_transform"] != "log1p" or state["hyperparameter_search"] is not False
                    or state["replaces_neural_context"] is not False or state["source_oof_fitting"] is not False
                    or state["validation_role"] != "source_validation; diagnostic only, no selection"
                    or state["source_station_ids"] != run["config"]["feature_definition"]["source_station_ids"]
                    or {k: v for k, v in params.items() if k != "n_jobs"}
                    != {k: v for k, v in parent_params.items() if k != "n_jobs"}):
                raise ValueError("Tree probe differs from the matched fixed-parameter definition")
            trees.append({**identity, "arm": arm, **{key: state[key] for key in
                ("n_features", "n_source_cells", "context_name", "validation_query_cells", "validation_mae_native")},
                "source_station_count": len(state["source_station_ids"]),
                "forest_parameters": json.dumps(params, sort_keys=True), "hyperparameter_search": False})
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
    return tuple(pd.DataFrame(rows) for rows in (training, choices, mixing, scores, trees))


def attach_strata(frame):
    """Use fixed, label-free current/history availability and visible-DOC age."""
    frame = frame.copy()
    count = frame.daily_numeric_valid_count.to_numpy()
    age = frame.observation_age_months.to_numpy()
    visible = frame.visible_target_history.to_numpy()
    history = frame.daily_history_valid_months.to_numpy()
    possible = frame.daily_history_possible_months.to_numpy()
    if (not np.isin(count, (0, 1, 2, 3)).all() or not np.isin(visible, (False, True)).all()
            or not np.isfinite(age).all() or (age < -1).any() or not (age == np.floor(age)).all()
            or not np.array_equal(age >= 0, visible.astype(bool))
            or not np.isin(possible, np.arange(1, 13)).all()
            or not np.isin(history, np.arange(13)).all() or (history > possible).any()):
        raise ValueError("Invalid fixed availability/visible-age descriptors")
    frame["daily_current_validity"] = np.select([count == 0, count < 3],
        ["none_valid", "partially_valid"], default="all_three_valid")
    frame["visible_target_age"] = np.select([age < 0, age == 0, age <= 3, age <= 12],
        ["never_visible", "age0", "age1_3", "age4_12"], default="age_over12")
    frame["daily_history_support"] = np.select([history == 0, history <= 5, history <= 11],
        ["history0", "history1_5", "history6_11"], default="history12")
    return frame


def stratified_profiles(panel, thresholds):
    """Descriptive K0/K5 strata, with empty groups explicit and no new CI gate."""
    rows, populations = [], []
    for axis, groups in STRATA.items():
        for group in groups:
            unique = panel[panel[axis].eq(group)].drop_duplicates("cell")
            occurrences = panel[panel[axis].eq(group)].drop_duplicates(["split_seed", "cell"])
            populations.append({"stratum_axis": axis, "stratum": group,
                "n_station_months_unique": len(unique), "n_stations_unique": unique.station.nunique(),
                "n_split_cell_occurrences": len(occurrences)})
    for (split, seed, model, k), run in panel[panel.k.isin((0, 5))].groupby(
            ["split_seed", "seed", "model_name", "k"]):
        for axis, groups in STRATA.items():
            for label in groups:
                group = run[run[axis].eq(label)]
                y, pred = group.y_true.to_numpy(), group.y_pred.to_numpy()
                high = y >= thresholds[(int(split), int(seed))]
                error = pred-y
                rows.append({"split_seed": split, "seed": seed, "model_name": model, "k": k,
                    "stratum_axis": axis, "stratum": label, "n_query_cells": len(group),
                    "n_stations": group.station.nunique(), "n_q90_cells": int(high.sum()),
                    "mae": float(np.abs(error).mean()) if len(group) else np.nan,
                    "rmse": float(np.sqrt(np.square(error).mean())) if len(group) else np.nan,
                    "log_mae": float(np.abs(np.log1p(pred)-np.log1p(y)).mean()) if len(group) else np.nan,
                    "signed_bias": float(error.mean()) if len(group) else np.nan,
                    "q90_mae": float(np.abs(error[high]).mean()) if high.any() else np.nan,
                    "nontail_mae": float(np.abs(error[~high]).mean()) if (~high).any() else np.nan,
                    "history_possible_months_mean": float(group.daily_history_possible_months.mean()) if len(group) else np.nan,
                    "q90_unstable": int(high.sum()) < 20})
    runs = pd.DataFrame(rows)
    keys = ["model_name", "k", "stratum_axis", "stratum"]
    scores = ("mae", "rmse", "log_mae", "signed_bias", "q90_mae", "nontail_mae", "history_possible_months_mean")
    parts = runs.groupby(["split_seed", *keys], as_index=False).agg(
        **{name: (name, "mean") for name in scores},
        **{name: (name, "first") for name in ("n_query_cells", "n_stations", "n_q90_cells")},
        n_nonempty_seeds=("mae", "count"), q90_unstable=("q90_unstable", "any"))
    summary = parts.groupby(keys, as_index=False).agg(
        **{name: (name, "mean") for name in scores}, n_nonempty_partitions=("mae", "count"),
        n_nonempty_q90_partitions=("q90_mae", "count"), q90_unstable_any=("q90_unstable", "any"))
    populations = pd.DataFrame(populations)
    summary = summary.merge(populations, on=["stratum_axis", "stratum"], validate="many_to_one")
    return runs, parts, summary, populations


def write_strata_report(out, summary, populations):
    lines = ["# Observation age and hydrological support: descriptive appendix", "",
        "Current daily validity uses the three numeric descriptor flags; it does not use descriptor magnitude.",
        "DOC age is elapsed months since a visible target observation, with -1 for never visible.",
        "The fixed age groups are never-visible, 0, 1–3, 4–12, and >12 months. Never-visible is not",
        "interpreted as calendar age. Under complete station holdout this axis may contain only",
        "never-visible cells; empty age strata are then not identifiable from this design.", "",
        "Daily history support counts months with any valid daily descriptor within the causal last",
        "12 calendar months. Groups are 0, 1–5, 6–11, and 12. The maximum available calendar history",
        "is min(month index+1,12), recorded separately so early padding is not mistaken for missing data.",
        "All arms carry the same data-availability descriptors, regardless of which inputs they zero.", "",
        "These are descriptive K0/K5 profiles, not additional model-selection endpoints or bootstrap tests.",
        "The estimator remains seed mean within partition, then equal mean of nonempty partitions.", "",
        "| Axis | Stratum | Unique station-months | Unique stations | Partition-cell occurrences |",
        "|---|---|---:|---:|---:|"]
    for r in populations.itertuples():
        lines.append(f"| {r.stratum_axis} | {r.stratum} | {r.n_station_months_unique} | {r.n_stations_unique} | {r.n_split_cell_occurrences} |")
    lines += ["", "| Model (GRU support basis) | K | Axis | Stratum | MAE | Log MAE | Q90 MAE | Ordinary MAE | Nonempty partitions |",
              "|---|---:|---|---|---:|---:|---:|---:|---:|"]
    for r in summary[summary.model_name.str.endswith("gru_tuned_anchor")].itertuples():
        lines.append(f"| {r.model_name} | {r.k} | {r.stratum_axis} | {r.stratum} | {r.mae:.4f} | "
                     f"{r.log_mae:.4f} | {r.q90_mae:.4f} | {r.nontail_mae:.4f} | {r.n_nonempty_partitions} |")
    (out / "strata_appendix.md").write_text("\n".join(lines)+"\n")


def write_report(out, curves, profiles, classification, effects, training, trees, mixing, controls, draws,
                 expected_epochs=120):
    copied = controls[controls.control_role.eq("copied_reference")]
    retrained = controls[controls.control_role.eq("off_retraining")]
    lines = ["# Daily hydrological state in the existing DOC temporal residual", "",
             "The three neural arms retain the same spatial/ecological encoder, original GRU",
             f"initialization, 38-channel scalar head, source OOF context base, {expected_epochs}-epoch cap and patience 5.",
             "All use cell-weighted native MAE with weight two at or above source-training Q90.", "",
             "- off: daily descriptors enter only the current scalar residual head.",
             "- current_only: a zero-initialized daily projection also enters the target GRU step.",
             "- full_history: the same projection enters every valid step of the causal 12-month GRU window.", "",
             "Current_only and full_history have the same additional 512 projection weights; their",
             "contrast isolates historical exposure with matched parameter count. Off has no projection",
             "and reproduces the previous daily-head-only model. Full_history versus off changes both",
             "the recurrent input and parameter count, so it does not isolate history by itself.", "",
             "Two ExtraTrees probes clone the previously selected context-forest parameters without",
             "new tuning. Both have 147 columns: unchanged context information plus twelve daily",
             "slots. The current probe zeros historical numeric/availability slots, retaining identical",
             "valid-date flags. The history probe retains all twelve causal slots. They are independent",
             "comparators and do not replace the neural forest base or its OOF residual targets.", "",
             "Current-month daily summaries support retrospective monthly reconstruction, not a forecast",
             "issued before that month. The original discharge footprint, source data and support basis remain fixed.",
             "The no-message spatial self path remains in use; this experiment does not establish river-message gain.", "",
             "Epoch and scale selection uses unweighted source-validation K0 MAE. Support and ecological",
             "mixing choices use the same validation episodes and grids, independently for each arm.",
             "All 24 models and all K curves are retained, including matched tree information controls.",
             f"Exactly {len(copied)} copied model/run checks and {len(retrained)} historical off-control checks are recorded.",
             "Historical equivalence is required for the formal matched budget; an explicit smoke-only",
             "unmatched-budget option records its different cap and does not assert retraining equivalence.", "",
             f"Intervals use {draws:,} paired whole-station bootstrap draws, jointly resampling the same",
             "station identities across partitions. Seed means are averaged within each partition and",
             "partitions receive equal weight. Negative error/FPR deltas favor the candidate; positive",
             "recall deltas favor detection. Q90 includes ties at the source-training threshold.",
             "These are reused development partitions; no new target outcome chooses a model, K,",
             "checkpoint, scale, support adapter, ecological mixture, or descriptive threshold.", "",
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
              "| Model | K | Q90 MAE | Q90 bias | Ordinary MAE | Ordinary bias | Q90 recall | Q90 precision | False-Q90 rate |",
              "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for model in MODELS:
        if not model.endswith("gru_tuned_anchor"):
            continue
        for k in (0, 5):
            profile = profiles[profiles.model_name.eq(model) & profiles.k.eq(k)].set_index("region")
            rates = classification[classification.model_name.eq(model) & classification.k.eq(k)].iloc[0]
            tail, ordinary = profile.loc["q90"], profile.loc["nontail"]
            lines.append(f"| {model} | {k} | {tail.mae:.4f} | {tail.signed_bias:+.4f} | {ordinary.mae:.4f} | "
                         f"{ordinary.signed_bias:+.4f} | {100*rates.q90_recall:.2f}% | {100*rates.q90_precision:.2f}% | {100*rates.q90_false_positive_rate:.2f}% |")
    lines += ["", "Recall conditions on true high DOC; false-Q90 rate conditions on ordinary observations.",
              "A lower false-positive rate alone is not better selectivity if recall also collapses. Precision is descriptive; paired intervals are reported for recall and false-positive rate.", ""]
    for role, title in (("matched_memory_input", "Current input and historical memory"),
                        ("matched_tree_information", "Neural models versus matched tree information"),
                        ("tree_history_information", "Historical daily information in tree models"),
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
              "| Hydrology arm | Selected epoch range | Total epochs executed | Scale counts | Mean selected validation MAE |",
              "|---|---:|---:|---|---:|"]
    for arm, group in training.groupby("arm", sort=False):
        scales = ", ".join(f"{scale:g}: {count}" for scale, count in group.selected_scale.value_counts().sort_index().items())
        lines.append(f"| {arm} | {group.best_epoch.min()}–{group.best_epoch.max()} | {group.epochs_run.sum()} | {scales} | {group.selected_validation_mae.mean():.6f} |")
    lines += ["", "| Tree probe | Source-validation MAE | Feature count | Number of fits |",
              "|---|---:|---:|---:|"]
    for arm, group in trees.groupby("arm", sort=False):
        lines.append(f"| {arm} | {group.validation_mae_native.mean():.6f} | {group.n_features.iloc[0]} | {len(group)} |")
    lines += ["", "Tree validation scores are descriptive and do not select probe parameters."]
    lines += ["", "All three arms use the same source-training objective. Encoder/GRU parameter movements and support choices are",
              "saved in the training and adapter CSVs.", "",
              "| Integrated model | K | Selected gamma counts |", "|---|---:|---|"]
    for (model, k), group in mixing.groupby(["model_name", "k"], sort=False):
        counts = ", ".join(f"{value:g}: {count}" for value, count in group.gamma.value_counts().sort_index().items())
        lines.append(f"| {model} | {k} | {counts} |")
    lines += ["", "Interpret direct feature contrasts before integrated ones: the latter also include changes",
              "in source-validation-selected ecological mixing and positive-K support calibration.",
              "A narrow or sign-changing contrast does not establish equivalence. Station gain/harm",
              "concentration and partition/seed directions are retained alongside the overall means.", ""]
    (out / "findings.md").write_text("\n".join(lines))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--bootstrap-draws", type=int, default=5000)
    parser.add_argument("--expected-epochs", type=int, default=120)
    parser.add_argument("--allow-unmatched-control", action="store_true",
                        help="Smoke-only: allow a different historical epoch ceiling")
    args = parser.parse_args()
    if args.bootstrap_draws < 1 or args.expected_epochs < 1:
        raise ValueError("Bootstrap draws and epoch ceiling must be positive")
    panel, thresholds, sources, states, controls = load_panel(args.root, args.expected_epochs, args.allow_unmatched_control)
    training, choices, mixing, gamma_scores, trees = selection_records(states, panel)
    runs, splits, curves = summarize_metrics(panel, thresholds)
    profile_runs, profile_splits, profiles = error_profiles(panel, thresholds)
    class_runs, class_splits, classification = classification_metrics(panel, thresholds)
    strata_runs, strata_splits, strata_summary, populations = stratified_profiles(panel, thresholds)
    definitions = comparison_definitions()
    effects, directions, partitions, stations, global_stations, concentration = compare(
        panel, thresholds, args.bootstrap_draws, definitions=definitions)
    effects = pd.concat([effects, recall_comparisons(panel, thresholds, args.bootstrap_draws, definitions)],
                        ignore_index=True)
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
               ("mixing_choices", mixing), ("gamma_scores", gamma_scores), ("reference_replication", controls),
               ("strata_profiles_by_run", strata_runs), ("strata_profiles_by_partition", strata_splits),
               ("strata_profiles", strata_summary), ("strata_populations", populations),
               ("tree_records", trees))
    for name, frame in outputs:
        frame.to_csv(out / f"{name}.csv", index=False)
    write_report(out, curves, profiles, classification, effects, training, trees, mixing, controls,
                 args.bootstrap_draws, args.expected_epochs)
    write_strata_report(out, strata_summary, populations)
    output_hashes = {f"{name}.csv": sha256(out / f"{name}.csv") for name, _ in outputs}
    output_hashes["findings.md"] = sha256(out / "findings.md")
    output_hashes["strata_appendix.md"] = sha256(out / "strata_appendix.md")
    # Bind shared analytical implementations as well as this study-specific entry point.
    helpers = ("analyze_doc_daily_hydro_fallback_v1.py", "analyze_doc_encoder_residual_v1.py", "analyze_doc_selective_residual_v1.py",
               "analyze_doc_tail_residual_v1.py", "analyze_unified_doc_spatial.py",
               "analyze_unified_doc_spatial_v2.py", "analyze_unified_doc_spatial_v3.py")
    for name in helpers:
        path = Path(__file__).parent / name
        sources.append({"path": str(path), "sha256": sha256(path)})
    (out / "analysis_manifest.json").write_text(json.dumps({
        "analysis_script": str(Path(__file__)), "analysis_script_sha256": sha256(Path(__file__)),
        "bootstrap_draws": args.bootstrap_draws, "models": MODELS, "k_values": KS,
        "expected_epochs": args.expected_epochs, "allow_unmatched_control": args.allow_unmatched_control, "comparison_count": len(definitions),
        "sources": sources, "outputs": output_hashes,
        "role": "same-cohort model development; no target-based model selection",
        "estimand": "cell-pooled within seed; seed mean within partition; equal partition mean",
    }, indent=2) + "\n")
    print(curves[curves.k.isin((0, 5))][["model_name", "k", "mae", "rmse"]].to_string(index=False))
    print(f"Saved three memory arms, two matched tree probes and {len(definitions)} fixed contrasts to {out}")


if __name__ == "__main__":
    main()
