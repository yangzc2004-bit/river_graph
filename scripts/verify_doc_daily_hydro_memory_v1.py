"""Replay daily-hydrology memory, matched tree probes and source-only adaptation."""
from __future__ import annotations

import argparse
import gc
import json
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
from run_unified_doc_spatial import digest, verify_files, verify_runtime_snapshot
from run_unified_doc_spatial_v2 import fit_adapters, make_predictions, read_source
from verify_doc_ecological_transfer_v2 import _selection_check, _validation_episodes
from verify_doc_encoder_residual_v1 import _encoder_scope, _raw_cache
from verify_doc_regime_residual_v1 import _feature_views
from verify_doc_tail_residual_v1 import (
    ATOL,
    RTOL,
    _close,
    _summary_matches,
)

from river_graph.experiments.provenance import run_identity_sha256, sha256_file
from river_graph.experiments.unified_spatial_protocol import support_query_cells
from river_graph.models.ecological_residual_transfer import EcologicalResidualTransfer
from river_graph.models.encoder_native_residual import EncoderNativeResidual
from river_graph.models.kgml_local_transport import (
    FIT_ROLES,
    build_rf_features,
    rf_feature_names,
)
from river_graph.models.support_aware_residual_transfer import (
    SupportAwareResidualTransfer,
)
from river_graph.models.support_shape_adapter import SupportShapeAdapter
from river_graph.models.unified_doc import UnifiedDOCReconstructor

SPLITS, SEEDS, KS = (142, 143, 144), (42, 43, 44), (0, 1, 3, 5)
ROOT = Path("experiments/phase4_transfer/doc_daily_hydro_memory_v1")
ARMS = ("off", "current_only", "full_history")
TREE_ARMS = ("tree_current", "tree_history")
SHAPES = ("constant", "gru_tuned_anchor")
INTERACTION_INDICES = (0, 2, 4, 28, 30, 31, 32)
DIRECT_MODELS = tuple(f"{arm}_{shape}" for arm in ("context", *ARMS, *TREE_ARMS) for shape in SHAPES)
INTEGRATED_MODELS = tuple(f"{arm}_integrated_{shape}" for arm in ARMS for shape in SHAPES)
REFERENCE_MODELS = tuple(f"prior_{arm}_{shape}" for arm in ("daily", "daily_integrated", "ecological_affine") for shape in SHAPES)
MODELS = (*DIRECT_MODELS, *INTEGRATED_MODELS, *REFERENCE_MODELS)


def _head_layout(model, inputs):
    prepared = model._prepare_inputs(inputs)
    cells = torch.as_tensor(np.unique(np.linspace(0, prepared["age"].numel()-1, 128, dtype=np.int64)))
    months = prepared["age"].shape[1]
    recorded = []
    hook = model.head.register_forward_pre_hook(lambda _module, args: recorded.append(args[0].detach().clone()))
    try:
        with torch.inference_mode():
            hidden = model._hidden_cells(prepared, cells)
            extra = prepared["extra"][cells // months, cells % months]
            interaction = (hidden[:, :, None] * extra[:, INTERACTION_INDICES][:, None, :]).flatten(1)
            expected = torch.cat([hidden, extra, interaction], dim=1)
            model._delta_cells(prepared, cells)
        if len(recorded) != 1:
            raise ValueError("Expected one scalar-head call")
        torch.testing.assert_close(recorded[0], expected, rtol=0, atol=0)
    finally:
        hook.remove()
    return {"sampled_cells": len(cells), "head_input_features": expected.shape[1],
            "ordering": "hidden-major outer product", "bitwise_exact": True}


def _bound_daily_pack(config, dataset, prior, cache):
    key = (config["daily_features_hash"], config["daily_metadata_hash"])
    report_path = prior.parent.parent / "verification/replay_checks.json"
    report = json.loads(report_path.read_text())
    matches = [row for row in report["results"] if row["run"] == prior.name]
    if (len(matches) != 1 or matches[0]["status"] != "verified"
            or matches[0]["completion_sha256"] != config["prior_completion_hash"]):
        raise ValueError("Previously verified daily package does not match this parent")
    checks = matches[0]["daily_feature_checks"]
    if (checks["feature_pack_sha256"] != key[0] or checks["metadata_sha256"] != key[1]
            or not checks["independent_formula_bitwise_exact"]):
        raise ValueError("The daily feature pack lacks its complete independent formula replay")
    prior_config = json.loads((prior / "config.json").read_text())
    for name in ("daily_features_path", "daily_features_hash", "daily_metadata_path", "daily_metadata_hash"):
        if config[name] != prior_config[name]:
            raise ValueError("The frozen daily pack identity changed")
    for name in ("daily_features", "daily_metadata"):
        if sha256_file(config[f"{name}_path"]) != config[f"{name}_hash"]:
            raise ValueError("The saved daily feature pack has changed")
    if key not in cache:
        metadata = json.loads(Path(config["daily_metadata_path"]).read_text())
        with np.load(config["daily_features_path"], allow_pickle=False) as saved:
            daily = saved["full"].copy()
            np.testing.assert_array_equal(saved["site_no"], np.asarray(dataset["site_no"]).astype(str))
            np.testing.assert_array_equal(saved["months"], [str(value) for value in dataset["months"]])
        if (daily.shape != (*dataset["y"].shape, 8) or daily.dtype != np.float32
                or metadata["dataset_hash"] != config["dataset_hash"]
                or not np.isfinite(daily).all() or not np.isin(daily[..., 5:8], [0, 1]).all()):
            raise ValueError("The daily grid is not aligned, finite, or correctly bound")
        cache[key] = daily, metadata
    daily, metadata = cache[key]
    return daily, metadata, {"feature_pack_sha256": key[0], "metadata_sha256": key[1],
        "previous_independent_formula_bitwise_exact": True, "raw_cache_rebuilt_this_replay": False,
        "parent_verification_path": str(report_path), "parent_verification_sha256": sha256_file(report_path)}


def _metadata(split, shape, daily):
    visible = np.zeros(shape, dtype=bool)
    visible.ravel()[split["train"]] = True
    age = np.full(shape, -1, dtype=np.int16)
    last = np.full(shape[0], -1, dtype=np.int64)
    for month in range(shape[1]):
        last[visible[:, month]] = month
        age[:, month] = np.where(last >= 0, month-last, -1)
    count = daily[..., 5:8].sum(-1).astype(np.int8)
    history = np.stack([(count[:, max(0, month-11):month+1] > 0).sum(1)
                        for month in range(shape[1])], axis=1).astype(np.int8)
    return {"observation_age_months": age, "visible_target_history": age >= 0,
            "daily_numeric_valid_count": count, "daily_history_valid_months": history,
            "daily_history_possible_months": np.broadcast_to(np.minimum(np.arange(shape[1])+1, 12), shape).astype(np.int8)}


def _hydro_memory(model, payload, inputs, arm, expert, config):
    if model.hydro_sequence_mode != arm:
        raise ValueError("Recurrent daily injection mode differs")
    initialized = EncoderNativeResidual(expert.residual.model.spatial, expert.residual.model.temporal,
        expert.residual.model.decay, encoder_mode="last_self_ecology", hydro_sequence_mode=arm,
        extra_dim=38, interaction_indices=INTERACTION_INDICES, seed=config["seed"])
    if any(torch.count_nonzero(value) for value in initialized.head.parameters()):
        raise ValueError("The initial residual head is not zero")
    prepared = model._prepare_inputs(inputs)
    months = prepared["age"].shape[1]
    cells = torch.tensor([0, 1, 11, 12, months+24], dtype=torch.long)
    if arm == "off":
        if model.hydro_projection is not None or any("hydro_projection" in key for key in payload):
            raise ValueError("Off control has acquired a recurrent hydrology branch")
        return {"projection_parameters": 0, "initial_head_zero": True, "recurrent_daily_input": "none"}
    projection = model.hydro_projection
    if (projection.bias is not None or projection.weight.shape != (64, 8)
            or not projection.weight.requires_grad
            or payload["summary"]["hydro_projection_trainable_parameter_count"] != 512
            or payload["summary"]["protocol"]["hydro_sequence_learning_rate"] != config["learning_rate"]):
        raise ValueError("Hydrology projection dimensions or optimizer scope differ")
    torch.testing.assert_close(payload["initial_hydro_projection"]["weight"],
                               torch.zeros_like(projection.weight), rtol=0, atol=0)
    torch.testing.assert_close(initialized.hydro_projection.weight,
                               payload["initial_hydro_projection"]["weight"], rtol=0, atol=0)
    captured = []
    hook = projection.register_forward_pre_hook(lambda _module, args: captured.append(args[0].clone()))
    try:
        with torch.inference_mode():
            baseline_hidden = model._hidden_cells(prepared, cells)
        expected = []
        for cell in cells.tolist():
            station, month = divmod(cell, months)
            dates = [month] if arm == "current_only" else range(max(0, month-11), month+1)
            expected.extend(prepared["daily_history"][station, date] for date in dates)
        torch.testing.assert_close(captured[0], torch.stack(expected), rtol=0, atol=0)
    finally:
        hook.remove()
    # The same early queries must ignore any later hydrology input.
    changed = dict(prepared)
    changed["daily_history"] = prepared["daily_history"].clone()
    changed["daily_history"][:, 25:] = 1-changed["daily_history"][:, 25:]
    with torch.inference_mode():
        torch.testing.assert_close(model._hidden_cells(changed, cells), baseline_hidden, rtol=0, atol=0)
        original_projection = projection.weight.clone()
        projection.weight.zero_()
        try:
            zero_hidden = model._hidden_cells(prepared, cells)
            model.hydro_projection = None
            no_projection = model._hidden_cells(prepared, cells)
            torch.testing.assert_close(zero_hidden, no_projection, rtol=0, atol=0)
        finally:
            model.hydro_projection = projection
            projection.weight.copy_(original_projection)
    return {"projection_parameters": 512, "initial_projection_zero": True,
        "initial_head_zero": True, "sampled_cells": len(cells),
        "projected_valid_rows": len(expected), "exact_allowed_dates_and_no_padding": True,
        "future_perturbation_invariant": True, "zero_projection_equals_old_recurrence": True}


def _trees(run, expert, parent_params, dataset, split, daily, metadata, full, bases):
    shape = tuple(dataset["y"].shape)
    source_y = np.zeros(shape, dtype=np.asarray(dataset["y"]).dtype)
    source_y.ravel()[split["train"]] = np.asarray(dataset["y"]).ravel()[split["train"]]
    base = build_rf_features({**dataset, "y": source_y}, split, ("train",),
                             target_transform="log1p", include_network=True)
    names = [*rf_feature_names(True), *[f"lag_{lag}_{name}" for lag in range(11, -1, -1)
             for name in (*metadata["feature_names"], "history_valid")]]
    _, validation_query = support_query_cells(split, target_role="val", k=0, n_months=shape[1])
    validation_y = np.asarray(dataset["y"], dtype=float).ravel()[validation_query]
    checks = []
    for arm in TREE_ARMS:
        mode = arm.removeprefix("tree_")
        matrix = np.zeros((*shape, 147), dtype=np.float32)
        matrix[..., :39] = base.reshape(*shape, 39)
        for month in range(shape[1]):
            for slot, lag in enumerate(range(11, -1, -1)):
                if month >= lag:
                    matrix[:, month, 39+slot*9+8] = 1
                    if mode == "history" or lag == 0:
                        matrix[:, month, 39+slot*9:39+slot*9+8] = daily[:, month-lag]
        forest = joblib.load(run / f"{arm}.joblib")
        record = json.loads((run / f"{arm}.json").read_text())
        expected_params = {**parent_params, "n_jobs": 2}
        expected = {"mode": mode, "lookback": 12, "feature_names": names, "n_features": 147,
            "n_source_cells": len(split["train"]),
            "source_station_ids": np.unique(np.asarray(split["train"])//shape[1]).tolist(),
            "context_name": expert.context_name, "selected_parent_parameters": parent_params,
            "forest_parameters": expected_params, "target_transform": "log1p",
            "visible_roles": ["train"], "inference_roles": ["train"],
            "hyperparameter_search": False, "replaces_neural_context": False, "source_oof_fitting": False,
            "validation_role": "source_validation; diagnostic only, no selection",
            "validation_query_cells": len(validation_query),
            "history_layout": "chronological oldest-to-current; eight daily channels then history_valid",
            "current_ablation": "older 11 daily slots zero; all history_valid bits retained"}
        if (forest.get_params(deep=False) != expected_params or forest.n_features_in_ != 147
                or any(record[key] != value for key, value in expected.items())):
            raise ValueError("Matched tree parameters, source cells or input definition differs")
        predicted = np.maximum(0, np.expm1(forest.predict(matrix.reshape(-1, 147))))
        stored = full[f"{arm}_pred"].to_numpy()
        np.testing.assert_allclose(predicted, stored, rtol=5e-14, atol=5e-14)
        validation_mae = float(np.abs(stored[validation_query]-validation_y).mean())
        if validation_mae != record["validation_mae_native"]:
            raise ValueError("Tree source-validation diagnostic does not match its saved predictions")
        for tree in forest.estimators_:
            if tree.tree_.n_node_samples[0] != len(split["train"]):
                raise ValueError("Matched ExtraTrees root sample count is not the source training count")
        bases[arm] = stored
        checks.append({"arm": arm, "feature_dimension": 147, "input_formula_rebuilt_independently": True,
            "fixed_parent_parameters": True, "source_training_root_counts_verified": True,
            "n_trees": len(forest.estimators_), "validation_record_exact": True,
            "prediction_max_abs_difference": float(np.abs(predicted-stored).max()),
            "prediction_bitwise_exact": bool(np.array_equal(predicted, stored)),
            "replay_rtol": 5e-14, "replay_atol": 5e-14, "forest_refitted": False})
        del matrix, forest
    return checks


def _off_control(run, prior, full, prior_full, queries, prior_queries):
    before = json.loads((prior / "daily.json").read_text())
    after = json.loads((run / "off.json").read_text())
    if ({key: value for key, value in before["config"].items() if key != "epochs"}
            != {key: value for key, value in after["config"].items() if key != "epochs"}):
        raise ValueError("Off control differs from the previous daily model beyond the epoch cap")
    prefix = min(len(before["trace"]), len(after["trace"]))
    if before["trace"][:prefix] != after["trace"][:prefix]:
        raise ValueError("Off control does not reproduce the previous daily training prefix")
    final = (before["config"]["epochs"] == after["config"]["epochs"] or
             (before["trace"][-1]["stale_epochs"] >= before["config"]["patience"]
              and after["config"]["epochs"] >= before["epochs_run"]))
    tensor_count = 0
    if final:
        stripped = lambda value: {key: item for key, item in value.items() if key != "config"}
        if stripped(before) != stripped(after):
            raise ValueError("Off control's complete selected training record changed")
        old = torch.load(prior / "daily.pt", weights_only=False, map_location="cpu")
        new = torch.load(run / "off.pt", weights_only=False, map_location="cpu")
        for module in ("spatial", "temporal", "decay", "head", "initial_spatial", "initial_temporal", "initial_decay"):
            for key, value in old[module].items():
                torch.testing.assert_close(value, new[module][key], rtol=0, atol=0)
                tensor_count += 1
        for suffix in ("delta", "pred", "integrated_k0_pred"):
            np.testing.assert_array_equal(prior_full[f"daily_{suffix}"], full[f"off_{suffix}"])
        for kind in ("adapters", "mixers"):
            old_states = json.loads((prior / f"{kind}.json").read_text())
            new_states = json.loads((run / f"{kind}.json").read_text())
            middle = "_integrated" if kind == "mixers" else ""
            for shape_name in SHAPES:
                if old_states[f"daily{middle}_{shape_name}"] != new_states[f"off{middle}_{shape_name}"]:
                    raise ValueError("Off control's validation-selected adapter changed")
                a = prior_queries[prior_queries.model_name.eq(f"daily{middle}_{shape_name}")].sort_values(["k", "cell"])
                b = queries[queries.model_name.eq(f"off{middle}_{shape_name}")].sort_values(["k", "cell"])
                for column in ("cell", "k", "y_pred", "base_pred", "adaptation_delta", "regional_gamma"):
                    np.testing.assert_array_equal(a[column], b[column])
    return {"common_trace_rows_bitwise_exact": prefix, "full_control_equality_required": final,
            "selected_tensor_checks": tensor_count, "final_predictions_exact_if_required": final,
            "earlier_cap": before["config"]["epochs"], "current_cap": after["config"]["epochs"]}


def _source_validation_csv(run, bases, context, memory, states, truth, split, months, metadata):
    _, query = support_query_cells(split, target_role="val", k=0, n_months=months)
    count = metadata["daily_numeric_valid_count"].ravel()[query]
    history = metadata["daily_history_valid_months"].ravel()[query]
    groups = {"all": np.ones(len(query), bool), "none_valid": count == 0,
        "some_valid": count > 0, "all_valid": count == 3, "history_0": history == 0,
        "history_1_5": (history >= 1) & (history <= 5),
        "history_6_11": (history >= 6) & (history <= 11), "history_12": history == 12}
    rows = []
    for arm, base in bases.items():
        for stage in (("direct", "integrated") if arm in ARMS else ("direct",)):
            gamma = 0
            prediction = base[query]
            if stage == "integrated":
                mixer = SupportAwareResidualTransfer.from_dict(states[f"{arm}_integrated_constant"])
                gamma = mixer.selected_gamma(0)
                prediction = mixer.selected_base(context[query], base[query], memory[query], k=0)
            for group, selected in groups.items():
                rows.append({"arm": arm, "stage": stage, "group": group, "n": int(selected.sum()),
                    "mae": float(np.abs(prediction[selected]-truth[query][selected]).mean()) if selected.any() else None,
                    "gamma_k0": gamma})
    expected = pd.DataFrame(rows)
    stored = pd.read_csv(run / "source_validation.csv")
    if len(stored) != 72:
        raise ValueError("Source-validation diagnostic table is not the complete 72-row panel")
    for column in ("arm", "stage", "group", "n", "gamma_k0"):
        np.testing.assert_array_equal(stored[column], expected[column])
    np.testing.assert_allclose(stored.mae, expected.mae, rtol=1e-14, atol=1e-14, equal_nan=True)
    return {"rows": len(stored), "only_validation_fixed_query_labels": True,
            "identity_counts_gamma_exact": True, "mae_csv_roundtrip_atol": 1e-14}


def _sidecar(run, filename, config, runtime, n_rows, completion):
    path = run / filename
    meta_path = path.with_suffix(".meta.json")
    sidecar = json.loads(meta_path.read_text())
    expected = {"config_hash": digest(config), "runtime_snapshot_hash": runtime,
                "dataset_sha256": config["dataset_hash"], "mask_sha256": config["mask_hash"],
                "prediction_sha256": sha256_file(path), "rows": n_rows,
                "selection_role": "source_validation",
                "run_identity_sha256": run_identity_sha256(digest(config), config["started_at"], runtime)}
    if sidecar["config"] != config:
        raise ValueError(f"Sidecar configuration differs: {meta_path}")
    for key, value in expected.items():
        if sidecar[key] != value:
            raise ValueError(f"Sidecar {key} differs: {meta_path}")
    required = {"adapters.json", "mixers.json", "feature_definition.json",
                *(f"{name}.{suffix}" for name in ARMS for suffix in ("pt", "json")),
                *(f"{name}.{suffix}" for name in TREE_ARMS for suffix in ("joblib", "json"))}
    if set(sidecar["model_files"]) != required:
        raise ValueError("Sidecar does not bind the complete regime model/feature state")
    for name, expected_hash in sidecar["model_files"].items():
        if sha256_file(run / name) != expected_hash or completion["files"].get(name) != expected_hash:
            raise ValueError(f"Changed model state: {run / name}")


def verify_one(root, split_seed, seed, runtime, daily_cache):
    run = root / "runs" / f"split{split_seed}_seed{seed}"
    if not (run / "complete.json").is_file():
        raise ValueError(f"Requested run is incomplete: {run}")
    config = json.loads((run / "config.json").read_text())
    if ((config["split_seed"], config["seed"]) != (split_seed, seed)
            or config["runtime_snapshot_hash"] != runtime):
        raise ValueError("Run identity or historical runtime differs")
    expected_interactions = list(INTERACTION_INDICES)
    if (set(config["models"]) != set(MODELS) or tuple(config["arms"]) != ARMS
            or tuple(config["tree_arms"]) != TREE_ARMS
            or config["interaction_indices"] != expected_interactions
            or config["interaction_order"] != "hidden-major outer product"
            or config["encoder_mode"] != "last_self_ecology"
            or config["tail_weight"] != 2
            or config["encoder_dropout"] != "off in every arm"
            or config["encoder_normalization"] != "unchanged historical expert input normalization"
            or config["train_memory"] is not True or config["extra_dim"] != 38
            or config["k_values"] != list(KS) or config["inference_roles"] != ["train"]):
        raise ValueError("Encoder head or visibility definitions differ")
    completion = verify_files(run, "complete.json", config)
    required = {"config.json", "adapters.json", "mixers.json", "feature_definition.json",
                "full_grid.parquet", "full_grid.meta.json", "predictions.parquet",
                "predictions.meta.json", "timing.json", "source_validation.csv"}
    required.update(f"{arm}.{suffix}" for arm in ARMS for suffix in ("pt", "json"))
    required.update(f"{arm}_trace.csv" for arm in ARMS)
    required.update(f"{arm}.{suffix}" for arm in TREE_ARMS for suffix in ("joblib", "json"))
    if not required.issubset(completion["files"]):
        raise ValueError("Completion record is missing required regime products")
    sources, source_configs = {}, {}
    for label in ("source", "prior", "basis", "oof", "memory"):
        package = Path(config[f"{label}_run"])
        package_config = json.loads((package / "config.json").read_text())
        if sha256_file(package / "complete.json") != config[f"{label}_completion_hash"]:
            raise ValueError(f"Changed {label} package binding")
        verify_files(package, "complete.json", package_config)
        sources[label], source_configs[label] = package, package_config
    for label in ("source", "basis", "oof"):
        if (Path(source_configs["prior"][f"{label}_run"]) != sources[label]
                or source_configs["prior"][f"{label}_completion_hash"] != config[f"{label}_completion_hash"]):
            raise ValueError(f"Prior and current {label} lineage differs")
    if (Path(source_configs["prior"]["memory_run"]) != sources["memory"]
            or source_configs["prior"]["memory_completion_hash"] != config["memory_completion_hash"]
            or sha256_file(sources["memory"] / "ecological_affine.json") != config["memory_file_hash"]):
        raise ValueError("Frozen ecological memory binding differs")
    old_config, _, dataset, split, old_full = read_source(sources["source"])
    for key in ("dataset_path", "dataset_hash", "mask_path", "mask_hash", "q90_threshold_train"):
        if config[key] != old_config[key]:
            raise ValueError(f"Current and source identities disagree: {key}")
    shape = tuple(dataset["y"].shape)
    months = shape[1]
    full = pd.read_parquet(run / "full_grid.parquet")
    queries = pd.read_parquet(run / "predictions.parquet")
    prior_full = pd.read_parquet(sources["prior"] / "full_grid.parquet")
    prior_queries = pd.read_parquet(sources["prior"] / "predictions.parquet")
    _sidecar(run, "full_grid.parquet", config, runtime, len(full), completion)
    _sidecar(run, "predictions.parquet", config, runtime, len(queries), completion)
    np.testing.assert_array_equal(full.cell, np.arange(np.prod(shape)))
    identity = ["cell", "station", "month", "analyte", "visibility_role"]
    pd.testing.assert_frame_equal(full[identity], old_full[identity], check_exact=True)
    pd.testing.assert_frame_equal(prior_full[identity], old_full[identity], check_exact=True)
    if (queries.duplicated(["model_name", "k", "cell"]).any()
            or set(map(tuple, queries[["model_name", "k"]].drop_duplicates().to_numpy()))
            != {(name, k) for name in MODELS for k in KS}
            or not queries.seed.eq(seed).all() or not queries.split_seed.eq(split_seed).all()):
        raise ValueError("Query/run identities differ or model/K combinations are incomplete")
    context = old_full.context_pred.to_numpy()
    np.testing.assert_array_equal(context, full.context_pred)
    np.testing.assert_array_equal(context, prior_full.context_pred)
    with np.load(sources["oof"] / "source_oof.npz", allow_pickle=False) as saved:
        oof = saved["pred_z"].copy()
    head_features, feature_checks = _feature_views(dataset, split, context, oof)
    daily, daily_meta, daily_checks = _bound_daily_pack(config, dataset, sources["prior"], daily_cache)
    metadata = _metadata(split, shape, daily)
    for name, values in metadata.items():
        np.testing.assert_array_equal(full[name], values.ravel())
        np.testing.assert_array_equal(queries[name], values.ravel()[queries.cell.to_numpy()])
    for index, name in enumerate(daily_meta["feature_names"]):
        np.testing.assert_array_equal(full[name], daily[..., index].ravel())
        np.testing.assert_array_equal(queries[name], daily[..., index].ravel()[queries.cell.to_numpy()])
    definition = {key: value for key, value in head_features.items()
                  if key not in ("source_extra", "full_extra", "source_station_ids")}
    definition["source_station_ids"] = head_features["source_station_ids"].tolist()
    definition.update({
        "feature_names": [*head_features["feature_names"], *daily_meta["feature_names"]],
        "daily_feature_names": daily_meta["feature_names"],
        "daily_value_feature_indices": [30+i for i in daily_meta["value_feature_indices"]],
        "daily_availability_feature_indices": [30+i for i in daily_meta["availability_feature_indices"]],
        "daily_feature_policy": daily_meta["policy"], "extra_dim": 38,
        "combined_interaction_indices": list(INTERACTION_INDICES)})
    if (definition != config["feature_definition"]
            or definition != json.loads((run / "feature_definition.json").read_text())
            or definition != json.loads((sources["prior"] / "feature_definition.json").read_text())):
        raise ValueError("Rebuilt and saved regime feature definitions differ")
    profile = EcologicalResidualTransfer.from_dict(
        json.loads((sources["memory"] / "ecological_affine.json").read_text()))
    memory = profile.predict_delta(context.reshape(shape)).ravel()
    np.testing.assert_array_equal(memory, full.ecological_memory)
    np.testing.assert_array_equal(memory, prior_full.ecological_memory)

    torch.set_num_threads(int(config["torch_threads"]))
    expert = UnifiedDOCReconstructor.load(sources["source"], dataset, split)
    np.testing.assert_array_equal(np.sort(np.concatenate(expert.rf.folds)), head_features["source_station_ids"])
    original_forest_params = expert.context_forest.get_params(deep=False)
    expert.context_forest.n_jobs = 1
    rf_features = build_rf_features(dataset, split, FIT_ROLES, target_transform="log1p", include_network=True)
    recomputed_context = np.maximum(0, np.expm1(expert.context_forest.predict(rf_features)))
    differences = {"context_pred": _close(recomputed_context, context)}
    del rf_features, oof
    full_inputs, raw_checks = _raw_cache(expert, dataset, split, head_features)
    bases, summaries, tensor_checks = {"context": context}, [], 0
    original_extra = full_inputs["extra"]
    source_ids = head_features["source_station_ids"]
    history_checks = {}
    full_inputs["extra"] = np.concatenate([original_extra, daily], axis=-1)
    full_inputs["daily_history"] = daily
    source_extra = np.concatenate([head_features["source_extra"], daily[source_ids]], axis=-1)
    np.testing.assert_array_equal(source_extra[..., :30], head_features["source_extra"])
    np.testing.assert_array_equal(source_extra[..., 30:], full_inputs["extra"][source_ids, :, 30:])
    for arm in ARMS:
        payload = torch.load(run / f"{arm}.pt", weights_only=False, map_location="cpu")
        summary = json.loads((run / f"{arm}.json").read_text())
        if payload["summary"] != summary:
            raise ValueError("Checkpoint summary and saved JSON differ")
        restored = EncoderNativeResidual.from_payload(payload)
        _summary_matches(restored.to_dict(), summary)
        history_checks[arm] = _hydro_memory(restored, payload, full_inputs, arm, expert, config)
        source_cells = np.asarray(split["train"], dtype=np.int64)
        source_y = torch.as_tensor(np.asarray(dataset["y"], dtype=float).ravel()[source_cells])
        source_tail = source_y >= config["q90_threshold_train"]
        weights = restored._source_weights(source_cells, source_y, source_tail, months)
        torch.testing.assert_close(weights, 1.0 + source_tail.double(), rtol=0, atol=0)
        if (restored.n_source_cells_ != len(source_cells)
                or restored.n_source_tail_cells_ != int(source_tail.sum())
                or summary["protocol"]["validation_loss"] != "unweighted pooled fixed-query raw MAE"):
            raise ValueError("Source tail weighting or validation objective differs")
        indices = INTERACTION_INDICES
        expected_width = 64 + 38 + 64 * len(indices)
        if (restored.hidden_size != 64 or restored.extra_dim != 38 or restored.head.in_features != expected_width
                or tuple(restored.interaction_indices) != indices or not restored.train_memory
                or summary["config"]["tail_weight"] != 2
                or summary["tail_threshold"] != config["q90_threshold_train"]
                or summary["protocol"]["tail_rule"] != "source_truth >= source_training_Q90"):
            raise ValueError("Regime readout dimensions, interactions or training objective differ")
        head_parameters = sum(p.numel() for p in restored.head.parameters())
        scope = _encoder_scope(restored, payload, "last_self_ecology")
        modules = (restored.temporal, restored.decay, restored.head, restored.spatial)
        if restored.hydro_projection is not None:
            modules += (restored.hydro_projection,)
        trainable = sum(p.numel() for module in modules for p in module.parameters() if p.requires_grad)
        if (head_parameters != expected_width + 1 or trainable != 26023 + scope["encoder_trainable_parameters"] + (0 if arm == "off" else 512)
                or summary["trainable_parameter_count"] != trainable
                or not all(p.requires_grad for module in (restored.temporal, restored.decay, restored.head)
                           for p in module.parameters())):
            raise ValueError("Regime model trainable parameter count differs")
        if (summary["config"]["encoder_mode"] != "last_self_ecology"
                or summary["encoder_trainable_parameter_count"] != scope["encoder_trainable_parameters"]
                or summary["protocol"]["spatial_mode"] != "eval throughout; empty edges; frozen input normalization"):
            raise ValueError("Encoder scope or fixed normalization protocol differs")
        for key in ("lookback", "epochs", "patience", "batch_size", "learning_rate", "head_learning_rate", "encoder_learning_rate"):
            if summary["config"][key] != config[key]:
                raise ValueError(f"Checkpoint setting differs from run: {key}")
        if summary["config"]["scales"] != config["residual_scales"]:
            raise ValueError("Residual scale grid differs")
        for module_name in ("temporal", "decay", "head", "spatial") + (() if arm == "off" else ("hydro_projection",)):
            for key, tensor in getattr(restored, module_name).state_dict().items():
                torch.testing.assert_close(tensor, payload[module_name][key], rtol=0, atol=0)
                tensor_checks += 1
        for module_name in ("temporal", "decay", "spatial"):
            for key, tensor in getattr(expert.residual.model, module_name).state_dict().items():
                torch.testing.assert_close(tensor, payload[f"initial_{module_name}"][key], rtol=0, atol=0)
                tensor_checks += 1
        layout = _head_layout(restored, full_inputs)
        delta = restored.predict_delta(full_inputs).ravel()
        prediction = restored.predict(full_inputs, context.reshape(shape)).ravel()
        np.testing.assert_array_equal(delta, full[f"{arm}_delta"])
        np.testing.assert_array_equal(prediction, full[f"{arm}_pred"])
        differences[f"{arm}_delta"] = _close(delta, full[f"{arm}_delta"])
        differences[f"{arm}_pred"] = _close(prediction, full[f"{arm}_pred"])
        bases[arm] = prediction
        summaries.append({"arm": arm, "best_epoch": summary["best_epoch"], "epochs_run": summary["epochs_run"],
                          "selected_scale": summary["selected_scale"], "head_input_features": expected_width,
                          "head_parameters": head_parameters, "trainable_parameters": trainable,
                          "interaction_layout": layout, "encoder_scope": scope, "checkpoint_summary_json_exact": True,
                          "reloaded_summary_tolerance_verified": True, "tensor_identity_matches": True,
                          "original_expert_initialization_exact": True, "full_grid_neural_replay_bitwise_exact": True})
        del restored, payload
    tree_checks = _trees(run, expert, original_forest_params, dataset, split, daily, daily_meta, full, bases)
    control_checks = _off_control(run, sources["prior"], full, prior_full, queries, prior_queries)
    del expert, full_inputs, head_features
    with np.load(sources["basis"] / "representations.npz", allow_pickle=False) as saved:
        shapes = {name: saved[name].copy() for name in SHAPES}
    adapter_states = json.loads((run / "adapters.json").read_text())
    if set(adapter_states) != set(DIRECT_MODELS):
        raise ValueError("Direct support adapters are incomplete")
    adapters = {name: SupportShapeAdapter.from_dict(state) for name, state in adapter_states.items()}
    # Only validation labels are populated for adapter/mixer reconstruction.
    validation_truth = np.full(np.prod(shape), np.nan)
    validation_truth[split["val"]] = np.asarray(dataset["y"], dtype=float).ravel()[split["val"]]
    refit_adapters = fit_adapters(bases, shapes, validation_truth, split, months)
    if {name: model.to_dict() for name, model in refit_adapters.items()} != adapter_states:
        raise ValueError("Validation-only direct-adapter refit differs")
    support, fixed_query = support_query_cells(split, target_role="test", k=5, n_months=months)
    if len(fixed_query) != config["query_cells"] or np.intersect1d(support, fixed_query).size:
        raise ValueError("Fixed-query count or reserved supports differ")
    labels = np.full(np.prod(shape), np.nan)
    labels[support] = np.asarray(dataset["y"]).ravel()[support]
    direct = make_predictions(old_full, bases, shapes, adapters, labels, split, months)
    direct["regional_gamma"] = 0.0
    mixer_states = json.loads((run / "mixers.json").read_text())
    if set(mixer_states) != set(INTEGRATED_MODELS):
        raise ValueError("Integrated support-aware wrappers are incomplete")
    _, val_query = support_query_cells(split, target_role="val", k=0, n_months=months)
    validation_dataset = {"y": validation_truth.reshape(shape)}
    integrated_frames, mixer_checks = [], []
    for arm in ARMS:
        scores = []
        for gamma in config["gamma_grid"]:
            candidate = (bases[arm][val_query] if gamma == 0 else np.maximum(
                0, context[val_query] + (1-gamma)*(bases[arm][val_query]-context[val_query]) + gamma*memory[val_query]))
            scores.append({"gamma": gamma, "mae": float(np.abs(candidate-validation_truth[val_query]).mean())})
        gamma_k0 = min(scores, key=lambda row: (row["mae"], row["gamma"]))["gamma"]
        for shape_name, basis in shapes.items():
            name = f"{arm}_integrated_{shape_name}"
            state = mixer_states[name]
            _selection_check(state, gamma_k0, shape_name, adapter_states[f"{arm}_{shape_name}"])
            model = SupportAwareResidualTransfer.from_dict(state)
            if model.to_dict() != state:
                raise ValueError("Integrated wrapper JSON roundtrip differs")
            episodes = _validation_episodes(validation_dataset, split, context, bases[arm], memory, basis)
            kwargs = {"ridge_strengths": (float("inf"),)} if shape_name == "constant" else {}
            refitted = SupportAwareResidualTransfer(months, **kwargs).fit(
                episodes, gamma_k0=gamma_k0, selection_role="source_validation")
            if refitted.to_dict() != state:
                raise ValueError("Validation-only integrated-wrapper refit differs")
            for k in KS:
                support_k, query = support_query_cells(split, target_role="test", k=k, n_months=months)
                base = model.selected_base(context, bases[arm], memory, k=k)
                np.testing.assert_array_equal(base, refitted.selected_base(context, bases[arm], memory, k=k))
                if k == 0:
                    np.testing.assert_array_equal(base, full[f"{arm}_integrated_k0_pred"])
                pred = model.adapt(context[query], bases[arm][query], memory[query], query,
                    context[support_k], bases[arm][support_k], memory[support_k], support_k,
                    labels[support_k], basis[query], basis[support_k], k=k)
                cached = model.adapters_by_gamma_[model.selected_gamma(k)]
                independent = cached.adapt(base[query], query, base[support_k], support_k, labels[support_k],
                    query_basis=basis[query], support_basis=basis[support_k], k=k)
                np.testing.assert_array_equal(pred, independent)
                frame = old_full.iloc[query][identity].copy()
                frame["model_name"], frame["k"] = name, k
                frame["y_pred"], frame["base_pred"] = pred, base[query]
                frame["adaptation_delta"] = np.log1p(pred)-np.log1p(base[query])
                frame["support_count"], frame["regional_gamma"] = k, model.selected_gamma(k)
                integrated_frames.append(frame)
            mixer_checks.append({"model_name": name, "gamma_k0": gamma_k0,
                                 "source_validation_k0_selection_verified": True,
                                 "validation_refit_state_exact": True, "full_grid_k0_replay_bitwise_exact": True})
    reference_mapping = {f"daily_{shape_name}": f"prior_daily_{shape_name}" for shape_name in SHAPES}
    reference_mapping.update({f"daily_integrated_{shape_name}": f"prior_daily_integrated_{shape_name}"
                              for shape_name in SHAPES})
    reference_mapping.update({f"prior_ecological_affine_{shape_name}": f"prior_ecological_affine_{shape_name}"
                              for shape_name in SHAPES})
    reference = prior_queries[prior_queries.model_name.isin(reference_mapping)].copy()
    reference["model_name"] = reference.model_name.map(reference_mapping)
    replay = pd.concat([direct, *integrated_frames, reference], ignore_index=True)
    query_checks = []
    for name in MODELS:
        for k in KS:
            actual = replay[replay.model_name.eq(name) & replay.k.eq(k)].sort_values("cell")
            stored = queries[queries.model_name.eq(name) & queries.k.eq(k)].sort_values("cell")
            np.testing.assert_array_equal(stored.cell, np.sort(fixed_query))
            np.testing.assert_array_equal(actual.cell, stored.cell)
            selected_support, selected_query = support_query_cells(split, target_role="test", k=k, n_months=months)
            np.testing.assert_array_equal(selected_query, fixed_query)
            if (len(selected_support) != k * len(np.unique(fixed_query // months))
                    or np.intersect1d(selected_support, fixed_query).size):
                raise ValueError("Support/query identities or per-station K differ")
            for column in ("station", "month", "analyte", "visibility_role", "support_count"):
                np.testing.assert_array_equal(actual[column], stored[column])
            np.testing.assert_array_equal(stored.y_true, np.asarray(dataset["y"]).ravel()[stored.cell])
            error = 0.0
            for column in ("y_pred", "base_pred", "adaptation_delta", "regional_gamma"):
                np.testing.assert_array_equal(actual[column], stored[column])
                error = max(error, _close(actual[column], stored[column]))
            if k == 0:
                np.testing.assert_array_equal(stored.y_pred, stored.base_pred)
            query_checks.append({"model_name": name, "k": k, "n_query": len(stored),
                                 "max_abs_difference": error, "bitwise_exact": True})
    validation_checks = _source_validation_csv(run, bases, context, memory, mixer_states, validation_truth, split, months, metadata)
    return {"run": run.name, "split_seed": split_seed, "seed": seed, "n_full_grid": len(full),
            "off_control_checks": control_checks, "tree_checks": tree_checks,
            "source_validation_csv_checks": validation_checks,
            "feature_checks": feature_checks, "daily_feature_checks": daily_checks,
            "daily_history_checks": history_checks, "raw_cache_checks": raw_checks, "full_grid_max_abs_difference": differences,
            "query_replays": query_checks, "checkpoint_summaries": summaries, "mixer_checks": mixer_checks,
            "direct_adapter_validation_refits": len(adapters), "exact_tensor_checks": tensor_checks,
            "context_forest_recomputation_bitwise_exact": bool(np.array_equal(recomputed_context, context)),
            "encoder_features_neural_outputs_and_adapted_queries_bitwise_exact": True,
            "frozen_prior_reference_queries_bitwise_unchanged": True, "source_bank_refitting": "none",
            "runtime_snapshot_hash": runtime, "completion_sha256": sha256_file(run / "complete.json"),
            "source_bindings_verified": True, "status": "verified"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--split-seeds", type=int, nargs="+", default=list(SPLITS))
    parser.add_argument("--seeds", type=int, nargs="+", default=list(SEEDS))
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    runtime = verify_runtime_snapshot(args.root)
    requested = [(split, seed) for split in args.split_seeds for seed in args.seeds]
    if not requested or len(set(requested)) != len(requested):
        raise ValueError("Requested replay identities must be nonempty and unique")
    results, daily_cache = [], {}
    for split, seed in requested:
        print(f"Replaying split{split}_seed{seed}", flush=True)
        results.append(verify_one(args.root, split, seed, runtime, daily_cache))
        gc.collect()
    completed = [(split, seed) for split in SPLITS for seed in SEEDS
                 if (args.root / "runs" / f"split{split}_seed{seed}" / "complete.json").is_file()]
    path = args.output or args.root / "verification" / "replay_checks.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({
        "checked_at": datetime.now(timezone.utc).isoformat(), "rtol": RTOL, "atol": ATOL,
        "tree_replay_rtol": 5e-14, "tree_replay_atol": 5e-14,
        "target_comparative_performance_analyzed": False,
        "requested_runs": len(requested), "verified_runs": len(results),
        "completed_production_runs_at_check": len(completed), "expected_production_runs": 9,
        "all_nine_replayed": set(requested) == {(split, seed) for split in SPLITS for seed in SEEDS},
        "scope": "only the explicitly listed runs were replayed; completion presence alone is not replay",
        "verifier_sha256": sha256_file(__file__),
        "shared_verifier_helpers_sha256": {
            name: sha256_file(name) for name in (
                "scripts/verify_doc_tail_residual_v1.py",
                "scripts/verify_doc_ecological_transfer_v2.py",
                "scripts/verify_doc_regime_residual_v1.py",
                "scripts/verify_doc_encoder_residual_v1.py",
                "src/river_graph/models/daily_hydro_tree.py",
            )
        },
        "results": results,
    }, indent=2, allow_nan=False) + "\n")
    print(path)


if __name__ == "__main__":
    main()
