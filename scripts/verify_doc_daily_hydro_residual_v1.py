"""Replay daily-hydrology DOC residuals and source-validation adaptation."""
from __future__ import annotations

import argparse
import gc
import json
from datetime import datetime, timezone
from pathlib import Path

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

from river_graph.data.nwis import load_daily_discharge
from river_graph.experiments.provenance import run_identity_sha256, sha256_file
from river_graph.experiments.unified_spatial_protocol import support_query_cells
from river_graph.models.ecological_residual_transfer import EcologicalResidualTransfer
from river_graph.models.encoder_native_residual import EncoderNativeResidual
from river_graph.models.kgml_local_transport import FIT_ROLES, build_rf_features
from river_graph.models.support_aware_residual_transfer import (
    SupportAwareResidualTransfer,
)
from river_graph.models.support_shape_adapter import SupportShapeAdapter
from river_graph.models.unified_doc import UnifiedDOCReconstructor

SPLITS, SEEDS, KS = (142, 143, 144), (42, 43, 44), (0, 1, 3, 5)
ROOT = Path("experiments/phase4_transfer/doc_daily_hydro_residual_v1")
ARMS = ("monthly", "availability", "daily")
SHAPES = ("constant", "gru_tuned_anchor")
INTERACTION_INDICES = (0, 2, 4, 28, 30, 31, 32)
DIRECT_MODELS = tuple(f"{arm}_{shape}" for arm in ("context", *ARMS) for shape in SHAPES)
INTEGRATED_MODELS = tuple(f"{arm}_integrated_{shape}" for arm in ARMS for shape in SHAPES)
REFERENCE_MODELS = tuple(f"prior_{arm}_{shape}" for arm in ("encoder", "encoder_integrated", "ecological_affine") for shape in SHAPES)
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


def _daily_ablation(daily, arm):
    result = daily.copy()
    if arm == "monthly":
        result[...] = 0
    elif arm == "availability":
        result[..., :3] = 0
    elif arm != "daily":
        raise ValueError("Unknown daily feature arm")
    return result


def _independent_daily_values(dataset, cache_dir):
    """Recompute descriptors from historical first-series daily values only."""
    sites = np.asarray(dataset["site_no"]).astype(str)
    months = pd.DatetimeIndex(dataset["months"]).to_period("M")
    if len(np.unique(sites)) != len(sites) or months.has_duplicates:
        raise ValueError("Dataset station/month identities are not unique")
    rows = load_daily_discharge(cache_dir)
    rows = rows[rows.site_no.isin(sites)].copy()
    numeric = rows.discharge_cfs.to_numpy(dtype=float)
    rows = rows[np.isfinite(numeric) & (np.abs(numeric) <= 3e6)].copy()
    rows["date"] = pd.to_datetime(rows.date).dt.normalize()
    rows["month"] = rows.date.dt.to_period("M")
    rows = rows[rows.month.isin(months)].copy()
    unique_values = rows.groupby(["site_no", "date"], sort=False).discharge_cfs.transform("nunique")
    conflicts = rows.loc[unique_values.gt(1), ["site_no", "date"]].drop_duplicates()
    rows = rows.loc[unique_values.eq(1)].drop_duplicates(["site_no", "date"]).copy()
    rows["station"] = rows.site_no.map({name: i for i, name in enumerate(sites)})
    rows["month_index"] = rows.month.map({value: i for i, value in enumerate(months)}).astype(int)
    rows["cell"] = rows.station * len(months) + rows.month_index
    rows = rows.sort_values(["station", "date"]).reset_index(drop=True)
    rows["abs_q"] = rows.discharge_cfs.abs()
    grouped = rows.groupby("cell", sort=True)
    counts = grouped.size()
    quantiles = grouped.discharge_cfs.quantile([.1, .9], interpolation="linear").unstack()
    widths = quantiles[.9] - quantiles[.1]
    scales = grouped.abs_q.mean()
    width_denominator = scales + widths
    width_values = np.divide(widths, width_denominator,
                             out=np.zeros(len(widths)), where=width_denominator.to_numpy() > 0)
    previous = rows.discharge_cfs.shift()
    pairs = rows.cell.eq(rows.cell.shift()) & rows.date.sub(rows.date.shift()).eq(pd.Timedelta(days=1))
    paired = rows.loc[pairs, ["cell"]].copy()
    paired["change"] = (rows.discharge_cfs - previous)[pairs]
    paired["abs_change"] = paired.change.abs()
    paired["scale"] = (rows.discharge_cfs.abs() + previous.abs())[pairs]
    paired["rising"] = paired.change.gt(0)
    pair_group = paired.groupby("cell", sort=True)
    pair_counts = pair_group.size()
    pair_scale = pair_group.scale.sum()
    changes = pair_group.abs_change.sum()
    flash_values = np.divide(changes, pair_scale, out=np.zeros(len(changes)),
                             where=pair_scale.to_numpy() > 0)
    rise_values = pair_group.rising.mean()
    n, t = len(sites), len(months)
    days = np.broadcast_to(months.days_in_month.to_numpy(), (n, t)).reshape(-1)
    daily = np.zeros((n*t, 8), dtype=np.float64)
    day_counts, actual_pairs = np.zeros(n*t), np.zeros(n*t)
    day_counts[counts.index] = counts
    actual_pairs[pair_counts.index] = pair_counts
    daily[:, 3] = day_counts / days
    daily[:, 4] = actual_pairs / (days - 1)
    day_valid = (daily[:, 3] >= .8) & (day_counts >= 2)
    pair_valid = day_valid & (daily[:, 4] >= .8) & (actual_pairs >= 1)
    daily[:, 5], daily[:, 6], daily[:, 7] = day_valid, pair_valid, pair_valid
    daily[counts.index, 0] = width_values
    daily[pair_counts.index, 1] = flash_values
    daily[pair_counts.index, 2] = rise_values
    daily[~day_valid, 0] = 0
    daily[~pair_valid, 1:3] = 0
    frozen_visible = np.asarray(dataset["x_mask"])[..., 1].astype(bool).reshape(-1)
    daily[~frozen_visible] = 0
    if not np.isfinite(daily).all() or np.any((daily < 0) | (daily > 1)):
        raise ValueError("Independently computed daily feature is nonfinite or out of bounds")
    return daily.reshape(n, t, 8).astype(np.float32), {
        "source": "independent historical first-series NWIS reader and calendar aggregation",
        "unique_station_days": len(rows), "conflicting_station_days_excluded": len(conflicts),
        "negative_station_days_retained": int(rows.discharge_cfs.lt(0).sum()),
        "monthly_frozen_hydro_mask_unchanged": True,
        "only_within_month_consecutive_pairs": True,
        "minimum_day_fraction": .8, "minimum_pair_fraction": .8,
        "normalization": "bounded dimensionless ratios; no fitted scaler",
        "target_doc_access": "none", "n_days": day_counts.reshape(n, t),
        "n_pairs": actual_pairs.reshape(n, t),
    }


def _daily_pack(config, dataset, cache):
    feature_path, metadata_path = Path(config["daily_features_path"]), Path(config["daily_metadata_path"])
    if (sha256_file(feature_path) != config["daily_features_hash"]
            or sha256_file(metadata_path) != config["daily_metadata_hash"]):
        raise ValueError("Bound daily feature pack or metadata changed")
    key = (config["daily_features_hash"], config["daily_metadata_hash"], config["dataset_hash"])
    if key in cache:
        return cache[key]
    metadata = json.loads(metadata_path.read_text())
    if (metadata["dataset_hash"] != config["dataset_hash"]
            or metadata["value_feature_indices"] != [0, 1, 2]
            or metadata["availability_feature_indices"] != [3, 4, 5, 6, 7]
            or len(metadata["feature_names"]) != 8
            or len(set(metadata["feature_names"])) != 8):
        raise ValueError("Daily feature semantics or dataset binding differs")
    policy = metadata["policy"]
    expected_policy = {"parameter_code": "00060", "statistic_code": "00003", "raw_unit": "cfs",
                       "coverage_fraction": .8, "quantile_method": "linear",
                       "fitted_statistics": "none", "target_label_dependency": "none",
                       "series_selection": "first matching _00060_00003 column in each RDB header; alternatives ignored",
                       "monthly_footprint": "all eight features zero wherever frozen x_mask[:,:,1] is zero"}
    if any(policy.get(name) != value for name, value in expected_policy.items()):
        raise ValueError("Daily feature policy differs from the matched monthly reconstruction experiment")
    expected_sites = [str(value) for value in dataset["site_no"]]
    expected_months = [str(value) for value in dataset["months"]]
    if (metadata["station_order"] != expected_sites or metadata["month_order"] != expected_months
            or metadata["feature_shape"] != [*dataset["y"].shape, 8]
            or metadata["dtype"] != "float32" or metadata["version"] != 1
            or metadata["feature_product_hash"] != config["daily_features_hash"]):
        raise ValueError("Daily metadata cell alignment or content binding differs")
    with np.load(feature_path, allow_pickle=False) as saved:
        full = saved["full"].copy()
        np.testing.assert_array_equal(saved["site_no"], expected_sites)
        np.testing.assert_array_equal(saved["months"], expected_months)
    if (full.shape != (*dataset["y"].shape, 8) or full.dtype != np.float32
            or not np.isfinite(full).all() or ((full < 0) | (full > 1)).any()):
        raise ValueError("Daily feature grid must be finite, bounded, and aligned")
    raw_files = metadata["raw_cache_files"]
    if not raw_files or len({row["path"] for row in raw_files}) != len(raw_files):
        raise ValueError("Daily raw-cache inventory must be nonempty and unique")
    for row in raw_files:
        if sha256_file(row["path"]) != row["sha256"]:
            raise ValueError(f"Changed daily raw cache: {row['path']}")
    raw_identity = {row["path"]: row["sha256"] for row in raw_files}
    snapshot = json.loads((feature_path.parent / "runtime_snapshot.json").read_text())
    expected_identity = {
        "dataset_hash": config["dataset_hash"], "raw_cache_hashes": raw_identity,
        "module_hash": snapshot["src/river_graph/models/daily_flow_features.py"],
        "builder_hash": snapshot["scripts/build_doc_daily_flow_features_v1.py"], "policy": policy}
    if metadata["identity"] != expected_identity:
        raise ValueError("Feature builder/cache identity differs from historical execution snapshot")
    parents = {Path(row["path"]).parent.resolve() for row in raw_files}
    if len(parents) != 1:
        raise ValueError("Expected one frozen daily cache directory")
    cache_dir = next(iter(parents))
    if {path.resolve() for path in cache_dir.glob("dv_*.rdb")} != {Path(row["path"]).resolve() for row in raw_files}:
        raise ValueError("Daily cache file inventory differs from the frozen feature pack")
    rebuilt, checks = _independent_daily_values(dataset, cache_dir)
    # Vectorized groupby summation can round differently from per-group NumPy
    # sums. Descriptor correctness uses float32 tolerance; neural replay uses
    # the original bound pack and is separately required to be bitwise exact.
    np.testing.assert_allclose(full[..., :3], rebuilt[..., :3], rtol=1e-6, atol=1e-7)
    np.testing.assert_array_equal(full[..., 3:], rebuilt[..., 3:])
    checks.pop("n_days")
    checks.pop("n_pairs")
    checks.update({"feature_pack_sha256": config["daily_features_hash"],
                   "metadata_sha256": config["daily_metadata_hash"],
                   "raw_cache_files_verified": len(raw_files),
                   "independent_formula_rtol": 1e-6, "independent_formula_atol": 1e-7,
                   "independent_formula_max_abs_difference": float(np.abs(full-rebuilt).max()),
                   "independent_formula_bitwise_exact": bool(np.array_equal(full, rebuilt)),
                   "coverage_and_validity_bitwise_exact": True,
                   "rebuild_scope": "all station-months, not a sample",
                   "new_features_are_appended_to_existing_thirty": True})
    cache[key] = (full, metadata, checks)
    return cache[key]


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
                *(f"{name}.{suffix}" for name in ARMS for suffix in ("pt", "json"))}
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
                "predictions.meta.json", "timing.json"}
    required.update(f"{arm}.{suffix}" for arm in ARMS for suffix in ("pt", "json"))
    required.update(f"{arm}_trace.csv" for arm in ARMS)
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
    daily, daily_meta, daily_checks = _daily_pack(config, dataset, daily_cache)
    for index, name in enumerate(daily_meta["feature_names"]):
        np.testing.assert_array_equal(full[name], daily[..., index].ravel())
        np.testing.assert_array_equal(queries[name], daily[..., index].ravel()[queries.cell.to_numpy()])
    definition = {key: value for key, value in head_features.items()
                  if key not in ("source_extra", "full_extra", "source_station_ids")}
    definition["source_station_ids"] = head_features["source_station_ids"].tolist()
    if definition != json.loads((sources["prior"] / "feature_definition.json").read_text()):
        raise ValueError("The daily comparison changed the original thirty feature definitions")
    definition.update({
        "feature_names": [*head_features["feature_names"], *daily_meta["feature_names"]],
        "daily_feature_names": daily_meta["feature_names"],
        "daily_value_feature_indices": [30+i for i in daily_meta["value_feature_indices"]],
        "daily_availability_feature_indices": [30+i for i in daily_meta["availability_feature_indices"]],
        "daily_feature_policy": daily_meta["policy"], "extra_dim": 38,
        "combined_interaction_indices": list(INTERACTION_INDICES)})
    if (definition != config["feature_definition"]
            or definition != json.loads((run / "feature_definition.json").read_text())):
        raise ValueError("Rebuilt and saved regime feature definitions differ")
    profile = EcologicalResidualTransfer.from_dict(
        json.loads((sources["memory"] / "ecological_affine.json").read_text()))
    memory = profile.predict_delta(context.reshape(shape)).ravel()
    np.testing.assert_array_equal(memory, full.ecological_memory)
    np.testing.assert_array_equal(memory, prior_full.ecological_memory)

    torch.set_num_threads(int(config["torch_threads"]))
    expert = UnifiedDOCReconstructor.load(sources["source"], dataset, split)
    np.testing.assert_array_equal(np.sort(np.concatenate(expert.rf.folds)), head_features["source_station_ids"])
    expert.context_forest.n_jobs = 1
    rf_features = build_rf_features(dataset, split, FIT_ROLES, target_transform="log1p", include_network=True)
    recomputed_context = np.maximum(0, np.expm1(expert.context_forest.predict(rf_features)))
    differences = {"context_pred": _close(recomputed_context, context)}
    del rf_features, oof
    full_inputs, raw_checks = _raw_cache(expert, dataset, split, head_features)
    bases, summaries, tensor_checks = {"context": context}, [], 0
    original_extra = full_inputs["extra"]
    source_ids = head_features["source_station_ids"]
    ablations = {}
    for arm in ARMS:
        block = _daily_ablation(daily, arm)
        full_inputs["extra"] = np.concatenate([original_extra, block], axis=-1)
        source_extra = np.concatenate([head_features["source_extra"], block[source_ids]], axis=-1)
        np.testing.assert_array_equal(full_inputs["extra"][..., :30], original_extra)
        np.testing.assert_array_equal(source_extra[..., :30], head_features["source_extra"])
        np.testing.assert_array_equal(source_extra[..., 30:], full_inputs["extra"][source_ids, :, 30:])
        if arm == "monthly" and np.count_nonzero(block):
            raise ValueError("Monthly control receives new daily features")
        if arm == "availability":
            np.testing.assert_array_equal(block[..., 3:], daily[..., 3:])
            if np.count_nonzero(block[..., :3]):
                raise ValueError("Availability arm receives daily descriptor values")
        ablations[arm] = {"source_subset_and_old_thirty_channels_exact": True,
                          "new_feature_dimension": 8, "total_feature_dimension": 38,
                          "zeroed_indices": list(range(8)) if arm == "monthly" else ([0, 1, 2] if arm == "availability" else [])}
        payload = torch.load(run / f"{arm}.pt", weights_only=False, map_location="cpu")
        summary = json.loads((run / f"{arm}.json").read_text())
        if payload["summary"] != summary:
            raise ValueError("Checkpoint summary and saved JSON differ")
        restored = EncoderNativeResidual.from_payload(payload)
        _summary_matches(restored.to_dict(), summary)
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
        trainable = sum(p.numel() for module in (restored.temporal, restored.decay, restored.head, restored.spatial)
                        for p in module.parameters() if p.requires_grad)
        if (head_parameters != expected_width + 1 or trainable != 26023 + scope["encoder_trainable_parameters"]
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
        for module_name in ("temporal", "decay", "head", "spatial"):
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
    reference_mapping = {f"last_self_ecology_{shape_name}": f"prior_encoder_{shape_name}" for shape_name in SHAPES}
    reference_mapping.update({f"last_self_ecology_integrated_{shape_name}": f"prior_encoder_integrated_{shape_name}"
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
    return {"run": run.name, "split_seed": split_seed, "seed": seed, "n_full_grid": len(full),
            "historical_optimization_trajectory_equality_required": False,
            "feature_checks": feature_checks, "daily_feature_checks": daily_checks,
            "daily_ablation_checks": ablations, "raw_cache_checks": raw_checks, "full_grid_max_abs_difference": differences,
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
                "src/river_graph/data/nwis.py",
            )
        },
        "results": results,
    }, indent=2, allow_nan=False) + "\n")
    print(path)


if __name__ == "__main__":
    main()
