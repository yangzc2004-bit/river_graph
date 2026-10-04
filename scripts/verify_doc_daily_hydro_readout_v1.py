"""Independently replay frozen DOC experts and their learned support readout."""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from run_unified_doc_spatial import verify_files, verify_runtime_snapshot
from run_unified_doc_spatial_v2 import read_source
from verify_doc_daily_hydro_support_basis_v1 import (
    _manual_base,
    _refit,
    _same,
    _sidecar,
)
from verify_doc_daily_hydro_support_basis_v1 import (
    _validation_rows as _support_validation_rows,
)
from verify_doc_regime_residual_v1 import _feature_views

from river_graph.experiments.provenance import sha256_file
from river_graph.experiments.spatial_fewshot import support_schedule
from river_graph.experiments.unified_spatial_protocol import support_query_cells
from river_graph.models.encoder_native_residual import EncoderNativeResidual
from river_graph.models.episodic_station_adapter import (
    _tensor_episode_loss,
    stratified_support_months,
)
from river_graph.models.episodic_support_readout import EpisodicSupportReadout
from river_graph.models.episodic_temporal_adapter import EpisodicTemporalAdapter
from river_graph.models.raw_temporal_features import extract_raw_temporal_inputs
from river_graph.models.unified_doc import UnifiedDOCReconstructor

ROOT = Path("experiments/phase4_transfer/doc_daily_hydro_readout_v1")
SPLITS, SEEDS, KS = (142, 143, 144), (42, 43, 44), (0, 1, 3, 5)
SHAPES = ("constant", "legacy", "refreshed_fixed", "refreshed_learned")
MODELS = tuple(f"off_{shape}_{stage}" for shape in SHAPES for stage in ("direct", "integrated"))


def _external(name):
    if name.startswith("off_integrated_"):
        return f"off_{name.removeprefix('off_integrated_')}_integrated"
    return f"{name}_direct"


def _array_digest(value):
    return hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()


def _hidden(model, inputs, batch_size=2048):
    prepared = model._prepare_inputs(inputs)
    n, months = prepared["age"].shape
    hidden = np.empty((n*months, model.hidden_size), dtype=np.float32)
    with torch.inference_mode():
        for start in range(0, n*months, batch_size):
            cells = torch.arange(start, min(n*months, start+batch_size))
            hidden[start:start+len(cells)] = model._hidden_cells(prepared, cells).numpy()
    if not np.isfinite(hidden).all():
        raise ValueError("Frozen expert hidden states are nonfinite")
    return hidden.reshape(n, months, -1)


def _source_and_full(memory_run, dataset, split, *, batch_size=2048):
    """Replay source-fold inputs with genuine source head features, not zeros."""
    config = json.loads((memory_run / "config.json").read_text())
    source = Path(config["source_run"])
    _, _, _, _, old_full = read_source(source)
    context = old_full.context_pred.to_numpy()
    with np.load(Path(config["oof_run"]) / "source_oof.npz", allow_pickle=False) as saved:
        oof = saved["pred_z"].copy()
    train_mask = np.zeros(dataset["y"].shape, dtype=bool)
    train_mask.ravel()[split["train"]] = True
    if not np.isfinite(oof[train_mask]).all() or not np.isnan(oof[~train_mask]).all():
        raise ValueError("OOF forest predictions do not cover exactly source training cells")
    extra, feature_checks = _feature_views(dataset, split, context, oof)
    with np.load(config["daily_features_path"], allow_pickle=False) as saved:
        daily = saved["full"].copy()
    expert = UnifiedDOCReconstructor.load(source, dataset, split)
    raw = extract_raw_temporal_inputs(expert, dataset, split)
    source_ids = raw["source_station_ids"]
    np.testing.assert_array_equal(source_ids, extra["source_station_ids"])
    np.testing.assert_array_equal(np.sort(np.concatenate(expert.rf.folds)), source_ids)
    if np.count_nonzero(raw["source_raw"][..., 8:10]):
        raise ValueError("Held source-station DOC values are visible in source raw features")
    held_ids = np.unique(np.concatenate((split["val"], split["test"]))//dataset["y"].shape[1])
    if np.count_nonzero(raw["full_raw"][held_ids, :, 8:10]):
        raise ValueError("Validation/test DOC values are visible in full raw features")
    model = EncoderNativeResidual.from_payload(torch.load(memory_run / "off.pt", weights_only=True))
    if model.hydro_sequence_mode != "off":
        raise ValueError("The fixed neural expert must be the off daily-head model")
    result = {"source_station_ids": source_ids}
    source_mask = train_mask[source_ids]
    for prefix in ("source", "full"):
        ids = source_ids if prefix == "source" else np.arange(len(train_mask))
        inputs = {key: raw[f"{prefix}_{key}"] for key in ("raw", "age", "support")}
        inputs["env"] = raw["source_env"] if prefix == "source" else raw["env"]
        inputs["extra"] = np.concatenate((extra[f"{prefix}_extra"], daily[ids]), axis=-1)
        inputs["daily_history"] = daily[ids]
        if prefix == "source":
            result["source_extra_sha256"] = _array_digest(inputs["extra"])
        hidden = _hidden(model, inputs, batch_size)
        delta = model.predict_delta(inputs, batch_size=batch_size)
        if prefix == "source":
            forest_base = np.maximum(0, np.expm1(oof[source_ids]))
            native = np.full_like(forest_base, np.nan)
            native[source_mask] = (forest_base[source_mask] if model.selected_scale_ == 0 else
                np.maximum(0, forest_base[source_mask]+model.selected_scale_*delta[source_mask]))
        else:
            forest_base = context.reshape(train_mask.shape)
            native = forest_base.copy() if model.selected_scale_ == 0 else np.maximum(0, forest_base+model.selected_scale_*delta)
            saved_full = pd.read_parquet(memory_run / "full_grid.parquet")
            _same(delta.ravel(), saved_full.off_delta)
            _same(native.ravel(), saved_full.off_pred)
        result[f"{prefix}_hidden"] = hidden
        result[f"{prefix}_native"] = native
        result[f"{prefix}_delta"] = delta
    if not np.isnan(result["source_native"][~source_mask]).all():
        raise ValueError("Unobserved source cells acquired invented forest baselines")
    result["source_mask"] = source_mask
    result["source_feature_checks"] = feature_checks
    result["selected_native_scale"] = model.selected_scale_
    return result


def _project(hidden, readout, anchors, floor, *, batch_size=2048):
    """Independent fixed-calendar scalar RMS projection, matching float64 math."""
    hidden = torch.as_tensor(hidden)
    readout = torch.as_tensor(readout, dtype=torch.float64)
    n, months, width = hidden.shape
    if width != 64 or readout.shape != (64, 2):
        raise ValueError("The support readout dimensions changed")
    raw = torch.empty((n*months, 2), dtype=torch.float64)
    with torch.inference_mode():
        flat = hidden.reshape(-1, width)
        for start in range(0, len(flat), batch_size):
            raw[start:start+batch_size] = flat[start:start+batch_size].double() @ readout
        raw = raw.reshape(n, months, 2)
        mean = raw[:, anchors].mean(dim=1, keepdim=True)
        variance = (raw[:, anchors]-mean).square().mean(dim=(1, 2), keepdim=True)
        scale = variance.clamp_min(floor**2).sqrt()
        basis = (raw-mean)/scale
    return basis.numpy().reshape(-1, 2), {"raw_basis": raw.numpy(),
        "station_anchor_mean": mean[:, 0].numpy(), "station_rms": variance.sqrt().reshape(-1).numpy(),
        "station_scale": scale.reshape(-1).numpy(), "floor_hit": variance.sqrt().reshape(-1).numpy() < floor}


def _readout_contract(model, payload, v4, replay, dataset, split):
    summary = model.to_dict()
    if summary != payload["summary"] or summary["trainable_parameter_count"] != 128:
        raise ValueError("Readout checkpoint roundtrip or parameter count differs")
    torch.testing.assert_close(payload["initial_readout"], v4.readout, rtol=0, atol=0)
    torch.testing.assert_close(model.readout_, v4.readout+v4.readout.norm()*payload["delta"], rtol=0, atol=0)
    protocol = summary["protocol"]
    expected = {"parameterization": "P=P0+frobenius_norm(P0)*D", "delta_initialization": "zero",
        "alpha": 1.0, "k_values": [3, 5], "ridge_strengths": [1.0, 10.0],
        "training_loss": "raw_mae_equal_station_mean_k_ridge",
        "validation_loss": "raw_mae_pooled_query_mean_k_ridge", "selection_role": "source_validation",
        "source_base_scale": "native", "projection_batch_size": 2048, "gradient_clip_norm": 1.0,
        "hidden_experts_updated": False, "pca_whitening_qr": False}
    if any(protocol[key] != value for key, value in expected.items()):
        raise ValueError("The supervised readout objective or frozen-expert contract changed")
    source_mask = replay["source_mask"]
    eligible = np.flatnonzero(source_mask.sum(1) > 5)
    np.testing.assert_array_equal(summary["source_station_ids"], eligible)
    np.testing.assert_array_equal(summary["excluded_source_station_ids"], np.flatnonzero(source_mask.sum(1) <= 5))
    if summary["n_source_query"] != int(source_mask[eligible].sum()-5*len(eligible)):
        raise ValueError("Source query count does not exclude every reserved support")
    schedules = summary["source_episode_schedules"]
    if len(schedules) != summary["epochs_run"]:
        raise ValueError("Source episode schedules do not cover every trained epoch")
    for epoch, row in enumerate(schedules, 1):
        expected_months = stratified_support_months(source_mask[eligible], seed=model.seed, epoch=epoch)
        if row["epoch"] != epoch:
            raise ValueError("Source episode epoch ordering changed")
        np.testing.assert_array_equal(row["support_months"], expected_months)
    months = dataset["y"].shape[1]
    validation_ids = np.unique(split["val"]//months)
    validation_mask = np.zeros(dataset["y"].shape, dtype=bool)
    validation_mask.ravel()[split["val"]] = True
    compact_mask = validation_mask[validation_ids]
    schedule, query = support_schedule(np.flatnonzero(compact_mask), months)
    supports = np.stack([schedule[index] % months for index in range(len(validation_ids))])
    np.testing.assert_array_equal(summary["validation_support_months"], supports)
    np.testing.assert_array_equal(summary["validation_query_cells"], query)
    global_query = validation_ids[query//months]*months + query % months
    global_support = (validation_ids[:, None]*months+supports).ravel()
    expected_support, expected_query = support_query_cells(split, target_role="val", k=5, n_months=months)
    np.testing.assert_array_equal(np.sort(global_support), np.sort(expected_support))
    np.testing.assert_array_equal(np.sort(global_query), np.sort(expected_query))
    if (summary["n_validation_query"] != len(query) or summary["n_validation_stations"] != len(validation_ids)
            or summary["n_source_stations"] != len(eligible)):
        raise ValueError("Readout source/validation identities disagree with actual splits")
    trace = summary["trace"]
    if (len(trace) != summary["epochs_run"]+1 or [row["epoch"] for row in trace] != list(range(len(trace)))
            or summary["best_epoch"] != min(range(len(trace)), key=lambda i: trace[i]["validation_mae"])
            or summary["validation_metrics"] != trace[summary["best_epoch"]]
            or trace[0]["delta_norm"] != 0 or trace[0]["readout_parameter_distance"] != 0):
        raise ValueError("Selected checkpoint or initial readout trace differs")
    truth = np.asarray(dataset["y"], dtype=float)[validation_ids]
    y = torch.as_tensor(np.where(compact_mask, truth, 0))
    native = replay["full_native"][validation_ids]
    z = torch.as_tensor(np.log1p(np.where(compact_mask, native, 0)))
    q_mask = torch.zeros(compact_mask.shape, dtype=torch.bool)
    q_mask.ravel()[query] = True
    source_hidden = replay["full_hidden"][validation_ids]
    score_checks = []
    for name, readout, row in (("initial", model.initial_readout, trace[0]),
                              ("selected", model.readout_, summary["validation_metrics"])):
        totals = np.zeros(4)
        for start in range(0, len(validation_ids), model.batch_size):
            sl = slice(start, start+model.batch_size)
            basis, _ = _project(source_hidden[sl], readout, model.anchor_months(months), model.scale_floor)
            with torch.inference_mode():
                _, losses = _tensor_episode_loss(torch.as_tensor(basis.reshape(-1, months, 2)),
                    z[sl], y[sl], torch.as_tensor(supports[sl]), q_mask[sl], pooled=True)
            totals += losses.numpy()*int(q_mask[sl].sum())
        components = totals/len(query)
        _same(components, [value["mae"] for value in row["validation_by_k_ridge"]])
        _same(float(components.mean()), row["validation_mae"])
        score_checks.append({"checkpoint": name, "validation_objective_bitwise_exact": True})
    # Refit only the inexpensive 128-parameter readout. All unselected labels
    # are NaN here, so exact fit reproduction also checks training isolation.
    source_truth = np.asarray(dataset["y"], dtype=float)[replay["source_station_ids"]]
    refit = EpisodicSupportReadout(model.initial_readout, **summary["config"]).fit(
        replay["source_hidden"], replay["source_native"],
        np.where(source_mask, source_truth, np.nan), source_mask,
        replay["full_hidden"][validation_ids], replay["full_native"][validation_ids],
        np.where(compact_mask, truth, np.nan), compact_mask, selection_role="source_validation")
    torch.testing.assert_close(refit.delta_, payload["delta"], rtol=0, atol=0)
    torch.testing.assert_close(refit.readout_, model.readout_, rtol=0, atol=0)
    if refit.to_dict() != summary:
        raise ValueError("Deterministic source/validation-only readout refit changed the selected checkpoint or trace")
    return {"optimized_parameters": 128, "original_readout_initialization_exact": True,
        "source_schedules_verified": len(schedules), "eligible_source_stations": len(eligible),
        "source_baseline_role": "station OOF forest plus frozen source-trained neural correction",
        "validation_support_query_global_identity_exact": True,
        "initial_selected_validation_objectives": score_checks,
        "deterministic_readout_refit_bitwise_exact": True,
        "entire_training_trace_and_selected_delta_exact": True,
        "non_source_non_validation_labels_in_refit": "none; unselected labels replaced with NaN"}


def _packages(config):
    packages = {}
    for name in ("prior", "expert", "source", "basis", "oof"):
        path = Path(config[f"{name}_run"])
        if sha256_file(path / "complete.json") != config[f"{name}_completion_hash"]:
            raise ValueError(f"Changed bound package: {name}")
        verify_files(path, "complete.json", json.loads((path / "config.json").read_text()))
        packages[name] = path
    pc = json.loads((packages["prior"] / "config.json").read_text())
    mc = json.loads((packages["expert"] / "config.json").read_text())
    if pc["prior_run"] != config["expert_run"] or pc["prior_completion_hash"] != config["expert_completion_hash"]:
        raise ValueError("The support-basis parent points to another native expert")
    for name in ("source", "basis"):
        if any(pc[f"{name}_{suffix}"] != config[f"{name}_{suffix}"] for suffix in ("run", "completion_hash")):
            raise ValueError(f"The parent's fixed {name} lineage changed")
    if mc["oof_run"] != config["oof_run"] or mc["oof_completion_hash"] != config["oof_completion_hash"]:
        raise ValueError("Native expert and current source OOF packages differ")
    for name in ("dataset", "mask", "daily_features", "daily_metadata"):
        if (config[f"{name}_path"] != pc[f"{name}_path"]
                or config[f"{name}_hash"] != pc[f"{name}_hash"]
                or sha256_file(config[f"{name}_path"]) != config[f"{name}_hash"]):
            raise ValueError(f"An unchanged input changed: {name}")
    parent_runtime = verify_runtime_snapshot(packages["prior"].parent.parent)
    report_path = packages["prior"].parent.parent / "verification/replay_checks.json"
    report = json.loads(report_path.read_text())
    verified = [row for row in report["results"] if row["run"] == packages["prior"].name]
    if (len(verified) != 1 or verified[0]["status"] != "verified"
            or verified[0]["completion_sha256"] != config["prior_completion_hash"]
            or verified[0]["runtime_snapshot_hash"] != parent_runtime):
        raise ValueError("The parent support-basis replay does not bind this exact package")
    packages["parent_verification"] = report_path
    expected_files = ((packages["expert"] / "off.pt", "parent_checkpoint_hash"),
        (packages["basis"] / "memory.pt", "basis_checkpoint_hash"),
        (packages["prior"] / "representations.npz", "parent_representation_hash"),
        (packages["prior"] / "full_grid.parquet", "parent_full_grid_hash"))
    for path, key in expected_files:
        if sha256_file(path) != config[key]:
            raise ValueError(f"Changed frozen parent artifact: {path}")
    return packages


def verify_one(root, partition, seed, runtime):
    run = root / "runs" / f"split{partition}_seed{seed}"
    if not (run / "complete.json").is_file():
        raise ValueError(f"Requested run is incomplete: {run}")
    config = json.loads((run / "config.json").read_text())
    required_config = {"anchor_count": 32, "scale_floor": 1e-4, "trainable_parameter_count": 128,
        "arms": ["off"], "basis_names": list(SHAPES), "k_values": list(KS),
        "inference_roles": ["train"], "backbone_retraining": False, "forest_retraining": False,
        "readout_fitting": True, "gamma_k0": "frozen parent source-validation choice",
        "selection_role": "source_validation", "train_k": [3, 5], "train_ridge": [1, 10], "train_alpha": 1,
        "learning_rate": .001, "gradient_clip_norm": 1, "hidden_batch_size": 2048, "projection_batch_size": 2048}
    if (any(config[key] != value for key, value in required_config.items())
            or (config["split_seed"], config["seed"]) != (partition, seed)
            or config["runtime_snapshot_hash"] != runtime or set(config["models"]) != set(MODELS)):
        raise ValueError("Readout task, information or training definitions differ")
    complete = verify_files(run, "complete.json", config)
    model_files = ("readout.pt", "readout.json", "input_definition.json", "source_training.npz",
                   "representations.npz", "basis_definition.json", "adapters.json", "mixers.json")
    required_files = {"config.json", *model_files, "readout_trace.csv", "source_validation.csv", "control_checks.json",
        "predictions.parquet", "predictions.meta.json", "full_grid.parquet", "full_grid.meta.json", "timing.json"}
    if set(complete["files"]) != required_files:
        raise ValueError("Readout completion record omits required products")
    packages = _packages(config)
    old_config, _, dataset, split, old_full = read_source(packages["source"])
    for key in ("dataset_path", "dataset_hash", "mask_path", "mask_hash", "q90_threshold_train"):
        if config[key] != old_config[key]:
            raise ValueError(f"Source and readout identity differ: {key}")
    full = pd.read_parquet(run / "full_grid.parquet")
    queries = pd.read_parquet(run / "predictions.parquet")
    previous = pd.read_parquet(packages["prior"] / "predictions.parquet")
    if sha256_file(run / "full_grid.parquet") != config["parent_full_grid_hash"]:
        raise ValueError("Native grid was not copied unchanged")
    for filename, frame in (("full_grid.parquet", full), ("predictions.parquet", queries)):
        _sidecar(run, filename, config, runtime, frame, complete, model_files)
    shape = tuple(dataset["y"].shape)
    months, cells = shape[1], int(np.prod(shape))
    np.testing.assert_array_equal(full.cell, np.arange(cells))
    identity = ["cell", "station", "month", "analyte", "visibility_role"]
    pd.testing.assert_frame_equal(full[identity], old_full[identity], check_exact=True)
    if (queries.duplicated(["model_name", "k", "cell"]).any()
            or set(map(tuple, queries[["model_name", "k"]].drop_duplicates().to_numpy()))
            != {(name, k) for name in MODELS for k in KS}
            or not queries.split_seed.eq(partition).all() or not queries.seed.eq(seed).all()):
        raise ValueError("The matched query panels are incomplete or duplicated")
    torch.set_num_threads(config["torch_threads"])
    replay = _source_and_full(packages["expert"], dataset, split, batch_size=config["hidden_batch_size"])
    validation_ids = np.unique(split["val"]//months)
    validation_mask = np.zeros(shape, dtype=bool)
    validation_mask.ravel()[split["val"]] = True
    with np.load(run / "source_training.npz", allow_pickle=False) as saved:
        expected_source = {"source_native": replay["source_native"], "source_mask": replay["source_mask"],
            "source_station_ids": replay["source_station_ids"], "validation_station_ids": validation_ids,
            "validation_mask": validation_mask[validation_ids]}
        if set(saved.files) != set(expected_source):
            raise ValueError("Compact source training cache has unexpected fields")
        for key, value in expected_source.items():
            np.testing.assert_array_equal(value, saved[key])
    input_definition = json.loads((run / "input_definition.json").read_text())
    expected_input = {"source_hidden_sha256": _array_digest(replay["source_hidden"]),
        "full_hidden_sha256": _array_digest(replay["full_hidden"]),
        "source_hidden_shape": list(replay["source_hidden"].shape),
        "full_hidden_shape": list(replay["full_hidden"].shape), "source_extra_sha256": replay["source_extra_sha256"],
        "source_station_ids": replay["source_station_ids"].tolist(),
        "validation_station_ids": validation_ids.tolist(), "forest_only_oof": True,
        "frozen_neural_source_trained": True}
    if input_definition != expected_input:
        raise ValueError("The independently reconstructed source/full inputs differ")
    payload = torch.load(run / "readout.pt", weights_only=True)
    model = EpisodicSupportReadout.from_payload(payload)
    summary = json.loads((run / "readout.json").read_text())
    if model.to_dict() != summary or summary != payload["summary"]:
        raise ValueError("Readout tensors and summaries disagree")
    if (run / "readout_trace.csv").read_text() != pd.DataFrame(summary["trace"]).to_csv(index=False):
        raise ValueError("The readable training trace differs from the selected checkpoint record")
    expected_settings = {"seed": seed, "max_epochs": config["epochs"], "patience": config["patience"],
        "lr": config["learning_rate"], "batch_size": config["batch_size"], "anchor_count": 32, "scale_floor": 1e-4}
    if summary["config"] != expected_settings:
        raise ValueError("Readout fitted settings differ from the run configuration")
    v4 = EpisodicTemporalAdapter.from_payload(torch.load(packages["basis"] / "memory.pt", weights_only=True))
    readout_checks = _readout_contract(model, payload, v4, replay, dataset, split)
    definition = json.loads((run / "basis_definition.json").read_text())
    if (definition["initial_readout"] != v4.readout.numpy().tolist()
            or definition["selected_readout"] != model.readout_.numpy().tolist()
            or definition["parent_checkpoint_hash"] != config["parent_checkpoint_hash"]
            or definition["anchor_count"] != 32 or definition["scale_floor"] != 1e-4
            or definition["checkpoint_keys"] != list(payload)):
        raise ValueError("Saved basis definition changed the selected or initial readout")
    with np.load(run / "representations.npz", allow_pickle=False) as saved:
        archive = {key: saved[key].copy() for key in saved.files}
    expected_keys = {*SHAPES, "anchor_months", *(f"{name}_{key}" for name in
        ("refreshed_fixed", "refreshed_learned") for key in
        ("raw_basis", "station_anchor_mean", "station_rms", "station_scale", "floor_hit"))}
    if set(archive) != expected_keys:
        raise ValueError("Saved projected-state archive is incomplete")
    anchors = np.linspace(0, months-1, min(32, months), dtype=np.int64)
    np.testing.assert_array_equal(archive["anchor_months"], anchors)
    basis_checks = []
    for name, matrix in (("refreshed_fixed", v4.readout), ("refreshed_learned", model.readout_)):
        basis, stats = _project(replay["full_hidden"], matrix, anchors, 1e-4)
        _same(basis, archive[name])
        for key, value in stats.items():
            _same(value, archive[f"{name}_{key}"])
        reloaded = model.transform_initial(replay["full_hidden"]) if name == "refreshed_fixed" else model.transform(replay["full_hidden"])
        _same(reloaded, basis)
        basis_checks.append({"basis": name, "grid_cells": cells, "projection_normalization_bitwise_exact": True,
                             "floor_hits": int(stats["floor_hit"].sum()), "max_abs_difference": 0.0})
    with np.load(packages["prior"] / "representations.npz", allow_pickle=False) as saved:
        for new, old in (("constant", "constant"), ("legacy", "legacy"), ("refreshed_fixed", "off_refreshed")):
            _same(archive[new], saved[old].reshape(cells, 2))
    _same(archive["constant"], np.zeros((cells, 2)))
    readout_initial_epoch0 = summary["trace"][0]
    if readout_initial_epoch0["delta_norm"] != 0 or readout_initial_epoch0["readout_parameter_distance"] != 0:
        raise ValueError("Epoch0 no longer represents the exact fixed refreshed control")
    source_feature_checks = replay["source_feature_checks"]
    del replay, model, payload, v4
    shapes = {name: archive[name] for name in SHAPES}
    context, native, memory = full.context_pred.to_numpy(), full.off_pred.to_numpy(), full.ecological_memory.to_numpy()
    adapter_states_external = json.loads((run / "adapters.json").read_text())
    mixer_states_external = json.loads((run / "mixers.json").read_text())
    adapter_states = {f"off_{name}": adapter_states_external[f"off_{name}_direct"] for name in SHAPES}
    mixer_states = {f"off_integrated_{name}": mixer_states_external[f"off_{name}_integrated"] for name in SHAPES}
    if set(adapter_states_external)|set(mixer_states_external) != set(MODELS):
        raise ValueError("Downstream adaptation state grid is incomplete")
    truth = np.asarray(dataset["y"], dtype=float).ravel()
    labels = np.full(cells, np.nan)
    labels[split["val"]] = truth[split["val"]]
    adapters, mixers = _refit("off", native, shapes, context, memory, labels, split, months, adapter_states, mixer_states)
    parent_mixers = json.loads((packages["prior"] / "mixers.json").read_text())
    parent_adapters = json.loads((packages["prior"] / "adapters.json").read_text())
    for name, old in (("constant", "constant"), ("legacy", "legacy"), ("refreshed_fixed", "refreshed")):
        if (adapter_states[f"off_{name}"] != parent_adapters[f"off_{old}"]
                or mixer_states[f"off_integrated_{name}"] != parent_mixers[f"off_integrated_{old}"]):
            raise ValueError("A fixed-control downstream support choice changed")
    if any(model.gamma_k0_ != parent_mixers["off_integrated_constant"]["gamma_k0"] for model in mixers.values()):
        raise ValueError("K0 integration changed the parent's frozen mixture")
    rows = _support_validation_rows("off", shapes, native, context, memory, labels, split, months, adapters, mixers)
    for row in rows:
        row["model_name"] = _external(row["model_name"])
    expected_validation = pd.DataFrame(rows)
    stored_validation = pd.read_csv(run / "source_validation.csv")
    if len(stored_validation) != 32:
        raise ValueError("Source-validation table omits a K/basis/pipeline combination")
    for column in expected_validation.columns.difference(["mae"]):
        np.testing.assert_array_equal(expected_validation[column], stored_validation[column])
    np.testing.assert_allclose(expected_validation.mae, stored_validation.mae, rtol=1e-14, atol=1e-14)
    query_checks, control_checks = _queries(queries, previous, full, truth, split, shapes, native,
                                         context, memory, adapters, mixers, months)
    if control_checks != json.loads((run / "control_checks.json").read_text()):
        raise ValueError("Saved fixed-control checks differ from independent replay")
    if sha256_file(packages["expert"] / "off.pt") != config["parent_checkpoint_hash"]:
        raise ValueError("The selected neural expert changed during replay")
    return {"run": run.name, "status": "verified", "runtime_snapshot_hash": runtime,
        "completion_sha256": sha256_file(run / "complete.json"), "n_full_grid": len(full),
        "full_grid_byte_identical_parent": True, "source_full_hidden_digests_exact": True,
        "source_native_and_masks_bitwise_exact": True, "source_feature_checks": source_feature_checks,
        "readout_checks": readout_checks, "basis_replays": basis_checks,
        "direct_validation_refits": len(adapters), "integrated_validation_refits": len(mixers),
        "source_validation_rows": len(rows), "fixed_control_panels_exact": len(control_checks),
        "query_replays": query_checks, "k0_k1_basis_invariant": True,
        "neural_forest_refitted_by_verifier": False, "deterministic_readout_refits": 1,
        "parent_verification_sha256": sha256_file(packages["parent_verification"])}


def _queries(queries, previous, full, truth, split, shapes, native, context, memory, adapters, mixers, months):
    reserved, fixed = support_query_cells(split, target_role="test", k=5, n_months=months)
    labels = np.full(len(truth), np.nan)
    labels[reserved] = truth[reserved]
    panels, controls = [], []
    old_names = {"constant": "constant", "legacy": "legacy", "refreshed_fixed": "refreshed"}
    for shape_name, basis in shapes.items():
        for stage in ("direct", "integrated"):
            name = f"off_{shape_name}_{stage}"
            internal = f"off{'_integrated' if stage == 'integrated' else ''}_{shape_name}"
            for k in KS:
                support, query = support_query_cells(split, target_role="test", k=k, n_months=months)
                np.testing.assert_array_equal(query, fixed)
                if (len(support) != k*len(np.unique(query//months)) or np.intersect1d(support, query).size):
                    raise ValueError("The target support/query schedule changed")
                stored = queries[queries.model_name.eq(name) & queries.k.eq(k)].sort_values("cell")
                np.testing.assert_array_equal(stored.cell, np.sort(query))
                query = stored.cell.to_numpy()
                if stage == "direct":
                    gamma, base = 0, native
                    prediction = adapters[internal].adapt(base[query], query, base[support], support,
                        labels[support], query_basis=basis[query], support_basis=basis[support], k=k)
                else:
                    model = mixers[internal]
                    gamma = model.selected_gamma(k)
                    base = _manual_base(context, native, memory, gamma)
                    _same(base, model.selected_base(context, native, memory, k=k))
                    prediction = model.adapt(context[query], native[query], memory[query], query,
                        context[support], native[support], memory[support], support, labels[support],
                        basis[query], basis[support], k=k)
                    if k == 0:
                        _same(base, full.off_integrated_k0_pred)
                for value, column in ((prediction, "y_pred"), (base[query], "base_pred"),
                    (np.log1p(prediction)-np.log1p(base[query]), "adaptation_delta"),
                    (np.full(len(query), gamma), "regional_gamma"), (np.full(len(query), k), "support_count"),
                    (truth[query], "y_true")):
                    _same(value, stored[column])
                for column in ("station", "month", "analyte", "visibility_role"):
                    np.testing.assert_array_equal(stored[column], full.iloc[query][column])
                if k == 0:
                    _same(prediction, base[query])
                if k in (0, 1):
                    reference = queries[queries.model_name.eq(f"off_legacy_{stage}") & queries.k.eq(k)].sort_values("cell")
                    _same(stored.y_pred, reference.y_pred)
                if shape_name in old_names:
                    old_name = f"off{'_integrated' if stage == 'integrated' else ''}_{old_names[shape_name]}"
                    old = previous[previous.model_name.eq(old_name) & previous.k.eq(k)].sort_values("cell")
                    np.testing.assert_array_equal(stored.cell, old.cell)
                    for column in ("y_pred", "base_pred", "adaptation_delta", "regional_gamma"):
                        _same(stored[column], old[column])
                    controls.append({"model_name": name, "parent_model": old_name, "k": k,
                                     "rows": len(stored), "bitwise_exact": True})
                panels.append({"model_name": name, "k": k, "n_query": len(stored),
                               "bitwise_exact": True, "max_abs_difference": 0.0})
    return panels, controls


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--split-seeds", nargs="+", type=int, default=list(SPLITS))
    parser.add_argument("--seeds", nargs="+", type=int, default=list(SEEDS))
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    runtime = verify_runtime_snapshot(args.root)
    requested = [(partition, seed) for partition in args.split_seeds for seed in args.seeds]
    if not requested or len(set(requested)) != len(requested):
        raise ValueError("Replay requests must be nonempty and unique")
    results = []
    for partition, seed in requested:
        print(f"Replaying split{partition}_seed{seed}", flush=True)
        results.append(verify_one(args.root, partition, seed, runtime))
        gc.collect()
    output = args.output or args.root / "verification/replay_checks.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps({"checked_at": datetime.now(timezone.utc).isoformat(),
        "requested_runs": len(requested), "verified_runs": len(results), "expected_production_runs": 9,
        "all_nine_replayed": set(requested) == {(partition, seed) for partition in SPLITS for seed in SEEDS},
        "verifier_sha256": sha256_file(__file__), "runtime_snapshot_hash": runtime,
        "target_comparative_performance_analyzed": False,
        "shared_helpers_sha256": {name: sha256_file(name) for name in (
            "scripts/verify_doc_daily_hydro_support_basis_v1.py", "scripts/verify_doc_regime_residual_v1.py",
            "scripts/verify_doc_ecological_transfer_v2.py", "scripts/run_unified_doc_spatial_v2.py",
            "scripts/run_unified_doc_spatial.py")}, "results": results}, indent=2, allow_nan=False)+"\n")
    print(output)


if __name__ == "__main__":
    main()
