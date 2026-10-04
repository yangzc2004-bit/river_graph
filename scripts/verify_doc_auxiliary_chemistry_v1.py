"""Replay measured-chemistry DOC heads, matched trees and final fallback."""
from __future__ import annotations

import argparse
import gc
import io
import json
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
from run_unified_doc_spatial import verify_files, verify_runtime_snapshot
from run_unified_doc_spatial_v2 import fit_adapters, read_source
from verify_doc_daily_hydro_support_basis_v1 import _manual_base, _same, _sidecar
from verify_doc_distribution_head_v1 import _array_digest, _head_feature_views

from river_graph.experiments.provenance import sha256_file
from river_graph.experiments.unified_spatial_protocol import support_query_cells
from river_graph.models.auxiliary_chemistry_features import (
    build_auxiliary_chemistry_features,
)
from river_graph.models.daily_hydro_tree import build_daily_tree_features
from river_graph.models.frozen_native_feature_head import FrozenNativeFeatureHead
from river_graph.models.kgml_local_transport import build_rf_features
from river_graph.models.support_aware_residual_transfer import (
    SupportAwareResidualTransfer,
    SupportAwareTransferEpisode,
)
from river_graph.models.support_shape_adapter import (
    SupportShapeAdapter,
    SupportShapeEpisode,
)

ROOT = Path("experiments/phase4_transfer/doc_auxiliary_chemistry_v1")
SPLITS, SEEDS, KS = (142, 143, 144), (42, 43, 44), (0, 1, 3, 5)
MODES = ("no_aux", "masks", "chemistry")
NEURAL = tuple(f"neural_{mode}" for mode in MODES)
TREES = tuple(f"tree_{mode}" for mode in MODES)
REFERENCES = ("point", "context", "tree_prior")
ARMS = (*REFERENCES, *NEURAL, *TREES)
INTEGRATED = ("point", *NEURAL)
BASIS = "gru_tuned_anchor"
MODELS = tuple(f"{arm}_{BASIS}" for arm in ARMS) + tuple(f"{arm}_integrated_{BASIS}" for arm in INTEGRATED)
FOREST_RTOL = FOREST_ATOL = 1e-12


def _mode(auxiliary, mode):
    block = auxiliary.copy()
    if mode == "no_aux":
        block[:] = 0
    elif mode == "masks":
        block[:, :2] = 0
    elif mode != "chemistry":
        raise ValueError("Unknown auxiliary mode")
    return block


def _augment(original, auxiliary):
    interactions = (original[:, :64, None]*auxiliary[:, None, :2]).reshape(len(original), 128)
    return np.concatenate([original, auxiliary, interactions], axis=1)


def _lineage(config):
    sources = {}
    for name in ("prior", "source", "basis", "oof"):
        path = Path(config[f"{name}_run"])
        if sha256_file(path / "complete.json") != config[f"{name}_completion_hash"]:
            raise ValueError(f"Changed bound {name} package")
        verify_files(path, "complete.json", json.loads((path / "config.json").read_text()))
        sources[name] = path
    pc = json.loads((sources["prior"] / "config.json").read_text())
    for name in ("source", "basis", "oof"):
        for suffix in ("run", "completion_hash"):
            if config[f"{name}_{suffix}"] != pc[f"{name}_{suffix}"]:
                raise ValueError("Frozen expert/source OOF/basis lineage differs")
    for name in ("dataset", "mask", "daily_features", "daily_metadata"):
        if sha256_file(config[f"{name}_path"]) != config[f"{name}_hash"]:
            raise ValueError(f"Changed {name} input")
        for suffix in ("path", "hash"):
            if config[f"{name}_{suffix}"] != pc[f"{name}_{suffix}"]:
                raise ValueError("Input grid differs from the parent")
    for filename, key in (("off.pt", "neural_parent_hash"), ("tree_current.joblib", "tree_parent_hash")):
        if sha256_file(sources["prior"] / filename) != config[key]:
            raise ValueError("Parent neural/tree checkpoint changed")
    audit = ROOT / "availability_audit.json"
    if sha256_file(audit) != config["availability_audit_hash"]:
        raise ValueError("Bound auxiliary availability/source audit changed")
    report_path = sources["prior"].parent.parent / "verification/replay_checks.json"
    report = json.loads(report_path.read_text())
    rows = [row for row in report["results"] if row["run"] == sources["prior"].name]
    parent_runtime = verify_runtime_snapshot(sources["prior"].parent.parent)
    if (len(rows) != 1 or rows[0]["status"] != "verified"
            or rows[0]["completion_sha256"] != config["prior_completion_hash"]
            or rows[0]["runtime_snapshot_hash"] != parent_runtime):
        raise ValueError("Parent numerical replay does not bind this package")
    return sources, report_path


def _auxiliary(config, dataset):
    auxiliary = {}
    for name in ("ph", "ec"):
        path = Path(config["auxiliary_paths"][name])
        if (sha256_file(path) != config["auxiliary_hashes"][name]
                or sha256_file(path.with_suffix(".provenance.json")) != config["auxiliary_provenance_hashes"][name]):
            raise ValueError("Auxiliary dataset/provenance has changed")
        auxiliary[name] = torch.load(path, weights_only=False)
        for key in ("site_no", "months", "x", "x_mask", "edge_index", "edge_attr"):
            a, b = np.asarray(dataset[key]), np.asarray(auxiliary[name][key])
            np.testing.assert_array_equal(a, b)
    ph_valid = np.asarray(auxiliary["ph"]["y_mask"], dtype=bool)
    ec_valid = np.asarray(auxiliary["ec"]["y_mask"], dtype=bool)
    ph = np.where(ph_valid, np.asarray(auxiliary["ph"]["y"]), 0).astype(float)
    ec = np.where(ec_valid, np.asarray(auxiliary["ec"]["y"]), 0).astype(float)
    if not np.isfinite(ph).all() or not np.isfinite(ec).all() or (ph < 0).any() or (ph > 14).any() or (ec < 0).any() or (ec > 100000).any():
        raise ValueError("Auxiliary values violate the unchanged unit/QC bounds")
    manual = np.stack([ph/14, np.log1p(ec), ph_valid, ec_valid], axis=-1).astype(np.float32)
    active = ph_valid | ec_valid
    official = build_auxiliary_chemistry_features(dataset, auxiliary["ph"], auxiliary["ec"])
    np.testing.assert_array_equal(official["full"], manual)
    np.testing.assert_array_equal(official["active"], active)
    if (official["policy"] != config["auxiliary_policy"]
            or official["feature_names"] != config["auxiliary_feature_names"]
            or _array_digest(manual.reshape(-1, 4)) != config["auxiliary_feature_hash"]
            or _array_digest(active.ravel()) != config["auxiliary_active_hash"]):
        raise ValueError("Auxiliary feature definitions or active footprint changed")
    changed_doc = {**dataset, "y": np.full_like(dataset["y"], np.nan), "y_mask": np.zeros_like(dataset["y_mask"])}
    perturbed = build_auxiliary_chemistry_features(changed_doc, auxiliary["ph"], auxiliary["ec"])
    np.testing.assert_array_equal(perturbed["full"], manual)
    np.testing.assert_array_equal(perturbed["active"], active)
    return manual.reshape(-1, 4), active.ravel()


def _head(run, mode, config, source, validation, source_base, val_base, source_y, val_y,
          source_active, val_active, original_full, auxiliary, full_base, full, active):
    name = f"neural_{mode}"
    payload = torch.load(run / f"{name}.pt", weights_only=True)
    summary = json.loads((run / f"{name}.json").read_text())
    model = FrozenNativeFeatureHead.from_payload(payload)
    if payload["summary"] != summary or model.to_dict() != summary:
        raise ValueError("Native-feature head checkpoint/summary roundtrip differs")
    if (summary["config"] != {"n_features": 682, **{key: config[key] for key in (
            "epochs", "patience", "batch_size", "seed", "learning_rate")}}
            or summary["trainable_parameter_count"] != 683 or sum(p.numel() for p in model.head.parameters()) != 683):
        raise ValueError("Head architecture, source budget or parameter count differs")
    for key, value in model.head.state_dict().items():
        torch.testing.assert_close(value, payload["head_state"][key], rtol=0, atol=0)
        if value.dtype != torch.float64:
            raise ValueError("Native-feature head must be float64")
    source_t = torch.as_tensor(source, dtype=torch.float64)
    mean, raw_std = source_t.mean(0), source_t.std(0, correction=0)
    scale = torch.where(raw_std < 1e-6, torch.ones_like(raw_std), raw_std)
    for key, value in (("feature_mean", mean), ("feature_raw_std", raw_std), ("feature_scale", scale)):
        np.testing.assert_array_equal(summary["normalization"][key], value.numpy())
    source_t = (source_t-mean)/scale
    val_t = (torch.as_tensor(validation, dtype=torch.float64)-mean)/scale
    weights = 1.+(source_y >= config["q90_threshold_train"]).astype(float)
    expected_counts = {"n_source_cells": len(source), "n_validation_query": len(validation),
        "n_source_tail_cells": int((weights == 2).sum()), "n_source_active": int(source_active.sum()),
        "n_validation_active": int(val_active.sum()), "source_weight_sum": float(weights.sum()),
        "source_weight_mean": float(weights.mean()), "tail_threshold": config["q90_threshold_train"]}
    if any(summary[key] != value for key, value in expected_counts.items()):
        raise ValueError("Source gate, tail weighting or population differs")
    np.testing.assert_allclose(summary["trace"][0]["training_loss"],
        (weights*np.abs(source_base-source_y)).sum()/weights.sum(), rtol=1e-14, atol=1e-14)
    if summary["trace"][0]["selected_scale"] != 0:
        raise ValueError("Zero-initialized head did not retain the exact epoch-zero base")
    with torch.inference_mode():
        source_delta = torch.nn.functional.linear(source_t, payload["head_state"]["weight"], payload["head_state"]["bias"]).squeeze(-1).numpy()
        val_delta = torch.nn.functional.linear(val_t, payload["head_state"]["weight"], payload["head_state"]["bias"]).squeeze(-1).numpy()
    source_delta, val_delta = np.where(source_active, source_delta, 0), np.where(val_active, val_delta, 0)
    for selected, key in ((model.selected_scale_, "selected_source_weighted_mae"),
                          (1., "selected_source_full_scale_weighted_mae")):
        prediction = source_base.copy() if selected == 0 else np.maximum(0, source_base+selected*source_delta)
        np.testing.assert_allclose((weights*np.abs(prediction-source_y)).sum()/weights.sum(), summary[key], rtol=1e-14, atol=1e-14)
    for correction, score in zip(config["correction_scales"], summary["validation_metrics"]["scale_scores"], strict=True):
        prediction = val_base.copy() if correction == 0 else np.maximum(0, val_base+correction*val_delta)
        if score["scale"] != correction:
            raise ValueError("Validation correction grid differs")
        np.testing.assert_allclose(np.abs(prediction-val_y).mean(), score["mae"], rtol=1e-14, atol=1e-14)
    trace = summary["trace"]
    best, stale = (float("inf"), float("inf"), float("inf")), 0
    if [row["epoch"] for row in trace] != list(range(summary["epochs_run"]+1)):
        raise ValueError("Missing training epochs")
    for row in trace:
        selected = min(row["scale_scores"], key=lambda x: (x["mae"], x["scale"]))
        rank = (selected["mae"], selected["scale"], row["epoch"])
        better = rank < best
        stale = 0 if better else stale+1
        if row["validation_mae"] != selected["mae"] or row["selected_scale"] != selected["scale"] or row["is_best"] != better or row["stale_epochs"] != stale:
            raise ValueError("Native head checkpoint/scale/patience logic differs")
        if better:
            best = rank
    if summary["best_epoch"] != best[2] or model.selected_scale_ != best[1]:
        raise ValueError("Selected head does not match its validation trace")
    if summary["epochs_run"] < config["epochs"] and trace[-1]["stale_epochs"] < config["patience"]:
        raise ValueError("Head stopped before the stated budget/patience")
    pd.testing.assert_frame_equal(pd.read_csv(run / f"{name}_trace.csv"),
        pd.read_csv(io.StringIO(pd.DataFrame(trace).to_csv(index=False))), check_dtype=False,
        check_exact=False, rtol=1e-14, atol=1e-14)
    predicted = np.empty(len(original_full))
    for start in range(0, len(predicted), 512):
        stop = min(start+512, len(predicted))
        values = _augment(original_full[start:stop], auxiliary[start:stop])
        gate = active[start:stop]
        delta = model.predict_delta(values, active=gate)
        with torch.inference_mode():
            normalized = (torch.as_tensor(values, dtype=torch.float64)-mean)/scale
            raw = torch.nn.functional.linear(normalized, payload["head_state"]["weight"], payload["head_state"]["bias"]).squeeze(-1).numpy()
        _same(delta, np.where(gate, raw, 0))
        predicted[start:stop] = model.predict(values, full_base[start:stop], active=gate)
        independent = full_base[start:stop].copy() if model.selected_scale_ == 0 else np.where(gate,
            np.maximum(0, full_base[start:stop]+model.selected_scale_*raw), full_base[start:stop])
        _same(predicted[start:stop], independent)
    _same(predicted, full[f"{name}_pred"])
    _same(predicted[~active], full_base[~active])
    return predicted, {"arm": name, "parameters": 683, "source_normalization_exact": True,
        "source_gate_and_tail_denominator_verified": True, "native_formula_and_saved_output_bitwise_exact": True,
        "inactive_native_fallback_exact": True, "selected_source_and_validation_losses_rtol_atol": 1e-14}


def _trees(run, prior, dataset, split, daily, auxiliary, active, full, config):
    shape = tuple(dataset["y"].shape)
    months = shape[1]
    source_y = np.zeros(shape, dtype=np.asarray(dataset["y"]).dtype)
    source_y.ravel()[split["train"]] = np.asarray(dataset["y"]).ravel()[split["train"]]
    context = build_rf_features({**dataset, "y": source_y}, split, ("train",), target_transform="log1p", include_network=True)
    matrix = np.zeros((*shape, 147), dtype=np.float32)
    matrix[..., :39] = context.reshape(*shape, 39)
    for slot, lag in enumerate(range(11, -1, -1)):
        if lag < months:
            matrix[:, lag:, 39+slot*9+8] = 1
    matrix[..., 39+11*9:39+11*9+8] = daily
    matrix = matrix.reshape(-1, 147)
    np.testing.assert_array_equal(matrix, build_daily_tree_features(dataset, split, daily, mode="current"))
    parent = joblib.load(prior / "tree_current.joblib")
    parent_parameters = parent.get_params(deep=False)
    prior_prediction = np.maximum(0, np.expm1(parent.predict(matrix)))
    canonical = full.tree_prior_pred.to_numpy()
    np.testing.assert_allclose(prior_prediction, canonical, rtol=FOREST_RTOL, atol=FOREST_ATOL)
    _, query = support_query_cells(split, target_role="val", k=0, n_months=months)
    truth = np.asarray(dataset["y"], dtype=float).ravel()
    checks = []
    for mode in MODES:
        name = f"tree_{mode}"
        features = np.concatenate([matrix, _mode(auxiliary, mode)], axis=1)
        tree = joblib.load(run / f"{name}.joblib")
        summary = json.loads((run / f"{name}.json").read_text())
        expected = {"mode": mode, "n_features": 151, "source_cells": len(split["train"]),
            "forest_parameters": {**parent_parameters, "n_jobs": config["tree_n_jobs"]},
            "parent_parameters": parent_parameters, "target_transform": "log1p",
            "inactive_fallback": "tree_current", "hyperparameter_search": False}
        if any(summary[key] != value for key, value in expected.items()) or tree.get_params(deep=False) != expected["forest_parameters"] or tree.n_features_in_ != 151:
            raise ValueError("Matched chemistry tree settings/features differ")
        for estimator in tree.estimators_:
            if estimator.tree_.n_node_samples[0] != len(split["train"]):
                raise ValueError("Tree root training count does not match source-only labels")
        recomputed = np.where(active, np.maximum(0, np.expm1(tree.predict(features))), canonical)
        stored = full[f"{name}_pred"].to_numpy()
        np.testing.assert_allclose(recomputed, stored, rtol=FOREST_RTOL, atol=FOREST_ATOL)
        _same(stored[~active], canonical[~active])
        if float(np.abs(stored[query]-truth[query]).mean()) != summary["validation_mae"]:
            raise ValueError("Tree validation diagnostic differs from canonical saved predictions")
        checks.append({"arm": name, "n_features": 151, "parent_parameters_matched": True,
            "source_sample_counts_verified": True, "max_abs_difference": float(np.abs(recomputed-stored).max()),
            "bitwise_exact": bool(np.array_equal(recomputed, stored)), "inactive_native_fallback_exact": True})
    return checks, float(np.abs(prior_prediction-canonical).max())


def _choices(bases, memory, basis, truth, split, months, active, states, mixer_states):
    context = bases["context"]
    adapters = fit_adapters({name: bases[name] for name in REFERENCES}, {BASIS: basis}, truth, split, months)
    full_tasks, active_tasks = [], []
    for k in KS:
        support, query = support_query_cells(split, target_role="val", k=k, n_months=months)
        full_tasks.append((k, support, query))
        selected_query = query[active[query]]
        selected_support = support[np.isin(support//months, np.unique(selected_query//months))]
        if len(selected_query) == 0 or np.intersect1d(selected_support, selected_query).size:
            raise ValueError("Invalid auxiliary-active source-validation episode")
        active_tasks.append((k, selected_support, selected_query))
    for arm in (*NEURAL, *TREES):
        episodes = [SupportShapeEpisode(k, query, truth[query], bases[arm][query], support,
            truth[support], bases[arm][support], basis[query], basis[support]) for k, support, query in active_tasks]
        adapters[f"{arm}_{BASIS}"] = SupportShapeAdapter(n_months=months).fit(episodes, selection_role="source_validation")
    if {name: model.to_dict() for name, model in adapters.items()} != states:
        raise ValueError("Independent active-only direct support selection differs")
    mixers = {}
    for arm in INTEGRATED:
        tasks = full_tasks if arm == "point" else active_tasks
        _, _, query = tasks[0]
        gamma = min((float(np.abs(_manual_base(context[query], bases[arm][query], memory[query], gamma)-truth[query]).mean()), gamma)
                    for gamma in (0, .25, .5, 1))[1]
        episodes = [SupportAwareTransferEpisode(k, query, truth[query], support, truth[support],
            context[query], bases[arm][query], memory[query], context[support], bases[arm][support], memory[support],
            basis[query], basis[support]) for k, support, query in tasks]
        mixers[f"{arm}_integrated_{BASIS}"] = SupportAwareResidualTransfer(months).fit(
            episodes, gamma_k0=gamma, selection_role="source_validation")
    if {name: model.to_dict() for name, model in mixers.items()} != mixer_states:
        raise ValueError("Independent active-only ecological/support selection differs")
    return adapters, mixers


def _final_panels(bases, memory, basis, adapters, mixers, labels, split, months, active, role):
    context, panels = bases["context"], {}
    for k in KS:
        support, query = support_query_cells(split, target_role=role, k=k, n_months=months)
        for arm in ARMS:
            name = f"{arm}_{BASIS}"
            base = bases[arm]
            prediction = adapters[name].adapt(base[query], query, base[support], support, labels[support],
                query_basis=basis[query], support_basis=basis[support], k=k)
            panels[(name, k)] = {"cell": query, "y_pred": prediction, "candidate_y_pred": prediction.copy(),
                "base_pred": base[query].copy(), "adaptation_delta": np.log1p(prediction)-np.log1p(base[query]),
                "regional_gamma": np.zeros(len(query)), "support_count": np.full(len(query), k),
                "aux_fallback": np.zeros(len(query), dtype=bool)}
        for arm in INTEGRATED:
            name = f"{arm}_integrated_{BASIS}"
            model = mixers[name]
            base = _manual_base(context, bases[arm], memory, model.selected_gamma(k))
            prediction = model.adapt(context[query], bases[arm][query], memory[query], query,
                context[support], bases[arm][support], memory[support], support, labels[support],
                basis[query], basis[support], k=k)
            panels[(name, k)] = {"cell": query, "y_pred": prediction, "candidate_y_pred": prediction.copy(),
                "base_pred": base[query].copy(), "adaptation_delta": np.log1p(prediction)-np.log1p(base[query]),
                "regional_gamma": np.full(len(query), model.selected_gamma(k)), "support_count": np.full(len(query), k),
                "aux_fallback": np.zeros(len(query), dtype=bool)}
        unavailable = ~active[query]
        for arm in (*NEURAL, *TREES):
            for suffix in (("", "_integrated") if arm in NEURAL else ("",)):
                candidate = panels[(f"{arm}{suffix}_{BASIS}", k)]
                reference = panels[(f"{'point' if arm in NEURAL else 'tree_prior'}{suffix}_{BASIS}", k)]
                for column in ("y_pred", "base_pred", "adaptation_delta", "regional_gamma"):
                    candidate[column][unavailable] = reference[column][unavailable]
                candidate["aux_fallback"][unavailable] = True
    return panels


def verify_one(root, split_seed, seed, runtime):
    run = root / "runs" / f"split{split_seed}_seed{seed}"
    config = json.loads((run / "config.json").read_text())
    expected = {"experiment": "doc_auxiliary_chemistry_v1", "split_seed": split_seed, "seed": seed,
        "runtime_snapshot_hash": runtime, "arms": list(ARMS), "neural_arms": list(NEURAL), "tree_arms": list(TREES),
        "modes": list(MODES), "basis_names": [BASIS], "k_values": list(KS), "inference_roles": ["train"],
        "selection_role": "source_validation", "feature_dim": 682, "head_parameters": 683,
        "tree_feature_dim": 151, "tree_hyperparameter_search": False, "backbone_retraining": False,
        "source_neural_is_oof": False, "tail_weight": 2, "correction_scales": [0, .25, .5, 1],
        "feature_std_floor": 1e-6, "learning_rate": .001, "batch_size": 512, "gradient_clip_norm": 1}
    if any(config[key] != value for key, value in expected.items()) or set(config["models"]) != set(MODELS):
        raise ValueError("Auxiliary study settings/model panel changed")
    completion = verify_files(run, "complete.json", config)
    model_files = {"input_definition.json", "source_training.npz", "adapters.json", "mixers.json",
        *(f"neural_{mode}.{suffix}" for mode in MODES for suffix in ("pt", "json")),
        *(f"tree_{mode}.{suffix}" for mode in MODES for suffix in ("joblib", "json"))}
    required = {*model_files, "config.json", "source_validation.csv", "reference_checks.json", "timing.json",
        "predictions.parquet", "predictions.meta.json", "full_grid.parquet", "full_grid.meta.json",
        *(f"neural_{mode}_trace.csv" for mode in MODES)}
    if set(completion["files"]) != required:
        raise ValueError("Completed auxiliary package product set differs")
    sources, parent_report = _lineage(config)
    old_config, _, dataset, split, old_full = read_source(sources["source"])
    for name in ("dataset_hash", "mask_hash", "q90_threshold_train", "query_cells"):
        if old_config[name] != config[name]:
            raise ValueError("Original DOC cohort/query protocol differs")
    auxiliary, active = _auxiliary(config, dataset)
    full, queries = (pd.read_parquet(run / name) for name in ("full_grid.parquet", "predictions.parquet"))
    parent = pd.read_parquet(sources["prior"] / "full_grid.parquet")
    previous = pd.read_parquet(sources["prior"] / "predictions.parquet")
    for name, frame in (("full_grid.parquet", full), ("predictions.parquet", queries)):
        _sidecar(run, name, config, runtime, frame, completion, model_files)
    shape, months = dataset["y"].shape, dataset["y"].shape[1]
    np.testing.assert_array_equal(full.cell, np.arange(np.prod(shape)))
    identity = ["cell", "station", "month", "analyte", "visibility_role", "ecological_novelty", "upstream_support"]
    pd.testing.assert_frame_equal(full[identity], old_full[identity], check_exact=True)
    _same(full.point_pred, parent.off_pred)
    _same(full.context_pred, old_full.context_pred)
    _same(full.tree_prior_pred, parent.tree_current_pred)
    _same(full.ecological_memory, parent.ecological_memory)
    if (queries.duplicated(["model_name", "k", "cell"]).any()
            or set(map(tuple, queries[["model_name", "k"]].drop_duplicates().to_numpy())) != {(name, k) for name in MODELS for k in KS}
            or not queries.seed.eq(seed).all() or not queries.split_seed.eq(split_seed).all()
            or not np.isfinite(full.select_dtypes(include="number")).all().all()):
        raise ValueError("Incomplete query panel or nonfinite full-grid product")
    flags = {"ph_available": auxiliary[:, 2].astype(bool), "ec_available": auxiliary[:, 3].astype(bool),
             "aux_available": active, "doc_observed": np.asarray(dataset["y_mask"]).ravel().astype(bool)}
    for name, value in flags.items():
        np.testing.assert_array_equal(full[name], value)
        np.testing.assert_array_equal(queries[name], value[queries.cell.to_numpy()])
    torch.set_num_threads(config["torch_threads"])
    replay = _head_feature_views(sources["prior"], dataset, split)
    source_ids = replay["source_station_ids"]
    local_cells = np.flatnonzero(replay["source_mask"])
    source_cells = source_ids[local_cells//months]*months+local_cells%months
    np.testing.assert_array_equal(source_cells, np.sort(split["train"]))
    _, val_cells = support_query_cells(split, target_role="val", k=0, n_months=months)
    source_base, native = replay["source_native"].ravel()[local_cells], replay["full_native"].ravel()
    with np.load(run / "source_training.npz", allow_pickle=False) as saved:
        expected_source = {"source_cells": source_cells, "source_local_cells": local_cells, "source_station_ids": source_ids,
            "source_base": source_base, "source_active": active[source_cells], "validation_cells": val_cells,
            "validation_base": native[val_cells], "validation_active": active[val_cells]}
        if set(saved.files) != set(expected_source):
            raise ValueError("Source training view schema differs")
        for name, value in expected_source.items():
            np.testing.assert_array_equal(saved[name], value)
    original_source = replay["source_features"].reshape(-1, 550)[local_cells]
    original_full = replay["full_features"].reshape(-1, 550)
    original_validation = original_full[val_cells]
    definitions = json.loads((run / "input_definition.json").read_text())
    if (definitions["source_original_feature_hash"] != _array_digest(original_source)
            or definitions["validation_original_feature_hash"] != _array_digest(original_validation)):
        raise ValueError("Frozen550 source/validation feature identities differ")
    truth = np.asarray(dataset["y"], dtype=float).ravel()
    bases = {name: full[f"{name}_pred"].to_numpy() for name in ARMS}
    head_checks = []
    for mode in MODES:
        block = _mode(auxiliary, mode)
        sx = _augment(original_source, block[source_cells])
        vx = _augment(original_validation, block[val_cells])
        if definitions["modes"][mode] != {"source_feature_hash": _array_digest(sx), "validation_feature_hash": _array_digest(vx)}:
            raise ValueError("Augmented682 input identity differs")
        output, checks = _head(run, mode, config, sx, vx, source_base, native[val_cells], truth[source_cells],
            truth[val_cells], active[source_cells], active[val_cells], original_full, block, native, full, active)
        bases[f"neural_{mode}"] = output
        head_checks.append(checks)
    del replay, original_source, original_full, original_validation, sx, vx
    gc.collect()
    with np.load(config["daily_features_path"], allow_pickle=False) as saved:
        daily = saved["full"].copy()
    tree_checks, parent_tree_difference = _trees(run, sources["prior"], dataset, split, daily,
        auxiliary, active, full, config)
    del daily
    with np.load(sources["basis"] / "representations.npz", allow_pickle=False) as saved:
        basis = saved[BASIS].copy()
    states, mixer_states = (json.loads((run / name).read_text()) for name in ("adapters.json", "mixers.json"))
    validation_truth = np.full(truth.shape, np.nan)
    validation_truth[split["val"]] = truth[split["val"]]
    memory = full.ecological_memory.to_numpy()
    adapters, mixers = _choices(bases, memory, basis, validation_truth, split, months, active, states, mixer_states)
    validation = _final_panels(bases, memory, basis, adapters, mixers, validation_truth, split, months, active, "val")
    rows = []
    for (name, k), panel in sorted(validation.items()):
        error = np.abs(panel["y_pred"]-validation_truth[panel["cell"]])
        selected = active[panel["cell"]]
        rows.append({"model_name": name, "k": k, "n": len(error), "n_active": int(selected.sum()),
            "mae": float(error.mean()), "active_mae": float(error[selected].mean())})
    pd.testing.assert_frame_equal(pd.read_csv(run / "source_validation.csv"), pd.DataFrame(rows),
                                  check_dtype=False, check_exact=False, rtol=1e-14, atol=1e-14)
    reserved, fixed_query = support_query_cells(split, target_role="test", k=5, n_months=months)
    if len(fixed_query) != config["query_cells"] or np.intersect1d(reserved, fixed_query).size:
        raise ValueError("Fixed target support/query population changed")
    labels = np.full(truth.shape, np.nan)
    labels[reserved] = truth[reserved]
    panels = _final_panels(bases, memory, basis, adapters, mixers, labels, split, months, active, "test")
    query_checks, reference_checks = [], []
    reference_mapping = {"point": "off", "point_integrated": "off_integrated", "context": "context", "tree_prior": "tree_current"}
    for (name, k), panel in panels.items():
        order = np.argsort(panel["cell"])
        cell = panel["cell"][order]
        stored = queries[queries.model_name.eq(name) & queries.k.eq(k)].sort_values("cell")
        np.testing.assert_array_equal(cell, np.sort(fixed_query))
        for column, value in panel.items():
            np.testing.assert_array_equal(stored[column], value[order])
        for column in identity[1:]:
            np.testing.assert_array_equal(stored[column], full.iloc[cell][column])
        _same(stored.y_true, truth[cell])
        if k == 0:
            _same(stored.y_pred, stored.base_pred)
            if "_integrated_" in name:
                arm = name.removesuffix(f"_integrated_{BASIS}")
                _same(stored.y_pred, full[f"{arm}_integrated_k0_pred"].to_numpy()[cell])
        reference_name = name.removesuffix(f"_{BASIS}")
        if reference_name in reference_mapping:
            prior = previous[previous.model_name.eq(f"{reference_mapping[reference_name]}_{BASIS}") & previous.k.eq(k)].sort_values("cell")
            for column in ("cell", "y_pred", "base_pred", "adaptation_delta", "regional_gamma"):
                np.testing.assert_array_equal(stored[column], prior[column])
            reference_checks.append({"model_name": name, "k": k, "rows": len(cell), "bitwise_exact": True})
        query_checks.append({"model_name": name, "k": k, "n_query": len(cell), "bitwise_exact": True,
            "inactive_final_fallback_rows": int(panel["aux_fallback"].sum()), "max_abs_difference": 0.})
    recorded = json.loads((run / "reference_checks.json").read_text())
    key = lambda row: (row["model_name"], row["k"])
    if sorted(recorded, key=key) != sorted(reference_checks, key=key):
        raise ValueError("Independent reference controls differ from recorded checks")
    for arm in INTEGRATED:
        candidate = _manual_base(bases["context"], bases[arm], memory, mixers[f"{arm}_integrated_{BASIS}"].selected_gamma(0))
        fallback = _manual_base(bases["context"], native, memory, mixers[f"point_integrated_{BASIS}"].selected_gamma(0))
        _same(full[f"{arm}_integrated_k0_pred"], candidate if arm == "point" else np.where(active, candidate, fallback))
    return {"run": run.name, "status": "verified", "completion_sha256": sha256_file(run / "complete.json"),
        "runtime_snapshot_hash": runtime, "n_full_grid": len(full), "query_replays": query_checks,
        "reference_controls": reference_checks, "head_checks": head_checks, "tree_checks": tree_checks,
        "parent_tree_max_abs_difference": parent_tree_difference, "source_validation_rows": len(rows),
        "direct_adapter_validation_refits": len(adapters), "integrated_mixer_validation_refits": len(mixers),
        "auxiliary_values_formula_and_doc_perturbation_exact": True,
        "source_550_and_augmented682_feature_digests_exact": True, "source_forest_only_oof": True,
        "active_only_calibration_independently_reconstructed": True,
        "inactive_final_prediction_base_delta_gamma_fallback_exact": True,
        "parent_verification_path": str(parent_report), "parent_verification_sha256": sha256_file(parent_report),
        "head_forest_backbone_refitted": False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--split-seeds", type=int, nargs="+", default=list(SPLITS))
    parser.add_argument("--seeds", type=int, nargs="+", default=list(SEEDS))
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    runtime = verify_runtime_snapshot(args.root)
    requested = [(s, r) for s in args.split_seeds for r in args.seeds]
    if not requested or len(requested) != len(set(requested)):
        raise ValueError("Replay requires a nonempty unique run panel")
    results = []
    for split, seed in requested:
        print(f"Replaying split{split}_seed{seed}", flush=True)
        results.append(verify_one(args.root, split, seed, runtime))
        gc.collect()
    output = args.output or args.root / "verification/replay_checks.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps({"checked_at": datetime.now(timezone.utc).isoformat(),
        "requested_runs": len(requested), "verified_runs": len(results), "expected_production_runs": 9,
        "all_nine_replayed": set(requested) == {(s, r) for s in SPLITS for r in SEEDS},
        "completed_production_runs_at_check": sum((args.root / "runs" / f"split{s}_seed{r}" / "complete.json").is_file()
            for s in SPLITS for r in SEEDS), "target_comparative_performance_analyzed": False,
        "head_forest_backbone_refitted": False, "neural_query_and_fallback_replay_rtol_atol": 0,
        "forest_replay_rtol": FOREST_RTOL, "forest_replay_atol": FOREST_ATOL,
        "verifier_sha256": sha256_file(__file__), "shared_verifier_helpers_sha256": {name: sha256_file(name) for name in (
            "scripts/verify_doc_distribution_head_v1.py", "scripts/verify_doc_daily_hydro_readout_v1.py",
            "scripts/verify_doc_regime_residual_v1.py", "scripts/verify_doc_daily_hydro_support_basis_v1.py")},
        "results": results}, indent=2)+"\n")
    print(f"Verified {len(results)} complete packages: {output}", flush=True)


if __name__ == "__main__":
    main()
