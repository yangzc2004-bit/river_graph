"""Replay matched source-loss DOC residuals without neural refitting."""
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
from verify_doc_encoder_residual_v1 import _encoder_scope, _head_layout, _raw_cache
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
from river_graph.models.kgml_local_transport import FIT_ROLES, build_rf_features
from river_graph.models.selective_encoder_residual import SelectiveEncoderResidual
from river_graph.models.support_aware_residual_transfer import (
    SupportAwareResidualTransfer,
)
from river_graph.models.support_shape_adapter import SupportShapeAdapter
from river_graph.models.unified_doc import UnifiedDOCReconstructor

SPLITS, SEEDS, KS = (142, 143, 144), (42, 43, 44), (0, 1, 3, 5)
ROOT = Path("experiments/phase4_transfer/doc_selective_residual_v1")
LOSS_SPECS = {
    "tail2": {"tail_weight": 2, "source_weighting": "cell", "ordinary_overprediction_penalty": 0.0},
    "mae": {"tail_weight": 1, "source_weighting": "cell", "ordinary_overprediction_penalty": 0.0},
    "selective": {"tail_weight": 2, "source_weighting": "cell", "ordinary_overprediction_penalty": 0.5},
    "station": {"tail_weight": 2, "source_weighting": "station", "ordinary_overprediction_penalty": 0.0},
}
ARMS = tuple(LOSS_SPECS)
SHAPES = ("constant", "gru_tuned_anchor")
INTERACTION_INDICES = (0, 2, 4, 28)
DIRECT_MODELS = tuple(f"{arm}_{shape}" for arm in ("context", *ARMS) for shape in SHAPES)
INTEGRATED_MODELS = tuple(f"{arm}_integrated_{shape}" for arm in ARMS for shape in SHAPES)
REFERENCE_MODELS = tuple(f"prior_{arm}_{shape}" for arm in ("encoder", "encoder_integrated", "ecological_affine") for shape in SHAPES)
MODELS = (*DIRECT_MODELS, *INTEGRATED_MODELS, *REFERENCE_MODELS)


def _source_objective(model, dataset, split, config, arm):
    spec = LOSS_SPECS[arm]
    for key, value in spec.items():
        if model._config()[key] != value:
            raise ValueError(f"Saved loss setting differs: {arm}/{key}")
    protocol = model.to_dict()["protocol"]
    if (protocol["ordinary_rule"] != "source_truth < source_training_Q90; training loss only"
            or protocol["penalty_prediction"] != "final nonnegative native concentration at residual scale 1"
            or protocol["normalization"] != "fixed full-source mean weight; never a minibatch weight mean"
            or protocol["validation_loss"] != "unweighted pooled fixed-query raw MAE"):
        raise ValueError("Source objective or validation selection semantics differ")
    cells = np.asarray(split["train"], dtype=np.int64)
    truth = np.asarray(dataset["y"], dtype=np.float64).ravel()[cells]
    source_y = torch.as_tensor(truth)
    tail = source_y >= config["q90_threshold_train"]
    months = dataset["y"].shape[1]
    if model.n_source_cells_ != len(cells) or model.n_source_tail_cells_ != int(tail.sum()):
        raise ValueError("Checkpoint source/tail counts differ from the selected training labels")
    # Fold-hidden caches reindex source stations contiguously. The weighting
    # depends only on station membership, so global cell identities are valid.
    weights = model._source_weights(cells, source_y, tail, months)
    expected = 1.0 + (spec["tail_weight"] - 1.0) * tail.numpy()
    stations, inverse = np.unique(cells // months, return_inverse=True)
    if spec["source_weighting"] == "station":
        totals = np.bincount(inverse, weights=expected)
        expected = expected / totals[inverse]
    np.testing.assert_array_equal(weights.numpy(), expected)
    station_totals = np.bincount(inverse, weights=weights.numpy())
    if spec["source_weighting"] == "station":
        np.testing.assert_allclose(station_totals, np.ones(len(stations)), rtol=0, atol=1e-12)
    # Check both error directions on source values without consulting query
    # labels or evaluating target performance. This does not fit any model.
    perturbation = torch.where(torch.arange(len(source_y)) % 2 == 0, .5, -.5).double()
    prediction = torch.maximum(source_y + perturbation, torch.zeros_like(source_y))
    error = prediction - source_y
    expected_error = weights * error.abs()
    if spec["ordinary_overprediction_penalty"]:
        expected_error = expected_error + .5 * (~tail).double() * torch.relu(error)
    actual_error = model._training_errors(prediction, source_y, tail, weights)
    torch.testing.assert_close(actual_error, expected_error, rtol=0, atol=0)
    np.testing.assert_allclose(float(actual_error.mean() / weights.mean()),
                               float(expected_error.sum() / weights.sum()), rtol=1e-14, atol=1e-14)
    # Pin the equality boundary: a true Q90 value belongs to the tail, not
    # the ordinary-overprediction group.
    boundary_y = torch.tensor([config["q90_threshold_train"]], dtype=torch.float64)
    boundary_tail = boundary_y >= config["q90_threshold_train"]
    boundary_weight = torch.tensor([float(spec["tail_weight"])])
    boundary_loss = model._training_errors(boundary_y + .5, boundary_y, boundary_tail, boundary_weight)
    torch.testing.assert_close(boundary_loss, .5 * boundary_weight.double(), rtol=0, atol=0)
    return {"source_cells": len(cells), "source_stations": len(stations), "source_tail_cells": int(tail.sum()),
            "specification": spec, "weights_match_independent_formula": True,
            "station_weight_total_min": float(station_totals.min()), "station_weight_total_max": float(station_totals.max()),
            "fixed_source_mean_weight": float(weights.mean()), "loss_terms_exact": True,
            "q90_equality_treated_as_tail": True, "inference_requires_group_labels": False}


def _tail2_control(run, prior, full, prior_full, queries, prior_queries):
    before = json.loads((prior / "last_self_ecology.json").read_text())
    after = json.loads((run / "tail2.json").read_text())
    old_config, new_config = before["config"], after["config"]
    common_new = {key: value for key, value in new_config.items()
                  if key not in ("source_weighting", "ordinary_overprediction_penalty")}
    if {**common_new, "epochs": old_config["epochs"]} != old_config:
        raise ValueError("Tail2 control changed settings beyond the maximum epoch cap")
    prefix = min(len(before["trace"]), len(after["trace"]))
    if before["trace"][:prefix] != after["trace"][:prefix]:
        raise ValueError("Tail2 control no longer reproduces the prior training trajectory")
    matched = common_new["epochs"] == old_config["epochs"]
    prior_stopped = before["trace"][-1]["stale_epochs"] >= old_config["patience"]
    cap_covers_stop = common_new["epochs"] >= before["epochs_run"]
    complete_equality = matched or (prior_stopped and cap_covers_stop)
    tensor_count = 0
    if complete_equality:
        for key, value in before.items():
            if key not in ("model_class", "config", "protocol") and after[key] != value:
                raise ValueError(f"Matched tail2 control changed summary field: {key}")
        for key, value in before["protocol"].items():
            if key != "training_loss" and after["protocol"][key] != value:
                raise ValueError(f"Matched tail2 control changed protocol: {key}")
        old_payload = torch.load(prior / "last_self_ecology.pt", weights_only=False, map_location="cpu")
        new_payload = torch.load(run / "tail2.pt", weights_only=False, map_location="cpu")
        for module in ("spatial", "temporal", "decay", "head", "initial_spatial", "initial_temporal", "initial_decay"):
            if set(old_payload[module]) != set(new_payload[module]):
                raise ValueError("Tail2 control tensor keys changed")
            for key, value in old_payload[module].items():
                torch.testing.assert_close(value, new_payload[module][key], rtol=0, atol=0)
                tensor_count += 1
        for suffix in ("delta", "pred", "integrated_k0_pred"):
            np.testing.assert_array_equal(full[f"tail2_{suffix}"], prior_full[f"last_self_ecology_{suffix}"])
        for shape_name in SHAPES:
            for middle, filename in (("", "adapters.json"), ("_integrated", "mixers.json")):
                old_name, new_name = f"last_self_ecology{middle}_{shape_name}", f"tail2{middle}_{shape_name}"
                a = prior_queries[prior_queries.model_name.eq(old_name)].copy()
                b = queries[queries.model_name.eq(new_name)].copy()
                a["model_name"] = new_name
                pd.testing.assert_frame_equal(a.sort_values(["k", "cell"]).reset_index(drop=True),
                                              b.sort_values(["k", "cell"]).reset_index(drop=True), check_exact=True)
                if json.loads((prior / filename).read_text())[old_name] != json.loads((run / filename).read_text())[new_name]:
                    raise ValueError("Tail2 control changed its validation support adaptation")
    return {"prior_run": str(prior), "prior_model": "last_self_ecology", "matched_epoch_cap": matched,
            "prior_patience_exhausted": prior_stopped, "current_cap_covers_prior_stopping_epoch": cap_covers_stop,
            "complete_control_equality_required": complete_equality,
            "exact_trace_prefix_rows": prefix, "trace_prefix_exact": True,
            "exact_control_tensor_checks": tensor_count,
            "full_grid_queries_adapters_unchanged": True if complete_equality else None,
            "scope": ("complete control, including an unchanged prior early stop under a larger cap"
                      if complete_equality else "trace prefix only; budgets do not imply equal final checkpoints")}


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


def verify_one(root, split_seed, seed, runtime):
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
            or config["loss_specs"] != LOSS_SPECS
            or config["encoder_dropout"] != "off in every arm"
            or config["encoder_normalization"] != "unchanged historical expert input normalization"
            or config["train_memory"] is not True or config["extra_dim"] != 30
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
    definition = {key: value for key, value in head_features.items()
                  if key not in ("source_extra", "full_extra", "source_station_ids")}
    definition["source_station_ids"] = head_features["source_station_ids"].tolist()
    if (definition != config["feature_definition"]
            or definition != json.loads((run / "feature_definition.json").read_text())):
        raise ValueError("Rebuilt and saved regime feature definitions differ")
    if definition != json.loads((sources["prior"] / "feature_definition.json").read_text()):
        raise ValueError("The loss comparison changed the prior feature definitions")
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
    for arm in ARMS:
        payload = torch.load(run / f"{arm}.pt", weights_only=False, map_location="cpu")
        summary = json.loads((run / f"{arm}.json").read_text())
        if payload["summary"] != summary:
            raise ValueError("Checkpoint summary and saved JSON differ")
        restored = SelectiveEncoderResidual.from_payload(payload)
        _summary_matches(restored.to_dict(), summary)
        loss_check = _source_objective(restored, dataset, split, config, arm)
        indices = INTERACTION_INDICES
        expected_width = 64 + 30 + 64 * len(indices)
        if (restored.hidden_size != 64 or restored.extra_dim != 30 or restored.head.in_features != expected_width
                or tuple(restored.interaction_indices) != indices or not restored.train_memory
                or summary["config"]["tail_weight"] != LOSS_SPECS[arm]["tail_weight"]
                or summary["tail_threshold"] != config["q90_threshold_train"]
                or summary["protocol"]["tail_rule"] != "source_truth >= source_training_Q90"):
            raise ValueError("Regime readout dimensions, interactions or training objective differ")
        head_parameters = sum(p.numel() for p in restored.head.parameters())
        scope = _encoder_scope(restored, payload, "last_self_ecology")
        trainable = sum(p.numel() for module in (restored.temporal, restored.decay, restored.head, restored.spatial)
                        for p in module.parameters() if p.requires_grad)
        if (head_parameters != expected_width + 1 or trainable != 25823 + scope["encoder_trainable_parameters"]
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
                          "interaction_layout": layout, "encoder_scope": scope, "source_objective": loss_check, "checkpoint_summary_json_exact": True,
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
    control = _tail2_control(run, sources["prior"], full, prior_full, queries, prior_queries)
    return {"run": run.name, "split_seed": split_seed, "seed": seed, "n_full_grid": len(full),
            "tail2_control": control,
            "feature_checks": feature_checks, "raw_cache_checks": raw_checks, "full_grid_max_abs_difference": differences,
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
    results = []
    for split, seed in requested:
        print(f"Replaying split{split}_seed{seed}", flush=True)
        results.append(verify_one(args.root, split, seed, runtime))
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
            )
        },
        "results": results,
    }, indent=2, allow_nan=False) + "\n")
    print(path)


if __name__ == "__main__":
    main()
