"""Replay saved regime-conditioned DOC residuals without neural refitting."""
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
from verify_doc_tail_residual_v1 import (
    ATOL,
    RTOL,
    _close,
    _full_inputs,
    _summary_matches,
)

from river_graph.experiments.provenance import run_identity_sha256, sha256_file
from river_graph.experiments.unified_spatial_protocol import support_query_cells
from river_graph.models.causal_flow_features import build_causal_flow_features
from river_graph.models.ecological_residual_transfer import EcologicalResidualTransfer
from river_graph.models.episodic_temporal_adapter import gathered_rolling_states
from river_graph.models.kgml_local_transport import FIT_ROLES, build_rf_features
from river_graph.models.native_temporal_residual import NativeTemporalResidual
from river_graph.models.regime_head_features import build_regime_head_features
from river_graph.models.support_aware_residual_transfer import (
    SupportAwareResidualTransfer,
)
from river_graph.models.support_shape_adapter import SupportShapeAdapter
from river_graph.models.unified_doc import UnifiedDOCReconstructor

SPLITS, SEEDS, KS = (142, 143, 144), (42, 43, 44), (0, 1, 3, 5)
ROOT = Path("experiments/phase4_transfer/doc_regime_residual_v1")
ARMS = ("additive", "concentration", "ecological")
SHAPES = ("constant", "gru_tuned_anchor")
INTERACTION_INDICES = {"additive": (0, 2, 4), "concentration": (0, 2, 4, 28),
                       "ecological": (0, 2, 4, *range(10, 19), 28)}
TRAINABLE_PARAMETERS = {"additive": 25759, "concentration": 25823, "ecological": 26399}
DIRECT_MODELS = tuple(f"{arm}_{shape}" for arm in ("context", *ARMS) for shape in SHAPES)
INTEGRATED_MODELS = tuple(f"{arm}_ecology_{shape}" for arm in ARMS for shape in SHAPES)
REFERENCE_MODELS = tuple(f"prior_{arm}_{shape}" for arm in ("interaction", "ecological_affine") for shape in SHAPES)
MODELS = (*DIRECT_MODELS, *INTEGRATED_MODELS, *REFERENCE_MODELS)


def _interaction_layout(model, inputs, indices):
    """Independently check sampled head inputs against the named outer product."""
    prepared = model._prepare_inputs(inputs)
    n, months = prepared["age"].shape
    cells = torch.as_tensor(np.unique(np.linspace(0, n * months - 1, 64, dtype=int)))
    observed = []
    hook = model.head.register_forward_pre_hook(
        lambda _module, args: observed.append(args[0].detach().clone()))
    try:
        with torch.inference_mode():
            hidden = gathered_rolling_states(
                model.temporal, model.decay, prepared, cells // months, cells % months,
                lookback=model.lookback)
            extra = prepared["extra"][cells // months, cells % months]
            # Extra-feature indices vary fastest within each hidden unit.
            interaction = (hidden[:, :, None] * extra[:, indices][:, None, :]).flatten(1)
            expected = torch.cat((hidden, extra, interaction), dim=1)
            model._delta_cells(prepared, cells)
        if len(observed) != 1:
            raise ValueError("Expected one scalar-head call for layout verification")
        torch.testing.assert_close(observed[0], expected, rtol=0, atol=0)
    finally:
        hook.remove()
    return {"sampled_cells": len(cells), "input_features": expected.shape[1],
            "ordering": "hidden-major outer product", "bitwise_exact": True}


def _feature_views(dataset, split, context, oof_z):
    """Reconstruct and independently check source-OOF and full-context views."""
    n, months = dataset["y"].shape
    cells = np.asarray(split["train"], dtype=np.int64)
    source_mask = np.zeros(n * months, dtype=bool)
    source_mask[cells] = True
    if (oof_z.shape != (n, months) or not np.isfinite(oof_z.ravel()[source_mask]).all()
            or not np.isnan(oof_z.ravel()[~source_mask]).all()):
        raise ValueError("Context OOF coverage differs from source training cells")
    source_prediction = np.maximum(0, np.expm1(oof_z.ravel()[cells]))
    flow = build_causal_flow_features(dataset)
    features = build_regime_head_features(
        np.asarray(dataset["regime"]), cells, source_prediction,
        context.reshape(n, months), flow["full"], n_months=months)
    source_ids, inverse, counts = np.unique(cells // months, return_inverse=True, return_counts=True)
    np.testing.assert_array_equal(features["source_station_ids"], source_ids)
    full, source = features["full_extra"], features["source_extra"]
    if (full.shape != (n, months, 30) or source.shape != (len(source_ids), months, 30)
            or full.dtype != np.float32 or source.dtype != np.float32
            or not np.isfinite(full).all() or not np.isfinite(source).all()):
        raise ValueError("Expected finite float32 source/full feature views with thirty channels")
    np.testing.assert_array_equal(full[..., :10], flow["full"])
    np.testing.assert_array_equal(source[..., :28], full[source_ids, :, :28])
    raw = np.asarray(dataset["regime"], dtype=np.float64)[:, 4:13]
    valid = np.isfinite(raw) & (raw != -1.0)
    median, iqr, active = np.zeros(9), np.ones(9), np.zeros(9, dtype=bool)
    for j in range(9):
        observed = raw[source_ids, j][valid[source_ids, j]]
        if len(observed):
            active[j] = True
            median[j] = np.median(observed)
            lo, hi = np.quantile(observed, [.25, .75])
            iqr[j] = hi - lo if hi > lo else 1.0
    ecology_difference = np.where(valid, raw, median) - median
    compressed = ecology_difference / (iqr + np.abs(ecology_difference))
    compressed[:, ~active] = 0.0
    compressed = compressed.astype(np.float32)
    np.testing.assert_array_equal(full[..., 10:19], np.broadcast_to(compressed[:, None], (n, months, 9)))
    np.testing.assert_array_equal(full[..., 19:28], np.broadcast_to(valid[:, None], (n, months, 9)))
    weights = 1.0 / (len(source_ids) * counts[inverse])
    log_context = np.log1p(source_prediction)
    mean = float(weights @ log_context)
    sd = max(float(np.sqrt(weights @ ((log_context - mean) ** 2))), 1e-6)
    full_difference = np.log1p(context.reshape(n, months)) - mean
    expected_full = (full_difference / (sd + np.abs(full_difference))).astype(np.float32)
    np.testing.assert_array_equal(full[..., 28], expected_full)
    np.testing.assert_array_equal(full[..., 29], np.ones((n, months), dtype=np.float32))
    source_difference = log_context - mean
    expected_source = np.zeros((len(source_ids), months), dtype=np.float32)
    expected_valid = np.zeros_like(expected_source)
    positions = np.searchsorted(source_ids, cells // months)
    expected_source[positions, cells % months] = source_difference / (sd + np.abs(source_difference))
    expected_valid[positions, cells % months] = 1.0
    np.testing.assert_array_equal(source[..., 28], expected_source)
    np.testing.assert_array_equal(source[..., 29], expected_valid)
    np.testing.assert_array_equal(features["normalization"]["ecology"]["median"], median)
    np.testing.assert_array_equal(features["normalization"]["ecology"]["iqr"], iqr)
    np.testing.assert_array_equal(features["normalization"]["ecology"]["active"], active)
    np.testing.assert_array_equal(features["normalization"]["context"]["log_mean"], mean)
    np.testing.assert_array_equal(features["normalization"]["context"]["log_sd"], sd)
    if features["interaction_indices"] != {name: list(indices) for name, indices in INTERACTION_INDICES.items()}:
        raise ValueError("Feature interactions differ from the three planned head contrasts")
    return features, {"source_stations": len(source_ids), "source_cells": len(cells),
                      "source_only_ecology_and_context_scaling_verified": True,
                      "source_context_is_oof_at_loss_cells": True,
                      "off_loss_source_context_and_validity_zero": True,
                      "full_context_valid_everywhere": True,
                      "missing_ecology_station_ids": np.flatnonzero(~valid.any(axis=1)).tolist(),
                      "full_and_source_extra_finite": True}


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
    expected_interactions = {name: list(indices) for name, indices in INTERACTION_INDICES.items()}
    if (set(config["models"]) != set(MODELS) or tuple(config["arms"]) != ARMS
            or config["interaction_indices"] != expected_interactions
            or config["interaction_order"] != "hidden-major outer product"
            or config["train_memory"] is not True or config["extra_dim"] != 30
            or config["k_values"] != list(KS) or config["inference_roles"] != ["train"]):
        raise ValueError("Regime head or visibility definitions differ")
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
    if (Path(source_configs["prior"]["prior_run"]) != sources["memory"]
            or source_configs["prior"]["prior_completion_hash"] != config["memory_completion_hash"]
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
    profile = EcologicalResidualTransfer.from_dict(
        json.loads((sources["memory"] / "ecological_affine.json").read_text()))
    memory = profile.predict_delta(context.reshape(shape)).ravel()
    np.testing.assert_array_equal(memory, full.ecological_memory)
    np.testing.assert_array_equal(memory, prior_full.ecological_affine_memory)

    torch.set_num_threads(int(config["torch_threads"]))
    expert = UnifiedDOCReconstructor.load(sources["source"], dataset, split)
    np.testing.assert_array_equal(np.sort(np.concatenate(expert.rf.folds)), head_features["source_station_ids"])
    expert.context_forest.n_jobs = 1
    rf_features = build_rf_features(dataset, split, FIT_ROLES, target_transform="log1p", include_network=True)
    recomputed_context = np.maximum(0, np.expm1(expert.context_forest.predict(rf_features)))
    differences = {"context_pred": _close(recomputed_context, context)}
    del rf_features, oof
    full_inputs = {**_full_inputs(expert), "extra": head_features["full_extra"]}
    bases, summaries, tensor_checks = {"context": context}, [], 0
    for arm in ARMS:
        payload = torch.load(run / f"{arm}.pt", weights_only=False, map_location="cpu")
        summary = json.loads((run / f"{arm}.json").read_text())
        if payload["summary"] != summary:
            raise ValueError("Checkpoint summary and saved JSON differ")
        restored = NativeTemporalResidual.from_payload(payload)
        _summary_matches(restored.to_dict(), summary)
        indices = INTERACTION_INDICES[arm]
        expected_width = 64 + 30 + 64 * len(indices)
        if (restored.hidden_size != 64 or restored.extra_dim != 30 or restored.head.in_features != expected_width
                or tuple(restored.interaction_indices) != indices or not restored.train_memory
                or summary["config"]["tail_weight"] != 2 or config["tail_weight"] != 2
                or summary["tail_threshold"] != config["q90_threshold_train"]
                or summary["protocol"]["tail_rule"] != "source_truth >= source_training_Q90"):
            raise ValueError("Regime readout dimensions, interactions or training objective differ")
        head_parameters = sum(p.numel() for p in restored.head.parameters())
        trainable = sum(p.numel() for module in (restored.temporal, restored.decay, restored.head)
                        for p in module.parameters() if p.requires_grad)
        if (head_parameters != expected_width + 1 or trainable != TRAINABLE_PARAMETERS[arm]
                or summary["trainable_parameter_count"] != trainable
                or not all(p.requires_grad for module in (restored.temporal, restored.decay, restored.head)
                           for p in module.parameters())):
            raise ValueError("Regime model trainable parameter count differs")
        for key in ("lookback", "epochs", "patience", "batch_size", "learning_rate", "head_learning_rate"):
            if summary["config"][key] != config[key]:
                raise ValueError(f"Checkpoint setting differs from run: {key}")
        if summary["config"]["scales"] != config["residual_scales"]:
            raise ValueError("Residual scale grid differs")
        for module_name in ("temporal", "decay", "head"):
            for key, tensor in getattr(restored, module_name).state_dict().items():
                torch.testing.assert_close(tensor, payload[module_name][key], rtol=0, atol=0)
                tensor_checks += 1
        for module_name in ("temporal", "decay"):
            for key, tensor in getattr(expert.residual.model, module_name).state_dict().items():
                torch.testing.assert_close(tensor, payload[f"initial_{module_name}"][key], rtol=0, atol=0)
                tensor_checks += 1
        layout = _interaction_layout(restored, full_inputs, indices)
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
                          "interaction_layout": layout, "checkpoint_summary_json_exact": True,
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
            name = f"{arm}_ecology_{shape_name}"
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
                    np.testing.assert_array_equal(base, full[f"{arm}_ecology_k0_pred"])
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
    reference = prior_queries[prior_queries.model_name.isin(
        [name.removeprefix("prior_") for name in REFERENCE_MODELS])].copy()
    reference["model_name"] = "prior_" + reference.model_name
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
            "feature_checks": feature_checks, "full_grid_max_abs_difference": differences,
            "query_replays": query_checks, "checkpoint_summaries": summaries, "mixer_checks": mixer_checks,
            "direct_adapter_validation_refits": len(adapters), "exact_tensor_checks": tensor_checks,
            "context_forest_recomputation_bitwise_exact": bool(np.array_equal(recomputed_context, context)),
            "regime_features_neural_outputs_and_adapted_queries_bitwise_exact": True,
            "frozen_v2_reference_queries_bitwise_unchanged": True, "source_bank_refitting": "none",
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
            )
        },
        "results": results,
    }, indent=2, allow_nan=False) + "\n")
    print(path)


if __name__ == "__main__":
    main()
