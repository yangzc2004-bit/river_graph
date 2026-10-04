"""Independent frozen-feature, distribution and support-adaptation replay."""
from __future__ import annotations

import argparse
import gc
import hashlib
import io
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from run_unified_doc_spatial import verify_files, verify_runtime_snapshot
from run_unified_doc_spatial_v2 import fit_adapters, read_source
from scipy.special import log_ndtr, logsumexp
from verify_doc_daily_hydro_readout_v1 import _source_and_full
from verify_doc_daily_hydro_support_basis_v1 import (
    _manual_base,
    _refit,
    _same,
    _sidecar,
    _validation_rows,
)
from verify_doc_recurrent_clock_v1 import _lineage
from verify_doc_regime_residual_v1 import _feature_views

from river_graph.experiments.provenance import sha256_file
from river_graph.experiments.unified_spatial_protocol import support_query_cells
from river_graph.models.distributional_residual_head import DistributionalResidualHead

ROOT = Path("experiments/phase4_transfer/doc_distribution_head_v1")
SPLITS, SEEDS, KS = (142, 143, 144), (42, 43, 44), (0, 1, 3, 5)
ARMS, SHAPES = ("point", "single", "mixture"), ("constant", "gru_tuned_anchor")
DIRECT = tuple(f"{arm}_{shape}" for arm in ("context", *ARMS) for shape in SHAPES)
INTEGRATED = tuple(f"{arm}_integrated_{shape}" for arm in ARMS for shape in SHAPES)
MODELS = (*DIRECT, *INTEGRATED)


def _array_digest(value):
    return hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()


def _head_feature_views(prior, dataset, split):
    """Rebuild actual source/full scalar-head vectors from the frozen expert."""
    config = json.loads((prior / "config.json").read_text())
    replay = _source_and_full(prior, dataset, split, batch_size=512)
    context = pd.read_parquet(prior / "full_grid.parquet", columns=["context_pred"]).context_pred.to_numpy()
    with np.load(Path(config["oof_run"]) / "source_oof.npz", allow_pickle=False) as saved:
        oof = saved["pred_z"].copy()
    extra, checks = _feature_views(dataset, split, context, oof)
    with np.load(config["daily_features_path"], allow_pickle=False) as saved:
        daily = saved["full"].copy()
    indices = (0, 2, 4, 28, 30, 31, 32)
    for prefix in ("source", "full"):
        daily_rows = daily[replay["source_station_ids"]] if prefix == "source" else daily
        extended = np.concatenate([extra[f"{prefix}_extra"], daily_rows], axis=-1)
        hidden = replay[f"{prefix}_hidden"]
        interaction = (hidden[..., :, None] * extended[..., indices][..., None, :]).reshape(
            *hidden.shape[:2], -1)
        features = np.concatenate([hidden, extended, interaction], axis=-1)
        if features.shape[-1] != 550 or features.dtype != np.float32 or not np.isfinite(features).all():
            raise ValueError("Expected finite frozen float32 scalar-head inputs with 550 columns")
        if prefix == "source" and _array_digest(extended) != replay["source_extra_sha256"]:
            raise ValueError("Source head extras differ from the actual source-native baseline inputs")
        replay[f"{prefix}_features"] = features
    replay["source_head_feature_checks"] = checks
    return replay


def _mixture_median(means, scales, weights):
    """Independent normal-mixture median via a component-median CDF bracket."""
    means, scales, weights = (np.asarray(x, dtype=np.float64) for x in (means, scales, weights))
    if (means.shape != scales.shape or means.shape != weights.shape or means.shape[-1] != 2
            or not all(np.isfinite(x).all() for x in (means, scales, weights))
            or (scales <= 0).any() or (weights < 0).any()):
        raise ValueError("Invalid two-normal distribution parameters")
    np.testing.assert_allclose(weights.sum(-1), 1, rtol=1e-14, atol=1e-14)
    lower, upper = means.min(-1), means.max(-1)
    # A convex combination of CDFs crosses one half between component medians.
    for _ in range(80):
        middle = lower + .5 * (upper-lower)
        z = (middle[..., None]-means)/scales
        below = z >= 0
        center = np.where(below, weights, 0).sum(-1)-.5
        with np.errstate(divide="ignore"):
            log_weight = np.log(weights)
            positive = np.concatenate([np.where(~below, log_weight+log_ndtr(z), -np.inf),
                np.where(center > 0, np.log(np.abs(center)), -np.inf)[..., None]], axis=-1)
            negative = np.concatenate([np.where(below, log_weight+log_ndtr(-z), -np.inf),
                np.where(center < 0, np.log(np.abs(center)), -np.inf)[..., None]], axis=-1)
        less = logsumexp(positive, axis=-1) < logsumexp(negative, axis=-1)
        lower = np.where(less, middle, lower)
        upper = np.where(less, upper, middle)
    return lower + .5 * (upper-lower)


def _distribution_math(raw, residual=None):
    """Evaluate Gaussian parameters and residual-space NLL independently."""
    raw = np.asarray(raw, dtype=np.float64)
    if raw.ndim != 2 or raw.shape[1] not in (2, 5) or not np.isfinite(raw).all():
        raise ValueError("A density head must provide two or five finite parameters")
    if raw.shape[1] == 2:
        means = raw[:, :1]
        scales = .03 + np.where(raw[:, 1:2] > 20, raw[:, 1:2], np.logaddexp(0, raw[:, 1:2]))
        weights = np.ones_like(means)
        log_weights = np.zeros_like(means)
        median = means[:, 0]
    else:
        gap = np.where(raw[:, 1] > 20, raw[:, 1], np.logaddexp(0, raw[:, 1]))
        means = np.stack([raw[:, 0], raw[:, 0]+gap], axis=-1)
        scales = .03 + np.where(raw[:, 2:4] > 20, raw[:, 2:4], np.logaddexp(0, raw[:, 2:4]))
        log_weights = np.stack([-np.logaddexp(0, raw[:, 4]), -np.logaddexp(0, -raw[:, 4])], axis=-1)
        weights = np.exp(log_weights)
        median = _mixture_median(means, scales, weights)
    result = {"means": means, "scales": scales, "weights": weights, "median": median}
    if residual is not None:
        residual = np.asarray(residual, dtype=np.float64)
        if residual.shape != (len(raw),) or not np.isfinite(residual).all():
            raise ValueError("Residual labels must be selected finite source/validation cells")
        normal_logp = -.5 * ((residual[:, None]-means)/scales)**2-np.log(scales)-.5*np.log(2*np.pi)
        result["nll"] = -logsumexp(log_weights+normal_logp, axis=-1)
    return result


def _density_replay(run, arm, config, full, full_features, full_base,
                    source_features, source_base, source_truth,
                    validation_features, validation_base, validation_truth):
    payload = torch.load(run / f"{arm}.pt", weights_only=True, map_location="cpu")
    summary = json.loads((run / f"{arm}.json").read_text())
    model = DistributionalResidualHead.from_payload(payload)
    components = 1 if arm == "single" else 2
    if (payload["summary"] != summary or model.to_dict() != summary
            or payload["model_class"] != "DistributionalResidualHead" or payload["schema_version"] != 1):
        raise ValueError("Density checkpoint/summary JSON roundtrip differs")
    expected_config = {"n_features": 550, "components": components, **{name: config[name]
        for name in ("epochs", "patience", "batch_size", "seed", "learning_rate")}}
    if (summary["config"] != expected_config or summary["selection_role"] != "source_validation"
            or summary["n_source_cells"] != len(source_features)
            or summary["n_validation_query"] != len(validation_features)
            or summary["trainable_parameter_count"] != config["head_parameters"][arm]
            or sum(p.numel() for p in model.head.parameters()) != config["head_parameters"][arm]):
        raise ValueError("Density dimensions, training population or parameter count differs")
    expected_definition = {"residual": "log1p(truth)-log1p(fixed_native_base)",
        "objective": "unweighted_source_gaussian_residual_nll", "point": "mixture_median",
        "median_iterations": 64, "median_bracket": "min(mu-12*sigma), max(mu+12*sigma)",
        "sigma_floor": .03, "sigma_upper_cap": None, "correction_scales": [0, .25, .5, 1],
        "scale_action": "max(0,expm1(log1p(base)+scale*median_residual))",
        "selection": "pooled_source_validation_native_mae", "selection_ties": "lower_scale_then_earlier_epoch",
        "trace_training_nll": "online_batch_loss; epoch0_full_source",
        "normalization_role": "source_training", "feature_std_floor": 1e-6}
    if summary["definition"] != expected_definition:
        raise ValueError("Density likelihood, median or selection definition changed")
    for key, value in model.head.state_dict().items():
        torch.testing.assert_close(value, payload["head_state"][key], rtol=0, atol=0)
        if value.dtype != torch.float64 or not torch.isfinite(value).all():
            raise ValueError("Head weights must be finite float64 tensors")
    source_tensor = torch.as_tensor(source_features, dtype=torch.float64)
    mean = source_tensor.mean(0)
    raw_std = source_tensor.std(0, correction=0)
    scale = torch.where(raw_std < 1e-6, torch.ones_like(raw_std), raw_std)
    for key, expected in (("feature_mean", mean), ("feature_raw_std", raw_std), ("feature_scale", scale)):
        np.testing.assert_array_equal(summary["normalization"][key], expected.numpy())
    if summary["normalization"]["unit_scale_feature_count"] != int((raw_std < 1e-6).sum()):
        raise ValueError("Source normalization floor count differs")
    source_residual = (torch.log1p(torch.as_tensor(source_truth))
                       - torch.log1p(torch.as_tensor(source_base))).numpy()
    sigma = max(float(source_residual.std()), .05)
    inverse_softplus = lambda value: float(value + np.log(-np.expm1(-value)))
    if components == 1:
        means = [float(source_residual.mean())]
        bias = [means[0], inverse_softplus(sigma-.03)]
    else:
        low, high = np.quantile(source_residual, [.25, .75], method="linear")
        high = max(float(high), float(low)+.05)
        means = [float(low), high]
        bias = [means[0], inverse_softplus(high-low), inverse_softplus(sigma-.03), inverse_softplus(sigma-.03), 0.0]
    initialization = {"means": means, "scales": [sigma]*components, "weights": [1/components]*components,
        "residual_mean": float(source_residual.mean()), "residual_std": float(source_residual.std()), "raw_bias": bias}
    if summary["initialization"] != initialization:
        raise ValueError("Density intercept initialization is not from the source residuals")
    validation_tensor = (torch.as_tensor(validation_features, dtype=torch.float64)-mean)/scale
    source_tensor = (source_tensor-mean)/scale
    with torch.inference_mode():
        source_raw = torch.nn.functional.linear(source_tensor, payload["head_state"]["weight"],
                                                payload["head_state"]["bias"]).numpy()
        validation_raw = torch.nn.functional.linear(validation_tensor, payload["head_state"]["weight"],
                                                    payload["head_state"]["bias"]).numpy()
    validation_residual = (torch.log1p(torch.as_tensor(validation_truth))
                           - torch.log1p(torch.as_tensor(validation_base))).numpy()
    independent_source = _distribution_math(source_raw, source_residual)
    independent_validation = _distribution_math(validation_raw, validation_residual)
    np.testing.assert_allclose(independent_source["nll"].mean(), summary["selected_source_nll"], rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(independent_validation["nll"].mean(), summary["validation_metrics"]["validation_nll"], rtol=1e-12, atol=1e-12)
    initial_source = _distribution_math(np.broadcast_to(bias, source_raw.shape).copy(), source_residual)
    initial_validation = _distribution_math(np.broadcast_to(bias, validation_raw.shape).copy(), validation_residual)
    np.testing.assert_allclose(initial_source["nll"].mean(), summary["trace"][0]["training_nll"], rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(initial_validation["nll"].mean(), summary["trace"][0]["validation_nll"], rtol=1e-12, atol=1e-12)
    for result, record in ((initial_validation, summary["trace"][0]),
                           (independent_validation, summary["validation_metrics"])):
        for correction, row in zip(config["correction_scales"], record["scale_scores"], strict=True):
            native = validation_base.copy() if correction == 0 else np.maximum(0,
                np.expm1(np.log1p(validation_base)+correction*result["median"]))
            if row["scale"] != correction:
                raise ValueError("Validation correction-scale order changed")
            np.testing.assert_allclose(np.abs(native-validation_truth).mean(), row["mae"], rtol=1e-11, atol=1e-11)
    trace = summary["trace"]
    if [row["epoch"] for row in trace] != list(range(summary["epochs_run"]+1)):
        raise ValueError("Density trace omits or duplicates an epoch")
    best, stale = (float("inf"), float("inf"), float("inf")), 0
    for row in trace:
        chosen = min(row["scale_scores"], key=lambda x: (x["mae"], x["scale"]))
        if chosen["mae"] != row["validation_mae"] or chosen["scale"] != row["selected_scale"]:
            raise ValueError("Per-epoch correction scale was not validation-MAE selected")
        rank = (chosen["mae"], chosen["scale"], row["epoch"])
        better = rank < best
        stale = 0 if better else stale+1
        if row["is_best"] != better or row["stale_epochs"] != stale:
            raise ValueError("Checkpoint selection/patience does not follow MAE/scale/epoch ties")
        if better:
            best = rank
    if (summary["best_epoch"] != best[2] or summary["selected_scale"] != best[1]
            or summary["validation_metrics"]["validation_mae"] != best[0]):
        raise ValueError("Saved checkpoint selection differs from its full validation trace")
    if summary["epochs_run"] < config["epochs"] and trace[-1]["stale_epochs"] < config["patience"]:
        raise ValueError("Density training ended before the stated budget/patience")
    pd.testing.assert_frame_equal(pd.read_csv(run / f"{arm}_trace.csv"),
        pd.read_csv(io.StringIO(pd.DataFrame(trace).to_csv(index=False))), check_dtype=False,
        check_exact=False, rtol=1e-14, atol=1e-14)
    for prefix, result in (("source", independent_source), ("validation", independent_validation)):
        recorded = summary[f"{prefix}_distribution"]
        if recorded["n_rows"] != len(result["median"]):
            raise ValueError("Selected density diagnostics use another population")
        for key in ("means", "scales", "weights"):
            for component in range(components):
                array = result[key][:, component]
                metrics = {"mean": float(array.mean()), "min": float(array.min()),
                    "p10": float(np.quantile(array, .1)), "median": float(np.median(array)),
                    "p90": float(np.quantile(array, .9)), "max": float(array.max())}
                np.testing.assert_allclose(list(metrics.values()), list(recorded[key][component].values()), rtol=1e-12, atol=1e-12)
    native = np.empty(len(full_features), dtype=np.float64)
    differences = {"means": 0., "scales": 0., "weights": 0., "median": 0., "native": 0.}
    for start in range(0, len(full_features), 512):
        stop = min(start+512, len(full_features))
        features = full_features[start:stop]
        distribution = model.predict_distribution(features)
        native[start:stop] = model.predict(features, full_base[start:stop])
        with torch.inference_mode():
            standardized = (torch.as_tensor(features, dtype=torch.float64)-mean)/scale
            raw = torch.nn.functional.linear(standardized, payload["head_state"]["weight"],
                                             payload["head_state"]["bias"]).numpy()
        independent = _distribution_math(raw)
        for key in ("means", "scales", "weights"):
            np.testing.assert_allclose(independent[key], distribution[key], rtol=1e-12, atol=1e-12)
            differences[key] = max(differences[key], float(np.abs(independent[key]-distribution[key]).max()))
            for component in range(components):
                _same(distribution[key][:, component], full[f"{arm}_{key}_{component}"].to_numpy()[start:stop])
        np.testing.assert_allclose(independent["median"], distribution["median_residual"], rtol=1e-10, atol=1e-10)
        differences["median"] = max(differences["median"], float(np.abs(independent["median"]-distribution["median_residual"]).max()))
        _same(distribution["median_residual"], full[f"{arm}_median_residual"].to_numpy()[start:stop])
        computed = full_base[start:stop].copy() if model.selected_scale_ == 0 else np.maximum(0,
            np.expm1(np.log1p(full_base[start:stop])+model.selected_scale_*independent["median"]))
        np.testing.assert_allclose(computed, native[start:stop], rtol=1e-10, atol=1e-10)
        differences["native"] = max(differences["native"], float(np.abs(computed-native[start:stop]).max()))
    _same(native, full[f"{arm}_pred"])
    return {"arm": arm, "components": components, "trainable_parameters": summary["trainable_parameter_count"],
        "checkpoint_tensor_identity_exact": True, "source_only_normalization_exact": True,
        "source_only_initialization_exact": True, "n_source_cells": len(source_features),
        "n_validation_query": len(validation_features), "independent_likelihood_rtol": 1e-12,
        "independent_likelihood_atol": 1e-12,
        "independent_median_solver": ("single Gaussian location" if components == 1 else
                                      "80-step component-median bracket, scipy log-tail CDF"),
        "independent_median_native_rtol": 1e-10, "independent_median_native_atol": 1e-10,
        "independent_max_abs_difference": differences, "full_grid_loaded_state_replay_bitwise_exact": True,
        "epoch_zero_and_selected_validation_choices_verified": True,
        "online_training_nll_not_treated_as_final_source_nll": True}, native


def _adaptation_replay(run, prior, basis_run, full, queries, dataset, split, bases):
    """Refit only validation adapters and replay the unchanged target supports."""
    shape = tuple(dataset["y"].shape)
    months = shape[1]
    context, memory = full.context_pred.to_numpy(), full.ecological_memory.to_numpy()
    with np.load(basis_run / "representations.npz", allow_pickle=False) as saved:
        shapes = {name: saved[name].copy() for name in SHAPES}
    direct_states = json.loads((run / "adapters.json").read_text())
    mixer_states = json.loads((run / "mixers.json").read_text())
    if set(direct_states) != set(DIRECT) or set(mixer_states) != set(INTEGRATED):
        raise ValueError("Incomplete fixed-basis support adaptation panel")
    validation = np.full(np.prod(shape), np.nan)
    validation[split["val"]] = np.asarray(dataset["y"]).ravel()[split["val"]]
    direct = fit_adapters({"context": context}, shapes, validation, split, months)
    if any(model.to_dict() != direct_states[name] for name, model in direct.items()):
        raise ValueError("Validation-only context adapter refit differs")
    mixers, val_rows = {}, []
    for arm in ARMS:
        adapters, mixture = _refit(arm, bases[arm], shapes, context, memory, validation,
            split, months, direct_states, mixer_states)
        direct.update(adapters)
        mixers.update(mixture)
        val_rows.extend(_validation_rows(arm, shapes, bases[arm], context, memory, validation,
            split, months, direct, mixers))
    stored_validation = pd.read_csv(run / "source_validation.csv")
    if len(stored_validation) != 48:
        raise ValueError("Expected 48 source-validation adaptation panels")
    pd.testing.assert_frame_equal(stored_validation, pd.DataFrame(val_rows), check_dtype=False,
                                  check_exact=False, rtol=1e-14, atol=1e-14)
    old_direct = json.loads((prior / "adapters.json").read_text())
    old_mixers = json.loads((prior / "mixers.json").read_text())
    for name in SHAPES:
        if (direct_states[f"point_{name}"] != old_direct[f"off_{name}"]
                or mixer_states[f"point_integrated_{name}"] != old_mixers[f"off_integrated_{name}"]):
            raise ValueError("The fixed point model's support or ecological adjustment changed")
    previous = pd.read_parquet(prior / "predictions.parquet")
    reserved, fixed_query = support_query_cells(split, target_role="test", k=5, n_months=months)
    labels = np.full(np.prod(shape), np.nan)
    labels[reserved] = np.asarray(dataset["y"]).ravel()[reserved]
    panels, controls = [], []
    for arm in ("context", *ARMS):
        for shape_name, basis in shapes.items():
            for stage in (("",) if arm == "context" else ("", "_integrated")):
                name = f"{arm}{stage}_{shape_name}"
                for k in KS:
                    support, query = support_query_cells(split, target_role="test", k=k, n_months=months)
                    np.testing.assert_array_equal(query, fixed_query)
                    if len(support) != k*len(np.unique(query//months)) or np.intersect1d(support, query).size:
                        raise ValueError("Support identity or fixed target query changed")
                    stored = queries[queries.model_name.eq(name) & queries.k.eq(k)].sort_values("cell")
                    np.testing.assert_array_equal(stored.cell, np.sort(query))
                    query = stored.cell.to_numpy()
                    if stage:
                        model = mixers[name]
                        gamma = model.selected_gamma(k)
                        base = _manual_base(context, bases[arm], memory, gamma)
                        _same(base, model.selected_base(context, bases[arm], memory, k=k))
                        pred = model.adapt(context[query], bases[arm][query], memory[query], query,
                            context[support], bases[arm][support], memory[support], support,
                            labels[support], basis[query], basis[support], k=k)
                        if k == 0:
                            _same(base, full[f"{arm}_integrated_k0_pred"])
                    else:
                        gamma, base = 0, bases[arm]
                        pred = direct[name].adapt(base[query], query, base[support], support, labels[support],
                            query_basis=basis[query], support_basis=basis[support], k=k)
                    for expected, column in ((pred, "y_pred"), (base[query], "base_pred"),
                            (np.log1p(pred)-np.log1p(base[query]), "adaptation_delta"),
                            (np.full(len(query), gamma), "regional_gamma"), (np.full(len(query), k), "support_count"),
                            (np.asarray(dataset["y"]).ravel()[query], "y_true")):
                        _same(expected, stored[column])
                    for column in ("station", "month", "analyte", "visibility_role"):
                        np.testing.assert_array_equal(stored[column], full.iloc[query][column])
                    if k == 0:
                        _same(pred, base[query])
                    if arm == "point":
                        parent = previous[previous.model_name.eq(f"off{stage}_{shape_name}")
                                          & previous.k.eq(k)].sort_values("cell")
                        for column in ("cell", "y_pred", "base_pred", "adaptation_delta", "regional_gamma"):
                            np.testing.assert_array_equal(stored[column], parent[column])
                        controls.append({"model_name": name, "k": k, "rows": len(query), "bitwise_exact": True})
                    panels.append({"model_name": name, "k": k, "n_query": len(query),
                                   "bitwise_exact": True, "max_abs_difference": 0.0})
    return {"query_replays": panels, "point_controls": controls,
            "direct_adapter_validation_refits": len(direct), "integrated_mixer_validation_refits": len(mixers),
            "source_validation_rows": len(stored_validation), "n_fixed_query": len(fixed_query)}


def verify_one(root, split_seed, seed, runtime):
    run = root / "runs" / f"split{split_seed}_seed{seed}"
    config = json.loads((run / "config.json").read_text())
    expected = {"experiment": "doc_distribution_head_v1", "split_seed": split_seed, "seed": seed,
        "runtime_snapshot_hash": runtime, "arms": list(ARMS), "basis_names": list(SHAPES),
        "k_values": list(KS), "feature_dim": 550, "feature_batch_size": 512, "extra_dim": 38,
        "interaction_indices": [0, 2, 4, 28, 30, 31, 32], "inference_roles": ["train"],
        "selection_role": "source_validation", "sigma_floor": .03, "sigma_initial_floor": .05,
        "feature_std_floor": 1e-6, "median_iterations": 64, "median_bracket_sigma": 12,
        "correction_scales": [0, .25, .5, 1], "head_parameters": {"single": 1102, "mixture": 2755},
        "backbone_retraining": False, "forest_retraining": False, "readout_fitting": False,
        "learning_rate": .001, "batch_size": 512, "gradient_clip_norm": 1}
    if any(config[name] != value for name, value in expected.items()) or set(config["models"]) != set(MODELS):
        raise ValueError("Frozen density experiment identity or definitions changed")
    completion = verify_files(run, "complete.json", config)
    model_files = {"single.pt", "mixture.pt", "single.json", "mixture.json", "adapters.json",
        "mixers.json", "feature_definition.json", "source_training.npz", "input_definition.json"}
    required = {*model_files, "config.json", "single_trace.csv", "mixture_trace.csv", "source_validation.csv",
        "point_checks.json", "predictions.parquet", "predictions.meta.json", "full_grid.parquet",
        "full_grid.meta.json", "timing.json"}
    if set(completion["files"]) != required:
        raise ValueError("Density package omits or adds an unexpected product")
    sources, pc, parent_report = _lineage(config)
    old_config, _, dataset, split, original_full = read_source(sources["source"])
    for name in ("dataset_hash", "mask_hash", "q90_threshold_train", "query_cells"):
        if config[name] != old_config[name]:
            raise ValueError("Cohort, query population or source threshold changed")
    shape = tuple(dataset["y"].shape)
    months = shape[1]
    full = pd.read_parquet(run / "full_grid.parquet")
    queries = pd.read_parquet(run / "predictions.parquet")
    previous = pd.read_parquet(sources["prior"] / "full_grid.parquet")
    for filename, frame in (("full_grid.parquet", full), ("predictions.parquet", queries)):
        _sidecar(run, filename, config, runtime, frame, completion, model_files)
    np.testing.assert_array_equal(full.cell, np.arange(np.prod(shape)))
    identities = ["cell", "station", "month", "analyte", "visibility_role", "ecological_novelty", "upstream_support"]
    pd.testing.assert_frame_equal(full[identities], original_full[identities], check_exact=True)
    if (queries.duplicated(["model_name", "k", "cell"]).any()
            or set(map(tuple, queries[["model_name", "k"]].drop_duplicates().to_numpy()))
            != {(name, k) for name in MODELS for k in KS}
            or not queries.seed.eq(seed).all() or not queries.split_seed.eq(split_seed).all()
            or not np.isfinite(full.select_dtypes(include="number")).all().all()):
        raise ValueError("Full-grid values or complete query identities differ")
    _same(full.context_pred, previous.context_pred)
    _same(full.context_pred, original_full.context_pred)
    _same(full.ecological_memory, previous.ecological_memory)
    _same(full.point_pred, previous.off_pred)
    if (config["feature_definition"] != pc["feature_definition"]
            or (run / "feature_definition.json").read_bytes() != (sources["prior"] / "feature_definition.json").read_bytes()):
        raise ValueError("Frozen native feature definition changed")
    torch.set_num_threads(int(config["torch_threads"]))
    replay = _head_feature_views(sources["prior"], dataset, split)
    source_ids = replay["source_station_ids"]
    local_cells = np.flatnonzero(replay["source_mask"])
    source_cells = source_ids[local_cells // months] * months + local_cells % months
    np.testing.assert_array_equal(source_cells, np.sort(split["train"]))
    _, validation_cells = support_query_cells(split, target_role="val", k=0, n_months=months)
    source_features = replay["source_features"].reshape(-1, 550)[local_cells]
    validation_features = replay["full_features"].reshape(-1, 550)[validation_cells]
    source_base = replay["source_native"].ravel()[local_cells]
    full_base = replay["full_native"].ravel()
    validation_base = full_base[validation_cells]
    with np.load(run / "source_training.npz", allow_pickle=False) as saved:
        expected_arrays = {"source_cells": source_cells, "source_local_cells": local_cells,
            "source_station_ids": source_ids, "source_base": source_base,
            "validation_cells": validation_cells, "validation_base": validation_base}
        if set(saved.files) != set(expected_arrays):
            raise ValueError("Saved source/validation native input schema changed")
        for name, value in expected_arrays.items():
            np.testing.assert_array_equal(saved[name], value)
    input_definition = json.loads((run / "input_definition.json").read_text())
    expected_definition = {"feature_dim": 550, "feature_order": "hidden64,extra38,hidden-major64x7 interactions",
        "source_feature_sha256": _array_digest(source_features),
        "validation_feature_sha256": _array_digest(validation_features),
        "source_base_sha256": _array_digest(source_base), "source_cells_sha256": _array_digest(source_cells),
        "source_feature_shape": list(source_features.shape), "validation_feature_shape": list(validation_features.shape),
        "source_station_ids": source_ids.tolist(), "source_neural_is_oof": False,
        "source_visibility": "receiving station-fold hidden", "full_visibility": "train DOC only"}
    if expected_definition != input_definition:
        raise ValueError("Independently rebuilt source/base/feature identities differ")
    truth = np.asarray(dataset["y"], dtype=float).ravel()
    density_checks, bases = [], {"context": full.context_pred.to_numpy(), "point": full_base}
    for arm in ("single", "mixture"):
        check, native = _density_replay(run, arm, config, full, replay["full_features"].reshape(-1, 550),
            full_base, source_features, source_base, truth[source_cells], validation_features,
            validation_base, truth[validation_cells])
        density_checks.append(check)
        bases[arm] = native
    del replay, source_features, validation_features
    adaptation = _adaptation_replay(run, sources["prior"], sources["basis"], full, queries, dataset, split, bases)
    if adaptation["n_fixed_query"] != config["query_cells"]:
        raise ValueError("Fixed target query count differs")
    recorded = json.loads((run / "point_checks.json").read_text())
    by_identity = lambda row: (row["model_name"], row["k"])
    if sorted(recorded, key=by_identity) != sorted(adaptation["point_controls"], key=by_identity):
        raise ValueError("Recorded point controls differ from independently replayed controls")
    return {"run": run.name, "status": "verified", "completion_sha256": sha256_file(run / "complete.json"),
        "runtime_snapshot_hash": runtime, "n_full_grid": len(full), **adaptation,
        "density_head_checks": density_checks, "source_features_and_native_baseline_bitwise_exact": True,
        "source_station_fold_hidden": True, "forest_only_oof": True,
        "source_validation_refits_receive_nan_elsewhere": True,
        "parent_verification_path": str(parent_report), "parent_verification_sha256": sha256_file(parent_report),
        "neural_forest_or_density_refitted": False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--split-seeds", type=int, nargs="+", default=list(SPLITS))
    parser.add_argument("--seeds", type=int, nargs="+", default=list(SEEDS))
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    runtime = verify_runtime_snapshot(args.root)
    requested = [(split, seed) for split in args.split_seeds for seed in args.seeds]
    if not requested or len(requested) != len(set(requested)):
        raise ValueError("Replay must request a nonempty unique run panel")
    results = []
    for split, seed in requested:
        print(f"Replaying split{split}_seed{seed}", flush=True)
        results.append(verify_one(args.root, split, seed, runtime))
        gc.collect()
    output = args.output or args.root / "verification/replay_checks.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps({"checked_at": datetime.now(timezone.utc).isoformat(),
        "requested_runs": len(requested), "verified_runs": len(results),
        "all_nine_replayed": set(requested) == {(s, r) for s in SPLITS for r in SEEDS},
        "expected_production_runs": 9, "completed_production_runs_at_check": sum(
            (args.root / "runs" / f"split{s}_seed{r}" / "complete.json").is_file() for s in SPLITS for r in SEEDS),
        "target_comparative_performance_analyzed": False, "neural_forest_or_density_refitted": False,
        "verifier_sha256": sha256_file(__file__), "shared_verifier_helpers_sha256": {
            name: sha256_file(name) for name in ("scripts/verify_doc_daily_hydro_readout_v1.py",
                "scripts/verify_doc_daily_hydro_support_basis_v1.py", "scripts/verify_doc_regime_residual_v1.py",
                "scripts/verify_doc_recurrent_clock_v1.py", "scripts/verify_doc_ecological_transfer_v2.py")},
        "results": results}, indent=2)+"\n")
    print(f"Verified {len(results)} complete packages: {output}", flush=True)


if __name__ == "__main__":
    main()
