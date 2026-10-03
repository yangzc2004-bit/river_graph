"""Rebuild ecological DOC residual transfer from source data and replay products.

Existing forest and neural packages are checked through their bound completion
records. This verifier refits only the small ecological transfer models from
source OOF residuals and source-validation queries; target queries are never
used for fitting or selection.
"""
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
from run_unified_doc_spatial_v2 import make_predictions, read_source

from river_graph.experiments.provenance import run_identity_sha256, sha256_file
from river_graph.experiments.unified_spatial_protocol import support_query_cells
from river_graph.models.ecological_residual_transfer import EcologicalResidualTransfer
from river_graph.models.support_shape_adapter import SupportShapeAdapter

ROOT = Path("experiments/phase4_transfer/doc_ecological_transfer_v1")
ARMS = ("global_bias", "ecological_bias", "global_affine", "ecological_affine")
SHAPES = ("constant", "gru_tuned_anchor")
MODELS = tuple(f"{base}_{shape}" for base in ("interaction", *ARMS)
               for shape in SHAPES)
SPLITS, SEEDS, KS = (142, 143, 144), (42, 43, 44), (0, 1, 3, 5)


def _same(actual, expected):
    actual, expected = np.asarray(actual), np.asarray(expected)
    if not np.isfinite(actual).all() or not np.isfinite(expected).all():
        raise ValueError("Nonfinite replay or saved prediction")
    np.testing.assert_array_equal(actual, expected)
    return float(np.max(np.abs(actual - expected))) if actual.size else 0.0


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
    required = {"adapters.json", *(f"{name}.json" for name in ARMS)}
    if set(sidecar["model_files"]) != required:
        raise ValueError("Sidecar does not bind every ecological transfer model and support adapter")
    for name, expected_hash in sidecar["model_files"].items():
        if sha256_file(run / name) != expected_hash or completion["files"].get(name) != expected_hash:
            raise ValueError(f"Changed model state: {run / name}")


def _packages(config):
    sources, source_configs = {}, {}
    for label in ("source", "prior", "basis", "oof"):
        package = Path(config[f"{label}_run"])
        package_config = json.loads((package / "config.json").read_text())
        if sha256_file(package / "complete.json") != config[f"{label}_completion_hash"]:
            raise ValueError(f"Changed {label} package binding")
        verify_files(package, "complete.json", package_config)
        sources[label], source_configs[label] = package, package_config
    for label in ("source", "basis", "oof"):
        if (Path(source_configs["prior"][f"{label}_run"]) != sources[label]
                or source_configs["prior"][f"{label}_completion_hash"]
                != config[f"{label}_completion_hash"]):
            raise ValueError(f"Prior and current {label} lineage differ")
    return sources, source_configs


def _source_bank(sources, split, dataset):
    shape = tuple(dataset["y"].shape)
    months = shape[1]
    train = np.asarray(split["train"], dtype=np.int64)
    source_ids = np.unique(train // months)
    with np.load(sources["oof"] / "source_oof.npz", allow_pickle=False) as saved:
        z = saved["pred_z"].copy()
    if z.shape != shape:
        raise ValueError("Source OOF grid dimensions differ")
    source_mask = np.zeros(np.prod(shape), dtype=bool)
    source_mask[train] = True
    if not np.isfinite(z.ravel()[source_mask]).all() or not np.isnan(z.ravel()[~source_mask]).all():
        raise ValueError("Context OOF predictions must cover exactly the source training cells")
    folds = json.loads((sources["oof"] / "oof_folds.json").read_text())
    held = np.concatenate([row["held_station_ids"] for row in folds])
    if len(folds) != 5 or not np.array_equal(np.sort(held), source_ids):
        raise ValueError("Source OOF folds do not partition source stations exactly once")
    for row in folds:
        held, fitting = set(row["held_station_ids"]), set(row["training_station_ids"])
        if held & fitting or held | fitting != set(source_ids):
            raise ValueError("OOF held/fitting station identities disagree")
    return train, np.maximum(0, np.expm1(z.ravel()[train])), np.asarray(dataset["y"]).ravel()[train]


def _scales_and_donors(state, regime, source_cells, source_prediction, source_y):
    """Independently reconstruct static scaling, donors and per-cell weights."""
    months, n = state["n_months"], state["n_nodes"]
    order = np.argsort(source_cells)
    source = np.asarray(source_cells)[order]
    prediction = np.asarray(source_prediction, dtype=np.float64)[order]
    truth = np.asarray(source_y, dtype=np.float64)[order]
    source_ids, inverse, counts = np.unique(source // months, return_inverse=True, return_counts=True)
    np.testing.assert_array_equal(state["source_cells"], source)
    np.testing.assert_array_equal(state["source_station_ids"], source_ids)
    np.testing.assert_array_equal(state["source_station_counts"], counts)
    weights = 1.0 / (len(source_ids) * counts[inverse])
    z = np.log1p(prediction)
    mean = float(weights @ z)
    sd = max(float(np.sqrt(weights @ ((z - mean) ** 2))), 1e-6)
    residual_scale = max(float(weights @ np.abs(truth - prediction)), 1e-6)
    for key, expected in (("log_context_mean", mean), ("log_context_sd", sd),
                          ("residual_scale", residual_scale)):
        _same(state["normalization"][key], expected)

    raw = np.asarray(regime, dtype=np.float64)[:, 4:13]
    valid = np.isfinite(raw) & (raw != -1.0)
    median, iqr, active = np.zeros(9), np.ones(9), np.zeros(9, dtype=bool)
    for column in range(9):
        observed = raw[source_ids, column][valid[source_ids, column]]
        if len(observed):
            active[column] = True
            median[column] = np.median(observed)
            lower, upper = np.quantile(observed, [.25, .75])
            iqr[column] = upper - lower if upper > lower else 1.0
    scaler = state["ecology_scaler"]
    if scaler["columns"] != list(range(4, 13)) or scaler["eligible_min_valid_features"] != 5:
        raise ValueError("Ecological columns or minimum coverage changed")
    _same(scaler["median"], median)
    _same(scaler["iqr"], iqr)
    np.testing.assert_array_equal(scaler["active"], active)
    standardized = (np.where(valid, raw, median) - median) / iqr
    standardized[:, ~active] = 0.0
    eligible = valid.sum(axis=1) >= 5
    eligible_ids = source_ids[eligible[source_ids]]
    source_counts = dict(zip(source_ids, counts, strict=True))
    selected = state["selected"]
    if len(state["station_diagnostics"]) != n:
        raise ValueError("A donor diagnostic is required for every station")
    fallbacks, max_weight_error = {}, 0.0
    for station, diagnostic in enumerate(state["station_diagnostics"]):
        if diagnostic["station"] != station:
            raise ValueError("Donor diagnostics are not station aligned")
        remaining = source_ids[source_ids != station]
        fallback, nearest, radius = None, None, None
        if state["mode"].startswith("global"):
            expected = remaining
        else:
            pool = eligible_ids[eligible_ids != station]
            if not eligible[station]:
                expected, fallback = remaining, "target_insufficient_ecology"
            elif not len(pool):
                expected, fallback = remaining, "no_eligible_source_donors"
            else:
                distances = np.mean((standardized[pool] - standardized[station]) ** 2, axis=1)
                nearest_order = np.lexsort((pool, distances))[:min(selected["k"], len(pool))]
                expected = pool[nearest_order]
                nearest, radius = float(distances[nearest_order[0]]), float(distances[nearest_order[-1]])
        donors = np.asarray(diagnostic["donor_ids"], dtype=np.int64)
        np.testing.assert_array_equal(donors, expected)
        if station in donors or not len(donors) or not np.isin(donors, source_ids).all():
            raise ValueError("Invalid donor identity or receiving-station leakage")
        expected_counts = np.array([source_counts[donor] for donor in donors])
        np.testing.assert_array_equal(diagnostic["donor_source_counts"], expected_counts)
        _same(diagnostic["donor_weights"], np.full(len(donors), 1.0 / len(donors)))
        # Summing each station's uniform cell weights recovers its equal total
        # weight even when source monitoring records have different lengths.
        cell_weights = np.concatenate([np.full(count, 1.0 / (len(donors) * count))
                                       for count in expected_counts])
        weight_error = abs(float(cell_weights.sum()) - 1.0)
        max_weight_error = max(max_weight_error, weight_error)
        np.testing.assert_allclose(cell_weights.sum(), 1.0, rtol=0, atol=1e-14)
        np.testing.assert_allclose(np.sum(diagnostic["donor_weights"]), 1.0, rtol=0, atol=1e-14)
        if (diagnostic["donor_count"] != len(donors) or diagnostic["effective_count"] != len(donors)
                or diagnostic["max_weight"] != 1.0 / len(donors)
                or diagnostic["fallback"] != fallback
                or diagnostic["ecology_valid_features"] != int(valid[station].sum())
                or diagnostic["source_self_excluded"] != bool(station in source_ids)
                or diagnostic["eligible_source_donors"] != int(np.sum(eligible_ids != station))):
            raise ValueError("Donor weights, eligibility or fallback diagnostics differ")
        for name, expected_distance in (("nearest_distance", nearest), ("radius_distance", radius)):
            if expected_distance is None:
                if diagnostic[name] is not None:
                    raise ValueError("Global/fallback donor pool must not report ecological distances")
            else:
                _same(diagnostic[name], expected_distance)
        if fallback:
            fallbacks[fallback] = fallbacks.get(fallback, 0) + 1
    if state["mode"].endswith("bias"):
        _same(np.asarray(state["coefficients"])[:, 1], np.zeros(n))
    return {"source_stations": len(source_ids), "source_cells": len(source),
            "source_only_scales_bitwise_exact": True, "station_checks": n,
            "donor_identity_and_self_exclusion_verified": True,
            "uniform_station_and_within_station_cell_weights_verified": True,
            "max_cell_weight_sum_error": max_weight_error,
            "fallback_counts": fallbacks, "insufficient_ecology_station_ids": np.flatnonzero(~eligible).tolist()}


def _selection_check(state, validation_truth, validation_temporal):
    attempts = [{"maxiter": 200, "ftol": 1e-12, "gtol": 1e-8, "maxls": 50},
                {"maxiter": 200, "ftol": 1e-12, "gtol": 1e-7, "maxls": 200}]
    expected_optimizer = {"name": "L-BFGS-B", "attempt_settings": attempts,
                          "initialization": "exact zero for each attempt", "dtype": "float64",
                          "acceptance": "optimizer success and finite coefficients/objective/gradient"}
    if state["optimizer"] != expected_optimizer:
        raise ValueError("Ecological optimizer settings differ")
    for diagnostic in state["station_diagnostics"]:
        optimizer = diagnostic["optimizer"]
        performed = optimizer["attempts"]
        if (len(performed) not in (1, 2) or not performed[-1]["success"] or not performed[-1]["finite"]
                or [row["settings"] for row in performed] != attempts[:len(performed)]
                or not optimizer["success"]):
            raise ValueError("Ecological profile was not accepted after the specified optimizer attempts")
        for key in ("objective", "iterations", "gradient_max_abs", "message"):
            if optimizer[key] != performed[-1][key]:
                raise ValueError("Saved coefficient diagnostic is not the successful optimizer result")
    if state["objective"] != (
            "station-balanced mean sqrt(scaled_error^2 + 0.05^2) + ridge * squared coefficients"):
        raise ValueError("Ecological smooth loss or ridge center differs")
    if state["selection_ties"] != "gamma zero, then larger ridge, then larger k, then smaller gamma":
        raise ValueError("Ecological tie preferences differ")
    baseline = float(np.mean(np.abs(validation_temporal - validation_truth)))
    _same(state["baseline_validation_mae"], baseline)
    trace = state["validation_trace"]
    ks = (None,) if state["mode"].startswith("global") else tuple(state["k_grid"])
    expected = {(k, ridge, gamma) for k in ks for ridge in state["ridge_grid"]
                for gamma in state["gamma_grid"]}
    actual = [(row["k"], row["ridge"], row["gamma"]) for row in trace]
    if len(actual) != len(expected) or set(actual) != expected:
        raise ValueError("Validation selection trace omits or repeats a configured candidate")
    for row in trace:
        if not np.isfinite(row["validation_mae"]):
            raise ValueError("Nonfinite validation selection score")
        if row["gamma"] == 0:
            _same(row["validation_mae"], baseline)
    chosen = min(trace, key=lambda row: (row["validation_mae"], row["gamma"] != 0,
                                        -row["ridge"], -(row["k"] or 0), row["gamma"]))
    if state["selected"] != chosen:
        raise ValueError("Saved selection disagrees with candidate scores and tie preferences")


def _query_replay(run, sources, config, dataset, split, old_full, bases, queries):
    shape, months = tuple(dataset["y"].shape), dataset["y"].shape[1]
    with np.load(sources["basis"] / "representations.npz", allow_pickle=False) as saved:
        shapes = {name: saved[name].copy() for name in SHAPES}
    adapters = {name: SupportShapeAdapter.from_dict(state)
                for name, state in json.loads((run / "adapters.json").read_text()).items()}
    if set(adapters) != set(MODELS):
        raise ValueError("Saved support adapters are incomplete")
    support, fixed_query = support_query_cells(split, target_role="test", k=5, n_months=months)
    if len(fixed_query) != config["query_cells"] or np.intersect1d(support, fixed_query).size:
        raise ValueError("Query count or support separation differs")
    labels = np.full(np.prod(shape), np.nan)
    labels[support] = np.asarray(dataset["y"]).ravel()[support]
    replay = make_predictions(old_full, bases, shapes, adapters, labels, split, months)
    previous = pd.read_parquet(sources["prior"] / "predictions.parquet")
    checks = []
    for model_name in MODELS:
        for k in KS:
            actual = replay[replay.model_name.eq(model_name) & replay.k.eq(k)].sort_values("cell")
            stored = queries[queries.model_name.eq(model_name) & queries.k.eq(k)].sort_values("cell")
            np.testing.assert_array_equal(stored.cell, np.sort(fixed_query))
            np.testing.assert_array_equal(actual.cell, stored.cell)
            selected_support, selected_query = support_query_cells(
                split, target_role="test", k=k, n_months=months)
            np.testing.assert_array_equal(selected_query, fixed_query)
            if (len(selected_support) != k * len(np.unique(fixed_query // months))
                    or np.intersect1d(selected_support, fixed_query).size):
                raise ValueError("Adaptation support count or query separation differs")
            for column in ("station", "month", "support_count", "visibility_role"):
                np.testing.assert_array_equal(actual[column], stored[column])
            _same(stored.y_true, np.asarray(dataset["y"]).ravel()[stored.cell])
            error = max(_same(actual[column], stored[column])
                        for column in ("y_pred", "base_pred", "adaptation_delta"))
            if k == 0:
                np.testing.assert_array_equal(stored.y_pred, stored.base_pred)
            if model_name.startswith("interaction_"):
                previous_name = model_name.replace("interaction_", "interaction_tuned_", 1)
                reference = previous[previous.model_name.eq(previous_name) & previous.k.eq(k)].sort_values("cell")
                np.testing.assert_array_equal(stored.cell, reference.cell)
                _same(stored.y_pred, reference.y_pred)
            checks.append({"model_name": model_name, "k": k, "n_query": len(stored),
                           "max_abs_difference": error, "bitwise_exact": True})
    return checks


def verify_one(root, split_seed, seed, runtime):
    run = root / "runs" / f"split{split_seed}_seed{seed}"
    if not (run / "complete.json").is_file():
        raise ValueError(f"Requested run is incomplete: {run}")
    config = json.loads((run / "config.json").read_text())
    if (config["split_seed"], config["seed"]) != (split_seed, seed):
        raise ValueError("Run identity does not match its directory")
    if config["runtime_snapshot_hash"] != runtime:
        raise ValueError("Run runtime does not match its saved source snapshot")
    if set(config["models"]) != set(MODELS) or tuple(config["arms"]) != ARMS:
        raise ValueError("Run ecological model definitions differ")
    required_config = {"ecological_columns": list(range(4, 13)), "neighbor_grid": [20, 40, 80],
                       "memory_ridge_grid": [.1, 1], "gamma_grid": [0, .25, .5, 1],
                       "smooth_epsilon_scaled": .05, "memory_max_iter": 200,
                       "minimum_valid_ecology": 5, "exclude_receiving_station": True,
                       "inference_roles": ["train"], "k_values": list(KS)}
    for key, expected in required_config.items():
        if config[key] != expected:
            raise ValueError(f"Changed ecological experiment setting: {key}")
    completion = verify_files(run, "complete.json", config)
    required = {"config.json", "adapters.json", "full_grid.parquet", "full_grid.meta.json",
                "predictions.parquet", "predictions.meta.json", "timing.json",
                *(f"{name}.json" for name in ARMS)}
    if not required.issubset(completion["files"]):
        raise ValueError("Completion record is missing required ecological products")
    sources, _ = _packages(config)
    old_config, _, dataset, split, old_full = read_source(sources["source"])
    for name in ("dataset_path", "dataset_hash", "mask_path", "mask_hash", "q90_threshold_train"):
        if config[name] != old_config[name]:
            raise ValueError(f"New and source configuration disagree on {name}")
    shape = tuple(dataset["y"].shape)
    months = shape[1]
    full = pd.read_parquet(run / "full_grid.parquet")
    queries = pd.read_parquet(run / "predictions.parquet")
    previous = pd.read_parquet(sources["prior"] / "full_grid.parquet")
    _sidecar(run, "full_grid.parquet", config, runtime, len(full), completion)
    _sidecar(run, "predictions.parquet", config, runtime, len(queries), completion)
    np.testing.assert_array_equal(full.cell.to_numpy(), np.arange(np.prod(shape)))
    identity = ["cell", "station", "month", "analyte", "visibility_role"]
    pd.testing.assert_frame_equal(full[identity], old_full[identity], check_exact=True)
    pd.testing.assert_frame_equal(previous[identity], old_full[identity], check_exact=True)
    for column in ("ecological_novelty", "upstream_support"):
        np.testing.assert_array_equal(full[column], old_full[column])
    if queries.duplicated(["model_name", "k", "cell"]).any():
        raise ValueError("Duplicate adapted query identities")
    if set(map(tuple, queries[["model_name", "k"]].drop_duplicates().to_numpy())) != {
            (name, k) for name in MODELS for k in KS}:
        raise ValueError("Query product does not contain all ten models and four K values")
    if not queries.split_seed.eq(split_seed).all() or not queries.seed.eq(seed).all():
        raise ValueError("Query/run identity differs")
    context = old_full.context_pred.to_numpy().reshape(shape)
    interaction = previous.interaction_tuned_pred.to_numpy().reshape(shape)
    _same(previous.context_pred, context.ravel())
    _same(full.context_pred, context.ravel())
    _same(full.interaction_pred, interaction.ravel())
    source_cells, source_context, source_y = _source_bank(sources, split, dataset)
    _, validation_cells = support_query_cells(split, target_role="val", k=0, n_months=months)
    if (np.intersect1d(source_cells // months, validation_cells // months).size
            or np.intersect1d(validation_cells, split["test"]).size):
        raise ValueError("Source-validation fitting identities overlap source or target test")
    validation_y = np.asarray(dataset["y"], dtype=np.float64).ravel()[validation_cells]
    regime = np.asarray(dataset["regime"])
    torch.set_num_threads(int(config["torch_threads"]))
    bases, checks, differences = {"interaction": interaction.ravel()}, [], {}
    for mode in ARMS:
        state = json.loads((run / f"{mode}.json").read_text())
        if (state["mode"] != mode or state["n_nodes"] != shape[0] or state["n_months"] != months
                or state["k_grid"] != config["neighbor_grid"]
                or state["ridge_grid"] != config["memory_ridge_grid"]
                or state["gamma_grid"] != config["gamma_grid"]):
            raise ValueError("Serialized ecological model disagrees with run configuration")
        np.testing.assert_array_equal(state["validation_cells"], np.sort(validation_cells))
        np.testing.assert_array_equal(state["validation_station_ids"], np.unique(validation_cells // months))
        restored = EcologicalResidualTransfer.from_dict(state)
        if restored.to_dict() != state:
            raise ValueError("Ecological model JSON round trip is not exact")
        donor_check = _scales_and_donors(state, regime, source_cells, source_context, source_y)
        _selection_check(state, validation_y, interaction.ravel()[validation_cells])
        # Refit the small residual memory, with no query labels or neural/forest
        # fitting. Source and validation vectors are explicitly sliced here.
        refitted = EcologicalResidualTransfer(
            mode=mode, k_grid=config["neighbor_grid"], ridge_grid=config["memory_ridge_grid"],
            gamma_grid=config["gamma_grid"],
        ).fit(regime, source_cells, source_context, source_y, n_months=months,
              validation_cells=validation_cells, validation_y=validation_y,
              validation_context=context.ravel()[validation_cells],
              validation_temporal=interaction.ravel()[validation_cells], selection_role="source_validation")
        refitted_state = refitted.to_dict()
        if refitted_state != state:
            changed = [key for key in state if refitted_state.get(key) != state[key]]
            raise ValueError(f"Source/validation refit changed saved ecological state: {mode}: {changed}")
        delta = restored.predict_delta(context)
        prediction = restored.predict(context, interaction)
        coefficients = np.asarray(state["coefficients"])
        norms = state["normalization"]
        u = (np.log1p(context) - norms["log_context_mean"]) / norms["log_context_sd"]
        manual_delta = norms["residual_scale"] * (coefficients[:, :1] + coefficients[:, 1:] * u)
        _same(delta, manual_delta)
        gamma = state["selected"]["gamma"]
        manual_prediction = (interaction.copy() if gamma == 0 else np.maximum(
            0, context + (1 - gamma) * (interaction - context) + gamma * manual_delta))
        _same(prediction, manual_prediction)
        _same(refitted.predict_delta(context), delta)
        _same(refitted.predict(context, interaction), prediction)
        differences[f"{mode}_memory"] = _same(delta.ravel(), full[f"{mode}_memory"])
        differences[f"{mode}_pred"] = _same(prediction.ravel(), full[f"{mode}_pred"])
        if gamma == 0:
            _same(prediction, interaction)
        bases[mode] = prediction.ravel()
        checks.append({"mode": mode, "selected": state["selected"],
                       "saved_state_roundtrip_exact": True, "source_validation_refit_exact": True,
                       "manual_delta_and_mixing_exact": True, "full_grid_replay_bitwise_exact": True,
                       "gamma_zero_prior_identity": True if gamma == 0 else None,
                       "scaling_and_donors": donor_check})
        del restored, refitted
    query_checks = _query_replay(run, sources, config, dataset, split, old_full, bases, queries)
    return {"run": run.name, "split_seed": split_seed, "seed": seed, "n_full_grid": len(full),
            "memory_checks": checks, "full_grid_max_abs_difference": differences,
            "query_replays": query_checks, "all_memory_and_adapted_predictions_bitwise_exact": True,
            "source_context_oof_coverage_verified": True, "old_context_component_bitwise_unchanged": True,
            "old_interaction_component_and_adapted_queries_bitwise_unchanged": True,
            "existing_forests_and_neural_models": "completion bindings verified; not refitted or replayed here",
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
        print(f"Rebuilding and replaying split{split}_seed{seed}", flush=True)
        results.append(verify_one(args.root, split, seed, runtime))
        gc.collect()
    completed = [(split, seed) for split in SPLITS for seed in SEEDS
                 if (args.root / "runs" / f"split{split}_seed{seed}" / "complete.json").is_file()]
    path = args.output or args.root / "verification" / "replay_checks.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({
        "checked_at": datetime.now(timezone.utc).isoformat(),
        "requested_runs": len(requested), "verified_runs": len(results),
        "completed_production_runs_at_check": len(completed), "expected_production_runs": 9,
        "all_nine_replayed": set(requested) == {(split, seed) for split in SPLITS for seed in SEEDS},
        "scope": "only the listed runs were replayed; existing forest/neural fits were bound, not refitted",
        "verifier_sha256": sha256_file(__file__), "results": results,
    }, indent=2, allow_nan=False) + "\n")
    print(path)


if __name__ == "__main__":
    main()
