"""Replay support-aware ecological transfer using frozen v1 residual profiles.

Only the validation support/mixing wrappers are refitted. The source residual
bank, ecological profiles, forest, recurrent model and support basis remain
fixed and are verified through their saved package bindings.
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
from run_unified_doc_spatial_v2 import read_source
from verify_doc_ecological_transfer_v1 import _packages, _same

from river_graph.experiments.provenance import run_identity_sha256, sha256_file
from river_graph.experiments.unified_spatial_protocol import support_query_cells
from river_graph.models.ecological_residual_transfer import EcologicalResidualTransfer
from river_graph.models.support_aware_residual_transfer import (
    SupportAwareResidualTransfer,
    SupportAwareTransferEpisode,
)
from river_graph.models.support_shape_adapter import SupportShapeAdapter

ROOT = Path("experiments/phase4_transfer/doc_ecological_transfer_v2")
ARMS = ("global_bias", "ecological_bias", "global_affine", "ecological_affine")
SHAPES = ("constant", "gru_tuned_anchor")
MODELS = tuple(f"{base}_{shape}" for base in ("interaction", *ARMS) for shape in SHAPES)
SPLITS, SEEDS, KS = (142, 143, 144), (42, 43, 44), (0, 1, 3, 5)


def _validation_episodes(dataset, split, context, interaction, memory, basis):
    """Construct fitting vectors from source-validation identities only."""
    months = dataset["y"].shape[1]
    episodes = []
    for k in KS:
        support, query = support_query_cells(split, target_role="val", k=k, n_months=months)
        if (not np.isin(query, split["val"]).all() or not np.isin(support, split["val"]).all()
                or np.intersect1d(query, split["test"]).size
                or np.intersect1d(support, split["test"]).size
                or np.intersect1d(support, query).size):
            raise ValueError("Validation wrapper received non-validation or overlapping cells")
        episodes.append(SupportAwareTransferEpisode(
            k=k, query_cells=query, query_values=np.asarray(dataset["y"], dtype=float).ravel()[query],
            support_cells=support, support_values=np.asarray(dataset["y"], dtype=float).ravel()[support],
            query_context=context[query], query_temporal=interaction[query], query_memory=memory[query],
            support_context=context[support], support_temporal=interaction[support], support_memory=memory[support],
            query_basis=basis[query], support_basis=basis[support],
        ))
    return episodes


def _selection_check(state, prior_gamma, shape_name, reference_adapter):
    """Check both nested support choices and the joint gamma winner."""
    if (state["gamma_k0"] != prior_gamma or state["gamma_values"] != [0, .25, .5, 1]
            or state["selection_role"] != "source_validation"
            or state["tie_rule"] != "gamma zero, then locked K0 gamma, then smaller gamma"
            or set(state["selection_by_k"]) != {str(k) for k in KS}):
        raise ValueError("Wrapper selection contract or locked K0 gamma differs")
    ridges = ["infinity"] if shape_name == "constant" else [.1, 1.0, 10.0, "infinity"]
    if state["ridge_strengths"] != ridges:
        raise ValueError("Wrapper does not use the matched shape/constant ridge grid")
    if state["adapters_by_gamma"]["0.0"] != reference_adapter:
        raise ValueError("Gamma-zero adapter changed the unchanged interaction calibration")
    gamma_scores = state["gamma_scores"]
    expected = {(gamma, k) for gamma in state["gamma_values"] for k in KS}
    actual = [(row["gamma"], row["k"]) for row in gamma_scores]
    if len(actual) != len(expected) or set(actual) != expected:
        raise ValueError("Gamma selection trace is not a complete unique gamma-by-K grid")
    for gamma in state["gamma_values"]:
        adapter = state["adapters_by_gamma"][str(float(gamma))]
        if (adapter["alpha_values"] != [0, .25, .5, .75, 1]
                or adapter["ridge_strengths"] != ridges):
            raise ValueError("A candidate changed the matched support hyperparameter grid")
        for k in KS:
            scores = [row for row in adapter["selection_scores"] if row["k"] == k and row["valid"]]
            if not scores:
                raise ValueError("Missing finite support candidate")
            chosen = min(scores, key=lambda row: (
                row["mae"], row["alpha"],
                -float("inf") if row["ridge_strength"] == "infinity" else -row["ridge_strength"]))
            selected = adapter["selection_by_k"][str(k)]
            if selected != {name: chosen[name] for name in ("alpha", "ridge_strength")}:
                raise ValueError("Support choice disagrees with saved losses or tie rule")
            joint = next(row for row in gamma_scores if row["gamma"] == gamma and row["k"] == k)
            if joint != {**chosen, "gamma": gamma}:
                raise ValueError("Joint gamma score does not use its selected support adapter")
    for k in KS:
        candidates = [row for row in gamma_scores if row["k"] == k
                      and (k != 0 or row["gamma"] == prior_gamma)]
        chosen = min(candidates, key=lambda row: (
            row["mae"], row["gamma"] != 0, row["gamma"] != prior_gamma, row["gamma"]))
        if state["selection_by_k"][str(k)] != {**chosen, "locked": k == 0}:
            raise ValueError("Joint gamma selection differs from its validation losses or lock")


def _manual_base(context, interaction, memory, gamma):
    if gamma == 0:
        return interaction.copy()
    return np.maximum(0.0, context + (1 - gamma) * (interaction - context) + gamma * memory)


def _sidecar(run, filename, config, runtime, n_rows, completion):
    path = run / filename
    sidecar = json.loads(path.with_suffix(".meta.json").read_text())
    expected = {"config_hash": digest(config), "runtime_snapshot_hash": runtime,
                "dataset_sha256": config["dataset_hash"], "mask_sha256": config["mask_hash"],
                "prediction_sha256": sha256_file(path), "rows": n_rows,
                "selection_role": "source_validation",
                "run_identity_sha256": run_identity_sha256(digest(config), config["started_at"], runtime)}
    if sidecar["config"] != config:
        raise ValueError("Prediction sidecar configuration differs")
    for key, value in expected.items():
        if sidecar[key] != value:
            raise ValueError(f"Sidecar {key} differs: {path}")
    if set(sidecar["model_files"]) != {"mixers.json", "profile_bindings.json"}:
        raise ValueError("Sidecar omits the support wrapper or frozen-profile binding")
    for name, bound_hash in sidecar["model_files"].items():
        if sha256_file(run / name) != bound_hash or completion["files"].get(name) != bound_hash:
            raise ValueError(f"Changed saved wrapper/binding: {name}")


def verify_one(root, split_seed, seed, runtime):
    run = root / "runs" / f"split{split_seed}_seed{seed}"
    if not (run / "complete.json").is_file():
        raise ValueError(f"Requested run is incomplete: {run}")
    config = json.loads((run / "config.json").read_text())
    if ((config["split_seed"], config["seed"]) != (split_seed, seed)
            or config["runtime_snapshot_hash"] != runtime):
        raise ValueError("Run or historical runtime identity differs")
    required_config = {"profile_fitting": "none; frozen v1 memories", "inference_roles": ["train"],
                       "k_values": list(KS), "gamma_grid": [0, .25, .5, 1],
                       "alpha_grid": [0, .25, .5, .75, 1], "ridge_grid": [.1, 1, 10, "infinity"],
                       "gamma_k0": "locked to frozen v1 source-validation choice",
                       "selection_role": "source_validation"}
    if set(config["models"]) != set(MODELS) or tuple(config["arms"]) != ARMS:
        raise ValueError("Support-aware model definitions differ")
    for key, value in required_config.items():
        if config[key] != value:
            raise ValueError(f"Support-aware configuration differs: {key}")
    completion = verify_files(run, "complete.json", config)
    required_files = {"config.json", "mixers.json", "profile_bindings.json", "full_grid.parquet",
                      "full_grid.meta.json", "predictions.parquet", "predictions.meta.json", "timing.json"}
    if not required_files.issubset(completion["files"]):
        raise ValueError("Completion omits a required support-aware product")
    sources, _ = _packages(config)
    old_config, _, dataset, split, old_full = read_source(sources["source"])
    for key in ("dataset_path", "dataset_hash", "mask_path", "mask_hash", "q90_threshold_train"):
        if config[key] != old_config[key]:
            raise ValueError(f"Source/current identity differs: {key}")
    binding = json.loads((run / "profile_bindings.json").read_text())
    if binding != {"prior_run": str(sources["prior"]), "prior_completion_hash": config["prior_completion_hash"],
                   "profiles": config["profile_files"]} or set(config["profile_files"]) != set(ARMS):
        raise ValueError("Frozen-profile bindings disagree with run configuration")
    previous_complete = json.loads((sources["prior"] / "complete.json").read_text())
    for mode, expected_hash in config["profile_files"].items():
        if (sha256_file(sources["prior"] / f"{mode}.json") != expected_hash
                or previous_complete["files"].get(f"{mode}.json") != expected_hash):
            raise ValueError(f"Frozen ecological profile changed: {mode}")

    full = pd.read_parquet(run / "full_grid.parquet")
    queries = pd.read_parquet(run / "predictions.parquet")
    previous_full = pd.read_parquet(sources["prior"] / "full_grid.parquet")
    previous_queries = pd.read_parquet(sources["prior"] / "predictions.parquet")
    _sidecar(run, "full_grid.parquet", config, runtime, len(full), completion)
    _sidecar(run, "predictions.parquet", config, runtime, len(queries), completion)
    shape = tuple(dataset["y"].shape)
    n, months = shape
    np.testing.assert_array_equal(full.cell, np.arange(n * months))
    pd.testing.assert_frame_equal(full, previous_full, check_exact=True)
    identity = ["cell", "station", "month", "analyte", "visibility_role"]
    pd.testing.assert_frame_equal(full[identity], old_full[identity], check_exact=True)
    if (queries.duplicated(["model_name", "k", "cell"]).any()
            or set(map(tuple, queries[["model_name", "k"]].drop_duplicates().to_numpy()))
            != {(name, k) for name in MODELS for k in KS}
            or not queries.seed.eq(seed).all() or not queries.split_seed.eq(split_seed).all()):
        raise ValueError("Support-aware query identities are incomplete or duplicated")
    context, interaction = full.context_pred.to_numpy(), full.interaction_pred.to_numpy()
    _same(context, old_full.context_pred.to_numpy())
    with np.load(sources["basis"] / "representations.npz", allow_pickle=False) as saved:
        shapes = {name: saved[name].copy() for name in SHAPES}
    for name, basis in shapes.items():
        if basis.ndim != 2 or basis.shape[0] != n * months or not np.isfinite(basis).all():
            raise ValueError(f"Frozen support basis is not aligned with the source grid: {name}")
    previous_adapters = json.loads((sources["prior"] / "adapters.json").read_text())
    states = json.loads((run / "mixers.json").read_text())
    if set(states) != {f"{mode}_{shape_name}" for mode in ARMS for shape_name in SHAPES}:
        raise ValueError("Saved wrappers are incomplete")
    reserved_support, fixed_query = support_query_cells(split, target_role="test", k=5, n_months=months)
    if (len(fixed_query) != config["query_cells"]
            or np.intersect1d(reserved_support, fixed_query).size):
        raise ValueError("Fixed queries and reserved supports differ")
    support_truth = np.full(n * months, np.nan)
    support_truth[reserved_support] = np.asarray(dataset["y"], dtype=float).ravel()[reserved_support]
    torch.set_num_threads(int(config["torch_threads"]))
    wrapper_checks, query_checks = [], []
    for mode in ARMS:
        profile_state = json.loads((sources["prior"] / f"{mode}.json").read_text())
        profile = EcologicalResidualTransfer.from_dict(profile_state)
        if profile.to_dict() != profile_state:
            raise ValueError("Frozen ecological profile JSON round trip differs")
        memory = profile.predict_delta(context.reshape(shape)).ravel()
        _same(memory, full[f"{mode}_memory"])
        for shape_name, basis in shapes.items():
            name = f"{mode}_{shape_name}"
            state = states[name]
            gamma_k0 = profile_state["selected"]["gamma"]
            reference_adapter = previous_adapters[f"interaction_{shape_name}"]
            _selection_check(state, gamma_k0, shape_name, reference_adapter)
            model = SupportAwareResidualTransfer.from_dict(state)
            if model.to_dict() != state:
                raise ValueError("Saved wrapper JSON round trip differs")
            episodes = _validation_episodes(dataset, split, context, interaction, memory, basis)
            kwargs = {"ridge_strengths": (float("inf"),)} if shape_name == "constant" else {}
            refitted = SupportAwareResidualTransfer(months, **kwargs).fit(
                episodes, gamma_k0=gamma_k0, selection_role="source_validation")
            if refitted.to_dict() != state:
                changed = [key for key in state if refitted.to_dict().get(key) != state[key]]
                raise ValueError(f"Validation-only wrapper refit changed saved state: {name}: {changed}")
            grid_checks = []
            for k in KS:
                support, query = support_query_cells(split, target_role="test", k=k, n_months=months)
                np.testing.assert_array_equal(query, fixed_query)
                if (len(support) != k * len(np.unique(query // months))
                        or np.intersect1d(support, query).size):
                    raise ValueError("Support/query identities or per-station K differ")
                base = model.selected_base(context, interaction, memory, k=k)
                gamma = model.selected_gamma(k)
                _same(base, _manual_base(context, interaction, memory, gamma))
                _same(base, refitted.selected_base(context, interaction, memory, k=k))
                if k == 0:
                    _same(base, full[f"{mode}_pred"])
                prediction = model.adapt(
                    context[query], interaction[query], memory[query], query,
                    context[support], interaction[support], memory[support], support,
                    support_truth[support], basis[query], basis[support], k=k)
                replay = refitted.adapt(
                    context[query], interaction[query], memory[query], query,
                    context[support], interaction[support], memory[support], support,
                    support_truth[support], basis[query], basis[support], k=k)
                _same(prediction, replay)
                # Reapply the cached shape adapter directly to prove that the
                # same selected gamma supplies both support and query bases.
                cached = SupportShapeAdapter.from_dict(state["adapters_by_gamma"][str(float(gamma))])
                direct = cached.adapt(base[query], query, base[support], support, support_truth[support],
                                      query_basis=basis[query], support_basis=basis[support], k=k)
                _same(prediction, direct)
                stored = queries[queries.model_name.eq(name) & queries.k.eq(k)].sort_values("cell")
                positions = np.argsort(query)
                np.testing.assert_array_equal(stored.cell, query[positions])
                for column in ("station", "month", "analyte", "visibility_role",
                               "ecological_novelty", "upstream_support"):
                    np.testing.assert_array_equal(stored[column], old_full.iloc[stored.cell][column])
                np.testing.assert_array_equal(stored.support_count, np.full(len(query), k))
                _same(stored.regional_gamma, np.full(len(query), gamma))
                error = max(_same(stored.y_pred, prediction[positions]),
                            _same(stored.base_pred, base[query][positions]),
                            _same(stored.adaptation_delta, (np.log1p(prediction) - np.log1p(base[query]))[positions]))
                _same(stored.y_true, np.asarray(dataset["y"]).ravel()[stored.cell])
                if k == 0:
                    _same(stored.y_pred, stored.base_pred)
                    previous = previous_queries[previous_queries.model_name.eq(name)
                                                & previous_queries.k.eq(0)].sort_values("cell")
                    np.testing.assert_array_equal(stored.cell, previous.cell)
                    _same(stored.y_pred, previous.y_pred)
                if gamma == 0:
                    reference = previous_queries[previous_queries.model_name.eq(f"interaction_{shape_name}")
                                                 & previous_queries.k.eq(k)].sort_values("cell")
                    np.testing.assert_array_equal(stored.cell, reference.cell)
                    _same(stored.y_pred, reference.y_pred)
                grid_checks.append({"k": k, "gamma": gamma, "n_cells": len(base),
                                    "formula_and_refit_bitwise_exact": True})
                query_checks.append({"model_name": name, "k": k, "n_query": len(query),
                                     "max_abs_difference": error, "bitwise_exact": True})
            wrapper_checks.append({"model_name": name, "gamma_k0": gamma_k0,
                                   "state_roundtrip_exact": True, "validation_refit_state_exact": True,
                                   "gamma_zero_adapter_matches_interaction": True,
                                   "profile_sha256": config["profile_files"][mode],
                                   "full_grid_selected_bases": grid_checks})
    for shape_name in SHAPES:
        name = f"interaction_{shape_name}"
        for k in KS:
            current = queries[queries.model_name.eq(name) & queries.k.eq(k)].sort_values("cell")
            prior = previous_queries[previous_queries.model_name.eq(name) & previous_queries.k.eq(k)].sort_values("cell")
            np.testing.assert_array_equal(current.cell, np.sort(fixed_query))
            pd.testing.assert_frame_equal(current[prior.columns].reset_index(drop=True),
                                          prior.reset_index(drop=True), check_exact=True)
            _same(current.regional_gamma, np.zeros(len(current)))
            query_checks.append({"model_name": name, "k": k, "n_query": len(current),
                                 "max_abs_difference": 0.0, "bitwise_exact": True})
    return {"run": run.name, "split_seed": split_seed, "seed": seed,
            "n_frozen_full_grid_rows": len(full), "wrapper_checks": wrapper_checks,
            "query_replays": query_checks, "all_predictions_and_wrapper_refits_bitwise_exact": True,
            "k0_matches_v1_bitwise": True, "old_interaction_queries_bitwise_unchanged": True,
            "frozen_v1_grid_bitwise_unchanged": True,
            "full_grid_product_role": "unchanged v1 components; K-dependent bases reconstructed from saved wrappers",
            "profile_fitting": "none; source bank/scalers/coefficients held fixed",
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
        print(f"Refitting validation wrappers and replaying split{split}_seed{seed}", flush=True)
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
        "scope": "listed runs only; validation wrappers refitted, source profiles and prior experts held fixed",
        "verifier_sha256": sha256_file(__file__),
        "v1_shared_verifier_helpers_sha256": sha256_file("scripts/verify_doc_ecological_transfer_v1.py"),
        "results": results,
    }, indent=2, allow_nan=False) + "\n")
    print(path)


if __name__ == "__main__":
    main()
