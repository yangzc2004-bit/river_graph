"""Replay fixed daily-availability routing and source-validation adaptation.

No forest, neural model or source residual bank is fitted. The prior verified
expert products are bound unchanged; only source-validation adapters are refit.
"""
from __future__ import annotations

import argparse
import gc
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from run_unified_doc_spatial import digest, verify_files, verify_runtime_snapshot
from run_unified_doc_spatial_v2 import fit_adapters, make_predictions, read_source
from verify_doc_ecological_transfer_v2 import _selection_check, _validation_episodes
from verify_doc_tail_residual_v1 import ATOL, RTOL, _close

from river_graph.experiments.provenance import run_identity_sha256, sha256_file
from river_graph.experiments.unified_spatial_protocol import support_query_cells
from river_graph.models.daily_hydro_router import DailyHydroRouter
from river_graph.models.support_aware_residual_transfer import (
    SupportAwareResidualTransfer,
)
from river_graph.models.support_shape_adapter import SupportShapeAdapter

SPLITS, SEEDS, KS = (142, 143, 144), (42, 43, 44), (0, 1, 3, 5)
ROOT = Path("experiments/phase4_transfer/doc_daily_hydro_fallback_v1")
ARMS = ("monthly", "daily", "hybrid")
SHAPES = ("constant", "gru_tuned_anchor")
DIRECT_MODELS = tuple(f"{arm}_{shape}" for arm in ("context", *ARMS) for shape in SHAPES)
INTEGRATED_MODELS = tuple(f"{arm}_integrated_{shape}" for arm in ARMS for shape in SHAPES)
REFERENCE_MODELS = tuple(f"prior_{arm}_{shape}" for arm in ("encoder", "encoder_integrated", "ecological_affine") for shape in SHAPES)
MODELS = (*DIRECT_MODELS, *INTEGRATED_MODELS, *REFERENCE_MODELS)


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
    required = {"adapters.json", "mixers.json", "router.json"}
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
            or config["runtime_snapshot_hash"] != runtime
            or config["expert_retraining"] is not False
            or tuple(config["arms"]) != ARMS or set(config["models"]) != set(MODELS)
            or config["k_values"] != list(KS) or config["inference_roles"] != ["train"]
            or config["gamma_grid"] != [0, .25, .5, 1]
            or config["alpha_grid"] != [0, .25, .5, .75, 1]
            or config["ridge_grid"] != [.1, 1, 10, "infinity"]):
        raise ValueError("Fixed-route configuration or experiment identities differ")
    completion = verify_files(run, "complete.json", config)
    required = {"config.json", "adapters.json", "mixers.json", "router.json", "source_validation.csv",
                "full_grid.parquet", "full_grid.meta.json", "predictions.parquet",
                "predictions.meta.json", "timing.json"}
    if not required.issubset(completion["files"]):
        raise ValueError("Completion record omits required routing products")
    router_state = json.loads((run / "router.json").read_text())
    if config["route_definition"] != router_state:
        raise ValueError("Saved routing rule and configuration disagree")
    router = DailyHydroRouter.from_dict(router_state)
    sources, source_configs = {}, {}
    for label in ("source", "prior", "basis", "oof", "memory"):
        package = Path(config[f"{label}_run"])
        package_config = json.loads((package / "config.json").read_text())
        if sha256_file(package / "complete.json") != config[f"{label}_completion_hash"]:
            raise ValueError(f"Changed {label} package binding")
        verify_files(package, "complete.json", package_config)
        sources[label], source_configs[label] = package, package_config
    for label in ("source", "basis", "oof", "memory"):
        if (Path(source_configs["prior"][f"{label}_run"]) != sources[label]
                or source_configs["prior"][f"{label}_completion_hash"] != config[f"{label}_completion_hash"]):
            raise ValueError(f"Parent and current {label} lineage differs")
    parent_runtime = verify_runtime_snapshot(sources["prior"].parent.parent)
    if source_configs["prior"]["runtime_snapshot_hash"] != parent_runtime:
        raise ValueError("Parent execution archive identity differs")
    verification_path = Path(config["parent_verification_path"])
    if (verification_path != sources["prior"].parent.parent / "verification" / "replay_checks.json"
            or sha256_file(verification_path) != config["parent_verification_hash"]):
        raise ValueError("Bound parent replay report differs")
    parent_report = json.loads(verification_path.read_text())
    verified = [item for item in parent_report["results"] if item["run"] == sources["prior"].name]
    if (len(verified) != 1 or verified[0]["status"] != "verified"
            or verified[0]["completion_sha256"] != config["prior_completion_hash"]
            or verified[0]["runtime_snapshot_hash"] != parent_runtime
            or not verified[0]["encoder_features_neural_outputs_and_adapted_queries_bitwise_exact"]):
        raise ValueError("Parent replay report does not verify these exact expert products")
    for key in ("daily_features_path", "daily_features_hash", "daily_metadata_path", "daily_metadata_hash"):
        if config[key] != source_configs["prior"][key]:
            raise ValueError("Routed product changed the parent's daily feature package")
    if (sha256_file(config["daily_features_path"]) != config["daily_features_hash"]
            or sha256_file(config["daily_metadata_path"]) != config["daily_metadata_hash"]
            or verified[0]["daily_feature_checks"]["feature_pack_sha256"] != config["daily_features_hash"]
            or verified[0]["daily_feature_checks"]["metadata_sha256"] != config["daily_metadata_hash"]
            or not verified[0]["daily_feature_checks"]["independent_formula_bitwise_exact"]):
        raise ValueError("The exact independently rebuilt parent daily pack is not retained")
    if sha256_file(sources["memory"] / "ecological_affine.json") != config["memory_file_hash"]:
        raise ValueError("Frozen ecological profile differs")
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
        raise ValueError("Query identities differ or model/K combinations are incomplete")
    context = old_full.context_pred.to_numpy()
    np.testing.assert_array_equal(context, full.context_pred)
    np.testing.assert_array_equal(context, prior_full.context_pred)
    memory = prior_full.ecological_memory.to_numpy()
    np.testing.assert_array_equal(memory, full.ecological_memory)
    feature_names = router_state["daily_feature_names"]
    features = prior_full[feature_names].to_numpy()
    with np.load(config["daily_features_path"], allow_pickle=False) as pack:
        np.testing.assert_array_equal(features, pack["full"].reshape(-1, 8))
        np.testing.assert_array_equal(pack["site_no"], np.asarray(dataset["site_no"]).astype(str))
        np.testing.assert_array_equal(pack["months"], [str(value) for value in dataset["months"]])
    for index, name in enumerate(feature_names):
        np.testing.assert_array_equal(full[name], features[:, index])
        np.testing.assert_array_equal(queries[name], features[queries.cell.to_numpy(), index])
    flags = features[:, [5, 6, 7]]
    if not np.isin(flags, [0, 1]).all():
        raise ValueError("Routing flags are not binary")
    uses_daily = flags.astype(bool).any(axis=1)
    valid_count = flags.sum(axis=1).astype(np.int8)
    monthly, daily = prior_full.monthly_pred.to_numpy(), prior_full.daily_pred.to_numpy()
    hybrid = np.where(uses_daily, daily, monthly)
    np.testing.assert_array_equal(router.predict(monthly, daily, features), hybrid)
    np.testing.assert_array_equal(full.uses_daily, uses_daily)
    np.testing.assert_array_equal(full.daily_numeric_valid_count, valid_count)
    np.testing.assert_array_equal(queries.uses_daily, uses_daily[queries.cell.to_numpy()])
    np.testing.assert_array_equal(queries.daily_numeric_valid_count, valid_count[queries.cell.to_numpy()])
    bases = {"context": context, "monthly": monthly, "daily": daily, "hybrid": hybrid}
    differences = {}
    for arm in ARMS:
        np.testing.assert_array_equal(full[f"{arm}_pred"], bases[arm])
        differences[f"{arm}_pred"] = _close(full[f"{arm}_pred"], bases[arm])
    np.testing.assert_array_equal(hybrid[~uses_daily], monthly[~uses_daily])
    np.testing.assert_array_equal(hybrid[uses_daily], daily[uses_daily])
    if not all(np.isfinite(value).all() for value in bases.values()):
        raise ValueError("Nonfinite routed base prediction")
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
    parent_adapters = json.loads((sources["prior"] / "adapters.json").read_text())
    for base in ("context", "monthly", "daily"):
        for shape_name in SHAPES:
            name = f"{base}_{shape_name}"
            if adapter_states[name] != parent_adapters[name]:
                raise ValueError("Unchanged expert's direct adapter differs from its parent")
    support, fixed_query = support_query_cells(split, target_role="test", k=5, n_months=months)
    if len(fixed_query) != config["query_cells"] or np.intersect1d(support, fixed_query).size:
        raise ValueError("Fixed-query count or reserved supports differ")
    labels = np.full(np.prod(shape), np.nan)
    labels[support] = np.asarray(dataset["y"]).ravel()[support]
    direct = make_predictions(old_full, bases, shapes, adapters, labels, split, months)
    direct["regional_gamma"] = 0.0
    mixer_states = json.loads((run / "mixers.json").read_text())
    parent_mixers = json.loads((sources["prior"] / "mixers.json").read_text())
    for base in ("monthly", "daily"):
        for shape_name in SHAPES:
            name = f"{base}_integrated_{shape_name}"
            if mixer_states[name] != parent_mixers[name]:
                raise ValueError("Unchanged expert's ecological mixer differs from its parent")
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
    groups = {"all": np.ones(len(val_query), dtype=bool), "none_valid": valid_count[val_query] == 0,
              "some_valid": valid_count[val_query] > 0, "all_valid": valid_count[val_query] == 3,
              "partially_valid": (valid_count[val_query] > 0) & (valid_count[val_query] < 3)}
    validation_rows = []
    for arm in ARMS:
        selected = SupportAwareResidualTransfer.from_dict(mixer_states[f"{arm}_integrated_constant"])
        for stage in ("direct", "integrated"):
            prediction = (bases[arm][val_query] if stage == "direct" else selected.selected_base(
                context[val_query], bases[arm][val_query], memory[val_query], k=0))
            for group, mask in groups.items():
                n = int(mask.sum())
                mae = float(np.abs(prediction[mask] - validation_truth[val_query][mask]).mean()) if n else None
                validation_rows.append({"arm": arm, "stage": stage, "group": group, "n": n,
                                        "mae": mae, "gamma_k0": 0 if stage == "direct" else selected.selected_gamma(0)})
    stored_validation = pd.read_csv(run / "source_validation.csv")
    pd.testing.assert_frame_equal(stored_validation, pd.DataFrame(validation_rows), check_exact=False,
                                  rtol=1e-12, atol=1e-12)
    reference = prior_queries[prior_queries.model_name.isin(REFERENCE_MODELS)].copy()
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
    parent_controls = (*[f"{base}_{shape_name}" for base in ("context", "monthly", "daily") for shape_name in SHAPES],
                       *[f"{base}_integrated_{shape_name}" for base in ("monthly", "daily") for shape_name in SHAPES])
    copied_columns = ["cell", "station", "month", "analyte", "visibility_role", "model_name", "k", "y_pred",
                      "base_pred", "adaptation_delta", "support_count", "regional_gamma", "y_true", "split_seed", "seed"]
    for name in (*parent_controls, *REFERENCE_MODELS):
        before = prior_queries[prior_queries.model_name.eq(name)].sort_values(["k", "cell"])
        after = queries[queries.model_name.eq(name)].sort_values(["k", "cell"])
        pd.testing.assert_frame_equal(before[copied_columns].reset_index(drop=True),
                                      after[copied_columns].reset_index(drop=True), check_exact=True)
    return {"run": run.name, "split_seed": split_seed, "seed": seed, "n_full_grid": len(full),
            "uses_daily_cells": int(uses_daily.sum()), "monthly_fallback_cells": int((~uses_daily).sum()),
            "routing_stage": "native base before ecological mixture and station support adaptation",
            "native_fallback_bitwise_exact": True, "router_has_fitted_parameters": False,
            "parent_control_models_unchanged": len(parent_controls), "parent_reference_models_unchanged": len(REFERENCE_MODELS),
            "full_grid_max_abs_difference": differences, "query_replays": query_checks,
            "direct_adapter_validation_refits": len(adapters), "mixer_checks": mixer_checks,
            "expert_refits": 0, "source_bank_refitting": "none",
            "source_validation_rows_recomputed": len(validation_rows),
            "query_labels_not_supplied_to_predict": True,
            "parent_full_neural_replay_report_bound": True, "neural_replay_repeated": False,
            "parent_verification_sha256": config["parent_verification_hash"],
            "runtime_snapshot_hash": runtime, "parent_runtime_snapshot_hash": parent_runtime,
            "completion_sha256": sha256_file(run / "complete.json"),
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
