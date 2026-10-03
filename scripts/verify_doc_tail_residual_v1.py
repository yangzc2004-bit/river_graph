"""Replay saved native-DOC residuals and support adaptation without fitting."""
from __future__ import annotations

import argparse
import gc
import json
from datetime import datetime, timezone
from numbers import Real
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from run_doc_tail_residual_v1 import ARMS, MODELS, ROOT, SHAPES
from run_unified_doc_spatial import digest, verify_files, verify_runtime_snapshot
from run_unified_doc_spatial_v2 import make_predictions, read_source

from river_graph.experiments.provenance import run_identity_sha256, sha256_file
from river_graph.experiments.unified_spatial_protocol import support_query_cells
from river_graph.models.kgml_local_transport import FIT_ROLES, build_rf_features
from river_graph.models.native_temporal_residual import NativeTemporalResidual
from river_graph.models.support_shape_adapter import SupportShapeAdapter
from river_graph.models.unified_doc import UnifiedDOCReconstructor

SPLITS, SEEDS, KS = (142, 143, 144), (42, 43, 44), (0, 1, 3, 5)
RTOL, ATOL = 1e-12, 1e-12


def _close(actual, expected):
    actual, expected = np.asarray(actual), np.asarray(expected)
    if not np.isfinite(actual).all() or not np.isfinite(expected).all():
        raise ValueError("Nonfinite replay or saved prediction")
    np.testing.assert_allclose(actual, expected, rtol=RTOL, atol=ATOL)
    return float(np.max(np.abs(actual - expected))) if actual.size else 0.0


def _summary_matches(actual, expected):
    if isinstance(expected, dict):
        if set(actual) != set(expected):
            raise ValueError("Reloaded summary keys changed")
        for key in expected:
            _summary_matches(actual[key], expected[key])
    elif isinstance(expected, (list, tuple)):
        if len(actual) != len(expected):
            raise ValueError("Reloaded summary length changed")
        for left, right in zip(actual, expected, strict=True):
            _summary_matches(left, right)
    elif isinstance(expected, Real) and not isinstance(expected, bool):
        _close(actual, expected)
    elif actual != expected:
        raise ValueError("Reloaded summary value changed")


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
    required = {"adapters.json", *(f"{name}.{suffix}" for name in ARMS for suffix in ("pt", "json"))}
    if set(sidecar["model_files"]) != required:
        raise ValueError("Sidecar does not bind the complete native model state")
    for name, expected_hash in sidecar["model_files"].items():
        if sha256_file(run / name) != expected_hash or completion["files"].get(name) != expected_hash:
            raise ValueError(f"Changed model state: {run / name}")


def _full_inputs(expert):
    residual, module = expert.residual, expert.residual.model
    prior = [(child, child.training) for child in module.modules()]
    module.eval()
    try:
        with torch.inference_mode():
            x, age = residual.input_view(expert.split, FIT_ROLES)
            encoded = module.encode_months(x, module._edge_index, module._edge_attr, residual.inputs.env_raw)
            support = torch.stack([x[..., 9], x[..., -2], x[..., -1]], dim=-1)
            return {"encoded": encoded.permute(1, 0, 2).contiguous().numpy().copy(),
                    "age": age.T.contiguous().numpy().copy(),
                    "support": support.permute(1, 0, 2).contiguous().numpy().copy()}
    finally:
        for child, training in prior:
            child.training = training


def verify_one(root, split_seed, seed, runtime):
    run = root / "runs" / f"split{split_seed}_seed{seed}"
    if not (run / "complete.json").is_file():
        raise ValueError(f"Requested run is incomplete: {run}")
    config = json.loads((run / "config.json").read_text())
    if (config["split_seed"], config["seed"]) != (split_seed, seed):
        raise ValueError("Run identity does not match its directory")
    if config["runtime_snapshot_hash"] != runtime:
        raise ValueError("Run runtime does not match its saved source snapshot")
    completion = verify_files(run, "complete.json", config)
    required = {"config.json", "adapters.json", "full_grid.parquet", "full_grid.meta.json",
                "predictions.parquet", "predictions.meta.json", "timing.json"}
    required.update(f"{name}.{suffix}" for name in ARMS for suffix in ("pt", "json"))
    required.update(f"{name}_trace.csv" for name in ARMS)
    if not required.issubset(completion["files"]):
        raise ValueError("Completion record is missing required model/products")
    sources = {}
    for label in ("source", "prior", "oof"):
        package = Path(config[f"{label}_run"])
        package_config = json.loads((package / "config.json").read_text())
        if sha256_file(package / "complete.json") != config[f"{label}_completion_hash"]:
            raise ValueError(f"Changed {label} package binding")
        verify_files(package, "complete.json", package_config)
        sources[label] = package
    old_config, _, dataset, split, old_full = read_source(sources["source"])
    for name in ("dataset_path", "dataset_hash", "mask_path", "mask_hash", "q90_threshold_train"):
        if config[name] != old_config[name]:
            raise ValueError(f"New and source configuration disagree on {name}")
    shape = tuple(dataset["y"].shape)
    months = shape[1]
    full = pd.read_parquet(run / "full_grid.parquet")
    queries = pd.read_parquet(run / "predictions.parquet")
    _sidecar(run, "full_grid.parquet", config, runtime, len(full), completion)
    _sidecar(run, "predictions.parquet", config, runtime, len(queries), completion)
    np.testing.assert_array_equal(full.cell.to_numpy(), np.arange(np.prod(shape)))
    identity_columns = ["cell", "station", "month", "analyte", "visibility_role"]
    pd.testing.assert_frame_equal(full[identity_columns], old_full[identity_columns], check_exact=True)
    numerical = ["context_pred", *(f"{name}_{suffix}" for name in ARMS for suffix in ("delta", "pred"))]
    if not np.isfinite(full[numerical].to_numpy()).all():
        raise ValueError("Nonfinite full-grid components")
    if queries.duplicated(["model_name", "k", "cell"]).any():
        raise ValueError("Duplicate adapted query identities")
    if set(map(tuple, queries[["model_name", "k"]].drop_duplicates().to_numpy())) != {
            (name, k) for name in MODELS for k in KS}:
        raise ValueError("Query product does not contain all six arms and four K values")
    if not queries.split_seed.eq(split_seed).all() or not queries.seed.eq(seed).all():
        raise ValueError("Query/run identity differs")

    # Match saved numerical settings; inference is sequential and forest
    # parallelism is limited independently. This performs no optimizer update.
    torch.set_num_threads(int(config["torch_threads"]))
    expert = UnifiedDOCReconstructor.load(sources["source"], dataset, split)
    expert.context_forest.n_jobs = 1
    features = build_rf_features(dataset, split, FIT_ROLES, target_transform="log1p", include_network=True)
    recomputed_context = np.maximum(0, np.expm1(expert.context_forest.predict(features)))
    differences = {"context_pred": _close(recomputed_context, full.context_pred.to_numpy())}
    _close(recomputed_context, old_full.context_pred.to_numpy())
    # The residual consumes the frozen source-product forest prediction. Keep
    # that exact input for tensor replay; recomputation above is separately
    # tolerance-checked because parallel forest reductions can differ slightly.
    context = old_full.context_pred.to_numpy()
    np.testing.assert_array_equal(context, full.context_pred.to_numpy())
    del features
    inputs = _full_inputs(expert)
    bases, summaries = {"context": context}, []
    tensor_checks = 0
    for name in ARMS:
        payload = torch.load(run / f"{name}.pt", weights_only=False, map_location="cpu")
        summary = json.loads((run / f"{name}.json").read_text())
        if payload["summary"] != summary:
            raise ValueError("Checkpoint summary and saved JSON differ")
        restored = NativeTemporalResidual.from_payload(payload)
        _summary_matches(restored.to_dict(), summary)
        for module_name in ("temporal", "decay", "head"):
            for key, value in getattr(restored, module_name).state_dict().items():
                torch.testing.assert_close(value, payload[module_name][key], rtol=0, atol=0)
                tensor_checks += 1
        for module_name in ("temporal", "decay"):
            for key, value in getattr(expert.residual.model, module_name).state_dict().items():
                torch.testing.assert_close(value, payload[f"initial_{module_name}"][key], rtol=0, atol=0)
                tensor_checks += 1
        if (summary["config"]["tail_weight"] != ARMS[name]
                or summary["tail_threshold"] != config["q90_threshold_train"]
                or summary["protocol"]["tail_rule"] != "source_truth >= source_training_Q90"):
            raise ValueError("Residual objective or tail definition differs from the run")
        delta = restored.predict_delta(inputs).ravel()
        prediction = restored.predict(inputs, context.reshape(shape)).ravel()
        np.testing.assert_array_equal(delta, full[f"{name}_delta"].to_numpy())
        np.testing.assert_array_equal(prediction, full[f"{name}_pred"].to_numpy())
        differences[f"{name}_delta"] = _close(delta, full[f"{name}_delta"].to_numpy())
        differences[f"{name}_pred"] = _close(prediction, full[f"{name}_pred"].to_numpy())
        expected = context if summary["selected_scale"] == 0 else np.maximum(
            0, context + summary["selected_scale"] * delta)
        _close(prediction, expected)
        bases[name] = prediction
        summaries.append({"arm": name, "best_epoch": summary["best_epoch"],
                          "epochs_run": summary["epochs_run"], "selected_scale": summary["selected_scale"],
                          "checkpoint_summary_json_exact": True, "reloaded_summary_tolerance_verified": True,
                          "tensor_identity_matches": True, "full_grid_residual_replay_bitwise_exact": True})
        del restored, payload
    with np.load(sources["prior"] / "representations.npz", allow_pickle=False) as saved:
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
    old_queries = pd.read_parquet(sources["prior"] / "predictions.parquet")
    query_checks = []
    for model_name in MODELS:
        for k in KS:
            actual = replay[replay.model_name.eq(model_name) & replay.k.eq(k)].sort_values("cell")
            stored = queries[queries.model_name.eq(model_name) & queries.k.eq(k)].sort_values("cell")
            np.testing.assert_array_equal(stored.cell, np.sort(fixed_query))
            np.testing.assert_array_equal(actual.cell, stored.cell)
            selected_support, _ = support_query_cells(split, target_role="test", k=k, n_months=months)
            if len(selected_support) != k * len(np.unique(fixed_query // months)):
                raise ValueError("Adaptation support count differs from K per station")
            for column in ("station", "month", "support_count"):
                np.testing.assert_array_equal(actual[column], stored[column])
            _close(stored.y_true, np.asarray(dataset["y"]).ravel()[stored.cell])
            error = max(_close(actual[column], stored[column])
                        for column in ("y_pred", "base_pred", "adaptation_delta"))
            for column in ("y_pred", "base_pred", "adaptation_delta"):
                np.testing.assert_array_equal(actual[column], stored[column])
            if k == 0:
                np.testing.assert_array_equal(stored.y_pred, stored.base_pred)
            if model_name.startswith("context_"):
                previous = old_queries[old_queries.model_name.eq(model_name) & old_queries.k.eq(k)].sort_values("cell")
                np.testing.assert_array_equal(stored.cell, previous.cell)
                np.testing.assert_array_equal(stored.y_pred, previous.y_pred)
            query_checks.append({"model_name": model_name, "k": k, "n_query": len(stored),
                                 "max_abs_difference": error, "bitwise_exact": True})
    return {"run": run.name, "split_seed": split_seed, "seed": seed, "n_full_grid": len(full),
            "full_grid_max_abs_difference": differences, "query_replays": query_checks,
            "checkpoint_summaries": summaries, "exact_tensor_checks": tensor_checks,
            "context_forest_recomputation_bitwise_exact": bool(np.array_equal(recomputed_context, context)),
            "native_residuals_and_adapted_queries_bitwise_exact": True,
            "old_v4_context_predictions_bitwise_unchanged": True,
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
        "verifier_sha256": sha256_file(__file__), "results": results,
    }, indent=2, allow_nan=False) + "\n")
    print(path)


if __name__ == "__main__":
    main()
