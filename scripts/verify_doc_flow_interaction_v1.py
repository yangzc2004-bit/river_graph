"""Replay saved state-by-flow DOC residuals and adaptation without fitting."""
from __future__ import annotations

import argparse
import gc
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from run_doc_flow_interaction_v1 import ARMS, MODELS, ROOT, SHAPES, arm_extra
from run_unified_doc_spatial import digest, verify_files, verify_runtime_snapshot
from run_unified_doc_spatial_v2 import make_predictions, read_source
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
from river_graph.models.episodic_temporal_adapter import gathered_rolling_states
from river_graph.models.kgml_local_transport import FIT_ROLES, build_rf_features
from river_graph.models.native_temporal_residual import NativeTemporalResidual
from river_graph.models.support_shape_adapter import SupportShapeAdapter
from river_graph.models.unified_doc import UnifiedDOCReconstructor

SPLITS, SEEDS, KS = (142, 143, 144), (42, 43, 44), (0, 1, 3, 5)
INTERACTION_INDICES = (0, 2, 4)
TRAIN_MEMORY = {"interaction_frozen": False, "interaction_tuned": True}
TRAINABLE_PARAMETERS = {"interaction_frozen": 267, "interaction_tuned": 25739}


def _interaction_layout(model, inputs):
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
            # The first numeric flow value varies fastest within each hidden unit.
            interaction = (hidden[:, :, None] * extra[:, INTERACTION_INDICES][:, None, :]).flatten(1)
            expected = torch.cat((hidden, extra, interaction), dim=1)
            model._delta_cells(prepared, cells)
        if len(observed) != 1:
            raise ValueError("Expected one scalar-head call for layout verification")
        torch.testing.assert_close(observed[0], expected, rtol=0, atol=0)
    finally:
        hook.remove()
    return {"sampled_cells": len(cells), "input_features": expected.shape[1],
            "ordering": "hidden-major outer product", "bitwise_exact": True}


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
    required = {"adapters.json", "flow_features.json",
                *(f"{name}.{suffix}" for name in ARMS for suffix in ("pt", "json"))}
    if set(sidecar["model_files"]) != required:
        raise ValueError("Sidecar does not bind the complete flow model/feature state")
    for name, expected_hash in sidecar["model_files"].items():
        if sha256_file(run / name) != expected_hash or completion["files"].get(name) != expected_hash:
            raise ValueError(f"Changed model state: {run / name}")


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
        raise ValueError("Run model or flow-arm definitions differ")
    if (tuple(config["interaction_indices"]) != INTERACTION_INDICES
            or config["train_memory_by_arm"] != TRAIN_MEMORY
            or config["interaction_order"] != "hidden-major outer product"):
        raise ValueError("Run interaction or memory-training definitions differ")
    completion = verify_files(run, "complete.json", config)
    required = {"config.json", "adapters.json", "flow_features.json", "full_grid.parquet",
                "full_grid.meta.json", "predictions.parquet", "predictions.meta.json", "timing.json"}
    required.update(f"{name}.{suffix}" for name in ARMS for suffix in ("pt", "json"))
    required.update(f"{name}_trace.csv" for name in ARMS)
    if not required.issubset(completion["files"]):
        raise ValueError("Completion record is missing required model/feature products")
    sources, source_configs = {}, {}
    for label in ("source", "prior", "basis", "oof"):
        package = Path(config[f"{label}_run"])
        package_config = json.loads((package / "config.json").read_text())
        if sha256_file(package / "complete.json") != config[f"{label}_completion_hash"]:
            raise ValueError(f"Changed {label} package binding")
        verify_files(package, "complete.json", package_config)
        sources[label], source_configs[label] = package, package_config
    for label, parent_key in (("source", "source"), ("basis", "basis"), ("oof", "oof")):
        if (Path(source_configs["prior"][f"{parent_key}_run"]) != sources[label]
                or source_configs["prior"][f"{parent_key}_completion_hash"]
                != config[f"{label}_completion_hash"]):
            raise ValueError(f"Prior and current {label} lineage differ")
    old_config, _, dataset, split, old_full = read_source(sources["source"])
    for name in ("dataset_path", "dataset_hash", "mask_path", "mask_hash", "q90_threshold_train"):
        if config[name] != old_config[name]:
            raise ValueError(f"New and source configuration disagree on {name}")
    shape, months = tuple(dataset["y"].shape), dataset["y"].shape[1]
    full = pd.read_parquet(run / "full_grid.parquet")
    queries = pd.read_parquet(run / "predictions.parquet")
    _sidecar(run, "full_grid.parquet", config, runtime, len(full), completion)
    _sidecar(run, "predictions.parquet", config, runtime, len(queries), completion)
    np.testing.assert_array_equal(full.cell.to_numpy(), np.arange(np.prod(shape)))
    identity = ["cell", "station", "month", "analyte", "visibility_role"]
    pd.testing.assert_frame_equal(full[identity], old_full[identity], check_exact=True)
    if queries.duplicated(["model_name", "k", "cell"]).any():
        raise ValueError("Duplicate adapted query identities")
    if set(map(tuple, queries[["model_name", "k"]].drop_duplicates().to_numpy())) != {
            (name, k) for name in MODELS for k in KS}:
        raise ValueError("Query product does not contain all six models and four K values")
    if not queries.split_seed.eq(split_seed).all() or not queries.seed.eq(seed).all():
        raise ValueError("Query/run identity differs")

    flow = build_causal_flow_features(dataset)
    definition = {key: value for key, value in flow.items() if key != "full"}
    if definition != config["flow_definition"] or definition != json.loads((run / "flow_features.json").read_text()):
        raise ValueError("Recomputed and saved flow definitions differ")
    if flow["full"].shape != (*shape, 10) or config["extra_dim"] != 10:
        raise ValueError("Expected ten aligned causal flow channels")
    if tuple(flow["value_feature_indices"]) != INTERACTION_INDICES:
        raise ValueError("Interactions must use exactly the three numeric flow channels")
    if (source_configs["prior"]["flow_definition"] != definition
            or source_configs["prior"]["extra_dim"] != 10):
        raise ValueError("Prior additive and current interaction flow features differ")
    # Inspect the bound additive reference, without refitting or replaying it.
    additive_payload = torch.load(sources["prior"] / "flow_values.pt",
                                  weights_only=False, map_location="cpu")
    if (tuple(additive_payload["head"]["weight"].shape) != (1, 74)
            or additive_payload["config"].get("interaction_indices", ())
            or not additive_payload["config"].get("train_memory", True)):
        raise ValueError("Expected the saved additive 74-input trainable-memory reference")
    del additive_payload
    flow_checks = []
    for j, name in enumerate(flow["feature_names"]):
        expected = full[f"flow_{name}"].to_numpy()
        actual = flow["full"][..., j].ravel()
        np.testing.assert_array_equal(actual, expected)
        flow_checks.append({"feature": name, "bitwise_exact": True,
                            "max_abs_difference": _close(actual, expected)})
    numerical = ["context_pred", *(f"{name}_{suffix}" for name in ARMS for suffix in ("delta", "pred"))]
    if not np.isfinite(full[numerical].to_numpy()).all():
        raise ValueError("Nonfinite full-grid components")

    torch.set_num_threads(int(config["torch_threads"]))
    expert = UnifiedDOCReconstructor.load(sources["source"], dataset, split)
    expert.context_forest.n_jobs = 1
    features = build_rf_features(dataset, split, FIT_ROLES, target_transform="log1p", include_network=True)
    recomputed_context = np.maximum(0, np.expm1(expert.context_forest.predict(features)))
    differences = {"context_pred": _close(recomputed_context, full.context_pred.to_numpy())}
    _close(recomputed_context, old_full.context_pred.to_numpy())
    # Forest parallel reductions can differ slightly; the exact saved context
    # is the actual input to residual replay, checked separately by tolerance.
    context = old_full.context_pred.to_numpy()
    np.testing.assert_array_equal(context, full.context_pred.to_numpy())
    del features
    recurrent_inputs = _full_inputs(expert)
    bases, summaries, tensor_checks = {"context": context}, [], 0
    for name in ARMS:
        extra = arm_extra(flow, name)
        np.testing.assert_array_equal(extra, flow["full"])
        inputs = {**recurrent_inputs, "extra": extra}
        payload = torch.load(run / f"{name}.pt", weights_only=False, map_location="cpu")
        summary = json.loads((run / f"{name}.json").read_text())
        if payload["summary"] != summary:
            raise ValueError("Checkpoint summary and saved JSON differ")
        restored = NativeTemporalResidual.from_payload(payload)
        _summary_matches(restored.to_dict(), summary)
        if (restored.hidden_size != 64 or restored.extra_dim != 10 or restored.head.in_features != 266
                or tuple(restored.interaction_indices) != INTERACTION_INDICES
                or restored.train_memory is not TRAIN_MEMORY[name]
                or summary["config"]["tail_weight"] != 2.0 or config["tail_weight"] != 2.0
                or summary["tail_threshold"] != config["q90_threshold_train"]
                or summary["protocol"]["tail_rule"] != "source_truth >= source_training_Q90"):
            raise ValueError("Flow readout or residual objective differs from the run")
        head_parameters = sum(value.numel() for value in restored.head.parameters())
        trainable_parameters = sum(value.numel()
                                   for module in (restored.temporal, restored.decay, restored.head)
                                   for value in module.parameters() if value.requires_grad)
        if (head_parameters != 267 or trainable_parameters != TRAINABLE_PARAMETERS[name]
                or summary["trainable_parameter_count"] != trainable_parameters):
            raise ValueError("Interaction head or trainable parameter count differs")
        if (not all(parameter.requires_grad == TRAIN_MEMORY[name]
                    for module in (restored.temporal, restored.decay) for parameter in module.parameters())
                or not all(parameter.requires_grad for parameter in restored.head.parameters())):
            raise ValueError("Memory/head parameter training flags differ")
        for key in ("lookback", "epochs", "patience", "batch_size", "learning_rate", "head_learning_rate"):
            if summary["config"][key] != config[key]:
                raise ValueError(f"Checkpoint setting differs from run configuration: {key}")
        if summary["config"]["scales"] != config["residual_scales"]:
            raise ValueError("Residual selection scales differ")
        for module_name in ("temporal", "decay", "head"):
            for key, value in getattr(restored, module_name).state_dict().items():
                torch.testing.assert_close(value, payload[module_name][key], rtol=0, atol=0)
                tensor_checks += 1
        for module_name in ("temporal", "decay"):
            for key, value in getattr(expert.residual.model, module_name).state_dict().items():
                torch.testing.assert_close(value, payload[f"initial_{module_name}"][key], rtol=0, atol=0)
                tensor_checks += 1
                if not TRAIN_MEMORY[name]:
                    torch.testing.assert_close(value, payload[module_name][key], rtol=0, atol=0)
                    tensor_checks += 1
        if not TRAIN_MEMORY[name]:
            for state in (summary, summary["validation_metrics"], *summary["trace"]):
                if state["temporal_parameter_distance"] != 0.0 or state["decay_parameter_distance"] != 0.0:
                    raise ValueError("Frozen memory changed in the selected checkpoint or training trace")
        interaction_layout = _interaction_layout(restored, inputs)
        delta = restored.predict_delta(inputs).ravel()
        prediction = restored.predict(inputs, context.reshape(shape)).ravel()
        np.testing.assert_array_equal(delta, full[f"{name}_delta"].to_numpy())
        np.testing.assert_array_equal(prediction, full[f"{name}_pred"].to_numpy())
        differences[f"{name}_delta"] = _close(delta, full[f"{name}_delta"].to_numpy())
        differences[f"{name}_pred"] = _close(prediction, full[f"{name}_pred"].to_numpy())
        bases[name] = prediction
        summaries.append({"arm": name, "best_epoch": summary["best_epoch"],
                          "epochs_run": summary["epochs_run"], "selected_scale": summary["selected_scale"],
                          "head_input_features": restored.head.in_features, "head_parameters": head_parameters,
                          "train_memory": restored.train_memory, "trainable_parameters": trainable_parameters,
                          "interaction_layout": interaction_layout,
                          "frozen_memory_equals_original": True if not TRAIN_MEMORY[name] else None,
                          "checkpoint_summary_json_exact": True, "reloaded_summary_tolerance_verified": True,
                          "tensor_identity_matches": True, "original_expert_initialization_exact": True,
                          "full_grid_residual_replay_bitwise_exact": True})
        del restored, payload, inputs, extra
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
            "flow_feature_replays": flow_checks, "full_grid_max_abs_difference": differences,
            "query_replays": query_checks, "checkpoint_summaries": summaries,
            "exact_tensor_checks": tensor_checks,
            "context_forest_recomputation_bitwise_exact": bool(np.array_equal(recomputed_context, context)),
            "flow_features_residuals_and_adapted_queries_bitwise_exact": True,
            "both_arms_use_identical_full_flow_features": True,
            "frozen_memory_bitwise_unchanged": True, "prior_additive_head_input_features": 74,
            "old_context_predictions_bitwise_unchanged": True,
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
        "shared_verifier_helpers_sha256": sha256_file("scripts/verify_doc_tail_residual_v1.py"),
        "results": results,
    }, indent=2, allow_nan=False) + "\n")
    print(path)


if __name__ == "__main__":
    main()
