"""Independently replay recurrent clocks, native predictions and adaptation."""
from __future__ import annotations

import argparse
import gc
import io
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from run_unified_doc_spatial import verify_files, verify_runtime_snapshot
from run_unified_doc_spatial_v2 import fit_adapters, read_source
from verify_doc_daily_hydro_memory_v1 import _head_layout, _metadata
from verify_doc_daily_hydro_support_basis_v1 import (
    _manual_base,
    _refit,
    _same,
    _sidecar,
    _validation_rows,
)
from verify_doc_encoder_residual_v1 import _encoder_scope, _raw_cache
from verify_doc_regime_residual_v1 import _feature_views

from river_graph.experiments.provenance import sha256_file
from river_graph.experiments.unified_spatial_protocol import support_query_cells
from river_graph.models.recurrent_clock_residual import ClockNativeResidual
from river_graph.models.unified_doc import UnifiedDOCReconstructor

ROOT = Path("experiments/phase4_transfer/doc_recurrent_clock_v1")
SPLITS, SEEDS, KS = (142, 143, 144), (42, 43, 44), (0, 1, 3, 5)
ARMS = ("legacy", "unseen_neutral", "flow_window")
SHAPES = ("constant", "gru_tuned_anchor")
DIRECT = tuple(f"{arm}_{basis}" for arm in ("context", *ARMS) for basis in SHAPES)
INTEGRATED = tuple(f"{arm}_integrated_{basis}" for arm in ARMS for basis in SHAPES)
MODELS = (*DIRECT, *INTEGRATED)
INTERACTIONS = (0, 2, 4, 28, 30, 31, 32)


def _clock_formula(model, inputs, cells):
    """Build calendar windows and integer flow ages without the model clock."""
    prepared = model._prepare_inputs(inputs)
    cells = np.asarray(cells, dtype=np.int64)
    months = prepared["age"].shape[1]
    station = torch.as_tensor(cells // months)
    dates = cells[:, None] % months - np.arange(11, -1, -1)[None, :]
    valid = torch.as_tensor(dates >= 0)
    dates = torch.as_tensor(np.maximum(dates, 0))
    stations = station[:, None].expand_as(dates)
    legacy = prepared["age"][stations, dates]
    if model.decay_clock == "legacy":
        clock = legacy.clone()
    elif model.decay_clock == "unseen_neutral":
        seen = prepared["raw"][stations, dates, -7].bool()
        clock = torch.where(valid & seen, legacy, torch.zeros_like(legacy))
    else:
        visible = prepared["raw"][stations, dates, 3].numpy().astype(bool)
        age = np.full(dates.shape, 12, dtype=np.int64)
        for row in range(len(cells)):
            current = 12
            for step in range(12):
                if valid[row, step]:
                    current = 0 if visible[row, step] else min(12, current + 1)
                age[row, step] = current
        clock = torch.log1p(torch.as_tensor(age, dtype=model.dtype)) / float(np.log1p(12))
        clock = torch.where(valid, clock, torch.zeros_like(clock))
    features = torch.cat([clock[..., None], prepared["support"][stations, dates]], dim=-1)
    with torch.inference_mode():
        gamma = torch.exp(-torch.relu(torch.nn.functional.linear(
            features, model.decay.weight, model.decay.bias)))
    block = {"clock": clock.numpy(), "legacy_clock": legacy.numpy(), "valid": valid.numpy(),
             "month_indices": dates.numpy(), "station_indices": station.numpy(), "gamma": gamma.numpy()}
    actual = model.clock_window(inputs, cells, include_gamma=True)
    for name, value in block.items():
        np.testing.assert_array_equal(actual[name], value)
    return prepared, stations, dates, valid, gamma, block


def _recurrence_check(model, inputs):
    """Check that the actual GRU consumes the independently reconstructed gamma."""
    n, months = inputs["age"].shape
    cells = np.unique(np.concatenate((np.array([0, 1, 11, 12, months + 24]),
        np.linspace(0, n * months - 1, 64, dtype=np.int64))))
    prepared, stations, dates, valid, gamma, _ = _clock_formula(model, inputs, cells)
    with torch.inference_mode():
        encoded = model.spatial.encode_nodes(prepared["raw"][stations[valid], dates[valid]],
            torch.empty((2, 0), dtype=torch.long),
            torch.empty((0, model.spatial_architecture["edge_dim"]), dtype=model.dtype),
            prepared["env"][stations[valid]])
        sequence = encoded.new_zeros((len(cells), 12, 64))
        sequence[valid] = encoded
        state = encoded.new_zeros((len(cells), 64))
        for step in range(12):
            indicator = valid[:, step, None]
            x = torch.cat([sequence[:, step] * indicator, indicator.to(model.dtype)], dim=-1)
            updated = model.temporal(x, gamma[:, step] * state)
            state = torch.where(indicator, updated, state)
        torch.testing.assert_close(model._hidden_cells(prepared, torch.as_tensor(cells)), state, rtol=0, atol=0)
        early = torch.as_tensor([0, 1, 11, 12, months + 24])
        baseline = model._hidden_cells(prepared, early)
        changed = {name: value.clone() for name, value in prepared.items()}
        for name in ("raw", "age", "support", "extra"):
            changed[name][:, 25:] = 0
        torch.testing.assert_close(model._hidden_cells(changed, early), baseline, rtol=0, atol=0)
    return {"sampled_windows": len(cells), "clock_gamma_and_recurrence_bitwise_exact": True,
            "padding_and_window_bounds_checked": True, "future_feature_perturbation_invariant": True,
            "neutral_raw_last_valid_index": -7, "flow_raw_visibility_index": 3}


def _diagnostic_replay(model, inputs, query_mask):
    candidates = np.flatnonzero(query_mask.ravel())
    cells = candidates[np.unique(np.linspace(0, len(candidates)-1, min(2048, len(candidates)), dtype=int))]
    *_, block = _clock_formula(model, inputs, cells)
    valid, clock, gamma = block["valid"], block["clock"], block["gamma"]
    result = {"sample_cells": cells.tolist(), "valid_steps": int(valid.sum()),
              "clock_mean": float(clock[valid].mean()),
              "clock_quantiles": np.quantile(clock[valid], [.1, .5, .9]).tolist(),
              "mean_gamma": float(gamma[valid].mean()), "retention": {}}
    for lag in (3, 6, 11):
        eligible = valid[:, -lag:].all(1)
        result["retention"][str(lag)] = float(gamma[eligible, -lag:].prod(1).mean()) if eligible.any() else None
    return result


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
            if pc[f"{name}_{suffix}"] != config[f"{name}_{suffix}"]:
                raise ValueError("Parent expert, basis or source OOF package changed")
    for name in ("dataset", "mask", "daily_features", "daily_metadata"):
        if sha256_file(config[f"{name}_path"]) != config[f"{name}_hash"]:
            raise ValueError(f"Changed frozen {name}")
        for suffix in ("path", "hash"):
            if config[f"{name}_{suffix}"] != pc[f"{name}_{suffix}"]:
                raise ValueError(f"Changed parent {name} definition")
    for filename, key in (("off.pt", "parent_checkpoint_hash"), ("full_grid.parquet", "parent_full_grid_hash")):
        if sha256_file(sources["prior"] / filename) != config[key]:
            raise ValueError("Selected parent native model changed")
    report_path = sources["prior"].parent.parent / "verification/replay_checks.json"
    report = json.loads(report_path.read_text())
    rows = [row for row in report["results"] if row["run"] == sources["prior"].name]
    parent_runtime = verify_runtime_snapshot(sources["prior"].parent.parent)
    if (len(rows) != 1 or rows[0]["status"] != "verified"
            or rows[0]["completion_sha256"] != config["prior_completion_hash"]
            or rows[0]["runtime_snapshot_hash"] != parent_runtime):
        raise ValueError("Parent expert package lacks its bound independent replay")
    return sources, pc, report_path


def verify_one(root, split_seed, seed, runtime):
    run = root / "runs" / f"split{split_seed}_seed{seed}"
    config = json.loads((run / "config.json").read_text())
    if ((config["split_seed"], config["seed"]) != (split_seed, seed)
            or config["runtime_snapshot_hash"] != runtime
            or tuple(config["arms"]) != ARMS or set(config["models"]) != set(MODELS)
            or config["k_values"] != list(KS) or config["basis_names"] != list(SHAPES)
            or config["inference_roles"] != ["train"] or config["extra_dim"] != 38
            or config["interaction_indices"] != list(INTERACTIONS)
            or config["hydro_sequence_mode"] != "off" or config["encoder_mode"] != "last_self_ecology"
            or config["tail_weight"] != 2 or config["new_neural_fits"] != 2
            or config["forest_retraining"] or config["readout_fitting"]):
        raise ValueError("Clock, architecture, information or run identity differs")
    completion = verify_files(run, "complete.json", config)
    model_files = {"adapters.json", "mixers.json", "feature_definition.json", "clock_diagnostics.json",
                   *(f"{arm}.{suffix}" for arm in ARMS for suffix in ("pt", "json"))}
    required = {*model_files, "config.json", "source_validation.csv", "legacy_checks.json",
                "full_grid.parquet", "full_grid.meta.json", "predictions.parquet", "predictions.meta.json",
                "timing.json", *(f"{arm}_trace.csv" for arm in ARMS[1:])}
    if set(completion["files"]) != required:
        raise ValueError("Completed clock product set differs")
    sources, pc, parent_report = _lineage(config)
    if root == ROOT and any(config[key] != pc[key] for key in ("epochs", "patience")):
        raise ValueError("Formal clock experiment does not match the parent's training budget")
    old_config, _, dataset, split, old_full = read_source(sources["source"])
    for key in ("dataset_hash", "mask_hash", "q90_threshold_train", "query_cells"):
        if config[key] != old_config[key]:
            raise ValueError("Original cohort, split or source threshold changed")
    shape = tuple(dataset["y"].shape)
    months = shape[1]
    full = pd.read_parquet(run / "full_grid.parquet")
    queries = pd.read_parquet(run / "predictions.parquet")
    prior_full = pd.read_parquet(sources["prior"] / "full_grid.parquet")
    prior_queries = pd.read_parquet(sources["prior"] / "predictions.parquet")
    for filename, frame in (("full_grid.parquet", full), ("predictions.parquet", queries)):
        _sidecar(run, filename, config, runtime, frame, completion, model_files)
    identity = ["cell", "station", "month", "analyte", "visibility_role"]
    np.testing.assert_array_equal(full.cell, np.arange(np.prod(shape)))
    pd.testing.assert_frame_equal(full[identity], old_full[identity], check_exact=True)
    if (queries.duplicated(["model_name", "k", "cell"]).any()
            or set(map(tuple, queries[["model_name", "k"]].drop_duplicates().to_numpy()))
            != {(name, k) for name in MODELS for k in KS}
            or not queries.seed.eq(seed).all() or not queries.split_seed.eq(split_seed).all()):
        raise ValueError("Incomplete or duplicate query panels")
    context, memory = full.context_pred.to_numpy(), full.ecological_memory.to_numpy()
    _same(context, old_full.context_pred)
    _same(context, prior_full.context_pred)
    _same(memory, prior_full.ecological_memory)
    with np.load(sources["oof"] / "source_oof.npz", allow_pickle=False) as saved:
        oof = saved["pred_z"].copy()
    features, feature_checks = _feature_views(dataset, split, context, oof)
    with np.load(config["daily_features_path"], allow_pickle=False) as saved:
        daily = saved["full"].copy()
        np.testing.assert_array_equal(saved["site_no"], np.asarray(dataset["site_no"]).astype(str))
        np.testing.assert_array_equal(saved["months"], list(map(str, dataset["months"])))
    metadata = json.loads(Path(config["daily_metadata_path"]).read_text())
    if (daily.shape != (*shape, 8) or daily.dtype != np.float32 or not np.isfinite(daily).all()
            or not np.isin(daily[..., 5:8], [0, 1]).all() or metadata["dataset_hash"] != config["dataset_hash"]):
        raise ValueError("Daily pack alignment or validity changed")
    definition = {key: value for key, value in features.items()
                  if key not in ("source_extra", "full_extra", "source_station_ids")}
    definition["source_station_ids"] = features["source_station_ids"].tolist()
    definition.update({"feature_names": [*features["feature_names"], *metadata["feature_names"]],
        "daily_feature_names": metadata["feature_names"],
        "daily_value_feature_indices": [30+i for i in metadata["value_feature_indices"]],
        "daily_availability_feature_indices": [30+i for i in metadata["availability_feature_indices"]],
        "daily_feature_policy": metadata["policy"], "extra_dim": 38,
        "combined_interaction_indices": list(INTERACTIONS)})
    if (definition != config["feature_definition"] or definition != pc["feature_definition"]
            or (run / "feature_definition.json").read_bytes() != (sources["prior"] / "feature_definition.json").read_bytes()):
        raise ValueError("Rebuilt source-normalized head features differ")
    for name, value in _metadata(split, shape, daily).items():
        np.testing.assert_array_equal(full[name], value.ravel())
    torch.set_num_threads(int(config["torch_threads"]))
    expert = UnifiedDOCReconstructor.load(sources["source"], dataset, split)
    inputs, raw_checks = _raw_cache(expert, dataset, split, features)
    inputs["extra"] = np.concatenate([inputs["extra"], daily], axis=-1)
    val_ids = np.unique(split["val"] // months)
    val_inputs = {key: value[val_ids] for key, value in inputs.items()}
    _, val_query = support_query_cells(split, target_role="val", k=0, n_months=months)
    val_mask = np.zeros(shape, dtype=bool)
    val_mask.ravel()[val_query] = True
    val_mask = val_mask[val_ids]
    if (np.count_nonzero(inputs["raw"][val_ids, :, 8:10])
            or np.count_nonzero(inputs["raw"][np.unique(split["test"] // months), :, 8:10])):
        raise ValueError("Validation/target DOC labels entered base inputs")
    bases, summaries, tensor_count = {"context": context}, [], 0
    diagnostics = json.loads((run / "clock_diagnostics.json").read_text())
    for arm in ARMS:
        payload = torch.load(run / f"{arm}.pt", map_location="cpu", weights_only=True)
        summary = json.loads((run / f"{arm}.json").read_text())
        if payload["summary"] != summary:
            raise ValueError("Checkpoint and summary differ")
        model = ClockNativeResidual.from_payload(payload)
        if (model.decay_clock != arm or model.hydro_projection is not None
                or model.hydro_sequence_mode != "off" or model.head.in_features != 550
                or model.extra_dim != 38 or tuple(model.interaction_indices) != INTERACTIONS):
            raise ValueError("Loaded clock or matched native architecture differs")
        scope = _encoder_scope(model, payload, "last_self_ecology")
        trainable = sum(p.numel() for module in (model.spatial, model.temporal, model.decay, model.head)
                        for p in module.parameters() if p.requires_grad)
        if trainable != 31559 or summary["trainable_parameter_count"] != trainable:
            raise ValueError("Clock arm parameter allocation changed")
        for name in ("spatial", "temporal", "decay", "head"):
            for key, value in getattr(model, name).state_dict().items():
                torch.testing.assert_close(value, payload[name][key], rtol=0, atol=0)
                tensor_count += 1
        for name in ("spatial", "temporal", "decay"):
            for key, value in getattr(expert.residual.model, name).state_dict().items():
                torch.testing.assert_close(value, payload[f"initial_{name}"][key], rtol=0, atol=0)
                tensor_count += 1
        if arm == "legacy":
            if ((run / "legacy.pt").read_bytes() != (sources["prior"] / "off.pt").read_bytes()
                    or summary != json.loads((sources["prior"] / "off.json").read_text())):
                raise ValueError("Legacy checkpoint is not the unchanged parent off model")
        else:
            if payload["model_class"] != "ClockNativeResidual" or payload["config"]["decay_clock"] != arm:
                raise ValueError("New arm lacks its explicit serialized clock identity")
            for key in ("epochs", "patience", "lookback", "batch_size", "learning_rate", "head_learning_rate", "encoder_learning_rate"):
                if summary["config"][key] != config[key]:
                    raise ValueError(f"New arm setting differs: {key}")
            trace = pd.read_csv(run / f"{arm}_trace.csv")
            expected_trace = pd.read_csv(io.StringIO(pd.DataFrame(summary["trace"]).to_csv(index=False)))
            pd.testing.assert_frame_equal(trace, expected_trace, check_dtype=False,
                                          check_exact=False, rtol=1e-14, atol=1e-14)
        if (summary["config"]["tail_weight"] != 2
                or summary["config"]["scales"] != config["residual_scales"]
                or summary["tail_threshold"] != config["q90_threshold_train"]):
            raise ValueError("Tail weighting, source threshold or residual scale grid changed")
        source_y = torch.as_tensor(np.asarray(dataset["y"], dtype=float).ravel()[split["train"]])
        tail = source_y >= config["q90_threshold_train"]
        torch.testing.assert_close(model._source_weights(split["train"], source_y, tail, months),
                                   1 + tail.double(), rtol=0, atol=0)
        if (summary["n_source_cells"] != len(split["train"])
                or summary["n_source_tail_cells"] != int(tail.sum())
                or summary["protocol"]["validation_loss"] != "unweighted pooled fixed-query raw MAE"):
            raise ValueError("Training counts, tail loss or checkpoint objective differs")
        recurrent = _recurrence_check(model, inputs)
        if _diagnostic_replay(model, val_inputs, val_mask) != diagnostics[arm]:
            raise ValueError("Saved validation clock/retention diagnostics differ")
        layout = _head_layout(model, inputs)
        delta = model.predict_delta(inputs).ravel()
        prediction = model.predict(inputs, context.reshape(shape)).ravel()
        _same(delta, full[f"{arm}_delta"])
        _same(prediction, full[f"{arm}_pred"])
        if arm == "legacy":
            _same(delta, prior_full.off_delta)
            _same(prediction, prior_full.off_pred)
        bases[arm] = prediction
        summaries.append({"arm": arm, "trainable_parameters": trainable, "encoder_scope": scope,
            "head_layout": layout, "recurrent_checks": recurrent,
            "validation_clock_diagnostics_bitwise_exact": True,
            "full_grid_native_replay_bitwise_exact": True, "max_abs_difference": 0.0})
        del model, payload
    del expert, inputs, val_inputs, features, daily, oof

    with np.load(sources["basis"] / "representations.npz", allow_pickle=False) as saved:
        shapes = {name: saved[name].copy() for name in SHAPES}
    direct_states = json.loads((run / "adapters.json").read_text())
    mixer_states = json.loads((run / "mixers.json").read_text())
    if set(direct_states) != set(DIRECT) or set(mixer_states) != set(INTEGRATED):
        raise ValueError("Incomplete source-selected adapter panel")
    labels = np.full(np.prod(shape), np.nan)
    labels[split["val"]] = np.asarray(dataset["y"]).ravel()[split["val"]]
    direct = fit_adapters({"context": context}, shapes, labels, split, months)
    if any(model.to_dict() != direct_states[name] for name, model in direct.items()):
        raise ValueError("Context validation-only adapter refit differs")
    mixers, val_rows = {}, []
    for arm in ARMS:
        a, m = _refit(arm, bases[arm], shapes, context, memory, labels, split, months,
                      direct_states, mixer_states)
        direct.update(a)
        mixers.update(m)
        val_rows.extend(_validation_rows(arm, shapes, bases[arm], context, memory, labels,
                                         split, months, direct, mixers))
    stored_val = pd.read_csv(run / "source_validation.csv")
    if len(stored_val) != 48:
        raise ValueError("Expected all 48 source-validation adaptation panels")
    pd.testing.assert_frame_equal(stored_val, pd.DataFrame(val_rows), check_dtype=False,
                                  check_exact=False, rtol=1e-14, atol=1e-14)
    old_direct = json.loads((sources["prior"] / "adapters.json").read_text())
    old_mixers = json.loads((sources["prior"] / "mixers.json").read_text())
    for basis in SHAPES:
        if (direct_states[f"legacy_{basis}"] != old_direct[f"off_{basis}"]
                or mixer_states[f"legacy_integrated_{basis}"] != old_mixers[f"off_integrated_{basis}"]):
            raise ValueError("Legacy support adapters or ecological mixers changed")
    reserved, fixed_query = support_query_cells(split, target_role="test", k=5, n_months=months)
    if len(fixed_query) != config["query_cells"]:
        raise ValueError("Fixed target query population differs")
    labels = np.full(np.prod(shape), np.nan)
    labels[reserved] = np.asarray(dataset["y"]).ravel()[reserved]
    panels, controls = [], []
    for arm in ("context", *ARMS):
        for basis_name, basis in shapes.items():
            for stage in (("",) if arm == "context" else ("", "_integrated")):
                name = f"{arm}{stage}_{basis_name}"
                for k in KS:
                    support, query = support_query_cells(split, target_role="test", k=k, n_months=months)
                    np.testing.assert_array_equal(query, fixed_query)
                    if len(support) != k * len(np.unique(query // months)) or np.intersect1d(support, query).size:
                        raise ValueError("Query/support identities changed")
                    saved = queries[queries.model_name.eq(name) & queries.k.eq(k)].sort_values("cell")
                    np.testing.assert_array_equal(saved.cell, np.sort(query))
                    query = saved.cell.to_numpy()
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
                        _same(expected, saved[column])
                    for column in identity[1:]:
                        np.testing.assert_array_equal(saved[column], full.iloc[query][column])
                    if k == 0:
                        _same(pred, base[query])
                    if arm == "legacy":
                        old = prior_queries[prior_queries.model_name.eq(f"off{stage}_{basis_name}")
                                            & prior_queries.k.eq(k)].sort_values("cell")
                        for column in ("cell", "y_pred", "base_pred", "adaptation_delta", "regional_gamma"):
                            np.testing.assert_array_equal(saved[column], old[column])
                        controls.append({"model_name": name, "k": k, "rows": len(query), "bitwise_exact": True})
                    panels.append({"model_name": name, "k": k, "n_query": len(query),
                                   "bitwise_exact": True, "max_abs_difference": 0.0})
    recorded_controls = json.loads((run / "legacy_checks.json").read_text())
    if sorted(controls, key=lambda x: (x["model_name"], x["k"])) != sorted(recorded_controls, key=lambda x: (x["model_name"], x["k"])):
        raise ValueError("Recorded legacy checks differ from independent control replay")
    return {"run": run.name, "status": "verified", "completion_sha256": sha256_file(run / "complete.json"),
        "runtime_snapshot_hash": runtime, "n_full_grid": len(full), "query_replays": panels,
        "legacy_controls": controls, "native_models": summaries, "exact_tensor_checks": tensor_count,
        "direct_adapter_validation_refits": len(direct), "integrated_mixer_validation_refits": len(mixers),
        "source_validation_rows": len(stored_val), "raw_cache_checks": raw_checks, "feature_checks": feature_checks,
        "source_labels_hidden_from_validation_and_target_base": True,
        "source_validation_refits_receive_nan_elsewhere": True,
        "legacy_checkpoint_byte_identical": True, "parent_verification_path": str(parent_report),
        "parent_verification_sha256": sha256_file(parent_report), "neural_or_forest_refitted": False}


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
        "target_comparative_performance_analyzed": False,
        "neural_or_forest_refitted": False, "native_and_query_replay_rtol": 0, "native_and_query_replay_atol": 0,
        "verifier_sha256": sha256_file(__file__), "shared_verifier_helpers_sha256": {
            name: sha256_file(name) for name in (
                "scripts/verify_doc_daily_hydro_memory_v1.py",
                "scripts/verify_doc_daily_hydro_support_basis_v1.py",
                "scripts/verify_doc_encoder_residual_v1.py", "scripts/verify_doc_regime_residual_v1.py",
                "scripts/verify_doc_ecological_transfer_v2.py")}, "results": results}, indent=2)+"\n")
    print(f"Verified {len(results)} complete packages: {output}", flush=True)


if __name__ == "__main__":
    main()
