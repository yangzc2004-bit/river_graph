"""Replay current-expert support bases and validation-only station adaptation."""
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
from run_unified_doc_spatial_v2 import fit_adapters, read_source
from verify_doc_ecological_transfer_v2 import _selection_check, _validation_episodes

from river_graph.experiments.provenance import run_identity_sha256, sha256_file
from river_graph.experiments.unified_spatial_protocol import support_query_cells
from river_graph.models.encoder_native_residual import EncoderNativeResidual
from river_graph.models.episodic_temporal_adapter import EpisodicTemporalAdapter
from river_graph.models.raw_temporal_features import extract_raw_temporal_inputs
from river_graph.models.support_aware_residual_transfer import (
    SupportAwareResidualTransfer,
)
from river_graph.models.support_shape_adapter import SupportShapeAdapter
from river_graph.models.unified_doc import UnifiedDOCReconstructor

ROOT = Path("experiments/phase4_transfer/doc_daily_hydro_support_basis_v1")
SPLITS, SEEDS, KS = (142, 143, 144), (42, 43, 44), (0, 1, 3, 5)
ARMS = ("off", "current_only", "full_history")
SHAPES = ("constant", "legacy", "refreshed")
MODELS = tuple(f"{arm}{stage}_{shape}" for arm in ARMS
               for stage in ("", "_integrated") for shape in SHAPES)


def _independent_projection(model, inputs, adapter, batch_size=2048):
    """Replay selected hidden states and anchor normalization without the helper."""
    prepared = model._prepare_inputs(inputs)
    n, months = prepared["age"].shape
    if adapter.readout.shape != (64, 2) or adapter.anchor_count != 32 or adapter.scale_floor != 1e-4:
        raise ValueError("Frozen v4 readout or calendar normalization changed")
    raw = torch.empty((n*months, 2), dtype=torch.float64)
    with torch.inference_mode():
        for start in range(0, n*months, batch_size):
            cells = torch.arange(start, min(n*months, start+batch_size))
            hidden = model._hidden_cells(prepared, cells)
            raw[start:start+len(cells)] = hidden.double() @ adapter.readout
        raw = raw.reshape(n, months, 2)
        anchors = np.linspace(0, months-1, min(months, 32), dtype=np.int64)
        anchored = raw[:, anchors]
        mean = anchored.mean(dim=1, keepdim=True)
        variance = (anchored-mean).square().mean(dim=(1, 2), keepdim=True)
        scale = variance.clamp_min(1e-8).sqrt()
        basis = (raw-mean)/scale
        rms = variance.sqrt().reshape(-1)
    if not torch.isfinite(basis).all():
        raise ValueError("Replayed refreshed basis is not finite")
    return {"basis": basis.numpy(), "raw_basis": raw.numpy(), "anchor_months": anchors,
            "station_anchor_mean": mean[:, 0].numpy(), "station_rms": rms.numpy(),
            "station_scale": scale.reshape(-1).numpy(), "floor_hit": rms.numpy() < 1e-4}


def _manual_base(context, native, memory, gamma):
    return native.copy() if gamma == 0 else np.maximum(0, context+(1-gamma)*(native-context)+gamma*memory)


def _same(a, b):
    a, b = np.asarray(a), np.asarray(b)
    if not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ValueError("Nonfinite saved or replayed product")
    np.testing.assert_array_equal(a, b)
    return 0.0


def _refit(arm, base, shapes, context, memory, labels, split, months, adapter_states, mixer_states):
    adapters = fit_adapters({arm: base}, shapes, labels, split, months)
    for name, adapter in adapters.items():
        if adapter.to_dict() != adapter_states[name]:
            raise ValueError(f"Validation-only direct adapter differs: {name}")
        if SupportShapeAdapter.from_dict(adapter_states[name]).to_dict() != adapter_states[name]:
            raise ValueError("Direct support adapter JSON roundtrip differs")
    _, query = support_query_cells(split, target_role="val", k=0, n_months=months)
    scores = [(float(np.abs(_manual_base(context[query], base[query], memory[query], gamma)
                           - labels[query]).mean()), gamma) for gamma in (0, .25, .5, 1)]
    _, gamma_k0 = min(scores)
    mixers = {}
    validation_dataset = {"y": labels.reshape(-1, months)}
    for shape_name, basis in shapes.items():
        name = f"{arm}_integrated_{shape_name}"
        state = mixer_states[name]
        _selection_check(state, gamma_k0, shape_name, adapter_states[f"{arm}_{shape_name}"])
        episodes = _validation_episodes(validation_dataset, split, context, base, memory, basis)
        kwargs = {"ridge_strengths": (float("inf"),)} if shape_name == "constant" else {}
        model = SupportAwareResidualTransfer(months, **kwargs).fit(
            episodes, gamma_k0=gamma_k0, selection_role="source_validation")
        if model.to_dict() != state or SupportAwareResidualTransfer.from_dict(state).to_dict() != state:
            raise ValueError(f"Validation-only mixer refit differs: {name}")
        mixers[name] = model
    return adapters, mixers


def _query_replays(queries, previous, full, dataset, split, bases, context, memory,
                   shapes_by_arm, adapters, mixers):
    months = dataset["y"].shape[1]
    reserved, fixed_query = support_query_cells(split, target_role="test", k=5, n_months=months)
    truth = np.asarray(dataset["y"]).ravel()
    labels = np.full(truth.size, np.nan)
    labels[reserved] = truth[reserved]
    panels, legacy_count = [], 0
    for arm in ARMS:
        for shape_name, basis in shapes_by_arm[arm].items():
            for stage in ("", "_integrated"):
                name = f"{arm}{stage}_{shape_name}"
                for k in KS:
                    support, query = support_query_cells(split, target_role="test", k=k, n_months=months)
                    np.testing.assert_array_equal(query, fixed_query)
                    if len(support) != k*len(np.unique(query//months)) or np.intersect1d(support, query).size:
                        raise ValueError("Support and fixed query identities differ")
                    stored = queries[queries.model_name.eq(name) & queries.k.eq(k)].sort_values("cell")
                    np.testing.assert_array_equal(stored.cell, np.sort(query))
                    query = stored.cell.to_numpy()
                    if stage:
                        model = mixers[name]
                        gamma = model.selected_gamma(k)
                        base = _manual_base(context, bases[arm], memory, gamma)
                        _same(base, model.selected_base(context, bases[arm], memory, k=k))
                        predicted = model.adapt(context[query], bases[arm][query], memory[query], query,
                            context[support], bases[arm][support], memory[support], support,
                            labels[support], basis[query], basis[support], k=k)
                        if k == 0:
                            _same(base, full[f"{arm}_integrated_k0_pred"])
                    else:
                        gamma, base = 0, bases[arm]
                        predicted = adapters[name].adapt(base[query], query, base[support], support,
                            labels[support], query_basis=basis[query], support_basis=basis[support], k=k)
                    _same(predicted, stored.y_pred)
                    _same(base[query], stored.base_pred)
                    _same(np.log1p(predicted)-np.log1p(base[query]), stored.adaptation_delta)
                    _same(np.full(len(query), gamma), stored.regional_gamma)
                    _same(np.full(len(query), k), stored.support_count)
                    _same(truth[query], stored.y_true)
                    for column in ("station", "month", "analyte", "visibility_role"):
                        np.testing.assert_array_equal(stored[column], full.iloc[query][column])
                    if k == 0:
                        _same(predicted, base[query])
                        constant = queries[queries.model_name.eq(f"{arm}{stage}_constant")
                                           & queries.k.eq(0)].sort_values("cell")
                        _same(predicted, constant.y_pred)
                    if shape_name in ("constant", "legacy"):
                        old_shape = "constant" if shape_name == "constant" else "gru_tuned_anchor"
                        old = previous[previous.model_name.eq(f"{arm}{stage}_{old_shape}")
                                       & previous.k.eq(k)].sort_values("cell")
                        np.testing.assert_array_equal(old.cell, query)
                        for column in ("y_pred", "base_pred", "adaptation_delta", "regional_gamma"):
                            _same(stored[column], old[column])
                        legacy_count += 1
                    panels.append({"model_name": name, "k": k, "n_query": len(query),
                                   "bitwise_exact": True, "max_abs_difference": 0.0})
    return panels, legacy_count


def _lineage(config):
    packages = {}
    for name in ("prior", "source", "basis"):
        path = Path(config[f"{name}_run"])
        if sha256_file(path / "complete.json") != config[f"{name}_completion_hash"]:
            raise ValueError(f"Changed parent package: {name}")
        state = json.loads((path / "config.json").read_text())
        verify_files(path, "complete.json", state)
        packages[name] = path
    prior_config = json.loads((packages["prior"] / "config.json").read_text())
    for name in ("source", "basis"):
        if any(config[f"{name}_{suffix}"] != prior_config[f"{name}_{suffix}"]
               for suffix in ("run", "completion_hash")):
            raise ValueError(f"The postprocessing changed its parent's {name} lineage")
    for key in ("daily_features_path", "daily_features_hash", "daily_metadata_path", "daily_metadata_hash",
                "dataset_path", "dataset_hash", "mask_path", "mask_hash"):
        if config[key] != prior_config[key]:
            raise ValueError(f"An unchanged parent input has changed: {key}")
    for name in ("daily_features", "daily_metadata", "dataset", "mask"):
        if sha256_file(config[f"{name}_path"]) != config[f"{name}_hash"]:
            raise ValueError(f"Changed bound input: {name}")
    runtime = verify_runtime_snapshot(packages["prior"].parent.parent)
    report_path = packages["prior"].parent.parent / "verification/replay_checks.json"
    report = json.loads(report_path.read_text())
    verified = [row for row in report["results"] if row["run"] == packages["prior"].name]
    if (len(verified) != 1 or verified[0]["status"] != "verified"
            or verified[0]["completion_sha256"] != config["prior_completion_hash"]
            or verified[0]["runtime_snapshot_hash"] != runtime):
        raise ValueError("The parent numerical replay does not bind the selected current experts")
    packages["parent_verification"] = report_path
    return packages


def _sidecar(run, filename, config, runtime, frame, completion, model_files):
    path = run / filename
    state = json.loads(path.with_suffix(".meta.json").read_text())
    expected = {"config": config, "config_hash": digest(config),
        "runtime_snapshot_hash": runtime, "dataset_sha256": config["dataset_hash"],
        "mask_sha256": config["mask_hash"], "prediction_sha256": sha256_file(path),
        "rows": len(frame), "selection_role": "source_validation",
        "run_identity_sha256": run_identity_sha256(digest(config), config["started_at"], runtime)}
    if any(state[key] != value for key, value in expected.items()):
        raise ValueError("Product sidecar does not match its complete experiment identity")
    if set(state["model_files"]) != set(model_files):
        raise ValueError("Product is not bound to every required basis and adapter state")
    for name, value in state["model_files"].items():
        if sha256_file(run / name) != value or completion["files"][name] != value:
            raise ValueError(f"Changed support state: {name}")


def _validation_rows(arm, shapes, base, context, memory, truth, split, months, adapters, mixers):
    rows = []
    for shape_name, basis in shapes.items():
        for k in KS:
            support, query = support_query_cells(split, target_role="val", k=k, n_months=months)
            for stage in ("direct", "integrated"):
                suffix = "" if stage == "direct" else "_integrated"
                name = f"{arm}{suffix}_{shape_name}"
                if stage == "direct":
                    prediction = adapters[name].adapt(base[query], query, base[support], support,
                        truth[support], query_basis=basis[query], support_basis=basis[support], k=k)
                else:
                    prediction = mixers[name].adapt(context[query], base[query], memory[query], query,
                        context[support], base[support], memory[support], support, truth[support],
                        basis[query], basis[support], k=k)
                rows.append({"arm": arm, "basis": shape_name, "k": k, "n": len(query),
                    "model_name": name, "stage": stage,
                    "mae": float(np.abs(prediction-truth[query]).mean())})
    return rows


def verify_one(root, partition, seed, runtime):
    run = root / "runs" / f"split{partition}_seed{seed}"
    if not (run / "complete.json").is_file():
        raise ValueError(f"Requested package is incomplete: {run}")
    config = json.loads((run / "config.json").read_text())
    if ((config["split_seed"], config["seed"]) != (partition, seed)
            or config["runtime_snapshot_hash"] != runtime or tuple(config["arms"]) != ARMS
            or tuple(config["basis_names"]) != SHAPES or set(config["models"]) != set(MODELS)
            or config["k_values"] != list(KS) or config["inference_roles"] != ["train"]
            or config["anchor_count"] != 32 or config["scale_floor"] != 1e-4
            or config["selection_role"] != "source_validation"
            or config["gamma_k0"] != "frozen parent source-validation choice"
            or any(config[key] is not False for key in ("backbone_retraining", "forest_retraining", "readout_fitting"))):
        raise ValueError("Support-basis experiment identities or fixed definitions differ")
    complete = verify_files(run, "complete.json", config)
    model_files = ("representations.npz", "basis_definition.json", "adapters.json", "mixers.json")
    required = {"config.json", *model_files, "source_validation.csv", "legacy_checks.json",
                "predictions.parquet", "predictions.meta.json", "full_grid.parquet", "full_grid.meta.json", "timing.json"}
    if set(complete["files"]) != required:
        raise ValueError("Support-basis completion record omits required products")
    packages = _lineage(config)
    prior, source, basis_run = (packages[name] for name in ("prior", "source", "basis"))
    old_config, _, dataset, split, old_full = read_source(source)
    for key in ("dataset_path", "dataset_hash", "mask_path", "mask_hash", "q90_threshold_train"):
        if config[key] != old_config[key]:
            raise ValueError(f"Source and current data identities differ: {key}")
    full = pd.read_parquet(run / "full_grid.parquet")
    queries = pd.read_parquet(run / "predictions.parquet")
    prior_full = pd.read_parquet(prior / "full_grid.parquet")
    previous = pd.read_parquet(prior / "predictions.parquet")
    if sha256_file(run / "full_grid.parquet") != config["parent_full_grid_hash"] or sha256_file(prior / "full_grid.parquet") != config["parent_full_grid_hash"]:
        raise ValueError("The inherited full grid is not byte-identical to the parent")
    pd.testing.assert_frame_equal(full, prior_full, check_exact=True)
    for filename, frame in (("full_grid.parquet", full), ("predictions.parquet", queries)):
        _sidecar(run, filename, config, runtime, frame, complete, model_files)
    shape = tuple(dataset["y"].shape)
    months, cells = shape[1], int(np.prod(shape))
    np.testing.assert_array_equal(full.cell, np.arange(cells))
    identity = ["cell", "station", "month", "analyte", "visibility_role"]
    pd.testing.assert_frame_equal(full[identity], old_full[identity], check_exact=True)
    if (queries.duplicated(["model_name", "k", "cell"]).any()
            or set(map(tuple, queries[["model_name", "k"]].drop_duplicates().to_numpy()))
            != {(name, k) for name in MODELS for k in KS}
            or not queries.seed.eq(seed).all() or not queries.split_seed.eq(partition).all()):
        raise ValueError("Query identities differ or model/K panels are incomplete")
    if (sha256_file(basis_run / "memory.pt") != config["basis_checkpoint_hash"]
            or sha256_file(basis_run / "representations.npz") != config["legacy_basis_hash"]
            or set(config["parent_checkpoint_hashes"]) != set(ARMS)):
        raise ValueError("Frozen v4 support readout/basis bindings changed")
    torch.set_num_threads(config["torch_threads"])
    v4 = EpisodicTemporalAdapter.from_payload(torch.load(basis_run / "memory.pt", weights_only=True))
    definition = json.loads((run / "basis_definition.json").read_text())
    expected_definition = {"readout": v4.readout.numpy().tolist(), "anchor_count": 32,
        "scale_floor": 1e-4, "readout_fitting": "none",
        "raw_state": "selected current encoder/GRU; no scalar head or extra-head channels",
        "normalization_role": "retrospective feature-only record normalization",
        "parent_checkpoint_hashes": config["parent_checkpoint_hashes"]}
    if definition != expected_definition:
        raise ValueError("Saved basis definition differs from the frozen readout")
    with np.load(run / "representations.npz", allow_pickle=False) as saved:
        archive = {key: saved[key].copy() for key in saved.files}
    expected_keys = {"constant", "legacy", "anchor_months",
        *(f"{arm}_{name}" for arm in ARMS for name in
          ("refreshed", "raw_basis", "station_anchor_mean", "station_rms", "station_scale", "floor_hit"))}
    if set(archive) != expected_keys:
        raise ValueError("The basis archive is incomplete")
    with np.load(basis_run / "representations.npz", allow_pickle=False) as saved:
        _same(archive["constant"], saved["constant"])
        _same(archive["legacy"], saved["gru_tuned_anchor"])
    _same(archive["constant"], np.zeros((cells, 2)))
    with np.load(config["daily_features_path"], allow_pickle=False) as saved:
        daily = saved["full"].copy()
        np.testing.assert_array_equal(saved["site_no"], np.asarray(dataset["site_no"]).astype(str))
        np.testing.assert_array_equal(saved["months"], [str(value) for value in dataset["months"]])
    expert = UnifiedDOCReconstructor.load(source, dataset, split)
    raw = extract_raw_temporal_inputs(expert, dataset, split)
    if np.count_nonzero(raw["source_raw"][..., 8:10]):
        raise ValueError("Source fold raw inputs contain held-station DOC")
    held_ids = np.unique(np.concatenate((split["val"], split["test"]))) // months
    if np.count_nonzero(raw["full_raw"][np.unique(held_ids), :, 8:10]):
        raise ValueError("Held validation/test DOC is visible in the model inputs")
    inputs = {key: raw[f"full_{key}"] for key in ("raw", "age", "support")}
    inputs.update(env=raw["env"], daily_history=daily,
                  extra=np.zeros((*shape, config["extra_dim"]), dtype=np.float32))
    del expert, raw
    basis_checks = []
    for arm in ARMS:
        checkpoint = prior / f"{arm}.pt"
        if sha256_file(checkpoint) != config["parent_checkpoint_hashes"][arm]:
            raise ValueError(f"Selected current model changed: {arm}")
        model = EncoderNativeResidual.from_payload(torch.load(checkpoint, weights_only=True))
        if model.hydro_sequence_mode != arm:
            raise ValueError("Loaded expert has the wrong hydrologic history mode")
        replay = _independent_projection(model, inputs, v4, config["batch_size"])
        for key, value in replay.items():
            saved_key = "anchor_months" if key == "anchor_months" else f"{arm}_{'refreshed' if key == 'basis' else key}"
            _same(value, archive[saved_key])
        if sha256_file(checkpoint) != config["parent_checkpoint_hashes"][arm]:
            raise ValueError("Basis extraction modified a frozen checkpoint")
        basis_checks.append({"arm": arm, "n_grid_cells": cells, "raw_projection_bitwise_exact": True,
            "basis_and_anchor_statistics_bitwise_exact": True, "checkpoint_unchanged": True,
            "n_anchors": len(replay["anchor_months"]), "floor_hits": int(replay["floor_hit"].sum()),
            "head_extra_not_used": True, "max_abs_difference": 0.0})
        del replay, model
    del inputs, daily, v4
    context, memory = full.context_pred.to_numpy(), full.ecological_memory.to_numpy()
    _same(context, old_full.context_pred)
    bases = {arm: full[f"{arm}_pred"].to_numpy() for arm in ARMS}
    shapes = {arm: {"constant": archive["constant"], "legacy": archive["legacy"],
                    "refreshed": archive[f"{arm}_refreshed"].reshape(-1, 2)} for arm in ARMS}
    labels = np.full(cells, np.nan)
    labels[split["val"]] = np.asarray(dataset["y"], dtype=float).ravel()[split["val"]]
    adapter_states = json.loads((run / "adapters.json").read_text())
    mixer_states = json.loads((run / "mixers.json").read_text())
    if (set(adapter_states) != {f"{arm}_{name}" for arm in ARMS for name in SHAPES}
            or set(mixer_states) != {f"{arm}_integrated_{name}" for arm in ARMS for name in SHAPES}):
        raise ValueError("Support adapter or mixer states are incomplete")
    previous_adapters = json.loads((prior / "adapters.json").read_text())
    previous_mixers = json.loads((prior / "mixers.json").read_text())
    adapters, mixers, validation = {}, {}, []
    for arm in ARMS:
        a, m = _refit(arm, bases[arm], shapes[arm], context, memory, labels, split, months, adapter_states, mixer_states)
        frozen_gamma = previous_mixers[f"{arm}_integrated_constant"]["gamma_k0"]
        if any(model.gamma_k0_ != frozen_gamma for model in m.values()):
            raise ValueError("K0 mixture was not locked to its parent source-validation choice")
        for name, previous_name in (("constant", "constant"), ("legacy", "gru_tuned_anchor")):
            if (a[f"{arm}_{name}"].to_dict() != previous_adapters[f"{arm}_{previous_name}"]
                    or m[f"{arm}_integrated_{name}"].to_dict() != previous_mixers[f"{arm}_integrated_{previous_name}"]):
                raise ValueError("The unchanged constant/legacy support fit changed")
        validation.extend(_validation_rows(arm, shapes[arm], bases[arm], context, memory, labels, split, months, a, m))
        adapters.update(a)
        mixers.update(m)
    expected_validation = pd.DataFrame(validation)
    stored_validation = pd.read_csv(run / "source_validation.csv")
    if len(stored_validation) != 72:
        raise ValueError("The source-validation K/basis/pipeline table is incomplete")
    for key in expected_validation.columns.difference(["mae"]):
        np.testing.assert_array_equal(expected_validation[key], stored_validation[key])
    np.testing.assert_allclose(expected_validation.mae, stored_validation.mae, rtol=1e-14, atol=1e-14)
    panels, legacy_count = _query_replays(queries, previous, full, dataset, split, bases, context,
                                        memory, shapes, adapters, mixers)
    saved_checks = json.loads((run / "legacy_checks.json").read_text())
    expected_checks = [{"model_name": f"{arm}{stage}_{name}", "parent_model": f"{arm}{stage}_{old}",
        "k": k, "rows": config["query_cells"], "bitwise_exact": True}
        for arm in ARMS for stage in ("", "_integrated")
        for name, old in (("constant", "constant"), ("legacy", "gru_tuned_anchor")) for k in KS]
    if saved_checks != expected_checks:
        raise ValueError("Stored legacy replication checks are incomplete or inconsistent")
    return {"run": run.name, "status": "verified", "runtime_snapshot_hash": runtime,
        "completion_sha256": sha256_file(run / "complete.json"),
        "parent_verification_path": str(packages["parent_verification"]),
        "parent_verification_sha256": sha256_file(packages["parent_verification"]),
        "n_full_grid": len(full), "parent_full_grid_byte_identical": True,
        "basis_replays": basis_checks, "direct_validation_refits": len(adapters),
        "integrated_validation_refits": len(mixers), "validation_rows_reproduced": len(validation),
        "constant_legacy_panels_bitwise_exact": legacy_count, "k0_basis_invariant": True,
        "query_replays": panels, "source_and_target_input_isolation_verified": True,
        "neural_forest_readout_training": "none"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--split-seeds", nargs="+", type=int, default=list(SPLITS))
    parser.add_argument("--seeds", nargs="+", type=int, default=list(SEEDS))
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    runtime = verify_runtime_snapshot(args.root)
    requested = [(split, seed) for split in args.split_seeds for seed in args.seeds]
    if not requested or len(set(requested)) != len(requested):
        raise ValueError("Replay identities must be nonempty and unique")
    results = []
    for partition, seed in requested:
        print(f"Replaying split{partition}_seed{seed}", flush=True)
        results.append(verify_one(args.root, partition, seed, runtime))
        gc.collect()
    output = args.output or args.root / "verification/replay_checks.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps({"checked_at": datetime.now(timezone.utc).isoformat(),
        "requested_runs": len(requested), "verified_runs": len(results), "expected_production_runs": 9,
        "all_nine_replayed": set(requested) == {(partition, seed) for partition in SPLITS for seed in SEEDS},
        "scope": "only explicitly listed completed packages were numerically replayed",
        "target_comparative_performance_analyzed": False,
        "verifier_sha256": sha256_file(__file__), "runtime_snapshot_hash": runtime,
        "shared_helpers_sha256": {name: sha256_file(name) for name in (
            "scripts/run_unified_doc_spatial.py", "scripts/run_unified_doc_spatial_v2.py",
            "scripts/verify_doc_ecological_transfer_v2.py")}, "results": results}, indent=2, allow_nan=False)+"\n")
    print(output)


if __name__ == "__main__":
    main()
