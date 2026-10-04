"""Independent saved-state replay of fresh station-role DOC confirmation.

No forest or neural optimization is run. Source moments, chemical PCA and
validation-only small support/calibration selections are recalculated. Native
source residuals remain explicitly source-trained over an OOF forest baseline.
"""
from __future__ import annotations

import argparse
import gc
import json
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
from run_unified_doc_spatial import (
    verify_files,
    verify_runtime_snapshot,
    write_json,
)
from verify_doc_auxiliary_chemistry_v1 import BASIS, MODES, NEURAL, _auxiliary, _mode
from verify_doc_chemistry_decoder_v1 import _head
from verify_doc_chemistry_support_v1 import (
    COMPONENTS,
    PIPELINES,
    VARIANTS,
    _add_selected,
    _basis,
    _candidate,
    _fit_source_adapters,
    _panels,
    _selection,
    _summary,
)
from verify_doc_daily_hydro_support_basis_v1 import _manual_base, _same, _sidecar
from verify_doc_ecological_transfer_v1 import _scales_and_donors, _selection_check
from verify_doc_encoder_residual_v1 import _encoder_scope, _raw_cache
from verify_doc_regime_residual_v1 import _feature_views

from river_graph.experiments.provenance import sha256_file
from river_graph.experiments.unified_spatial_protocol import (
    build_unified_spatial_split,
    support_query_cells,
)
from river_graph.models.daily_hydro_tree import build_daily_tree_features
from river_graph.models.ecological_residual_transfer import EcologicalResidualTransfer
from river_graph.models.encoder_native_residual import EncoderNativeResidual
from river_graph.models.episodic_station_adapter import EpisodicStationProjector
from river_graph.models.episodic_temporal_adapter import EpisodicTemporalAdapter
from river_graph.models.kgml_local_transport import FIT_ROLES, build_rf_features
from river_graph.models.native_temporal_residual import NativeTemporalResidual
from river_graph.models.raw_temporal_features import extract_raw_temporal_inputs
from river_graph.models.support_aware_residual_transfer import (
    SupportAwareResidualTransfer,
    SupportAwareTransferEpisode,
)
from river_graph.models.support_shape_adapter import (
    SupportShapeAdapter,
    SupportShapeEpisode,
)
from river_graph.models.unified_doc import UnifiedDOCReconstructor

ROOT = Path("experiments/phase4_transfer/doc_chemistry_confirmation_v1")
SPLITS, SEEDS, KS = (242, 243, 244), (42, 43, 44), (0, 1, 3, 5)
BASES = ("context", "point", "tree_prior", *NEURAL, "tree_chemistry")
INTEGRATED = ("point", *NEURAL)
CONTROLS = ("context", "point", "tree_prior", "neural_no_aux", "neural_no_aux_integrated",
            "neural_masks", "neural_masks_integrated")
MODELS = (tuple(f"{pipe}_{basis}" for pipe in PIPELINES.values() for basis in (*VARIANTS, "selected"))
          + ("point_integrated_legacy",) + tuple(f"{name}_legacy" for name in CONTROLS))
INDICES = (0, 2, 4, 28, 30, 31, 32)


def _json(path):
    return json.loads(Path(path).read_text())


def _npz(path):
    with np.load(path, allow_pickle=False) as saved:
        return {key: saved[key].copy() for key in saved.files}


def _close_tree(actual, expected):
    np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=1e-12)
    return float(np.max(np.abs(np.asarray(actual)-np.asarray(expected))))


def _native_prediction(base, delta, scale):
    return base.copy() if scale == 0 else np.maximum(0, base + scale * delta)


def _trace_selection(summary):
    """Native epoch selection uses strict MAE improvement, then lower scale."""
    best, epoch, scale, stale = np.inf, None, None, 0
    for row in summary["trace"]:
        candidate = min(row["validation_candidates"], key=lambda x: (x["mae"], x["scale"]))
        improved = candidate["mae"] < best
        stale = 0 if improved else stale + 1
        if improved:
            best, epoch, scale = candidate["mae"], row["epoch"], candidate["scale"]
        if (row["is_best"] != improved or row["stale_epochs"] != stale
                or row["validation_mae"] != candidate["mae"] or row["validation_scale"] != candidate["scale"]):
            raise ValueError("Native checkpoint/scale selection differs from its trace")
    if summary["best_epoch"] != epoch or summary["selected_scale"] != scale:
        raise ValueError("Native selected state differs from the validation winner")
    return {"best_epoch": epoch, "selected_scale": scale, "epochs_run": summary["epochs_run"]}


def _selected_native_validation(model, inputs, cells, base, truth):
    prepared = model._prepare_inputs(inputs)
    with torch.inference_mode():
        delta = torch.cat([model._delta_cells(prepared, cells[start:start+512])
                           for start in range(0, len(cells), 512)])
        b, y = torch.as_tensor(base, dtype=torch.float64), torch.as_tensor(truth, dtype=torch.float64)
        scores = [{"scale": scale, "mae": float((model._combine(b, delta, scale)-y).abs().mean())}
                  for scale in model.scales]
    saved = model.to_dict()["validation_metrics"]["validation_candidates"]
    if scores != saved:
        raise ValueError("Selected native checkpoint does not reproduce validation scale scores")
    return {"validation_candidates": scores, "selected_checkpoint_validation_exact": True}


def _observed_vectors(model, inputs, cells):
    """Reconstruct hidden-major scalar-head vectors with original512 row batches."""
    prepared = model._prepare_inputs(inputs)
    months = prepared["age"].shape[1]
    total = prepared["age"].numel()
    selected = np.zeros(total, dtype=bool)
    selected[cells] = True
    rows, features, residuals = [], [], []
    with torch.inference_mode():
        for start in range(0, total, 512):
            stop = min(start + 512, total)
            if not selected[start:stop].any():
                continue
            index = torch.arange(start, stop)
            hidden = model._hidden_cells(prepared, index)
            extra = prepared["extra"][index // months, index % months]
            interaction = (hidden[:, :, None] * extra[:, INDICES][:, None, :]).flatten(1)
            vectors = torch.cat([hidden, extra, interaction], dim=1)
            delta = torch.nn.functional.linear(vectors, model.head.weight, model.head.bias).squeeze(-1).double()
            chosen = selected[start:stop]
            rows.append(index.numpy()[chosen])
            features.append(vectors.numpy()[chosen].copy())
            residuals.append(delta.numpy()[chosen].copy())
    np.testing.assert_array_equal(np.concatenate(rows), np.sort(cells))
    return np.concatenate(features), np.concatenate(residuals)


def _full_vectors(model, inputs, output):
    prepared = model._prepare_inputs(inputs)
    months = prepared["age"].shape[1]
    with torch.inference_mode():
        for start in range(0, len(output), 512):
            cells = torch.arange(start, min(start + 512, len(output)))
            hidden = model._hidden_cells(prepared, cells)
            extra = prepared["extra"][cells // months, cells % months]
            interaction = (hidden[:, :, None] * extra[:, INDICES][:, None, :]).flatten(1)
            output[start:start+len(cells)] = torch.cat([hidden, extra, interaction], dim=1).numpy()
    output.flush()


def _legacy_choices(bases, memory, basis, labels, split, months, active):
    """Independently build full/active validation episodes and repeat selection."""
    context = bases["context"]
    full_tasks, active_tasks = [], []
    for k in KS:
        support, query = support_query_cells(split, target_role="val", k=k, n_months=months)
        full_tasks.append((k, support, query))
        selected = query[active[query]]
        active_tasks.append((k, support[np.isin(support // months, np.unique(selected // months))], selected))
    adapters, mixers = {}, {}
    for name, base in bases.items():
        tasks = full_tasks if name in ("context", "point", "tree_prior") else active_tasks
        episodes = [SupportShapeEpisode(k, q, labels[q], base[q], s, labels[s], base[s], basis[q], basis[s])
                    for k, s, q in tasks]
        adapters[f"{name}_{BASIS}"] = SupportShapeAdapter(n_months=months).fit(
            episodes, selection_role="source_validation")
    for name in INTEGRATED:
        tasks = full_tasks if name == "point" else active_tasks
        q = tasks[0][2]
        gamma = min((float(np.abs(_manual_base(context[q], bases[name][q], memory[q], g)-labels[q]).mean()), g)
                    for g in (0., .25, .5, 1.))[1]
        episodes = [SupportAwareTransferEpisode(k, q, labels[q], s, labels[s], context[q], bases[name][q], memory[q],
            context[s], bases[name][s], memory[s], basis[q], basis[s]) for k, s, q in tasks]
        mixers[f"{name}_integrated_{BASIS}"] = SupportAwareResidualTransfer(months).fit(
            episodes, gamma_k0=gamma, selection_role="source_validation")
    return adapters, mixers


def _control_panels(bases, memory, basis, adapters, mixers, labels, split, months, active, role):
    panels = {}
    context = bases["context"]
    for k in KS:
        support, query = support_query_cells(split, target_role=role, k=k, n_months=months)
        for name in (*BASES, *[f"{name}_integrated" for name in INTEGRATED]):
            if name.endswith("_integrated"):
                arm = name.removesuffix("_integrated")
                panel = _candidate("integrated", basis, k, support, query, {"neural": bases[arm]}, context,
                    memory, labels, mixers[f"{name}_{BASIS}"])
            else:
                panel = _candidate(name, basis, k, support, query, {name: bases[name]}, context,
                    memory, labels, adapters[f"{name}_{BASIS}"])
            panels[(name, k)] = {"cell": query, **panel, "aux_fallback": np.zeros(len(query), dtype=bool),
                                "basis_name": np.full(len(query), "legacy"),
                                "selected_basis": np.full(len(query), "legacy")}
        absent = ~active[query]
        for name in (*NEURAL, "tree_chemistry"):
            for suffix in (("", "_integrated") if name in NEURAL else ("",)):
                panel = panels[(name + suffix, k)]
                parent = panels[(("tree_prior" if name == "tree_chemistry" else "point") + suffix, k)]
                for column in COMPONENTS:
                    panel[column][absent] = parent[column][absent]
                panel["aux_fallback"][absent] = True
    return panels


def _add_controls(panels, controls):
    result = panels.copy()
    for name in CONTROLS:
        for k in KS:
            result[(f"{name}_legacy", k)] = controls[(name, k)]
    if {name for name, _ in result} != set(MODELS):
        raise ValueError("Confirmation model panel differs")
    return result


def _native_replay(run, dataset, split, full):
    """Restore fresh experts/basis, then replay source and full native views."""
    shape, months = tuple(dataset["y"].shape), dataset["y"].shape[1]
    expert = UnifiedDOCReconstructor.load(run / "initial", dataset, split)
    original = pd.read_parquet(run / "initial/full_grid.parquet")
    context = original.context_pred.to_numpy()
    features = build_rf_features(dataset, split, FIT_ROLES, target_transform="log1p", include_network=True)
    initial_prediction = np.maximum(0, np.expm1(expert.context_forest.predict(features)))
    forest_error = _close_tree(initial_prediction, context)
    del features
    initial_components = expert.predict_components()
    for name, values in initial_components.items():
        _close_tree(values.ravel(), original[name].to_numpy())
    _same(full.context_pred, context)
    oof = _npz(run / "source_oof/source_oof.npz")["pred_z"]
    train = np.sort(np.asarray(split["train"]))
    source_ids = np.unique(train // months)
    oof_mask = np.zeros(shape, dtype=bool)
    oof_mask.ravel()[train] = True
    if not np.isfinite(oof[oof_mask]).all() or not np.isnan(oof[~oof_mask]).all():
        raise ValueError("Context OOF file must cover source training cells only")
    folds = _json(run / "source_oof/folds.json")
    for row in folds:
        held, fitting = set(row["held_station_ids"]), set(row["training_station_ids"])
        if held & fitting or held | fitting != set(source_ids):
            raise ValueError("Source context OOF roles are not station disjoint")
    encoded = _npz(run / "basis/inputs/temporal_inputs.npz")
    np.testing.assert_array_equal(encoded["source_station_ids"], source_ids)
    projector = EpisodicStationProjector.from_dict(_json(run / "basis/projector/projector.json"))
    pca = projector.basis_
    weight = pca.components_.T / np.sqrt(np.maximum(pca.eigenvalues_, pca.eigenvalue_floor))
    weight[:, ~projector.active_modes_] = 0
    readout = weight @ projector.projection_.T
    _same(readout, _npz(run / "basis/projector/readout.npz")["readout"])
    memory_payload = torch.load(run / "basis/memory/memory.pt", weights_only=False)
    basis_model = EpisodicTemporalAdapter.from_payload(memory_payload)
    _same(basis_model.readout.numpy(), readout)
    if basis_model.to_dict() != _json(run / "basis/memory/memory.json"):
        raise ValueError("Legacy temporal basis state differs from saved summary")
    temporal_inputs = {key: encoded[f"full_{key}"] for key in ("encoded", "age", "support")}
    legacy = basis_model.transform(temporal_inputs).reshape(-1, 2)
    _same(legacy, _npz(run / "basis/memory/representations.npz")[BASIS])
    extra, extra_checks = _feature_views(dataset, split, context, oof)
    full_inputs, raw_checks = _raw_cache(expert, dataset, split, extra)
    raw = extract_raw_temporal_inputs(expert, dataset, split)
    source_inputs = {key: raw[f"source_{key}"] for key in ("raw", "age", "support", "env")}
    config = _json(run / "config.json")
    daily = _npz(config["daily_features_path"])["full"]
    source_inputs["extra"] = np.concatenate([extra["source_extra"], daily[source_ids]], axis=-1)
    full_inputs["extra"] = np.concatenate([extra["full_extra"], daily], axis=-1)
    compact = _npz(run / "native/source_views.npz")
    native = _npz(run / "native/native_components.npz")
    _same(native["context"], context)
    interaction_payload = torch.load(run / "native/interaction_tuned.pt", weights_only=False)
    interaction = NativeTemporalResidual.from_payload(interaction_payload)
    off_payload = torch.load(run / "native/off.pt", weights_only=False)
    off = EncoderNativeResidual.from_payload(off_payload)
    if (interaction.to_dict() != _json(run / "native/interaction_tuned.json")
            or off.to_dict() != _json(run / "native/off.json")):
        raise ValueError("Native checkpoint summaries do not roundtrip")
    expected_shared = {"lookback": 12, "patience": 5, "batch_size": 512, "seed": config["seed"],
        "learning_rate": 1e-4, "head_learning_rate": 1e-3, "tail_weight": 2., "scales": [0., .25, .5, 1.]}
    expected_interaction = {**expected_shared, "epochs": 1 if config["smoke"] else 30,
                           "extra_dim": 10, "interaction_indices": [0, 2, 4]}
    expected_off = {**expected_shared, "epochs": 1 if config["smoke"] else 120,
        "extra_dim": 38, "interaction_indices": list(INDICES), "encoder_mode": "last_self_ecology",
        "encoder_learning_rate": 1e-5}
    if interaction.to_dict()["config"] != expected_interaction or off.to_dict()["config"] != expected_off:
        raise ValueError("Retained interaction/native settings differ")
    scope = _encoder_scope(off, off_payload, "last_self_ecology")
    for payload in (interaction_payload, off_payload, memory_payload):
        for name, module in (("temporal", expert.residual.model.temporal), ("decay", expert.residual.model.decay)):
            for key, value in module.state_dict().items():
                torch.testing.assert_close(payload[f"initial_{name}"][key], value, rtol=0, atol=0)
    for key, value in expert.residual.model.spatial.state_dict().items():
        torch.testing.assert_close(off_payload["initial_spatial"][key], value, rtol=0, atol=0)
    initializer_inputs = {**temporal_inputs, "extra": extra["full_extra"][..., :10]}
    interaction_delta = interaction.predict_delta(initializer_inputs).ravel()
    _same(interaction_delta, native["interaction_delta"])
    _same(_native_prediction(context, interaction_delta, interaction.selected_scale_), native["interaction_tuned"])
    delta = off.predict_delta(full_inputs).ravel()
    _same(delta, native["point_delta"])
    _same(_native_prediction(context, delta, off.selected_scale_), native["point"])
    _same(native["point"], full.point_pred)
    source_cells = np.flatnonzero(oof_mask[source_ids])
    global_cells = source_ids[source_cells // months] * months + source_cells % months
    _same(global_cells, train)
    vectors, source_delta = _observed_vectors(off, source_inputs, source_cells)
    _same(vectors, compact["source_features"])
    _same(source_delta, compact["source_delta"])
    source_oof = np.maximum(0, np.expm1(oof.ravel()[train]))
    source_base = _native_prediction(source_oof, source_delta, off.selected_scale_)
    _same(source_base, compact["source_base"])
    _same(source_oof, compact["source_oof_native"])
    _, val_cells = support_query_cells(split, target_role="val", k=0, n_months=months)
    val_ids = np.unique(val_cells // months)
    local_val = np.searchsorted(val_ids, val_cells // months) * months + val_cells % months
    val_native_inputs = {key: values[val_ids] for key, values in full_inputs.items()}
    val_interaction_inputs = {key: values[val_ids] for key, values in initializer_inputs.items()}
    native_val_checks = {name: _selected_native_validation(model, view, local_val, context[val_cells],
                         np.asarray(dataset["y"], dtype=np.float64).ravel()[val_cells])
        for name, model, view in (("off", off, val_native_inputs),
                                  ("interaction_tuned", interaction, val_interaction_inputs))}
    val_vectors, _ = _observed_vectors(off, full_inputs, val_cells)
    _same(val_vectors, compact["validation_features"])
    for name, value in (("source_cells", train), ("source_local_cells", source_cells),
                        ("source_station_ids", source_ids), ("validation_cells", val_cells),
                        ("validation_base", native["point"][val_cells])):
        _same(compact[name], value)
    state = _json(run / "native/ecological_affine.json")
    profile = EcologicalResidualTransfer.from_dict(state)
    if profile.to_dict() != state:
        raise ValueError("Ecological profile state failed roundtrip")
    donor_check = _scales_and_donors(state, dataset["regime"], train, source_oof,
                                    np.asarray(dataset["y"]).ravel()[train])
    _selection_check(state, np.asarray(dataset["y"]).ravel()[val_cells], native["interaction_tuned"][val_cells])
    memory = profile.predict_delta(context.reshape(shape)).ravel()
    normalization, theta = state["normalization"], np.asarray(state["coefficients"])
    u = (np.log1p(context.reshape(shape))-normalization["log_context_mean"])/normalization["log_context_sd"]
    independent_memory = normalization["residual_scale"]*(theta[:, :1]+theta[:, 1:]*u)
    _same(memory, independent_memory.ravel())
    _same(memory, native["memory"])
    _same(memory, full.ecological_memory)
    # Perturb hidden values inside the already-fitted input cache, with no refit.
    y_model = expert.residual.inputs.y_model
    hidden = np.r_[split["val"], split["test"]]
    original_y = y_model.clone()
    try:
        y_model.reshape(-1)[hidden] += 10000
        alternate = extract_raw_temporal_inputs(expert, dataset, split)
        for key in raw:
            np.testing.assert_array_equal(raw[key], alternate[key])
    finally:
        y_model.copy_(original_y)
    checks = {"initial_context_forest_max_difference": forest_error,
        "initial_and_native_initialization_fresh_and_exact": True, "raw_input_checks": raw_checks,
        "native_source_and_full_replay_exact": True, "source_feature_dimension": 550,
        "source_extra_checks": extra_checks, "encoder_scope": scope,
        "ecological_donors": donor_check, "profile_selection_initializer": "interaction_tuned",
        "legacy_basis_replay_exact": True, "hidden_label_perturbation_raw_inputs_exact": True,
        "native_selected_validation": native_val_checks,
        "interaction_selection": _trace_selection(interaction.to_dict()),
        "off_selection": _trace_selection(off.to_dict()),
        "source_neural_is_oof": False, "source_forest_is_oof": True}
    return {"expert": expert, "off": off, "full_inputs": full_inputs, "legacy": legacy,
            "source_ids": source_ids, "source_cells": train, "val_cells": val_cells,
            "source_features": vectors, "val_features": val_vectors, "source_base": source_base,
            "daily": daily, "checks": checks}


def verify_one(root, partition, seed, runtime):
    run = root / "runs" / f"split{partition}_seed{seed}"
    config = _json(run / "config.json")
    expected = {"experiment": "doc_chemistry_confirmation_v1", "split_seed": partition, "seed": seed,
        "runtime_snapshot_hash": runtime, "models": list(MODELS), "k_values": list(KS),
        "inference_roles": ["train"], "selection_role": "source_validation",
        "historical_fitted_models_reused": False, "source_neural_is_oof": False, "source_forest_is_oof": True}
    if any(config[key] != value for key, value in expected.items()):
        raise ValueError("Fresh confirmation scope differs")
    torch.set_num_threads(config["torch_threads"])
    completion = verify_files(run, "complete.json", config)
    for name in ("dataset", "mask", "protocol", "daily_features", "daily_metadata"):
        if sha256_file(config[f"{name}_path"]) != config[f"{name}_hash"]:
            raise ValueError(f"Changed {name} input")
    subtrees = ("initial", "source_oof", "basis", "native", "chemical")
    model_files = {str(path.relative_to(run)) for directory in subtrees
                   for path in (run / directory).rglob("*") if path.is_file()}
    for stage in ("initial", "source_oof", "basis/inputs", "basis/projector", "basis/memory", "native"):
        stage_config = _json(run / stage / "config.json")
        verify_files(run / stage, "complete.json", stage_config)
    dataset = torch.load(config["dataset_path"], weights_only=False)
    split = _npz(config["mask_path"])
    rebuilt, protocol = build_unified_spatial_split(np.asarray(dataset["y_mask"]), seed=partition)
    for role in rebuilt:
        np.testing.assert_array_equal(split[role], rebuilt[role])
    if protocol != _json(config["protocol_path"]):
        raise ValueError("Station role assignment differs from its label-free seed")
    full, queries = pd.read_parquet(run / "full_grid.parquet"), pd.read_parquet(run / "predictions.parquet")
    for filename, frame in (("full_grid.parquet", full), ("predictions.parquet", queries)):
        _sidecar(run, filename, config, runtime, frame, completion, model_files)
    shape, months = tuple(dataset["y"].shape), dataset["y"].shape[1]
    if config["q90_threshold_train"] != float(np.quantile(np.asarray(dataset["y"]).ravel()[split["train"]], .9)):
        raise ValueError("Tail threshold differs from the fresh source population")
    total = int(np.prod(shape))
    _same(full.cell, np.arange(total))
    if "y_true" in full:
        raise ValueError("Full-grid components must not contain hidden DOC truth")
    for table in (full, queries):
        cells = table.cell.to_numpy(dtype=int)
        np.testing.assert_array_equal(table.station.astype(str), np.asarray(dataset["site_no"], str)[cells // months])
        np.testing.assert_array_equal(table.month.astype(str), np.asarray(dataset["months"], str)[cells % months])
        if not np.isfinite(table.select_dtypes(include="number")).all().all():
            raise ValueError("Nonfinite confirmation product")
    if queries.duplicated(["model_name", "k", "cell"]).any():
        raise ValueError("Repeated confirmation queries")
    stage = _json(run / "chemical/stage.json")
    auxiliary, active = _auxiliary({**config, **stage}, dataset)
    for key, values in {"ph_available": auxiliary[:, 2].astype(bool), "ec_available": auxiliary[:, 3].astype(bool),
                        "aux_available": active, "doc_observed": np.asarray(dataset["y_mask"]).ravel().astype(bool)}.items():
        np.testing.assert_array_equal(full[key], values)
        np.testing.assert_array_equal(queries[key], values[queries.cell.to_numpy()])
    print(f"{run.name}: source/native replay", flush=True)
    replay = _native_replay(run, dataset, split, full)
    truth = np.asarray(dataset["y"], dtype=np.float64).ravel()
    source, val = replay["source_cells"], replay["val_cells"]
    definitions = _json(run / "chemical/input_definition.json")
    for name, values in (("source_original_feature_hash", replay["source_features"]),
                         ("validation_original_feature_hash", replay["val_features"])):
        from verify_doc_distribution_head_v1 import _array_digest
        if definitions[name] != _array_digest(values):
            raise ValueError("Source/validation head vectors differ from training identities")
    head_checks = []
    cfg = {"epochs": 1 if config["smoke"] else 120, "patience": 10, "batch_size": 512,
           "seed": seed, "learning_rate": .001, "correction_scales": [0, .25, .5, 1],
           "q90_threshold_train": config["q90_threshold_train"]}
    print(f"{run.name}: nonlinear and tree replay", flush=True)
    with tempfile.TemporaryDirectory(prefix="doc-confirmation-replay-") as temporary:
        features = np.lib.format.open_memmap(Path(temporary) / "features.npy", mode="w+", dtype=np.float32, shape=(total, 550))
        _full_vectors(replay["off"], replay["full_inputs"], features)
        for mode in MODES:
            block = _mode(auxiliary, mode)
            sx = np.concatenate([replay["source_features"], block[source]], axis=1)
            vx = np.concatenate([replay["val_features"], block[val]], axis=1)
            _, check = _head(run / "chemical", mode, cfg, sx, vx, replay["source_base"], full.point_pred.to_numpy()[val],
                truth[source], truth[val], active[source], active[val], features, block,
                full.point_pred.to_numpy(), full, active)
            head_checks.append(check)
        del features
    matrix = build_daily_tree_features(dataset, split, replay["daily"], mode="current")
    tree_checks = []
    for name, inputs in (("current", matrix), ("chemistry", np.concatenate([matrix, auxiliary], axis=1))):
        forest = joblib.load(run / "chemical" / f"tree_{name}.joblib")
        expected_params = replay["expert"].context_forest.get_params(deep=False)
        expected_params["n_jobs"] = 2
        if forest.get_params(deep=False) != expected_params:
            raise ValueError("Confirmation tree did not retain selected context parameters")
        predicted = np.maximum(0, np.expm1(forest.predict(inputs)))
        if name == "chemistry":
            predicted[~active] = full.tree_prior_pred.to_numpy()[~active]
        column = "tree_prior_pred" if name == "current" else "tree_chemistry_pred"
        error = _close_tree(predicted, full[column])
        tree_checks.append({"arm": name, "feature_dim": inputs.shape[1], "max_abs_difference": error,
                            "rtol": 1e-12, "atol": 1e-12})
    del matrix
    payload = torch.load(run / "chemical/neural_chemistry.pt", weights_only=True)
    basis_states = _json(run / "chemical/basis_definition.json")
    shapes, archive, basis_checks = {"legacy": replay["legacy"]}, {
        "legacy": replay["legacy"], "active": active, "source_station_ids": replay["source_ids"]}, []
    for mode, name in (("masks", "masks_aug"), ("chemistry", "chemistry_aug")):
        chemical, augmented, check = _basis(basis_states[name], auxiliary.reshape(*shape, 4), replay["source_ids"], payload, replay["legacy"])
        shapes[name] = augmented.reshape(total, 4)
        archive[name], archive[f"{mode}_chemical"] = shapes[name], chemical.reshape(total, 2)
        basis_checks.append(check)
    saved_basis = _npz(run / "chemical/representations.npz")
    if set(saved_basis) != set(archive):
        raise ValueError("Chemical representation archive differs")
    for key, values in archive.items():
        np.testing.assert_array_equal(values, saved_basis[key])
    bases = {name: full[f"{name}_pred"].to_numpy() for name in BASES}
    memory = full.ecological_memory.to_numpy()
    labels = np.full(total, np.nan)
    labels[split["val"]] = truth[split["val"]]
    old_adapters, old_mixers = _legacy_choices(bases, memory, shapes["legacy"], labels, split, months, active)
    adapter_states, mixer_states = ({name: model.to_dict() for name, model in items.items()}
                                   for items in (old_adapters, old_mixers))
    if (adapter_states != _json(run / "chemical/legacy_adapters.json")
            or mixer_states != _json(run / "chemical/legacy_mixers.json")):
        raise ValueError("Fresh legacy source-validation calibration replay differs")
    gamma = mixer_states[f"neural_chemistry_integrated_{BASIS}"]["gamma_k0"]
    if stage["gamma_k0"] != gamma:
        raise ValueError("K0 ecology mixture changed")
    reduced = {"neural": bases["neural_chemistry"], "tree": bases["tree_chemistry"]}
    adapters, mixers = _fit_source_adapters(reduced, shapes, bases["context"], memory, labels, split, months, active, gamma)
    a_state = {f"{PIPELINES[pipe]}_{name}": model.to_dict() for (pipe, name), model in adapters.items()}
    m_state = {f"neural_chemistry_integrated_{name}": model.to_dict() for name, model in mixers.items()}
    if a_state != _json(run / "chemical/adapters.json") or m_state != _json(run / "chemical/mixers.json"):
        raise ValueError("Chemical coordinate calibration selection differs")
    def panels(role_labels, role):
        support_panels = _panels(full, reduced, shapes, adapters, mixers, adapter_states, mixer_states,
                                role_labels, split, months, active, role)
        controls = _control_panels(bases, memory, shapes["legacy"], old_adapters, old_mixers,
                                   role_labels, split, months, active, role)
        return support_panels, controls
    validation, controls = panels(labels, "val")
    selected = _selection(_summary(validation, labels, active))
    if selected != _json(run / "chemical/basis_selection.json"):
        raise ValueError("Selected representation is not the validation-only winner")
    validation = _add_controls(_add_selected(validation, selected), controls)
    summary = _summary(validation, labels, active)
    for path in (run / "source_validation.csv", run / "chemical/source_validation.csv"):
        pd.testing.assert_frame_equal(summary, pd.read_csv(path), check_dtype=False, check_exact=False, rtol=1e-14, atol=1e-14)
    support, query = support_query_cells(split, target_role="test", k=5, n_months=months)
    labels = np.full(total, np.nan)
    labels[support] = truth[support]
    target, controls = panels(labels, "test")
    target = _add_controls(_add_selected(target, selected), controls)
    checks = []
    for (name, k), panel in target.items():
        order = np.argsort(panel["cell"])
        stored = queries[queries.model_name.eq(name) & queries.k.eq(k)].sort_values("cell")
        np.testing.assert_array_equal(stored.cell, np.sort(query))
        for key, values in panel.items():
            np.testing.assert_array_equal(stored[key], values[order])
        _same(stored.y_true, truth[stored.cell.to_numpy()])
        if k == 0:
            _same(stored.y_pred, stored.base_pred)
        checks.append({"model_name": name, "k": k, "n_query": len(stored), "bitwise_exact": True,
                       "max_abs_difference": 0., "inactive_fallback_exact": True})
    if len(queries) != len(query) * len(MODELS) * len(KS):
        raise ValueError("Unexpected confirmation panel row count")
    invariant = 0
    for pipe in PIPELINES.values():
        for k in (0, 1):
            parent = target[(f"{pipe}_legacy", k)]
            for variant in ("masks_aug", "chemistry_aug", "selected"):
                for key in COMPONENTS:
                    _same(parent[key], target[(f"{pipe}_{variant}", k)][key])
                invariant += 1
    return {"run": run.name, "status": "verified", "completion_sha256": sha256_file(run / "complete.json"),
        "runtime_snapshot_hash": runtime, "full_grid_rows": len(full), "native_checks": replay["checks"],
        "head_replays": head_checks, "tree_replays": tree_checks, "basis_checks": basis_checks,
        "query_replays": checks, "source_validation_rows": len(summary), "low_k_invariant_panels": invariant,
        "validation_adapter_refits": len(old_adapters) + len(adapters),
        "validation_mixer_refits": len(old_mixers) + len(mixers), "basis_choices_replayed": 12,
        "neural_training_performed": False, "forest_training_performed": False,
        "ecological_optimizer_refitted": False, "scope": "saved-state numerical replay; not training-trajectory reproduction"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--split-seeds", type=int, nargs="+", default=list(SPLITS))
    parser.add_argument("--seeds", type=int, nargs="+", default=list(SEEDS))
    args = parser.parse_args()
    runtime = verify_runtime_snapshot(args.root)
    snapshot = _json(args.root / "runtime_snapshot.json")
    for name, expected in snapshot.items():
        if name.endswith(".py") and sha256_file(name) != expected:
            raise ValueError(f"Live replay source differs from archived execution source: {name}")
    results = []
    for partition in args.split_seeds:
        for seed in args.seeds:
            results.append(verify_one(args.root, partition, seed, runtime))
            print(f"split{partition}_seed{seed}: verified", flush=True)
            gc.collect()
    record = {"generated_at": datetime.now(timezone.utc).isoformat(), "status": "verified",
        "verifier": str(Path(__file__).resolve().relative_to(Path.cwd())), "verifier_sha256": sha256_file(__file__),
        "runtime_snapshot_hash": runtime, "results": results}
    write_json(args.root / "verification/replay_checks.json", record)


if __name__ == "__main__":
    main()
