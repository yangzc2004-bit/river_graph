"""Fresh chemical decoders, explicit trees and station calibration for DOC.

Called by the confirmation driver after its fresh initial/native stages. This
helper never loads an earlier fitted model. Only aligned covariate datasets
are reused. Test query truths are deliberately absent from returned products.
"""
from __future__ import annotations

import json
import time
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
from doc_chemistry_confirmation_initial import _content_hash
from run_doc_auxiliary_chemistry_v1 import (
    AUX_PATHS,
    BASIS,
    INTEGRATED,
    MODES,
    NEURAL,
    direct_frame,
    validation_tasks,
)
from run_doc_chemistry_support_v1 import (
    MODELS as SUPPORT_MODELS,
)
from run_doc_chemistry_support_v1 import (
    add_selected,
    choose_representations,
    fit_calibrators,
    invariance_checks,
    make_panels,
    validation_summary,
)
from run_doc_daily_hydro_readout_v1 import array_digest
from run_doc_daily_hydro_support_basis_v1 import fit_arm_mixers, integrated_frame
from run_doc_distribution_head_v1 import head_feature_blocks, observed_features
from run_unified_doc_spatial import (
    bind_files,
    covariate_diagnostics,
    digest,
    verify_files,
    write_json,
)
from run_unified_doc_spatial_v2 import fit_adapters
from sklearn.base import clone

from river_graph.experiments.provenance import runtime_code_snapshot, sha256_file
from river_graph.experiments.unified_spatial_protocol import support_query_cells
from river_graph.models.auxiliary_chemistry_features import (
    apply_auxiliary_mode,
    build_auxiliary_chemistry_features,
)
from river_graph.models.chemical_support_basis import ChemicalSupportBasis
from river_graph.models.daily_hydro_tree import (
    build_daily_tree_features,
    daily_tree_feature_names,
)
from river_graph.models.nonlinear_chemistry_head import NonlinearChemistryHead
from river_graph.models.support_aware_residual_transfer import (
    SupportAwareResidualTransfer,
    SupportAwareTransferEpisode,
)
from river_graph.models.support_shape_adapter import (
    SupportShapeAdapter,
    SupportShapeEpisode,
)

KS = (0, 1, 3, 5)
REFERENCES = ("point", "context", "tree_prior")
NEW_DIRECT = (*NEURAL, "tree_chemistry")
RAW_CONTROLS = ("context", "point", "tree_prior", "neural_no_aux", "neural_no_aux_integrated",
                "neural_masks", "neural_masks_integrated")
MODELS = (*SUPPORT_MODELS, *(f"{name}_legacy" for name in RAW_CONTROLS))
EXPECTED_FILES = {"config.json", "stage.json", "full_grid.parquet", "predictions.parquet",
    "source_validation.csv", "basis_selection.json", "source_training.npz", "input_definition.json",
    "tree_current.joblib", "tree_chemistry.joblib", "trees.json", "legacy_adapters.json",
    "legacy_mixers.json", "representations.npz", "basis_definition.json", "adapters.json", "mixers.json",
    "reference_checks.json", *(f"neural_{mode}{suffix}" for mode in MODES
                               for suffix in (".pt", ".json", "_trace.csv"))}


def retained_chemical_recipe(*, smoke=False):
    return {"modes": list(MODES), "models": list(MODELS), "k_values": list(KS),
        "epochs": 1 if smoke else 120, "patience": 10, "batch_size": 512, "learning_rate": .001,
        "tail_weight": 2, "correction_scales": [0, .25, .5, 1], "head_features": 554,
        "head_parameters": 1111, "tree_features": {"current": 147, "chemistry": 151},
        "tree_n_jobs": 2, "pca_eigenvalue_floor": 1e-8,
        "normalization_role": "source_training", "selection_role": "source_validation",
        "auxiliary_gate": "any actual auxiliary observation; same for all three heads",
        "final_fallback": "frozen fresh point/tree-current after support calibration and ecological mixing",
        "basis_tie_order": ["legacy", "masks_aug", "chemistry_aug"]}


def _fit_label_view(dataset, split):
    labels = np.full(np.prod(dataset["y"].shape), np.nan)
    for role in ("train", "val"):
        labels[split[role]] = np.asarray(dataset["y"]).ravel()[split[role]]
    if (np.intersect1d(split["train"], split["val"]).size
            or not np.isnan(labels[split["test"]]).all()):
        raise ValueError("Source, validation and target DOC roles must be disjoint")
    return labels


def _target_support_view(dataset, split):
    months = dataset["y"].shape[1]
    support, query = support_query_cells(split, target_role="test", k=5, n_months=months)
    labels = np.full(np.prod(dataset["y"].shape), np.nan)
    labels[support] = np.asarray(dataset["y"]).ravel()[support]
    if not np.isnan(labels[query]).all():
        raise ValueError("Target query labels entered the support view")
    return labels


def _chemical_config(run, dataset, split, seed, initial, native, *, smoke):
    dependencies = [*initial["stage_completion_files"], Path(native["native_dir"]) / "complete.json"]
    upstream = {}
    for completion in dependencies:
        completion = Path(completion)
        configuration = json.loads((completion.parent / "config.json").read_text())
        verify_files(completion.parent, completion.name, configuration)
        upstream[str(completion)] = sha256_file(completion)
    parent = json.loads((run / "config.json").read_text()) if (run / "config.json").exists() else None
    model = native["model"]
    # Whole-dataset hashing is opaque identity binding. Fitting still receives
    # only source/validation labels, and inference later receives support only.
    return {"stage": "fresh_chemistry_and_calibration", "seed": int(seed), "smoke": bool(smoke),
        "recipe": retained_chemical_recipe(smoke=smoke), "upstream": upstream,
        "dataset_content_hash": _content_hash(dataset), "split_content_hash": _content_hash(split),
        "driver_config_hash": digest(parent), "helper_hash": sha256_file(__file__),
        "library_snapshot_hash": digest(runtime_code_snapshot()), "torch_threads": torch.get_num_threads(),
        "auxiliary_hashes": {name: sha256_file(path) for name, path in AUX_PATHS.items()},
        "native_state_hash": _content_hash({name: getattr(model, name).state_dict()
                                           for name in ("spatial", "temporal", "decay", "head")}),
        "native_selected_scale": model.selected_scale_,
        "selected_context_parameters": initial["expert"].context_forest.get_params(deep=False),
        "input_hash": _content_hash({"context": initial["context"], "oof_z": initial["oof_z"],
            "legacy_basis": initial["legacy_basis"], "source_inputs": native["source_inputs"],
            "full_inputs": native["full_inputs"], "point": native["point"], "memory": native["memory"],
            "daily": native["daily"], "source_station_ids": native["source_station_ids"],
            "compact": {name: native[name] for name in ("source_features", "source_base", "source_cells",
                "source_local_cells", "validation_features", "validation_cells") if name in native}})}


def _load_chemical_result(output):
    return {"full_grid": pd.read_parquet(output / "full_grid.parquet"),
        "predictions": pd.read_parquet(output / "predictions.parquet"),
        "source_validation": pd.read_csv(output / "source_validation.csv"),
        "metadata": json.loads((output / "stage.json").read_text()),
        "selection": json.loads((output / "basis_selection.json").read_text()),
        "model_files": sorted(str(path.relative_to(output.parent)) for path in output.iterdir() if path.is_file())}


def _prepare_chemical_stage(output, config):
    """Verify a complete cache; preserve an incomplete stage before restarting."""
    if (output / "complete.json").exists():
        if json.loads((output / "config.json").read_text()) != config:
            raise ValueError("Completed chemical stage inputs or recipe changed")
        complete = verify_files(output, "complete.json", config)
        if set(complete["files"]) != EXPECTED_FILES:
            raise ValueError("Completed chemical stage is missing required products")
        print(f"{output.parent.name}: reused verified chemical stage", flush=True)
        return _load_chemical_result(output)
    if output.exists() and any(output.iterdir()):
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
        archived = output.with_name(f"chemical_partial_{stamp}")
        output.rename(archived)
        print(f"{output.parent.name}: preserved incomplete chemical stage at {archived.name}", flush=True)
    output.mkdir(parents=True, exist_ok=True)
    write_json(output / "config.json", config)
    return None


def _finish_chemical_stage(output, config, full, predictions):
    full.to_parquet(output / "full_grid.parquet", index=False)
    predictions.to_parquet(output / "predictions.parquet", index=False)
    actual = {path.name for path in output.iterdir() if path.is_file()}
    if actual != EXPECTED_FILES:
        raise ValueError(f"Unexpected chemical-stage artifacts: {sorted(actual ^ EXPECTED_FILES)}")
    bind_files(output, "complete.json", [output / name for name in sorted(EXPECTED_FILES)], config)
    return _load_chemical_result(output)


def _augment(vectors, auxiliary):
    result = np.concatenate((vectors, np.asarray(auxiliary, dtype=np.float32)), axis=1)
    if result.shape != (len(vectors), 554) or not np.isfinite(result).all():
        raise ValueError("Expected source-standardized native550 plus auxiliary4 layout")
    return result


def _full_grid(dataset, split, context, point, memory, auxiliary, active):
    n, months = dataset["y"].shape
    cells = np.arange(n * months)
    source_y = np.zeros((n, months), dtype=np.asarray(dataset["y"]).dtype)
    source_y.ravel()[split["train"]] = np.asarray(dataset["y"]).ravel()[split["train"]]
    novelty, upstream = covariate_diagnostics({**dataset, "y": source_y}, split)
    roles = np.full(n * months, "unobserved", dtype=object)
    for role, selected in split.items():
        roles[selected] = role
    return pd.DataFrame({"cell": cells,
        "station": np.asarray(dataset["site_no"], str)[cells // months],
        "month": np.asarray(dataset["months"], str)[cells % months],
        "analyte": "doc", "visibility_role": roles, "input_roles": "train",
        "ecological_novelty": novelty[cells // months], "upstream_support": upstream.ravel(),
        "context_pred": context, "point_pred": point, "ecological_memory": memory,
        "ph_available": auxiliary[:, 2].astype(bool), "ec_available": auxiliary[:, 3].astype(bool),
        "aux_available": active, "doc_observed": np.asarray(dataset["y_mask"]).ravel().astype(bool)})


def fit_legacy_choices(bases, context, memory, legacy, labels, split, months, active):
    """The executed decoder calibration recipe, omitting unneeded tree arms."""
    shapes = {BASIS: legacy}
    adapters = fit_adapters({name: bases[name] for name in REFERENCES}, shapes, labels, split, months)
    mixers = {}
    for name in NEW_DIRECT:
        episodes = [SupportShapeEpisode(k, query, labels[query], bases[name][query], support,
            labels[support], bases[name][support], legacy[query], legacy[support])
            for k, support, query in validation_tasks(split, months, active)]
        adapters[f"{name}_{BASIS}"] = SupportShapeAdapter(n_months=months).fit(
            episodes, selection_role="source_validation")
    _, full_query = support_query_cells(split, target_role="val", k=0, n_months=months)
    for name in INTEGRATED:
        query = full_query if name == "point" else full_query[active[full_query]]
        scores = []
        for gamma in (0., .25, .5, 1.):
            predicted = bases[name][query] if gamma == 0 else np.maximum(0,
                context[query] + (1 - gamma) * (bases[name][query] - context[query]) + gamma * memory[query])
            scores.append((float(np.abs(predicted - labels[query]).mean()), gamma))
        _, gamma = min(scores)
        if name == "point":
            mixers.update(fit_arm_mixers(name, context, bases[name], memory, shapes,
                                        labels, split, months, gamma))
        else:
            episodes = [SupportAwareTransferEpisode(k, query, labels[query], support, labels[support],
                context[query], bases[name][query], memory[query], context[support], bases[name][support],
                memory[support], legacy[query], legacy[support])
                for k, support, query in validation_tasks(split, months, active)]
            mixers[f"{name}_integrated_{BASIS}"] = SupportAwareResidualTransfer(months).fit(
                episodes, gamma_k0=gamma, selection_role="source_validation")
    return adapters, mixers


def legacy_panels(full, bases, legacy, adapters, mixers, labels, split, months, active, *, role):
    context, memory = full.context_pred.to_numpy(), full.ecological_memory.to_numpy()
    frame = direct_frame(full, bases, legacy, adapters, labels, split, months, role)
    frame = pd.concat([frame, *[integrated_frame(full, arm, context, bases[arm], memory,
        {BASIS: legacy}, mixers, labels, split, months, role=role) for arm in INTEGRATED]], ignore_index=True)
    frame["candidate_y_pred"], frame["aux_fallback"] = frame.y_pred.copy(), False
    for name in NEW_DIRECT:
        stages = ("", "_integrated") if name in NEURAL else ("",)
        for stage in stages:
            parent = f"{'point' if name in NEURAL else 'tree_prior'}{stage}_{BASIS}"
            for k in KS:
                selected = frame.model_name.eq(f"{name}{stage}_{BASIS}") & frame.k.eq(k)
                selected &= ~active[frame.cell.to_numpy()]
                cells = frame.loc[selected, "cell"].to_numpy()
                fallback = frame[frame.model_name.eq(parent) & frame.k.eq(k)].set_index("cell").loc[cells]
                for column in ("y_pred", "base_pred", "adaptation_delta", "regional_gamma"):
                    frame.loc[selected, column] = fallback[column].to_numpy()
                frame.loc[selected, "aux_fallback"] = True
    for column in ("ph_available", "ec_available", "aux_available", "doc_observed"):
        frame[column] = full[column].to_numpy()[frame.cell.to_numpy()]
    return frame


def _join_controls(support_frame, legacy_frame):
    names = {f"{name}_{BASIS}": f"{name}_legacy" for name in RAW_CONTROLS}
    controls = legacy_frame[legacy_frame.model_name.isin(names)].copy()
    controls["model_name"] = controls.model_name.map(names)
    controls["basis_name"], controls["selected_basis"] = "legacy", "legacy"
    result = pd.concat([support_frame, controls], ignore_index=True)
    if set(result.model_name) != set(MODELS) or result.duplicated(["model_name", "k", "cell"]).any():
        raise ValueError("Incomplete or repeated confirmation curve identities")
    if not np.isfinite(result.select_dtypes(include="number")).all().all():
        raise ValueError("Nonfinite confirmation prediction product")
    return result


def fit_chemistry_and_calibration(run, dataset, split, seed, initial, native, *, smoke=False):
    """Fit only fresh chemistry states and return source-selected DOC products.

    ``initial`` provides expert/context/oof_z/legacy_basis. ``native`` provides
    model, source_inputs/full_inputs, source_station_ids, point, memory, daily
    and q90_threshold_train. Arrays are in this dataset's station-major order.
    Driver adds test truths and provenance sidecars after this helper returns.
    """
    started = time.monotonic()
    run = Path(run)
    output = run / "chemical"
    config = _chemical_config(run, dataset, split, seed, initial, native, smoke=smoke)
    restored = _prepare_chemical_stage(output, config)
    if restored is not None:
        return restored
    n, months = dataset["y"].shape
    total = n * months
    labels = _fit_label_view(dataset, split)
    source_ids = np.asarray(native["source_station_ids"], dtype=np.int64)
    np.testing.assert_array_equal(source_ids, np.unique(split["train"] // months))
    if np.intersect1d(source_ids, np.unique(np.r_[split["val"], split["test"]] // months)).size:
        raise ValueError("Fresh source and held station populations overlap")
    context = np.asarray(initial["context"]).reshape(-1)
    point, memory = np.asarray(native["point"]).reshape(-1), np.asarray(native["memory"]).reshape(-1)
    legacy = np.asarray(initial["legacy_basis"]).reshape(total, 2)
    auxiliary_data = {name: torch.load(path, weights_only=False) for name, path in AUX_PATHS.items()}
    chemistry = build_auxiliary_chemistry_features(dataset, auxiliary_data["ph"], auxiliary_data["ec"])
    auxiliary, active = chemistry["full"].reshape(total, 4), chemistry["active"].ravel()
    full = _full_grid(dataset, split, context, point, memory, auxiliary, active)
    model = native["model"]
    for module in (model.spatial, model.temporal, model.decay, model.head):
        module.requires_grad_(False)
    train_mask = np.zeros((n, months), dtype=bool)
    train_mask.ravel()[split["train"]] = True
    local_cells = np.flatnonzero(train_mask[source_ids])
    source_cells = source_ids[local_cells // months] * months + local_cells % months
    np.testing.assert_array_equal(source_cells, np.sort(split["train"]))
    _, validation_cells = support_query_cells(split, target_role="val", k=0, n_months=months)
    if "source_features" in native:
        np.testing.assert_array_equal(native["source_cells"], source_cells)
        np.testing.assert_array_equal(native["source_local_cells"], local_cells)
        np.testing.assert_array_equal(native["validation_cells"], validation_cells)
        source_features = np.asarray(native["source_features"])
        validation_features = np.asarray(native["validation_features"])
        source_base = np.asarray(native["source_base"])
    else:
        source_features, source_delta = observed_features(model, native["source_inputs"], local_cells)
        validation_features, _ = observed_features(model, native["full_inputs"], validation_cells)
        source_context = np.maximum(0, np.expm1(np.asarray(initial["oof_z"]).ravel()[source_cells]))
        source_base = np.maximum(0, source_context + model.selected_scale_ * source_delta)
    if (source_features.shape != (len(source_cells), 550)
            or validation_features.shape != (len(validation_cells), 550)
            or source_base.shape != (len(source_cells),)):
        raise ValueError("Fresh native source/validation caches have incompatible shapes")
    np.savez_compressed(output / "source_training.npz", source_cells=source_cells,
        source_local_cells=local_cells, source_station_ids=source_ids, source_base=source_base,
        source_active=active[source_cells], validation_cells=validation_cells,
        validation_base=point[validation_cells], validation_active=active[validation_cells])
    identities = {"source_original_feature_hash": array_digest(source_features),
        "validation_original_feature_hash": array_digest(validation_features), "modes": {}}
    heads = {}
    for mode in MODES:
        block = apply_auxiliary_mode(chemistry["full"], mode).reshape(total, 4)
        sx, vx = _augment(source_features, block[source_cells]), _augment(validation_features, block[validation_cells])
        identities["modes"][mode] = {"source_feature_hash": array_digest(sx), "validation_feature_hash": array_digest(vx)}
        head = NonlinearChemistryHead(n_features=554, epochs=1 if smoke else 120, patience=10,
            batch_size=512, seed=seed, learning_rate=.001).fit(sx, source_base, labels[source_cells],
                vx, point[validation_cells], labels[validation_cells], source_active=active[source_cells],
                validation_active=active[validation_cells], tail_threshold=native["q90_threshold_train"],
                selection_role="source_validation", progress=lambda row, name=mode: print(
                    f"{run.name}/neural_{name}: {json.dumps(row)}", flush=True))
        if head.trainable_parameter_count_ != 1111:
            raise ValueError("The accepted nonlinear decoder must have 1111 parameters")
        heads[mode] = head
        torch.save(head.to_payload(), output / f"neural_{mode}.pt")
        write_json(output / f"neural_{mode}.json", head.to_dict())
        pd.DataFrame(head.to_dict()["trace"]).to_csv(output / f"neural_{mode}_trace.csv", index=False)
    write_json(output / "input_definition.json", identities)
    bases = {"point": point, "context": context}
    tree_features = build_daily_tree_features(dataset, split, native["daily"], mode="current")
    if tree_features.shape != (total, 147):
        raise ValueError("Current-daily tree feature definition changed")
    tree_records = {}
    current_tree = clone(initial["expert"].context_forest).set_params(n_jobs=2)
    current_tree.fit(tree_features[split["train"]], np.log1p(labels[split["train"]]))
    bases["tree_prior"] = np.maximum(0, np.expm1(current_tree.predict(tree_features)))
    joblib.dump(current_tree, output / "tree_current.joblib", compress=3)
    chemical_features = np.concatenate((tree_features, auxiliary), axis=1)
    chemical_tree = clone(current_tree).set_params(n_jobs=2)
    chemical_tree.fit(chemical_features[split["train"]], np.log1p(labels[split["train"]]))
    chemical_candidate = np.maximum(0, np.expm1(chemical_tree.predict(chemical_features)))
    bases["tree_chemistry"] = np.where(active, chemical_candidate, bases["tree_prior"])
    joblib.dump(chemical_tree, output / "tree_chemistry.joblib", compress=3)
    for name, forest, width in (("current", current_tree, 147), ("chemistry", chemical_tree, 151)):
        tree_records[name] = {"n_features": width, "forest_parameters": forest.get_params(deep=False),
            "selected_context_parameters": initial["expert"].context_forest.get_params(deep=False),
            "source_cells": len(split["train"]), "source_station_ids": source_ids.tolist(),
            "target_transform": "log1p", "visible_roles": ["train"], "hyperparameter_search": False}
    tree_records["current_feature_names"] = daily_tree_feature_names()
    tree_records["chemistry_extra_feature_names"] = chemistry["feature_names"]
    write_json(output / "trees.json", tree_records)
    del tree_features, chemical_features, source_features, validation_features, sx, vx
    for mode in MODES:
        bases[f"neural_{mode}"] = np.empty(total)
    for cells, vectors, _ in head_feature_blocks(model, native["full_inputs"]):
        for mode in MODES:
            block = apply_auxiliary_mode(auxiliary[cells].reshape(1, -1, 4), mode).reshape(-1, 4)
            bases[f"neural_{mode}"][cells] = heads[mode].predict(
                _augment(vectors, block), point[cells], active=active[cells])
    for name, prediction in bases.items():
        full[f"{name}_pred"] = prediction
    for name in NEURAL:
        np.testing.assert_array_equal(bases[name][~active], point[~active])
    np.testing.assert_array_equal(bases["tree_chemistry"][~active], bases["tree_prior"][~active])
    old_adapters, old_mixers = fit_legacy_choices(bases, context, memory, legacy, labels, split, months, active)
    old_adapter_states = {name: value.to_dict() for name, value in old_adapters.items()}
    old_mixer_states = {name: value.to_dict() for name, value in old_mixers.items()}
    write_json(output / "legacy_adapters.json", old_adapter_states)
    write_json(output / "legacy_mixers.json", old_mixer_states)
    for name in INTEGRATED:
        candidate = old_mixers[f"{name}_integrated_{BASIS}"].selected_base(context, bases[name], memory, k=0)
        fallback = old_mixers[f"point_integrated_{BASIS}"].selected_base(context, point, memory, k=0)
        full[f"{name}_integrated_k0_pred"] = candidate if name == "point" else np.where(active, candidate, fallback)
    shapes, definitions = {"legacy": legacy}, {}
    coordinates = {"legacy": legacy, "active": active, "source_station_ids": source_ids}
    for mode, name in (("masks", "masks_aug"), ("chemistry", "chemistry_aug")):
        projector = ChemicalSupportBasis(heads["chemistry"], eigenvalue_floor=1e-8).fit(
            chemistry["full"], source_ids, source_role="source_training", mode=mode)
        chemical = projector.transform(chemistry["full"]).reshape(total, 2)
        shapes[name] = np.concatenate((legacy, chemical), axis=1)
        coordinates[name], coordinates[f"{mode}_chemical"] = shapes[name], chemical
        definitions[name] = projector.to_dict()
    np.savez_compressed(output / "representations.npz", **coordinates)
    write_json(output / "basis_definition.json", definitions)
    gamma_k0 = old_mixer_states[f"neural_chemistry_integrated_{BASIS}"]["gamma_k0"]
    adapters, mixers = fit_calibrators(bases, context, memory, shapes, labels, split, months, active, gamma_k0)
    write_json(output / "adapters.json", {name: value.to_dict() for name, value in adapters.items()})
    write_json(output / "mixers.json", {name: value.to_dict() for name, value in mixers.items()})
    validation = make_panels(full, bases, shapes, adapters, mixers, old_adapter_states, old_mixer_states,
                             labels, split, months, active, role="val")
    selection = choose_representations(validation_summary(validation, labels))
    write_json(output / "basis_selection.json", selection)
    validation = add_selected(validation, selection)
    old_validation = legacy_panels(full, bases, legacy, old_adapters, old_mixers,
                                   labels, split, months, active, role="val")
    val_checks = invariance_checks(validation, old_validation, role="val")
    validation = _join_controls(validation, old_validation)
    summary = validation_summary(validation, labels)
    summary.to_csv(output / "source_validation.csv", index=False)
    print(f"{run.name}: fresh chemistry states and source-validation choices frozen", flush=True)

    # This is the first extraction of held target DOC values: support only.
    target_labels = _target_support_view(dataset, split)
    predictions = add_selected(make_panels(full, bases, shapes, adapters, mixers,
        old_adapter_states, old_mixer_states, target_labels, split, months, active, role="test"), selection)
    old_predictions = legacy_panels(full, bases, legacy, old_adapters, old_mixers,
                                    target_labels, split, months, active, role="test")
    test_checks = invariance_checks(predictions, old_predictions, role="test")
    predictions = _join_controls(predictions, old_predictions)
    write_json(output / "reference_checks.json", {"validation": val_checks, "test": test_checks})
    metadata = {"models": list(MODELS), "k_values": list(KS), "new_neural_fits": 3, "new_tree_fits": 2,
        "source_neural_is_oof": False, "source_forest_is_oof": True, "source_station_ids": source_ids.tolist(),
        "epochs": 1 if smoke else 120, "patience": 10, "feature_dim": 554, "head_parameters": 1111,
        "auxiliary_paths": {name: str(path) for name, path in AUX_PATHS.items()},
        "auxiliary_hashes": {name: sha256_file(path) for name, path in AUX_PATHS.items()},
        "auxiliary_provenance_hashes": {name: sha256_file(path.with_suffix(".provenance.json"))
                                        for name, path in AUX_PATHS.items()},
        "auxiliary_policy": chemistry["policy"], "auxiliary_feature_names": chemistry["feature_names"],
        "auxiliary_feature_hash": array_digest(auxiliary), "auxiliary_active_hash": array_digest(active),
        "target_query_labels_read": False, "basis_selection_role": "source_validation",
        "basis_selection_score": "final active validation MAE", "gamma_k0": gamma_k0,
        "elapsed_seconds": time.monotonic() - started}
    write_json(output / "stage.json", metadata)
    return _finish_chemical_stage(output, config, full, predictions)
