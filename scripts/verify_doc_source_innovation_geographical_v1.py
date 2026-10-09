"""Replay source libraries, nested references and fixed geographical predictions."""
from __future__ import annotations

import argparse
import json
from itertools import combinations
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
from run_doc_daily_hydro_residual_v1 import load_daily_pack
from run_doc_geographical_confirmation_v1 import DAILY_ROOT, support_curves
from run_doc_source_innovation_geographical_v1 import (
    ARMS,
    INPUT_FIELDS,
    MODELS,
    ROOT,
    geographic_inputs,
)
from run_unified_doc_spatial import (
    digest,
    verify_files,
    verify_runtime_snapshot,
    write_json,
)

from river_graph.experiments.provenance import run_identity_sha256, sha256_file
from river_graph.experiments.unmonitored_doc import (
    HUC4_BLOCKS,
    development_labels,
    strip_auxiliary_water,
)
from river_graph.models.ecological_residual_transfer import EcologicalResidualTransfer
from river_graph.models.encoder_native_residual import EncoderNativeResidual
from river_graph.models.kgml_local_transport import station_folds
from river_graph.models.source_doc_innovations import (
    SourceDOCInnovationLibrary,
    source_residual_grid,
)
from river_graph.models.source_innovation_training import (
    innovation_readout_features,
    nested_innovation_inputs,
)


def verify(run, runtime):
    config = json.loads((run/"config.json").read_text())
    if config["runtime_snapshot_hash"] != runtime:
        raise ValueError("geographical source-information runtime changed")
    for kind in ("dataset", "mask"):
        if sha256_file(config[f"{kind}_path"]) != config[f"{kind}_hash"]:
            raise ValueError("geographical input changed")
    parent = Path(config["parent_run"])
    old = json.loads((parent/"config.json").read_text())
    verify_files(parent, "complete.json", old)
    if (sha256_file(parent/"complete.json") != config["parent_completion_hash"]
            or sha256_file(config["source_decision"]) != config["source_decision_hash"]):
        raise ValueError("parent or source-selected decision changed")
    for stage in ("complete.json", "native_complete.json", "tree_complete.json", "point_complete.json"):
        verify_files(run, stage, config)
    original = torch.load(config["dataset_path"], weights_only=False, map_location="cpu")
    truth = np.asarray(original["y"], float).copy()
    months = truth.shape[1]
    with np.load(config["mask_path"], allow_pickle=False) as saved:
        split = {role: saved[role].copy() for role in ("train", "val", "test", "context")}
    dataset = strip_auxiliary_water(original)
    dataset["y"] = development_labels(dataset, split)
    train, test = split["train"], split["test"]
    ids, test_ids = np.unique(train//months), np.unique(test//months)
    folds = station_folds(train, months, config["seed"])
    references = {}
    for a, b in combinations(range(len(folds)), 2):
        prefix = f"pair_reference_{a}_{b}"
        verify_files(run, f"{prefix}_complete.json", config)
        with np.load(run/f"{prefix}.npz", allow_pickle=False) as saved:
            cells, features, expected = (saved[key].copy() for key in ("cells", "features", "pred_z"))
        forest = joblib.load(run/f"{prefix}.joblib")
        np.testing.assert_allclose(forest.predict(features), expected, rtol=1e-12, atol=1e-12)
        references[(a, b)] = {**json.loads((run/f"{prefix}.json").read_text()), "cells": cells, "pred_z": expected}
        del forest, features
    with np.load(parent/"oof.npz", allow_pickle=False) as saved:
        oof = saved["new"].copy()
    with np.load(run/"components.npz", allow_pickle=False) as saved:
        grids = {key: saved[key].copy() for key in ("environment", "native", "integrated", "enriched_trees")}
    daily, _, _ = load_daily_pack(DAILY_ROOT, config["dataset_hash"], truth.shape)
    _, full_inputs, _, _ = geographic_inputs(parent, dataset, split, daily, oof, grids["environment"])
    names = np.asarray(dataset["site_no"], str)
    parts, records = nested_innovation_inputs(dataset["y"], names, dataset["months"],
        full_inputs["env"], train, folds, references)
    if records != json.loads((run/"source_library_roles.json").read_text()):
        raise ValueError("nested query-fold exclusion changed")
    source_ids, residual = source_residual_grid(dataset["y"], oof, train)
    np.testing.assert_array_equal(source_ids, ids)
    expected_library = SourceDOCInnovationLibrary().fit(names[ids], dataset["months"], full_inputs["env"][ids], residual)
    library = SourceDOCInnovationLibrary.load(run/"source_library.npz")
    for key in ("source_names_", "months_", "ecology_", "seasonal_mean_", "innovations_", "sigma_"):
        np.testing.assert_array_equal(getattr(library, key), getattr(expected_library, key))
    full_parts = library.predict_components(names, full_inputs["env"], dataset["months"])
    full_features = innovation_readout_features(full_parts, "real", 1.)
    with np.load(run/"innovation_inputs.npz", allow_pickle=False) as saved:
        np.testing.assert_array_equal(saved["source_ids"], ids)
        np.testing.assert_array_equal(saved["source_features"], innovation_readout_features(parts, "real", 1.))
        np.testing.assert_array_equal(saved["full_features"], full_features)
        for key in ("support_count", "weight_mass"):
            np.testing.assert_array_equal(saved[key], full_parts[key])
    retained = EncoderNativeResidual.from_payload(torch.load(parent/"native.pt", weights_only=False, map_location="cpu"))
    model = EncoderNativeResidual.from_payload(torch.load(run/"native.pt", weights_only=False, map_location="cpu"))
    expected_config = {**retained._config(), "extra_dim": retained.extra_dim+3,
        "interaction_indices": (*retained.interaction_indices, retained.extra_dim), "epochs": 30, "patience": 5}
    if (digest(model._config()) != digest(expected_config)
            or model.trainable_parameter_count_ != retained.trainable_parameter_count_+67):
        raise ValueError("unplanned model/budget change")
    for name in ("spatial", "temporal", "decay"):
        for key, value in getattr(retained, f"_initial_{name}_state").items():
            torch.testing.assert_close(getattr(model, f"_initial_{name}_state")[key], value, rtol=0, atol=0)
    full_view = {**full_inputs, "extra": np.concatenate([full_inputs["extra"], full_features], -1)}
    np.testing.assert_array_equal(model.predict(full_view, grids["environment"]), grids["native"])
    with np.load(run/"test_inputs.npz", allow_pickle=False) as saved:
        np.testing.assert_array_equal(saved["cell"], test)
        for key in INPUT_FIELDS:
            np.testing.assert_array_equal(saved[key], full_view[key][test_ids])
        np.testing.assert_array_equal(saved["context"], grids["environment"][test_ids])
    memory = EcologicalResidualTransfer.from_dict(json.loads((run/"memory.json").read_text()))
    np.testing.assert_array_equal(memory.predict(grids["environment"], grids["native"]), grids["integrated"])
    tree = joblib.load(run/"enriched_tree.joblib")
    parent_tree = joblib.load(parent/"station_hidden_trees.joblib")
    if tree.get_params() != parent_tree.get_params():
        raise ValueError("enriched forest hyperparameters changed")
    with np.load(run/"tree_test_inputs.npz", allow_pickle=False) as saved:
        np.testing.assert_array_equal(saved["cells"], test)
        tree_prediction = np.maximum(0., np.expm1(tree.predict(saved["features"])))
        np.testing.assert_allclose(tree_prediction, grids["enriched_trees"].ravel()[test], rtol=1e-12, atol=1e-12)
    bases = dict(zip(ARMS, (grids["native"], grids["integrated"], grids["enriched_trees"]), strict=True))
    for filename in ("predictions.parquet", "support_curves.parquet"):
        frame, prior = (pd.read_parquet(path/filename) for path in (run, parent))
        original_rows = frame[frame.model_name.isin(prior.model_name.unique())].reset_index(drop=True)
        pd.testing.assert_frame_equal(original_rows, prior.reset_index(drop=True), check_exact=True)
        meta = json.loads((run/filename.replace(".parquet", ".meta.json")).read_text())
        if (meta["config_hash"] != digest(config) or meta["prediction_sha256"] != sha256_file(run/filename)
                or meta["run_identity_sha256"] != run_identity_sha256(digest(config), config["started_at"], runtime)):
            raise ValueError("geographical prediction identity changed")
        if (set(frame.model_name) != set(MODELS) or not frame.visibility_role.eq("test").all()
                or frame.duplicated(["model_name", "k", "cell"]).any() or not np.isfinite(frame.y_pred).all()):
            raise ValueError("incomplete geographical panel")
        query = None
        for _, group in frame.groupby(["model_name", "k"]):
            selected = group.cell.to_numpy()
            if query is None:
                query = selected
            np.testing.assert_array_equal(selected, query)
            np.testing.assert_array_equal(group.y_true, truth.ravel()[selected])
            np.testing.assert_array_equal(group.station, names[selected//months])
            np.testing.assert_array_equal(group.month, np.asarray(dataset["months"], str)[selected % months])
        if filename == "predictions.parquet":
            np.testing.assert_array_equal(query, test)
            for name, grid in bases.items():
                np.testing.assert_array_equal(frame[frame.model_name.eq(name)].y_pred, grid.ravel()[test])
    curves, adapters = support_curves(bases, dataset, split, truth)
    curves["split_seed"], curves["seed"], curves["target_huc4"] = config["split_seed"], config["seed"], config["target_huc4"]
    saved = pd.read_parquet(run/"support_curves.parquet")
    pd.testing.assert_frame_equal(saved[saved.model_name.isin(ARMS)].reset_index(drop=True), curves.reset_index(drop=True), check_exact=True)
    if adapters != json.loads((run/"support_adapters.json").read_text()):
        raise ValueError("validation-selected adapter changed")
    return {"run": run.name, "identity": True, "nested_source_exclusion": True,
        "library_features_native_memory_support_replay": "bitwise", "forests_replay": "1e-12 reduction tolerance",
        "initial_model": "retained", "target_cells": len(test), "scope": config["historical_scope"]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--huc4", nargs="+", default=list(HUC4_BLOCKS), choices=HUC4_BLOCKS)
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44, 45, 46])
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    torch.set_num_threads(2)
    runtime = verify_runtime_snapshot(args.root)
    paths = [args.root/"runs"/f"huc4_{region}_seed{seed}" for region in args.huc4 for seed in args.seeds]
    if any(not (path/"complete.json").exists() for path in paths):
        raise ValueError("requested geographical packages are incomplete")
    write_json(args.output or args.root/"verification/replay.json", [verify(path, runtime) for path in paths])


if __name__ == "__main__":
    main()
