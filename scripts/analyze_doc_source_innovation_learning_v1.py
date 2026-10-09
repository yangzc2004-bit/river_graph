"""Replay matched innovation readouts and trees, then report source DOC effects."""
from __future__ import annotations

import argparse
import json
from itertools import combinations
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
from analyze_doc_chemistry_confirmation_v1 import metric_summary
from analyze_doc_source_retrieval_v1 import load_panel, paired
from analyze_unified_doc_spatial import joint_station_bootstrap
from run_doc_source_innovation_learning_v1 import (
    ARMS,
    INPUT_FIELDS,
    MODES,
    ROOT,
    TREE_ARMS,
)
from run_unified_doc_spatial import (
    digest,
    verify_files,
    verify_runtime_snapshot,
    write_json,
)

from river_graph.experiments.provenance import run_identity_sha256, sha256_file
from river_graph.experiments.unmonitored_doc import development_labels
from river_graph.models.ecological_residual_transfer import EcologicalResidualTransfer
from river_graph.models.encoder_native_residual import EncoderNativeResidual
from river_graph.models.kgml_local_transport import station_folds
from river_graph.models.source_doc_innovations import SourceDOCInnovationLibrary
from river_graph.models.source_innovation_training import (
    innovation_readout_features,
    nested_innovation_inputs,
)


def verify(run, runtime):
    config = json.loads((run/"config.json").read_text())
    verify_files(run, "complete.json", config)
    if config["runtime_snapshot_hash"] != runtime:
        raise ValueError("innovation learning execution changed")
    for kind in ("dataset", "mask"):
        if sha256_file(config[f"{kind}_path"]) != config[f"{kind}_hash"]:
            raise ValueError("source data or station roles changed")
    parent, source, probe = (Path(config[key]) for key in ("parent_run", "source_inputs_run", "probe_run"))
    old = json.loads((parent/"config.json").read_text())
    verify_files(parent, "complete.json", old)
    if (sha256_file(parent/"complete.json") != config["parent_completion_hash"]
            or sha256_file(source/"joint_source_inputs.npz") != config["source_inputs_hash"]
            or sha256_file(probe/"complete.json") != config["probe_completion_hash"]):
        raise ValueError("source inputs or ancestor changed")
    meta = json.loads((run/"predictions.meta.json").read_text())
    identity = run_identity_sha256(digest(config), config["started_at"], runtime)
    if (meta["run_identity_sha256"] != identity or meta["config_hash"] != digest(config)
            or meta["prediction_sha256"] != sha256_file(run/"predictions.parquet")):
        raise ValueError("innovation product identity changed")
    data = torch.load(config["dataset_path"], weights_only=False, map_location="cpu")
    with np.load(config["mask_path"], allow_pickle=False) as saved:
        split = {role: saved[role].copy() for role in ("train", "val", "test", "context")}
    data["y"] = development_labels(data, split)
    months = data["y"].shape[1]
    train, val = split["train"], split["val"]
    source_ids, validation_ids = np.unique(train//months), np.unique(val//months)
    if np.intersect1d(source_ids, np.unique(np.r_[val, split["test"]]//months)).size:
        raise ValueError("overlapping source/receiving stations")
    folds, references = station_folds(train, months, config["seed"]), {}
    for a, b in combinations(range(len(folds)), 2):
        prefix = f"pair_reference_{a}_{b}"
        with np.load(run/f"{prefix}.npz", allow_pickle=False) as saved:
            cells, features, expected = saved["cells"].copy(), saved["features"].copy(), saved["pred_z"].copy()
        forest = joblib.load(run/f"{prefix}.joblib")
        np.testing.assert_allclose(forest.predict(features), expected, rtol=1e-12, atol=1e-12)
        references[(a, b)] = {**json.loads((run/f"{prefix}.json").read_text()), "cells": cells, "pred_z": expected}
        del forest, features
    with np.load(source/"joint_source_inputs.npz", allow_pickle=False) as saved:
        np.testing.assert_array_equal(saved["source_station_ids"], source_ids)
        source_ecology = saved["env"].copy()
    with np.load(parent/"validation_inputs.npz", allow_pickle=False) as saved:
        validation = {key: saved[key].copy() for key in INPUT_FIELDS}
        base = saved["context"].copy()
        np.testing.assert_array_equal(saved["cell"], val)
    ecology = np.full((len(data["site_no"]), 9), np.nan)
    ecology[source_ids] = source_ecology
    source_parts, records = nested_innovation_inputs(data["y"], data["site_no"], data["months"],
        ecology, train, folds, references)
    if records != json.loads((run/"source_library_roles.json").read_text()):
        raise ValueError("nested source role replay failed")
    library = SourceDOCInnovationLibrary.load(probe/"source_library.npz")
    validation_parts = library.predict_components(np.asarray(data["site_no"], str)[validation_ids],
        validation["env"], data["months"])
    scale = json.loads((run/"innovation_scale.json").read_text())
    if scale["scale"] != 1. or scale["fit_role"] != "none":
        raise ValueError("label-dependent cross-bank scaling introduced")
    with np.load(run/"innovation_inputs.npz", allow_pickle=False) as saved:
        np.testing.assert_array_equal(saved["source_ids"], source_ids)
        np.testing.assert_array_equal(saved["validation_ids"], validation_ids)
        np.testing.assert_array_equal(saved["cells"], val)
        for mode in MODES:
            np.testing.assert_array_equal(saved[f"source_{mode}"], innovation_readout_features(source_parts, mode, 1.))
            np.testing.assert_array_equal(saved[f"validation_{mode}"], innovation_readout_features(validation_parts, mode, 1.))
        val_features = {mode: saved[f"validation_{mode}"].copy() for mode in MODES}
    previous, panel = pd.read_parquet(parent/"predictions.parquet"), pd.read_parquet(run/"predictions.parquet")
    original = panel[panel.model_name.isin(previous.model_name.unique())][previous.columns]
    pd.testing.assert_frame_equal(original.reset_index(drop=True), previous.reset_index(drop=True), check_exact=True)
    old_model = EncoderNativeResidual.from_payload(torch.load(parent/"native.pt", weights_only=False, map_location="cpu"))
    expected_settings = {**old_model._config(), "extra_dim": old_model.extra_dim+3,
                        "interaction_indices": (*old_model.interaction_indices, old_model.extra_dim)}
    index = np.searchsorted(validation_ids, val//months), val % months
    trajectories = []
    for mode in MODES:
        prefix = f"innovation_{mode}"
        model = EncoderNativeResidual.from_payload(torch.load(run/f"{prefix}.pt", weights_only=False, map_location="cpu"))
        if model._config() != expected_settings or model.trainable_parameter_count_ != old_model.trainable_parameter_count_+67:
            raise ValueError("undeclared model/budget change")
        for name in ("spatial", "temporal", "decay"):
            for key, expected in getattr(old_model, f"_initial_{name}_state").items():
                torch.testing.assert_close(getattr(model, f"_initial_{name}_state")[key], expected, rtol=0, atol=0)
        view = {**validation, "extra": np.concatenate([validation["extra"], val_features[mode]], -1)}
        prediction = model.predict(view, base)
        np.testing.assert_array_equal(prediction[index], panel[panel.model_name.eq(f"{prefix}_native")].y_pred)
        memory = EcologicalResidualTransfer.from_dict(json.loads((run/f"{prefix}_memory.json").read_text()))
        full_base = np.zeros((len(data["site_no"]), months))
        full_base[validation_ids] = base
        native = full_base.copy()
        native[validation_ids] = prediction
        np.testing.assert_array_equal(memory.predict(full_base, native).ravel()[val],
            panel[panel.model_name.eq(f"{prefix}_integrated")].y_pred)
        trajectories.append({"mode": mode, "best_epoch": model.best_epoch_, "epochs_run": model.epochs_run_,
            "trainable_parameters": model.trainable_parameter_count_})
    parent_forest = joblib.load(Path(old["parent_run"])/"station_hidden_trees.joblib")
    with np.load(run/"tree_validation_inputs.npz", allow_pickle=False) as saved:
        core = saved["features"].copy()
        np.testing.assert_array_equal(saved["cells"], val)
    np.testing.assert_allclose(np.maximum(0., np.expm1(parent_forest.predict(core))),
        previous[previous.model_name.eq("station_hidden_trees")].y_pred, rtol=1e-12, atol=1e-12)
    for mode in ("real", "historical"):
        model = joblib.load(run/f"innovation_{mode}_trees.joblib")
        if model.get_params() != {**parent_forest.get_params(), "n_jobs": 2}:
            raise ValueError("enriched forest settings changed")
        expected = np.maximum(0., np.expm1(model.predict(np.column_stack([core, val_features[mode][index]]))))
        np.testing.assert_allclose(expected, panel[panel.model_name.eq(f"innovation_{mode}_trees")].y_pred,
            rtol=1e-12, atol=1e-12)
    for _, group in panel.groupby("model_name"):
        np.testing.assert_array_equal(group.cell, val)
    if not panel.visibility_role.eq("val").all() or not np.isfinite(panel.y_pred).all():
        raise ValueError("invalid innovation-learning source product")
    return {"run": run.name, "identity": True, "neural_and_memory_replay": "bitwise",
        "pair_and_enriched_forest_replay": "1e-12 tolerance", "nested_library_and_feature_replay": "bitwise",
        "unchanged_parent_predictions": True, "old_target_evaluation": False,
        "source_query_and_donor_fold_exclusion": True, "trajectories": trajectories}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--bootstrap-draws", type=int, default=5000)
    args = parser.parse_args()
    torch.set_num_threads(2)
    runtime = verify_runtime_snapshot(args.root)
    complete = sorted((args.root/"runs").glob("*/complete.json"))
    if {p.parent.name for p in complete} != {f"split{s}_seed{seed}" for s in (142, 143, 144) for seed in (42, 43, 44)}:
        raise ValueError("complete all nine innovation-learning source packages")
    verification = [verify(p.parent, runtime) for p in complete]
    write_json(args.root/"verification/replay.json", verification)
    panel, thresholds = load_panel(args.root)
    runs, partitions, summary = metric_summary(panel, thresholds)
    output = args.root/"analysis"
    output.mkdir(exist_ok=True)
    for name, frame in (("run_metrics", runs), ("partition_metrics", partitions), ("summary", summary)):
        frame.to_csv(output/f"{name}.csv", index=False)
    comparisons = [(arm, ref) for arm in (*ARMS, *TREE_ARMS) for ref in ("unmonitored_integrated", "station_hidden_trees")]
    comparisons += [("innovation_real_integrated", ref) for ref in ("innovation_historical_integrated", "innovation_availability_integrated", "innovation_real_trees")]
    comparisons += [("innovation_real_native", ref) for ref in ("unmonitored_residual", "innovation_historical_native", "innovation_availability_native")]
    comparisons.append(("innovation_real_trees", "innovation_historical_trees"))
    effects = []
    for candidate, reference in comparisons:
        pair = paired(panel, candidate, reference)
        for region in ("overall", "q90"):
            selected = pair
            if region == "q90":
                cutoff = np.array([thresholds[(p, s)] for p, s in zip(pair.split_seed, pair.seed, strict=True)])
                selected = pair[pair.y_true_candidate.to_numpy() >= cutoff]
            direction = selected.groupby(["split_seed", "seed"])[["candidate_error", "reference_error"]].mean()
            parts, seeds = direction.groupby("split_seed").mean(), direction.groupby("seed").mean()
            effects.append({"candidate": candidate, "reference": reference, "region": region,
                **joint_station_bootstrap(selected, draws=args.bootstrap_draws),
                "improved_packages": int((direction.candidate_error < direction.reference_error).sum()),
                "improved_partitions": int((parts.candidate_error < parts.reference_error).sum()),
                "improved_training_seeds": int((seeds.candidate_error < seeds.reference_error).sum())})
    pd.DataFrame(effects).to_csv(output/"paired_effects.csv", index=False)
    panel["absolute_error"], panel["signed_error"] = np.abs(panel.y_pred-panel.y_true), panel.y_pred-panel.y_true
    panel.groupby(["split_seed", "seed", "model_name", "station"], as_index=False).agg(
        mae=("absolute_error", "mean"), bias=("signed_error", "mean"), n_cells=("cell", "size"))\
        .to_csv(output/"station_metrics.csv", index=False)
    write_json(output/"sources.json", {"evaluation_role": "source_validation_selected_development",
        "bootstrap_draws": args.bootstrap_draws, "analyzer_sha256": sha256_file(__file__),
        "runs": {p.parent.name: sha256_file(p) for p in complete},
        "numerical_outputs": {p.name: sha256_file(p) for p in output.glob("*.csv")}})
    print(summary[["model_name", "mae", "signed_bias", "q90_mae"]].to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
