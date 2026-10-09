"""Evaluate the fixed five-member current-source release in HUC02040104.

This case was inspected for the previous release. Predict without receiving
chemistry, retain old products, and calibrate intervals/support on source val.
"""
from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from portable_current_source_doc_v2 import PortableCurrentSourceDOC
from run_doc_external_replication_v1 import (
    EXTERNAL_PROTOCOL,
    empirical_interval,
    source_ensemble_policy,
    source_inputs,
)
from run_unified_doc_spatial import digest, verify_files, write_json

from river_graph.experiments.provenance import sha256_file
from river_graph.experiments.spatial_fewshot import support_schedule
from river_graph.models.station_adapted_hybrid import station_residual_correction

ROOT = Path("experiments/phase4_transfer/doc_current_source_external_v2")
SOURCE = Path("experiments/phase4_transfer/doc_current_source_portable_v2_r1")
PREVIOUS = Path("experiments/phase4_transfer/doc_external_replication_v1")
SEEDS = (42, 43, 44, 45, 46)
NEW_MODELS = ("available_real_native", "available_real_integrated")
COMPONENTS = ("environment_pred", "local_temporal_correction", "source_observation_correction",
              "source_static_correction", "source_transfer_correction", "river_correction",
              "native_pred", "final_pred", "source_support_count", "attention_prior_mass", "attention_entropy")


def validate_previous_panel(panel, cells, query, truth):
    """Old comparisons must preserve the exact receiving population and truth."""
    if panel.duplicated(["population", "k", "model_name", "cell"]).any():
        raise ValueError("duplicate previous query identities")
    for (population, _, _), group in panel.groupby(["population", "k", "model_name"]):
        expected = cells if population == "all_observed_k0" else query
        ordered = group.sort_values("cell")
        np.testing.assert_array_equal(ordered.cell, expected)
        np.testing.assert_array_equal(ordered.y_true, truth[expected])


def generate_points(source_root, root):
    external = json.loads(EXTERNAL_PROTOCOL.read_text())
    runs = [source_root/"runs"/f"seed{seed}" for seed in SEEDS]
    for run in runs:
        config = json.loads((run/"config.json").read_text())
        verify_files(run, "complete.json", config)
    old_meta = json.loads((PREVIOUS/"predictions.meta.json").read_text())
    if sha256_file(PREVIOUS/"predictions.parquet") != old_meta["prediction_sha256"]:
        raise ValueError("previous external prediction content changed")
    if sha256_file(external["label_free_inputs"]) != external["input_hash"]:
        raise ValueError("receiving label-free input changed")
    previous_points = json.loads((PREVIOUS/"point_generation_complete.json").read_text())
    for name, expected in previous_points["files"].items():
        if sha256_file(PREVIOUS/name) != expected:
            raise ValueError("previous fixed point-generation product changed")
    config = {"source_root": str(source_root), "seeds": list(SEEDS),
        "new_models": list(NEW_MODELS), "primary_procedure": NEW_MODELS[1],
        "source_completions": {str(run/"complete.json"): sha256_file(run/"complete.json") for run in runs},
        "external_protocol_hash": sha256_file(EXTERNAL_PROTOCOL), "inputs_hash": external["input_hash"],
        "external_scoring_hash": external["dataset_hash"],
        "previous_predictions_hash": old_meta["prediction_sha256"],
        "previous_points_hash": sha256_file(PREVIOUS/"point_generation_complete.json"),
        "ensemble": "arithmetic mean of native mg/L point predictions across five fixed seeds",
        "external_exposure": "02040104 was inspected for the preceding release; updated replication, not new blind test",
        "model_selection": "fixed architecture from source roles and ST357 geographical confirmation",
        "runner_sha256": sha256_file(__file__),
        "portable_sha256": sha256_file("scripts/portable_current_source_doc_v2.py")}
    root.mkdir(parents=True, exist_ok=True)
    path = root/"point_generation_complete.json"
    if path.exists():
        saved = json.loads(path.read_text())
        if saved["config"] != config:
            raise ValueError("preserve external execution; changed generation requires a new version")
        for name, expected in saved["files"].items():
            if sha256_file(root/name) != expected:
                raise ValueError("saved external product changed")
        return config
    with np.load(external["label_free_inputs"], allow_pickle=False) as saved:
        inputs = {key: saved[key].copy() for key in ("site_no", "months", "x", "x_mask", "static", "regime")}
        inputs["daily_features"] = saved["daily"].copy()
    members = {name: {key: [] for key in COMPONENTS} for name in NEW_MODELS}
    val_members = {name: [] for name in NEW_MODELS}
    validation_cells = validation_truth = None
    replay, donor_records, source_rows = [], [], []
    for run in runs:
        predictor = PortableCurrentSourceDOC.load(run/"export")
        fit = json.loads((run/"config.json").read_text())
        dataset = torch.load(fit["dataset_path"], weights_only=False, map_location="cpu")
        with np.load(fit["mask_path"], allow_pickle=False) as saved:
            cells = saved["val"].copy()
        with np.load(fit["daily_path"], allow_pickle=False) as saved:
            daily = saved["full"].copy()
        months = dataset["y"].shape[1]
        ids = np.unique(cells//months)
        parts = predictor.predict_components(source_inputs(dataset, ids, daily))
        original = pd.read_parquet(run/"predictions.parquet")
        for name, key in zip(NEW_MODELS, ("native_pred", "final_pred"), strict=True):
            saved = original[original.model_name.eq(name)].sort_values("cell")
            np.testing.assert_array_equal(saved.cell, cells)
            values = parts[key][np.searchsorted(ids, cells//months), cells % months]
            np.testing.assert_allclose(values, saved.y_pred, rtol=1e-6, atol=1e-6)
            val_members[name].append(values)
            replay.append({"seed": fit["seed"], "model_name": name, "n_cells": len(cells),
                "max_abs_error": float(np.abs(values-saved.y_pred.to_numpy()).max())})
            if validation_cells is not None:
                np.testing.assert_array_equal(validation_cells, cells)
                np.testing.assert_array_equal(validation_truth, saved.y_true)
            validation_cells, validation_truth = cells, saved.y_true.to_numpy()
        source_rows.append(original)
        parts = predictor.predict_components(inputs)
        donor_records.extend({"seed": fit["seed"], **record} for record in parts["retrieval_sources"])
        for name in NEW_MODELS:
            for key in COMPONENTS:
                value = parts[key]
                if name == NEW_MODELS[0]:
                    if key == "final_pred":
                        value = parts["native_pred"]
                    elif key == "source_static_correction":
                        value = np.zeros_like(value)
                    elif key == "source_transfer_correction":
                        value = parts["source_observation_correction"]
                members[name][key].append(value)
        print(json.dumps({"stage": "label_free_external_prediction", "seed": fit["seed"]}), flush=True)
        del predictor, dataset, daily, parts
        gc.collect()
    with np.load(PREVIOUS/"seed_components.npz", allow_pickle=False) as old:
        arrays = {name: old[name].copy() for name in old.files}
    np.testing.assert_array_equal(arrays["station"], np.asarray(inputs["site_no"], str))
    np.testing.assert_array_equal(arrays["month"], np.asarray(inputs["months"], str))
    calibration = json.loads((PREVIOUS/"source_calibration.json").read_text())
    full_rows = []
    for name in NEW_MODELS:
        point = {key: np.asarray(value) for key, value in members[name].items()}
        arrays.update({f"{name}__{key}": value for key, value in point.items()})
        calibration["procedures"][name] = source_ensemble_policy(np.mean(val_members[name], axis=0),
            validation_cells, validation_truth, n_months=months)
        center = {key: value.mean(0) for key, value in point.items()}
        np.testing.assert_allclose(center["environment_pred"]+center["local_temporal_correction"]
            +center["source_transfer_correction"], center["final_pred"], atol=1e-12, rtol=1e-12)
        lower, upper = empirical_interval(center["final_pred"], calibration["procedures"][name]["primary_k0_log_half_width"])
        full_rows.append(pd.DataFrame({"station": np.repeat(arrays["station"], len(arrays["month"])),
            "month": np.tile(arrays["month"], len(arrays["station"])), "analyte": "doc", "model_name": name,
            **{key: value.ravel() for key, value in center.items() if value.ndim == 2},
            "attention_prior_mass_mean": center["attention_prior_mass"].mean(-1).ravel(),
            "attention_entropy_mean": center["attention_entropy"].mean(-1).ravel(),
            "pi_lower": lower.ravel(), "pi_upper": upper.ravel(),
            "hydro_channels_available": np.asarray(inputs["x_mask"]).sum(-1).ravel(),
            "visibility_role": "unmonitored_external", "support_count": 0}))
    calibration.update(external_DOC_used=False, external_support_used=False,
        new_release="current-source attention", external_exposure=config["external_exposure"])
    np.savez_compressed(root/"seed_components.npz", **arrays)
    write_json(root/"source_calibration.json", calibration)
    write_json(root/"portable_replay.json", replay)
    write_json(root/"retrieval_sources.json", donor_records)
    pd.concat(full_rows, ignore_index=True).to_parquet(root/"full_grid.parquet", index=False)
    pd.concat(source_rows, ignore_index=True).to_parquet(root/"source_validation.parquet", index=False)
    names = ("seed_components.npz", "source_calibration.json", "portable_replay.json",
             "retrieval_sources.json", "full_grid.parquet", "source_validation.parquet")
    write_json(path, {"config": config, "config_hash": digest(config), "external_DOC_read": False,
        "files": {name: sha256_file(root/name) for name in names}})
    return config


def score_points(root, config):
    external = json.loads(EXTERNAL_PROTOCOL.read_text())
    if sha256_file(external["dataset_path"]) != external["dataset_hash"]:
        raise ValueError("external scoring data changed")
    dataset = torch.load(external["dataset_path"], weights_only=False, map_location="cpu")
    truth = np.asarray(dataset["y"], float).ravel()
    cells = np.flatnonzero(np.asarray(dataset["y_mask"], bool))
    names, dates = np.asarray(dataset["site_no"], str), np.asarray(dataset["months"], str)
    months = len(dates)
    schedule, query = support_schedule(cells, months)
    old = pd.read_parquet(PREVIOUS/"predictions.parquet")
    validate_previous_panel(old, cells, query, truth)
    frames = [old]
    calibration = json.loads((root/"source_calibration.json").read_text())["procedures"]
    with np.load(root/"seed_components.npz", allow_pickle=False) as saved:
        np.testing.assert_array_equal(saved["station"], names)
        np.testing.assert_array_equal(saved["month"], dates)
        for name in NEW_MODELS:
            center = saved[f"{name}__final_pred"].mean(0).ravel()
            for population, k, selected in [("all_observed_k0", 0, cells),
                    *[("fixed_query_curve", k, query) for k in (0, 1, 3, 5)]]:
                support = np.sort(np.concatenate([value[:k] for value in schedule.values()]))
                policy = calibration[name]["curve"][str(k)]
                prediction = center[selected] if k == 0 else station_residual_correction(
                    center[selected], selected, center[support], support, truth[support],
                    n_months=months, alpha=policy["selected"]["alpha"])
                width = calibration[name]["primary_k0_log_half_width"] if population == "all_observed_k0" else policy["log_half_width"]
                lower, upper = empirical_interval(prediction, width)
                frames.append(pd.DataFrame({"cell": selected, "station": names[selected//months],
                    "month": dates[selected % months], "analyte": "doc", "basin": "02040104",
                    "model_name": name, "population": population, "k": k, "seed": -1,
                    "y_pred": prediction, "y_true": truth[selected], "pi_lower": lower, "pi_upper": upper,
                    "support_count": k, "visibility_role": "independent_external_test",
                    "support_alpha": policy["selected"]["alpha"] if k else 0.}))
    panel = pd.concat(frames, ignore_index=True)
    if not np.isfinite(panel[["y_pred", "pi_lower", "pi_upper"]]).all().all():
        raise ValueError("nonfinite external products")
    path = root/"predictions.parquet"
    panel.to_parquet(path, index=False)
    scoring_config = {**config, "procedures": list(calibration)}
    write_json(path.with_suffix(".meta.json"), {"config": scoring_config,
        "config_hash": digest(scoring_config), "prediction_sha256": sha256_file(path),
        "point_generation_sha256": sha256_file(root/"point_generation_complete.json"),
        "source_calibration_sha256": sha256_file(root/"source_calibration.json"),
        "dataset_sha256": external["dataset_hash"], "rows": len(panel),
        "external_selection": "availability only", "model_selection": "source validation only",
        "scope": config["external_exposure"], "external_K0_calibration": False})
    return panel


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, default=SOURCE)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--predict-only", action="store_true")
    args = parser.parse_args()
    torch.set_num_threads(2)
    config = generate_points(args.source_root, args.root)
    if not args.predict_only:
        panel = score_points(args.root, config)
        print(panel.groupby(["population", "model_name", "k"]).size().to_string(), flush=True)


if __name__ == "__main__":
    main()
