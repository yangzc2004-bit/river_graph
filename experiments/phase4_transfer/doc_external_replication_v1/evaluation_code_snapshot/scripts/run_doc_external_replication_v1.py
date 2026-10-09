"""Apply the fixed portable source ensemble to independent DOC stations.

Source fitting must be complete first. Save label-free point predictions and
source-validation calibration before opening external DOC values for scoring.
"""
from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from portable_doc_reconstructor_v1 import PROCEDURES, PortableDOCReconstructor
from run_doc_portable_source_fit_v1 import ROOT as SOURCE_ROOT
from run_unified_doc_spatial import digest, write_json

from river_graph.experiments.provenance import sha256_file
from river_graph.experiments.spatial_fewshot import support_schedule
from river_graph.models.station_adapted_hybrid import station_residual_correction

ROOT = Path("experiments/phase4_transfer/doc_external_replication_v1")
EXTERNAL_PROTOCOL = Path("experiments/phase4_transfer/doc_unmonitored_tasks_v1/external_protocol_v1.json")
COMPONENTS = ("environment_pred", "local_temporal_correction", "source_transfer_correction",
              "river_correction", "native_pred", "final_pred")


def log_residual_quantile(prediction, truth, probability=.9):
    """Finite-rank empirical validation calibration; no coverage guarantee."""
    prediction, truth = np.asarray(prediction, float), np.asarray(truth, float)
    if (prediction.shape != truth.shape or prediction.ndim != 1 or not len(truth)
            or not np.isfinite(prediction).all() or not np.isfinite(truth).all()
            or (prediction < 0).any() or (truth < 0).any() or not 0 < probability < 1):
        raise ValueError("calibration requires aligned finite nonnegative validation values")
    residual = np.abs(np.log1p(prediction)-np.log1p(truth))
    rank = min(len(residual), int(np.ceil((len(residual)+1)*probability)))
    return float(np.sort(residual)[rank-1])


def source_ensemble_policy(prediction, cells, truth, *, n_months):
    """Choose support alpha and log intervals from source validation only."""
    prediction, cells, truth = np.asarray(prediction), np.asarray(cells), np.asarray(truth)
    if (prediction.shape != cells.shape or truth.shape != cells.shape or cells.ndim != 1
            or cells.dtype.kind not in "iu" or len(np.unique(cells)) != len(cells)):
        raise ValueError("source ensemble arrays must have unique aligned cell identities")
    schedule, query = support_schedule(cells, n_months)
    lookup = {int(cell): i for i, cell in enumerate(cells)}
    qp = np.array([lookup[int(cell)] for cell in query])
    policy = {"primary_k0_log_half_width": log_residual_quantile(prediction, truth),
              "validation_cells": len(cells), "validation_stations": len(schedule), "curve": {}}
    for k in (0, 1, 3, 5):
        support = np.sort(np.concatenate([value[:k] for value in schedule.values()]))
        sp = np.array([lookup[int(cell)] for cell in support], dtype=np.int64)
        choices = []
        for alpha in ((0.,) if k == 0 else (0., .25, .5, .75, 1.)):
            calibrated = station_residual_correction(prediction[qp], query, prediction[sp], support,
                truth[sp], n_months=n_months, alpha=alpha)
            choices.append({"alpha": alpha, "validation_mae": float(np.abs(calibrated-truth[qp]).mean())})
        selected = min(choices, key=lambda value: (value["validation_mae"], value["alpha"]))
        calibrated = station_residual_correction(prediction[qp], query, prediction[sp], support,
            truth[sp], n_months=n_months, alpha=selected["alpha"])
        policy["curve"][str(k)] = {"selected": selected, "candidates": choices,
            "log_half_width": log_residual_quantile(calibrated, truth[qp]), "query_cells": len(query)}
    return policy


def empirical_interval(prediction, half_width):
    center = np.log1p(np.asarray(prediction, float))
    if (not np.isfinite(center).all() or np.any(np.asarray(prediction) < 0)
            or not np.isfinite(half_width) or half_width < 0):
        raise ValueError("interval center and width must be finite and nonnegative")
    lower = np.maximum(0, np.expm1(center-half_width))
    upper = np.expm1(center+half_width)
    if not np.isfinite(upper).all():
        raise ValueError("interval overflow; preserve the scientific failure")
    return lower, upper


def source_inputs(dataset, ids, daily):
    inputs = {key: np.asarray(dataset[key])[ids] for key in ("site_no", "x", "x_mask", "static", "regime")}
    inputs.update(months=np.asarray(dataset["months"]), daily_features=daily[ids])
    return inputs


def generate_points(source_root, root):
    """Generate every point estimate without opening external DOC labels."""
    protocol = json.loads((source_root/"deployment_protocol.json").read_text())
    external = json.loads(EXTERNAL_PROTOCOL.read_text())
    if (protocol["external_huc8"] != external["selected_huc8"]
            or protocol["procedures"] != list(PROCEDURES)
            or protocol["primary_procedure"] != "unmonitored_integrated"):
        raise ValueError("external and source deployment definitions disagree")
    root.mkdir(parents=True, exist_ok=True)
    runs = [source_root/"runs"/f"seed{seed}" for seed in protocol["seeds"]]
    if not all((run/"complete.json").exists() for run in runs):
        raise RuntimeError("all five fixed source fits must finish before external replication")
    if sha256_file(external["label_free_inputs"]) != external["input_hash"]:
        raise ValueError("external label-free inputs changed")
    with np.load(external["label_free_inputs"], allow_pickle=False) as saved:
        inputs = {key: saved[key].copy() for key in ("site_no", "months", "x", "x_mask", "static", "regime")}
        inputs["daily_features"] = saved["daily"].copy()
    config = {"source_root": str(source_root), "source_protocol_hash": sha256_file(source_root/"deployment_protocol.json"),
        "source_completions": {str(run/"complete.json"): sha256_file(run/"complete.json") for run in runs},
        "external_protocol_hash": sha256_file(EXTERNAL_PROTOCOL), "inputs_hash": external["input_hash"],
        "external_scoring_hash": external["dataset_hash"], "seeds": protocol["seeds"],
        "procedures": list(PROCEDURES), "point_ensemble": protocol["ensemble"],
        "runner_sha256": sha256_file(__file__),
        "portable_sha256": sha256_file("scripts/portable_doc_reconstructor_v1.py")}
    path = root/"point_generation_complete.json"
    if path.exists():
        previous = json.loads(path.read_text())
        if previous["config"] != config:
            raise ValueError("saved external point execution changed; preserve it and use a new version")
        for name, expected in previous["files"].items():
            if sha256_file(root/name) != expected:
                raise ValueError("saved external prediction product changed")
        return config
    all_points, validation, replay, donor_records = {}, {}, [], []
    validation_cells = validation_truth = None
    for procedure in PROCEDURES:
        point_components, val_members = {key: [] for key in COMPONENTS}, []
        for run in runs:
            predictor = PortableDOCReconstructor.from_fitted_run(run, procedure=procedure)
            source_config = json.loads((run/"config.json").read_text())
            dataset = torch.load(source_config["dataset_path"], weights_only=False, map_location="cpu")
            with np.load(source_config["mask_path"], allow_pickle=False) as saved:
                cells = saved["val"].copy()
            with np.load(source_config["daily_path"], allow_pickle=False) as saved:
                daily = saved["full"].copy()
            months = dataset["y"].shape[1]
            ids = np.unique(cells//months)
            receiving = source_inputs(dataset, ids, daily)
            predicted = predictor.predict(receiving)
            values = predicted[np.searchsorted(ids, cells//months), cells % months]
            original = pd.read_parquet(run/"predictions.parquet")
            original = original[original.model_name.eq(procedure)]
            np.testing.assert_array_equal(original.cell, cells)
            np.testing.assert_allclose(values, original.y_pred, rtol=1e-6, atol=1e-6)
            if validation_cells is not None:
                np.testing.assert_array_equal(cells, validation_cells)
                np.testing.assert_array_equal(original.y_true, validation_truth)
            validation_cells, validation_truth = cells, original.y_true.to_numpy()
            val_members.append(values)
            parts = predictor.predict_components(inputs)
            donor_records.extend({"run": run.name, "procedure": procedure, **record}
                                 for record in parts["retrieval_sources"])
            for key in COMPONENTS:
                point_components[key].append(parts[key])
            replay.append({"run": run.name, "procedure": procedure, "source_validation_cells": len(cells),
                "max_replay_difference": float(np.max(np.abs(values-original.y_pred.to_numpy())))})
            print(json.dumps({"stage": "label_free_external_prediction", "run": run.name, "procedure": procedure}), flush=True)
            del predictor, dataset, daily
            gc.collect()
        validation[procedure] = source_ensemble_policy(np.mean(val_members, axis=0),
            validation_cells, validation_truth, n_months=months)
        for key, members in point_components.items():
            all_points[f"{procedure}__{key}"] = np.asarray(members)
    np.savez_compressed(root/"seed_components.npz", **all_points,
                        station=np.asarray(inputs["site_no"], str), month=np.asarray(inputs["months"], str))
    write_json(root/"source_calibration.json", {"definition": "source-val-only empirical log1p residual calibration",
        "ensemble": protocol["ensemble"], "procedures": validation,
        "external_DOC_used": False, "external_support_used": False})
    write_json(root/"portable_replay.json", replay)
    write_json(root/"retrieval_sources.json", donor_records)
    station, month = np.asarray(inputs["site_no"], str), np.asarray(inputs["months"], str)
    full_rows = []
    for procedure in PROCEDURES:
        point = {key: all_points[f"{procedure}__{key}"].mean(0) for key in COMPONENTS}
        lower, upper = empirical_interval(point["final_pred"], validation[procedure]["primary_k0_log_half_width"])
        full_rows.append(pd.DataFrame({"station": np.repeat(station, len(month)),
            "month": np.tile(month, len(station)), "analyte": "doc", "model_name": procedure,
            **{key: values.ravel() for key, values in point.items()},
            "pi_lower": lower.ravel(), "pi_upper": upper.ravel(),
            "hydro_channels_available": np.asarray(inputs["x_mask"]).sum(-1).ravel(),
            "visibility_role": "unmonitored_external", "support_count": 0}))
    pd.concat(full_rows, ignore_index=True).to_parquet(root/"full_grid.parquet", index=False)
    names = ("seed_components.npz", "source_calibration.json", "portable_replay.json",
             "retrieval_sources.json", "full_grid.parquet")
    write_json(path, {"config": config, "config_hash": digest(config),
        "files": {name: sha256_file(root/name) for name in names}, "external_DOC_read": False})
    return config


def score_points(root, config):
    """Read external truth after point predictions and source choices are saved."""
    external = json.loads(EXTERNAL_PROTOCOL.read_text())
    if sha256_file(external["dataset_path"]) != external["dataset_hash"]:
        raise ValueError("external scoring dataset changed")
    dataset = torch.load(external["dataset_path"], weights_only=False, map_location="cpu")
    truth = np.asarray(dataset["y"], float).ravel()
    cells = np.flatnonzero(np.asarray(dataset["y_mask"], bool))
    names, dates = np.asarray(dataset["site_no"], str), np.asarray(dataset["months"], str)
    months = len(dates)
    schedule, query = support_schedule(cells, months)
    source = json.loads((root/"source_calibration.json").read_text())["procedures"]
    rows = []
    with np.load(root/"seed_components.npz", allow_pickle=False) as saved:
        np.testing.assert_array_equal(saved["station"], names)
        np.testing.assert_array_equal(saved["month"], dates)
        for name in PROCEDURES:
            members = saved[f"{name}__final_pred"]
            center = members.mean(0).ravel()
            for population, k, selected in [("all_observed_k0", 0, cells),
                    *[("fixed_query_curve", k, query) for k in (0, 1, 3, 5)]]:
                support = np.sort(np.concatenate([value[:k] for value in schedule.values()]))
                policy = source[name]["curve"][str(k)]
                prediction = (center[selected] if k == 0 else station_residual_correction(
                    center[selected], selected, center[support], support, truth[support],
                    n_months=months, alpha=policy["selected"]["alpha"]))
                half_width = (source[name]["primary_k0_log_half_width"] if population == "all_observed_k0"
                              else policy["log_half_width"])
                lower, upper = empirical_interval(prediction, half_width)
                rows.append(pd.DataFrame({"cell": selected, "station": names[selected//months],
                    "month": dates[selected % months], "analyte": "doc", "basin": "02040104",
                    "model_name": name, "population": population, "k": k, "seed": -1,
                    "y_pred": prediction, "y_true": truth[selected], "pi_lower": lower, "pi_upper": upper,
                    "support_count": k, "visibility_role": "independent_external_test",
                    "support_alpha": policy["selected"]["alpha"] if k else 0.}))
    path = root/"predictions.parquet"
    panel = pd.concat(rows, ignore_index=True)
    if not np.isfinite(panel[["y_pred", "pi_lower", "pi_upper"]]).all().all():
        raise ValueError("nonfinite external prediction or interval")
    panel.to_parquet(path, index=False)
    write_json(path.with_suffix(".meta.json"), {"config": config, "config_hash": digest(config),
        "prediction_sha256": sha256_file(path), "point_generation_sha256": sha256_file(root/"point_generation_complete.json"),
        "source_calibration_sha256": sha256_file(root/"source_calibration.json"),
        "dataset_sha256": external["dataset_hash"], "rows": len(panel),
        "external_selection": "availability only", "model_selection": "source validation only",
        "scope": "independent02040104 DOC validation; no external K0 calibration"})
    return panel


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, default=SOURCE_ROOT)
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
