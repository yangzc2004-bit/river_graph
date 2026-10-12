"""Independently recompute saved DOC benchmark metrics and provenance."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from run_doc_fusion_benchmark_development_v1 import DATASET, MASKS, verify
from run_doc_river_architecture_comparison_v1 import digest, write_json

from river_graph.experiments.doc_domain_work import CANDIDATES, DETERMINISTIC, MODELS
from river_graph.experiments.doc_fusion_benchmark import REGIONS, SEEDS
from river_graph.experiments.provenance import sha256_file

ROOT = Path("experiments/phase4_transfer/doc_domain_work_v1")


def independent_metrics(part):
    y, p = part.y_true.to_numpy(float), part.y_pred.to_numpy(float)
    error = p - y
    squared_total = np.square(y - y.mean()).sum()
    return {"mae": np.abs(error).mean(),
        "log1p_mae": np.abs(np.log1p(p) - np.log1p(y)).mean(),
        "rmse": np.sqrt(np.square(error).mean()),
        "r2": 1 - np.square(error).sum() / squared_total if squared_total else np.nan,
        "bias": error.mean(),
        "station_macro_mae": pd.Series(np.abs(error)).groupby(part.station.to_numpy()).mean().mean(),
        "source_q90_mae": np.abs(error[part.high_doc.to_numpy(bool)]).mean()
            if part.high_doc.any() else np.nan}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    args = parser.parse_args()
    root = args.root.resolve()
    protocol = json.loads((root / "protocol.json").read_text())
    manifest = json.loads((root / "manifest.json").read_text())
    if manifest["protocol_hash"] != digest(protocol) or protocol["candidates"] != CANDIDATES:
        raise ValueError("protocol/candidate hash mismatch")
    if sha256_file("scripts/run_ladder.py") != protocol["original_repository_dispatcher_sha256"]:
        raise ValueError("original dispatcher changed")
    for name, expected in protocol["runtime_snapshot"].items():
        if sha256_file(root / "code_snapshot" / name) != expected:
            raise ValueError("frozen code snapshot changed")
        if name != "scripts/run_ladder.py" and sha256_file(name) != expected:
            raise ValueError("execution source changed after freeze")
    deps = Path("cache/doc_domain_work_v1/python_deps")
    for name, expected in protocol["optional_package_files"].items():
        if sha256_file(deps / name) != expected:
            raise ValueError("optional dependency changed")
    if sha256_file(DATASET) != protocol["dataset_sha256"]:
        raise ValueError("dataset changed")
    data = torch.load(DATASET, weights_only=False, map_location="cpu")
    months = data["y"].shape[1]
    truth = np.asarray(data["y"]).ravel()
    frames, fits = [], 0
    gate = json.loads((root / "all_selections.json").read_text())
    if gate["protocol_hash"] != digest(protocol) or gate["test_used"]:
        raise ValueError("invalid source selection gate")
    for region in REGIONS:
        run = root / "regions" / f"huc4_{region}"
        mask = MASKS / f"huc4_{region}.npz"
        if sha256_file(mask) != protocol["mask_sha256"][region]:
            raise ValueError("mask changed")
        with np.load(mask, allow_pickle=False) as split:
            cells = split["test"].copy()
            train, val = split["train"].copy(), split["val"].copy()
        for a, b in ((train, val), (train, cells), (val, cells)):
            if np.intersect1d(a // months, b // months).size:
                raise ValueError("station roles overlap")
        verify(run, "selection_complete.json")
        if sha256_file(run / "selection_complete.json") != gate["regional_selections"][region]:
            raise ValueError("regional selection changed")
        selection = json.loads((run / "selection.json").read_text())
        if selection["test_used"]:
            raise ValueError("target labels used for selection")
        for relative, expected in selection["fit_receipts"].items():
            fit = run / relative
            if sha256_file(fit / "fit_complete.json") != expected:
                raise ValueError("fit receipt changed")
            verify(fit, "fit_complete.json")
            saved_val = pd.read_parquet(fit / "validation_predictions.parquet")
            np.testing.assert_array_equal(saved_val.cell, val)
            np.testing.assert_array_equal(saved_val.y_true, truth[val])
            if not saved_val.visibility_role.eq("val").all():
                raise ValueError("validation role changed")
            fits += 1
        verify(run, "test_complete.json")
        meta = json.loads((run / "predictions.meta.json").read_text())
        if meta["prediction_sha256"] != sha256_file(run / "predictions.parquet"):
            raise ValueError("prediction sidecar mismatch")
        if meta["config_hash"] != digest(meta["config"]) or meta["all_selections_sha256"] != sha256_file(root / "all_selections.json"):
            raise ValueError("prediction config or selection gate changed")
        frame = pd.read_parquet(run / "predictions.parquet")
        if not np.isfinite(frame[["y_true", "y_pred"]].to_numpy()).all() or (frame.y_pred < 0).any():
            raise ValueError("invalid concentration predictions")
        if not frame.visibility_role.eq("test").all() or not frame.k.eq(0).all():
            raise ValueError("unexpected target context")
        if set(frame.model_name) != set(MODELS) or set(frame.seed) != set(SEEDS):
            raise ValueError("missing model or evaluation seed")
        for name in MODELS:
            for seed in SEEDS:
                part = frame[(frame.model_name == name) & (frame.seed == seed)]
                np.testing.assert_array_equal(part.cell, cells)
                np.testing.assert_array_equal(part.y_true, truth[cells])
                np.testing.assert_array_equal(part.station, np.asarray(data["site_no"], str)[cells // months])
                np.testing.assert_array_equal(part.high_doc, truth[cells] >= np.quantile(truth[train], .9))
            if name in DETERMINISTIC:
                first = frame[(frame.model_name == name) & (frame.seed == SEEDS[0])].y_pred.to_numpy()
                for seed in SEEDS[1:]:
                    np.testing.assert_array_equal(first, frame[(frame.model_name == name) & (frame.seed == seed)].y_pred)
        frames.append(frame)
    if fits != 120:
        raise ValueError("declared fitting budget differs")
    full = pd.concat(frames, ignore_index=True)
    summary = pd.read_csv(root / "analysis/summary.csv").set_index("model_name")
    maximum = 0.
    for name in MODELS:
        part = full[full.model_name == name]
        values = pd.DataFrame([independent_metrics(group) for _, group in part.groupby("seed")]).mean()
        values["basin_macro_mae"] = np.mean([independent_metrics(group)["mae"]
            for _, group in part.groupby(["seed", "target_huc4"])])
        values["basin_macro_log1p_mae"] = np.mean([independent_metrics(group)["log1p_mae"]
            for _, group in part.groupby(["seed", "target_huc4"])])
        for key, value in values.items():
            difference = abs(float(summary.loc[name, key]) - value)
            maximum = max(maximum, difference)
            np.testing.assert_allclose(summary.loc[name, key], value, rtol=0, atol=1e-10, equal_nan=True)
    verify(root / "analysis", "complete.json", {"protocol_hash": digest(protocol)})
    batch = json.loads((root / "batch_complete.json").read_text())
    if batch["analysis_receipt_sha256"] != sha256_file(root / "analysis/complete.json"):
        raise ValueError("batch analysis receipt changed")
    result = {"verified": True, "unique_query_cells": int(full.cell.nunique()),
        "query_stations": int(full.station.nunique()), "unique_candidate_fits": fits,
        "models": list(MODELS), "maximum_independent_metric_difference": maximum,
        "joint_primary": protocol["joint_primary"], "adaptation_status": protocol["adaptation_status"],
        "scope": protocol["scope"], "sota_established": False,
        "sources": {"protocol": sha256_file(root / "protocol.json"),
            "batch": sha256_file(root / "batch_complete.json"),
            "analysis": sha256_file(root / "analysis/complete.json")}}
    write_json(root / "analysis/independent_audit.json", result)
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
