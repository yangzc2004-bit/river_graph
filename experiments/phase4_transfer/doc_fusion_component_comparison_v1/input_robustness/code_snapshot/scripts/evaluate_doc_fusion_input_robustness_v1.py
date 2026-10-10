"""Predeclared additional inference diagnostics; never change component selections."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from run_doc_fusion_component_comparison_v1 import (
    DATASET,
    MASKS,
    ROOT,
    load_model,
    verify_package,
)
from run_doc_river_architecture_comparison_v1 import (
    digest,
    native_prediction,
    predict,
    write_json,
)

from river_graph.experiments.provenance import sha256_file
from river_graph.models.river_architecture_comparison import (
    covariate_inputs,
    graph_view,
)

SCENARIOS = ("available_covariates", "target_temperature_hidden", "target_dynamic_hydrology_hidden")
ALIASES = ("selected__reference", "selected__fusion")


def masked_covariates(data, target_rows, scenario):
    """No DOC values/visibility passed into the covariate interface."""
    result = {key: value for key, value in data.items() if key not in ("y", "y_mask")}
    result["x_mask"] = np.asarray(data["x_mask"], dtype=bool).copy()
    if scenario == "target_temperature_hidden":
        result["x_mask"][target_rows, :, 0] = False
    elif scenario == "target_dynamic_hydrology_hidden":
        result["x_mask"][target_rows] = False
    elif scenario != "available_covariates":
        raise ValueError("unknown input diagnostic")
    return result


def freeze(root):
    parent = json.loads((root / "protocol.json").read_text())
    if sha256_file(DATASET) != parent["dataset_hash"]:
        raise ValueError("parent dataset changed")
    for region, expected in parent["mask_hashes"].items():
        if sha256_file(MASKS / f"huc4_{region}.npz") != expected:
            raise ValueError("parent mask changed")
    directory = root / "input_robustness"
    code = ("scripts/evaluate_doc_fusion_input_robustness_v1.py", "tests/test_doc_fusion_input_robustness.py",
            "src/river_graph/models/doc_fusion_comparison.py",
            "src/river_graph/models/river_architecture_comparison.py")
    protocol = {"parent_protocol_hash": digest(parent), "scenarios": list(SCENARIOS), "models": list(ALIASES),
        "regions": parent["regions"], "seeds": parent["seeds"], "dataset_hash": parent["dataset_hash"],
        "mask_hashes": parent["mask_hashes"], "runtime_snapshot": {name: sha256_file(name) for name in code},
        "model_selection": "reuse frozen validation-selected final pipeline and fixed MLP/GRU/concat reference",
        "intervention": "hide selected target dynamic measurement masks for all months; rebuild windows/ages; "
                        "source context, static features, season and source normalization remain available",
        "endpoint": "observed query native DOC MAE averaged over seed errors, descriptive additional diagnostic",
        "scope": "controlled missing-input stress test; not an exact simulator of naturally missing DOC months; "
                 "no retraining, new architecture selection or changes to parent primary endpoints"}
    destination = directory / "protocol.json"
    if destination.exists():
        if json.loads(destination.read_text()) != protocol:
            raise ValueError("frozen input diagnostic changed; use a new version")
        for name, expected in protocol["runtime_snapshot"].items():
            if sha256_file(directory / "code_snapshot" / name) != expected:
                raise ValueError("frozen input diagnostic code copy changed")
    else:
        write_json(destination, protocol)
        for name in code:
            path = directory / "code_snapshot" / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(Path(name).read_bytes())
    return parent, protocol


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--freeze-only", action="store_true")
    args = parser.parse_args()
    parent, protocol = freeze(args.root)
    out = args.root / "input_robustness"
    if args.freeze_only:
        print(json.dumps({"protocol_hash": digest(protocol), "additional_training_fits": 0}), flush=True)
        return
    torch.set_num_threads(parent["settings"]["torch_threads"])
    data = torch.load(DATASET, weights_only=False, map_location="cpu")
    t = data["y"].shape[1]
    frames, receipts = [], {}
    for region in parent["regions"]:
        run = args.root / "regions" / f"huc4_{region}"
        verify_package(run, "complete.json")
        selection = json.loads((run / "selections.json").read_text())
        with np.load(MASKS / f"huc4_{region}.npz", allow_pickle=False) as split:
            source, target = np.unique(split["train"] // t), np.unique(split["test"] // t)
            cells = split["test"].copy()
        original = pd.read_parquet(run / "predictions.parquet")
        inputs = {scenario: covariate_inputs(masked_covariates(data, target, scenario), source,
                                             lookback=parent["settings"]["lookback"]) for scenario in SCENARIOS}
        for scenario in SCENARIOS:
            if inputs[scenario]["normalization"] != inputs[SCENARIOS[0]]["normalization"]:
                raise ValueError("target masking changed source normalization")
        graph = graph_view(data["edge_index"], inputs[SCENARIOS[0]]["order"], np.sort(np.r_[source, target]))
        graph_rows = np.searchsorted(graph["rows"].numpy(), cells // t)
        for alias in ALIASES:
            key = selection["aliases"][alias]
            for seed in parent["seeds"]:
                directory = run / "fits" / key / f"seed{seed}"
                verify_package(directory, "fit_complete.json")
                verify_package(directory, "test_complete.json")
                model = load_model(directory)
                info = json.loads((directory / "info.json").read_text())
                part = original[(original.model_name == alias) & (original.seed == seed)].copy()
                np.testing.assert_array_equal(part.cell, cells)
                for scenario in SCENARIOS:
                    grid = native_prediction(predict(model, inputs[scenario], graph, np.arange(t),
                        parent["settings"]["batch_months"]), info["target_normalization"])
                    predicted = grid[cells % t, graph_rows]
                    if scenario == SCENARIOS[0]:
                        np.testing.assert_allclose(predicted, part.y_pred, atol=1e-6, rtol=1e-6)
                    result = part.copy()
                    result["y_pred"] = predicted
                    result["input_scenario"] = scenario
                    frames.append(result)
                receipts[str(directory / "checkpoint.pt")] = sha256_file(directory / "checkpoint.pt")
                del model
        receipts[str(run / "selections.json")] = sha256_file(run / "selections.json")
        print(json.dumps({"diagnostic_region_complete": region}), flush=True)
    frame = pd.concat(frames, ignore_index=True)
    if frame.duplicated(["input_scenario", "model_name", "seed", "cell"]).any():
        raise ValueError("duplicate diagnostic queries")
    frame.to_parquet(out / "predictions.parquet", index=False)
    config = {"diagnostic_protocol_hash": digest(protocol), "parent_protocol_hash": digest(parent)}
    write_json(out / "predictions.meta.json", {"config": config, "config_hash": digest(config),
        "dataset_sha256": protocol["dataset_hash"], "mask_sha256": protocol["mask_hashes"],
        "runtime_snapshot_hash": digest(protocol["runtime_snapshot"]),
        "prediction_sha256": sha256_file(out / "predictions.parquet"),
        "checkpoint_and_selection_hashes": receipts, "observed_query_coverage": 1., "rows": len(frame),
        "changes_component_choices": False, "full_grid_saved": False})
    scores = []
    for (scenario, alias), part in frame.groupby(["input_scenario", "model_name"]):
        error = part.y_pred.to_numpy(dtype=np.float64) - part.y_true.to_numpy(dtype=np.float64)
        scores.append({"input_scenario": scenario, "model_name": alias,
            "mae": float(np.abs(error).mean()), "rmse": float(np.sqrt(np.square(error).mean())),
            "cells": int(part.cell.nunique()), "stations": int(part.station.nunique())})
    table = pd.DataFrame(scores)
    table.to_csv(out / "summary.csv", index=False)
    per_basin = frame.assign(error=lambda part: np.abs(part.y_pred - part.y_true)).groupby(
        ["target_huc4", "input_scenario", "model_name"], as_index=False).error.mean()
    per_basin.rename(columns={"error": "mae"}).to_csv(out / "per_basin_metrics.csv", index=False)
    write_json(out / "complete.json", {"diagnostic_protocol_hash": digest(protocol),
        "files": {name: sha256_file(out / name) for name in
                  ("protocol.json", "predictions.parquet", "predictions.meta.json", "summary.csv", "per_basin_metrics.csv")}})
    print(table.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
