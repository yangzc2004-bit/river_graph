"""Fixed-checkpoint time-recipe comparison with current query covariates hidden.

Unlike all-history erasure, non-query historical measurements remain visible.
Query-month erasures also propagate consistently into subsequent windows.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from analyze_doc_fusion_component_comparison_v1 import interval, metrics
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
from river_graph.models.doc_fusion_comparison import TEMPORAL_CHOICES
from river_graph.models.river_architecture_comparison import (
    covariate_inputs,
    graph_view,
)

SCENARIOS = ("available_covariates", "query_current_temperature_hidden", "query_current_hydrology_hidden")


def query_masked_inputs(data, cells, scenario):
    result = {key: value for key, value in data.items() if key not in ("y", "y_mask")}
    result["x_mask"] = np.asarray(data["x_mask"], dtype=bool).copy()
    t = result["x_mask"].shape[1]
    rows, months = cells // t, cells % t
    if scenario == SCENARIOS[1]:
        result["x_mask"][rows, months, 0] = False
    elif scenario == SCENARIOS[2]:
        result["x_mask"][rows, months, :] = False
    elif scenario != SCENARIOS[0]:
        raise ValueError("unknown current-input intervention")
    return result


def freeze(root):
    parent = json.loads((root / "protocol.json").read_text())
    code = ("scripts/evaluate_doc_time_current_missing_v1.py", "tests/test_doc_time_current_missing.py",
            "scripts/analyze_doc_fusion_component_comparison_v1.py")
    protocol = {"parent_protocol_hash": digest(parent), "models": [f"time__{name}" for name in TEMPORAL_CHOICES],
        "scenarios": list(SCENARIOS), "regions": parent["regions"], "seeds": parent["seeds"],
        "dataset_hash": parent["dataset_hash"], "mask_hashes": parent["mask_hashes"],
        "runtime_snapshot": {name: sha256_file(name) for name in code},
        "parent_modules": "same validation-selected environment per outer fold; concatenation fusion; spatial trunk fixed",
        "intervention": "erase temperature, or both dynamic channels, at all target DOC query months; "
                        "non-query measurements remain; erased query months also disappear from later history; "
                        "rebuild ages/windows; source context and normalization unchanged",
        "comparisons": "all four historical recipes vs current-only MLP, within each availability scenario; "
                       "native cell-pooled seed-average MAE and exploratory paired five-basin bootstrap",
        "selection_changes": False, "additional_training_fits": 0,
        "scope": "additional controlled input-missing diagnostic, declared before pooled test scores; "
                 "not an exact simulator of naturally missing DOC months or a same-operator causal ablation"}
    out = root / "time_current_missing_v1"
    destination = out / "protocol.json"
    if sha256_file(DATASET) != protocol["dataset_hash"]:
        raise ValueError("parent dataset changed")
    for region, expected in protocol["mask_hashes"].items():
        if sha256_file(MASKS / f"huc4_{region}.npz") != expected:
            raise ValueError("parent mask changed")
    if destination.exists():
        if json.loads(destination.read_text()) != protocol:
            raise ValueError("current-input diagnostic changed; create a new version")
        for name, expected in protocol["runtime_snapshot"].items():
            if sha256_file(out / "code_snapshot" / name) != expected:
                raise ValueError("frozen diagnostic code copy changed")
    else:
        write_json(destination, protocol)
        for name in code:
            path = out / "code_snapshot" / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(Path(name).read_bytes())
    return parent, protocol, out


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--freeze-only", action="store_true")
    args = parser.parse_args()
    parent, protocol, out = freeze(args.root)
    if args.freeze_only:
        print(json.dumps({"protocol_hash": digest(protocol), "additional_training_fits": 0}), flush=True)
        return
    if (out / "complete.json").exists():
        record = verify_package(out, "complete.json")
        if record["protocol_hash"] != digest(protocol):
            raise ValueError("diagnostic completion has different identity")
        meta = json.loads((out / "predictions.meta.json").read_text())
        for name, expected in meta["source_hashes"].items():
            if sha256_file(name) != expected:
                raise ValueError("diagnostic checkpoint or selection changed")
        print("Verified completed current-input diagnostic", flush=True)
        return
    torch.set_num_threads(parent["settings"]["torch_threads"])
    data = torch.load(DATASET, weights_only=False, map_location="cpu")
    t, frames, receipts = data["y"].shape[1], [], {}
    for region in parent["regions"]:
        run = args.root / "regions" / f"huc4_{region}"
        verify_package(run, "complete.json")
        selection = json.loads((run / "selections.json").read_text())
        with np.load(MASKS / f"huc4_{region}.npz", allow_pickle=False) as split:
            source, target = np.unique(split["train"] // t), np.unique(split["test"] // t)
            cells = split["test"].copy()
        original = pd.read_parquet(run / "predictions.parquet")
        scenarios = {name: covariate_inputs(query_masked_inputs(data, cells, name), source,
                                            lookback=parent["settings"]["lookback"]) for name in SCENARIOS}
        for name in SCENARIOS:
            if scenarios[name]["normalization"] != scenarios[SCENARIOS[0]]["normalization"]:
                raise ValueError("source normalization changed")
        graph = graph_view(data["edge_index"], scenarios[SCENARIOS[0]]["order"], np.sort(np.r_[source, target]))
        graph_rows = np.searchsorted(graph["rows"].numpy(), cells // t)
        for name in protocol["models"]:
            key = selection["aliases"][name]
            for seed in protocol["seeds"]:
                directory = run / "fits" / key / f"seed{seed}"
                verify_package(directory, "fit_complete.json")
                model = load_model(directory)
                info = json.loads((directory / "info.json").read_text())
                part = original[(original.model_name == name) & (original.seed == seed)].copy()
                np.testing.assert_array_equal(part.cell, cells)
                np.testing.assert_array_equal(part.y_true, np.asarray(data["y"]).ravel()[cells])
                for scenario in SCENARIOS:
                    grid = native_prediction(predict(model, scenarios[scenario], graph, np.arange(t),
                        parent["settings"]["batch_months"]), info["target_normalization"])
                    prediction = grid[cells % t, graph_rows]
                    if scenario == SCENARIOS[0]:
                        np.testing.assert_allclose(prediction, part.y_pred, atol=1e-6, rtol=1e-6)
                    result = part.copy()
                    result["y_pred"], result["input_scenario"] = prediction, scenario
                    frames.append(result)
                receipts[str(directory / "checkpoint.pt")] = sha256_file(directory / "checkpoint.pt")
                del model
        receipts[str(run / "selections.json")] = sha256_file(run / "selections.json")
        print(json.dumps({"current_input_diagnostic_complete": region}), flush=True)
    frame = pd.concat(frames, ignore_index=True)
    if frame.duplicated(["input_scenario", "model_name", "seed", "cell"]).any():
        raise ValueError("duplicate diagnostic query")
    frame.to_parquet(out / "predictions.parquet", index=False)
    config = {"protocol_hash": digest(protocol), "parent_protocol_hash": digest(parent)}
    write_json(out / "predictions.meta.json", {"config": config, "config_hash": digest(config),
        "dataset_sha256": protocol["dataset_hash"], "mask_sha256": protocol["mask_hashes"],
        "runtime_snapshot_hash": digest(protocol["runtime_snapshot"]), "source_hashes": receipts,
        "prediction_sha256": sha256_file(out / "predictions.parquet"), "rows": len(frame), "coverage": 1.,
        "selection_changed": False, "full_grid_saved": False})
    summary = pd.DataFrame([{"input_scenario": scenario, "model_name": name, **metrics(part)}
        for (scenario, name), part in frame.groupby(["input_scenario", "model_name"])])
    summary.to_csv(out / "summary.csv", index=False)
    pairs = []
    for scenario, part in frame.groupby("input_scenario"):
        scores = summary[summary.input_scenario == scenario].set_index("model_name").mae
        for name in protocol["models"]:
            if name == "time__current":
                continue
            pairs.append({"input_scenario": scenario, "baseline": "time__current", "candidate": name,
                "gain_percent": 100 * (scores["time__current"] - scores[name]) / scores["time__current"],
                "basin_bootstrap": interval(part, "time__current", name)})
    write_json(out / "paired_comparisons.json", pairs)
    write_json(out / "complete.json", {"protocol_hash": digest(protocol), "files": {
        name: sha256_file(out / name) for name in ("protocol.json", "predictions.parquet", "predictions.meta.json",
                                                "summary.csv", "paired_comparisons.json")}})
    print(summary[["input_scenario", "model_name", "mae"]].to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
