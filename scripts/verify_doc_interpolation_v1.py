"""Independent truth/coverage/metric audit and saved-checkpoint replay."""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
import torch
from run_doc_fusion_benchmark_development_v1 import verify
from run_doc_interpolation_v1 import (
    CODE,
    DATASET,
    GEOMETRY,
    ROOT,
    concentration,
    configuration,
    native_prediction,
    predict,
    task_context,
    tasks,
    write_json,
)
from run_doc_river_architecture_comparison_v1 import digest
from threadpoolctl import threadpool_limits

from river_graph.experiments.provenance import sha256_file
from river_graph.models.river_architecture_comparison import (
    RiverArchitectureModel,
    covariate_inputs,
    graph_view,
)


def independent_metrics(y, p):
    difference = p-y
    return {"mae": np.abs(difference).mean(), "log1p_mae": np.abs(np.log1p(p)-np.log1p(y)).mean(),
            "rmse": np.sqrt(np.mean(difference**2)), "r2": 1-np.sum(difference**2)/np.sum((y-y.mean())**2),
            "bias": difference.mean()}


def neural_replay(directory, protocol, data, rows, cells):
    settings = protocol["settings"]
    inputs = covariate_inputs({k:v for k,v in data.items() if k not in ("y", "y_mask")}, rows)
    assert inputs["normalization"] == json.loads((directory/"normalization.json").read_text())
    graph = graph_view(data["edge_index"], inputs["order"], rows)
    model = RiverArchitectureModel("graph_transformer", **{k:settings[k] for k in ("hidden", "temporal_hidden", "heads", "layers", "dropout")})
    model.load_state_dict(torch.load(directory/"model.pt", weights_only=True, map_location="cpu"))
    info = json.loads((directory/"model_info.json").read_text())
    t = data["y"].shape[1]
    dates = np.unique(cells % t)
    grid = native_prediction(predict(model, inputs, graph, dates, settings["batch_months"]), info["target_normalization"])
    mapping = {int(s):i for i,s in enumerate(rows)}
    return grid[np.searchsorted(dates, cells % t), [mapping[int(c//t)] for c in cells]]


def main():
    protocol = json.loads((ROOT/"protocol.json").read_text())
    assert json.loads((ROOT/"batch_complete.json").read_text())["protocol_hash"] == digest(protocol)
    for name in CODE:
        assert sha256_file(ROOT/"code_snapshot"/name) == protocol["runtime_snapshot"][name]
        if name != "scripts/run_ladder.py":
            assert sha256_file(name) == protocol["runtime_snapshot"][name]
    assert sha256_file("scripts/run_ladder.py") == protocol["original_dispatcher_sha256"]
    assert sha256_file(DATASET) == protocol["dataset_sha256"]
    assert sha256_file(GEOMETRY) == protocol["geometry_sha256"]
    for path, expected in protocol["dependencies"].items():
        assert sha256_file(path) == expected
    gate = json.loads((ROOT/"all_selections.json").read_text())
    data = torch.load(DATASET, weights_only=False, map_location="cpu")
    flat = np.asarray(data["y"]).ravel()
    maximum_replay, candidates, frames = 0., 0, []
    torch.set_num_threads(1)
    with threadpool_limits(limits=1):
        for task in tasks():
            directory = ROOT/"tasks"/task
            assert sha256_file(directory/"selection_complete.json") == gate["selection_hashes"][task]
            verify(directory, "selection_complete.json", configuration(protocol, task))
            verify(directory, "test_complete.json", configuration(protocol, task))
            split, rows, interp = task_context(ROOT, task, data)
            assert not set(split["train"]) & set(split["test"])
            assert not set(split["train"]) & set(split["val"])
            selection = json.loads((directory/"selection.json").read_text())
            assert selection["test_used"] is False
            validation = pd.read_parquet(directory/"validation_predictions.parquet")
            output = pd.read_parquet(directory/"predictions.parquet")
            for (family, recipe_id), part in validation.groupby(["model_name", "recipe_id"]):
                np.testing.assert_array_equal(part.cell, split["val"])
                np.testing.assert_array_equal(part.y_true, flat[split["val"]])
                if family == "fixed_fusion":
                    values = neural_replay(directory, protocol, data, rows, split["val"])
                else:
                    recipe = next(r for r in protocol["recipes"][family] if r["id"] == recipe_id)
                    values, intrinsic, raw = interp.predict(family, recipe, split["val"])
                    np.testing.assert_array_equal(intrinsic, part.intrinsic)
                    np.testing.assert_allclose(raw, part.raw_prediction, equal_nan=True, atol=1e-8)
                maximum_replay = max(maximum_replay, float(np.max(np.abs(values-part.y_pred.to_numpy()))))
                np.testing.assert_allclose(values, part.y_pred, atol=1e-7, rtol=0)
                candidates += 1
            for family, selected in selection["selections"].items():
                records = [r for r in selection["validation_records"] if r["model"] == family]
                winner = min(records, key=lambda r:(r["metrics"]["mae"], r["metrics"]["log1p_mae"], r["recipe"]["id"]))
                assert selected == winner["recipe"]
            for (family, seed), part in output.groupby(["model_name", "seed"]):
                np.testing.assert_array_equal(part.cell, split["test"])
                np.testing.assert_array_equal(part.y_true, flat[split["test"]])
                assert np.isfinite(part.y_pred).all() and (part.y_pred >= 0).all()
                if family in selection["selections"]:
                    values, intrinsic, raw = interp.predict(family, selection["selections"][family], split["test"])
                    np.testing.assert_array_equal(intrinsic, part.intrinsic)
                    np.testing.assert_allclose(raw, part.raw_prediction, equal_nan=True, atol=1e-8)
                    np.testing.assert_allclose(values, part.y_pred, atol=1e-7, rtol=0)
                elif task.startswith("time"):
                    values = neural_replay(directory, protocol, data, rows, split["test"])
                    np.testing.assert_allclose(values, part.y_pred, atol=1e-7, rtol=0)
            if task.startswith("time"):
                grid = np.full((data["y"].shape[1], data["y"].shape[0]), np.nan, dtype=np.float32)
                grid[split["train"] % grid.shape[0], split["train"]//grid.shape[0]] = flat[split["train"]]
                ordered = grid[:, rows]
                source = torch.log1p(torch.from_numpy(ordered[np.isfinite(ordered)]))
                normalization = json.loads((directory/"model_info.json").read_text())["target_normalization"]
                np.testing.assert_allclose([normalization["mean"], normalization["std"]], [float(source.mean()), max(float(source.std()), .1)])
            output["task"] = task
            output["track"] = "time" if task.startswith("time") else "space"
            output["gap_months"] = int(task.split("_")[1][1:]) if task.startswith("time") else 0
            frames.append(output)
    all_predictions = pd.concat(frames, ignore_index=True)
    table = pd.read_csv(ROOT/"analysis/by_seed.csv")
    maximum_metrics = 0.
    for item in table.to_dict("records"):
        part = all_predictions[(all_predictions.track == item["track"]) & (all_predictions.gap_months == item["gap_months"]) &
                               (all_predictions.seed == item["seed"]) & (all_predictions.model_name == item["model_name"])]
        values = independent_metrics(part.y_true.to_numpy(float), part.y_pred.to_numpy(float))
        for name, value in values.items():
            maximum_metrics = max(maximum_metrics, abs(value-item[name]))
        assert len(part) == item["query_count"]
        np.testing.assert_allclose(part.intrinsic.mean(), item["intrinsic_coverage"])
    assert maximum_metrics < 1e-10
    common_table = pd.read_csv(ROOT/"analysis/common_subset_by_seed.csv")
    for (_, gap, seed), group in all_predictions.groupby(["track", "gap_months", "seed"]):
        keys = group.groupby(["task", "cell"]).intrinsic.all()
        keys = set(keys[keys].index)
        for model, part in group.groupby("model_name"):
            part = part[part[["task", "cell"]].apply(tuple, axis=1).isin(keys)]
            row = common_table[(common_table.gap_months == gap) & (common_table.seed == seed) & (common_table.model_name == model)]
            if len(part):
                assert len(row) == 1 and row.iloc[0].query_count == len(part)
                actual = independent_metrics(part.y_true.to_numpy(float), concentration(part.raw_prediction.to_numpy(float)))
                for key, value in actual.items():
                    np.testing.assert_allclose(value, row.iloc[0][key], atol=1e-10)
    verify(ROOT/"analysis", "complete.json", {"protocol_hash": digest(protocol)})
    write_json(ROOT/"analysis/independent_audit.json", {"status": "passed", "validation_replays": candidates,
        "new_fusion_fits": 9, "maximum_replay_difference": maximum_replay, "maximum_metric_difference": maximum_metrics,
        "coverage_and_fallback_verified": True, "common_subset_verified": True, "protocol_hash": digest(protocol)})
    print(json.dumps({"stage": "audit_passed", "validation_replays": candidates, "maximum_metric_difference": maximum_metrics}))


if __name__ == "__main__":
    main()
