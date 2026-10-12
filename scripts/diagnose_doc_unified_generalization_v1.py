"""Frozen-weight sensitivity tests for the already inspected Delaware case."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from run_doc_fusion_benchmark_development_v1 import bind, save_predictions, verify
from run_doc_unified_transfer_v2 import (
    DATA,
    context,
    controlled_graph,
    digest,
    external_archive,
    frame,
    load_model,
    metrics,
    native_prediction,
    portable_inputs,
    predict,
    write_json,
)
from run_doc_unified_transfer_v2 import (
    ROOT as PREVIOUS,
)

from river_graph.experiments.provenance import sha256_file

ROOT = Path("experiments/phase4_transfer/doc_unified_generalization_v1/diagnosis")
CASES = ("reference", "no_cross", "no_global", "no_directional", "coords_neutral", "eco_neutral", "hydro_missing", "source_context")
ENV_NAMES = ("stream_order", "log_area", "slope", "station_headwater", "forest", "crop", "urban", "wetland",
             "precipitation", "log_climate_temperature", "soil_organic_matter", "elevation", "baseflow_index", "latitude", "longitude")


def changed_graph(graph, case):
    result = dict(graph)
    allowed = graph["allowed"].clone()
    identity = torch.eye(allowed.shape[1], dtype=torch.bool)
    if case == "no_cross":
        allowed[:] = identity
    elif case == "no_global":
        allowed[2] = identity
    elif case == "no_directional":
        allowed[:2] = identity
    result["allowed"] = allowed
    return result


def joined_context(source_inputs, source_graph, source_months, target_inputs, target_graph):
    a, b = len(source_graph["rows"]), len(target_graph["rows"])
    relation = torch.full((a+b, a+b), 5, dtype=torch.long)
    relation[:a, :a], relation[a:, a:] = source_graph["relation"], target_graph["relation"]
    relation.fill_diagonal_(0)
    graph = {"rows": torch.arange(a+b), "relation": relation,
        "allowed": torch.stack([relation <= 2, (relation == 0) | (relation == 3) | (relation == 4), torch.ones_like(relation, dtype=torch.bool)])}
    rows = source_graph["rows"]
    inputs = {"windows": torch.cat([source_inputs["windows"][source_months][:, rows], target_inputs["windows"]], 1),
        "environment": torch.cat([source_inputs["environment"][rows], target_inputs["environment"]], 0),
        "season": target_inputs["season"]}
    return inputs, graph, a


def main():
    ROOT.mkdir(parents=True, exist_ok=True)
    old = json.loads((PREVIOUS/"protocol.json").read_text())
    tasks = [t for t in old["tasks"] if t["arm"] == "direction_heads"]
    protocol = {"cases": list(CASES), "tasks": tasks, "purpose": "inference sensitivity only, no fitting; Delaware is development data",
        "parent_protocol_sha256": sha256_file(PREVIOUS/"protocol.json"), "parent_audit_sha256": sha256_file(PREVIOUS/"analysis/independent_audit.json"),
        "source_sha256": sha256_file(DATA), "development_labels_sha256": old["external_labels_sha256"],
        "development_inputs_sha256": old["external_input_sha256"], "script_sha256": sha256_file(__file__),
        "interpretation": "input/mask interventions expose sensitivity, not causal biogeochemical attribution; no independent external confirmation"}
    if (ROOT/"protocol.json").exists() and json.loads((ROOT/"protocol.json").read_text()) != protocol:
        raise ValueError("diagnostic freeze changed")
    write_json(ROOT/"protocol.json", protocol)
    data = torch.load(DATA, weights_only=False, map_location="cpu")
    target = external_archive(PREVIOUS)
    labels = torch.load(old["external_labels"], weights_only=False, map_location="cpu")
    cells = np.flatnonzero(np.asarray(labels["y_mask"]).ravel())
    times = np.unique(cells % len(target["months"]))
    source_dates = np.asarray(data["months"], str)
    source_months = np.searchsorted(source_dates, target["months"])
    if not np.array_equal(source_dates[source_months], target["months"]):
        raise ValueError("receiving calendar does not align with source context")
    torch.set_num_threads(2)
    scores, profiles, hydro = [], [], []
    for task in tasks:
        source_directory = PREVIOUS/"fits"/task["id"]
        verify(source_directory, "selection_complete.json")
        model, info = load_model(source_directory, task, old)
        ctx = context(PREVIOUS, task, data)
        original = portable_inputs(target, ctx["inputs"]["normalization"])
        target_graph = controlled_graph(target["edge_index"], original["order"], np.arange(len(target["site_no"])), "direction_heads")
        source_raw = np.column_stack([np.asarray(data["regime"]), np.asarray(data["static"])])[ctx["rows"]["train"]]
        target_raw = np.column_stack([target["regime"], target["static"]])
        for index, name in enumerate(ENV_NAMES):
            lo, hi = np.nanmin(source_raw[:, index]), np.nanmax(source_raw[:, index])
            state = ctx["inputs"]["normalization"]["environment"]
            profiles.append({"split": task["split"], "seed": task["seed"], "feature": name,
                "source_median": float(np.nanmedian(source_raw[:, index])), "target_median": float(np.nanmedian(target_raw[:, index])),
                "outside_source_range": float(((target_raw[:, index] < lo) | (target_raw[:, index] > hi)).mean()),
                "robust_median_shift": float((np.nanmedian(target_raw[:, index])-state["median"][index])/state["iqr"][index]),
                "source_sentinel_minus1": float((source_raw[:, index] == -1).mean()),
                "target_sentinel_minus1": float((target_raw[:, index] == -1).mean())})
        source_mask = np.asarray(data["x_mask"])
        source_query = ctx["split"]["train"]
        for index, name in enumerate(("temperature", "discharge")):
            hydro.append({"split": task["split"], "seed": task["seed"], "channel": name,
                "source_grid_availability": float(source_mask[ctx["rows"]["train"], :, index].mean()),
                "target_grid_availability": float(target["x_mask"][..., index].mean()),
                "source_doc_availability": float(source_mask.reshape(-1, 2)[source_query, index].mean()),
                "target_doc_availability": float(target["x_mask"].reshape(-1, 2)[cells, index].mean())})
        for case in CASES:
            directory = ROOT/task["id"]/case
            config = {"protocol_hash": digest(protocol), "model_sha256": sha256_file(source_directory/"model.pt"),
                "info_sha256": sha256_file(source_directory/"info.json"), "normalization_sha256": sha256_file(source_directory/"normalization.json"),
                "runtime_snapshot_hash": digest(old["runtime_snapshot"]), "dataset_sha256": old["external_labels_sha256"],
                "mask_sha256": digest(cells.tolist()), "task": task["id"], "case": case}
            if not (directory/"complete.json").exists():
                directory.mkdir(parents=True, exist_ok=True)
                inputs = dict(original)
                graph = changed_graph(target_graph, case)
                offset = 0
                if case in ("coords_neutral", "eco_neutral"):
                    inputs["environment"] = original["environment"].clone()
                    if case == "coords_neutral":
                        inputs["environment"][:, 13:15] = 0
                    else:
                        inputs["environment"][:, 4:13] = 0
                elif case == "hydro_missing":
                    masked = {**target, "x_mask": np.zeros_like(target["x_mask"], dtype=bool)}
                    inputs = portable_inputs(masked, ctx["inputs"]["normalization"])
                elif case == "source_context":
                    inputs, graph, offset = joined_context(ctx["inputs"], ctx["graphs"]["train"], source_months, original, target_graph)
                grid = native_prediction(predict(model, inputs, graph, times, 32), info["target_normalization"])
                values = grid[np.searchsorted(times, cells % len(target["months"])), offset+cells//len(target["months"])]
                part = frame(labels, cells, values, task, "development_external")
                part["case"] = case
                save_predictions(directory, "predictions", part, config)
                bind(directory, "complete.json", config, ["predictions.parquet", "predictions.meta.json"])
            verify(directory, "complete.json", config)
            part = pd.read_parquet(directory/"predictions.parquet")
            scores.append({**task, "case": case, "domain": "development_external", **metrics(part)})
        # Same source weights, role labels and preprocessing: only remove the
        # context that was present in the previous source test.
        source_cells = ctx["split"]["test"]
        source_times = np.unique(source_cells % ctx["months"])
        graph = controlled_graph(data["edge_index"], ctx["inputs"]["order"], ctx["rows"]["test"], "direction_heads")
        grid = native_prediction(predict(model, ctx["inputs"], graph, source_times, 32), info["target_normalization"])
        values = grid[np.searchsorted(source_times, source_cells % ctx["months"]), np.searchsorted(graph["rows"].numpy(), source_cells//ctx["months"])]
        part = frame(data, source_cells, values, task, "source_target_only")
        directory = ROOT/task["id"]/"source_target_only"
        directory.mkdir(parents=True, exist_ok=True)
        config = {"protocol_hash": digest(protocol), "dataset_sha256": old["dataset_sha256"], "mask_sha256": old["mask_sha256"][str(task["split"])],
            "runtime_snapshot_hash": digest(old["runtime_snapshot"]), "model_sha256": sha256_file(source_directory/"model.pt"), "case": "source_target_only"}
        if not (directory/"complete.json").exists():
            save_predictions(directory, "predictions", part, config)
            bind(directory, "complete.json", config, ["predictions.parquet", "predictions.meta.json"])
        verify(directory, "complete.json", config)
        scores.append({**task, "case": "source_target_only", "domain": "source", **metrics(part)})
        original_source = pd.read_parquet(source_directory/"source_predictions.parquet")
        scores.append({**task, "case": "source_with_context", "domain": "source", **metrics(original_source)})
        print(json.dumps({"diagnosed": task["id"]}), flush=True)
    detail = pd.DataFrame(scores)
    detail.to_csv(ROOT/"by_fit.csv", index=False)
    detail.groupby(["domain", "case"], as_index=False).mean(numeric_only=True).to_csv(ROOT/"summary.csv", index=False)
    pd.DataFrame(profiles).groupby("feature", as_index=False).mean(numeric_only=True).to_csv(ROOT/"environment_shift.csv", index=False)
    pd.DataFrame(hydro).groupby("channel", as_index=False).mean(numeric_only=True).to_csv(ROOT/"hydro_shift.csv", index=False)
    bind(ROOT, "complete.json", {"protocol_hash": digest(protocol)}, ["by_fit.csv", "summary.csv", "environment_shift.csv", "hydro_shift.csv"])
    print((ROOT/"summary.csv").read_text())


if __name__ == "__main__":
    main()
