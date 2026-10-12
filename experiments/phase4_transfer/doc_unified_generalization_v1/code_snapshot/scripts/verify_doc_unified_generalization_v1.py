"""Independent source/receiving checkpoint and metric audit for transfer v1."""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
import torch
from run_doc_unified_generalization_v1 import (
    DATA,
    ROOT,
    archive,
    cell_prediction,
    configuration,
    context,
    controlled_graph,
    coordinate_policy,
    digest,
    load_model,
    native_prediction,
    portable_inputs,
    predict,
    tasks,
    verify,
    write_json,
)

from river_graph.experiments.provenance import sha256_file


def independent_metrics(part):
    y, p = (part[key].to_numpy(dtype=float) for key in ("y_true", "y_pred"))
    if not np.isfinite(y).all() or not np.isfinite(p).all() or (y < 0).any() or (p < 0).any():
        raise ValueError("invalid DOC query values")
    difference = p-y
    return {"mae": float(np.abs(difference).mean()), "log1p_mae": float(np.abs(np.log1p(p)-np.log1p(y)).mean()),
        "rmse": float(np.sqrt(np.square(difference).mean())),
        "r2": float(1-np.square(difference).sum()/np.square(y-y.mean()).sum()), "bias": float(difference.mean()),
        "station_macro_mae": float(pd.DataFrame({"station": part.station, "error": np.abs(difference)}).groupby("station").error.mean().mean())}


def main():
    root = ROOT
    protocol = json.loads((root/"protocol.json").read_text())
    manifest = json.loads((root/"manifest.json").read_text())
    if digest(protocol) != manifest["protocol_hash"]:
        raise ValueError("protocol manifest mismatch")
    for name, expected in protocol["runtime_snapshot"].items():
        if sha256_file(root/"code_snapshot"/name) != expected or (name != "scripts/run_ladder.py" and sha256_file(name) != expected):
            raise ValueError("optimization execution code changed")
    if sha256_file(DATA) != protocol["dataset_sha256"] or sha256_file(root/"source_domains.csv") != protocol["source_domains_sha256"]:
        raise ValueError("source data/domains changed")
    data = torch.load(DATA, weights_only=False, map_location="cpu")
    selection = json.loads((root/"all_selections.json").read_text())
    fields = json.loads((root/"all_fields.json").read_text())
    if fields["selection_gate_sha256"] != sha256_file(root/"all_selections.json") or selection["selected_model_sha256"] != sha256_file(root/"selected_model.json"):
        raise ValueError("source-only choice gate changed")
    detail = pd.read_csv(root/"analysis/by_fit.csv")
    receiving, labels = {}, {}
    for population in ("development", "confirmation"):
        if sha256_file(protocol[population]["inputs"]) != protocol[population]["input_sha256"] or sha256_file(protocol[population]["labels"]) != protocol[population]["labels_sha256"]:
            raise ValueError("receiving data changed")
        receiving[population] = archive(protocol[population]["inputs"])
        raw = torch.load(protocol[population]["labels"], weights_only=False, map_location="cpu")
        if population == "confirmation":
            with np.load(root/"confirmation/eligibility.npz") as eligibility:
                chosen = eligibility["original_rows"]
            raw = {**raw, "y": np.asarray(raw["y"])[chosen], "y_mask": np.asarray(raw["y_mask"])[chosen],
                "site_no": np.asarray(raw["site_no"])[chosen].tolist()}
        labels[population] = raw
    torch.set_num_threads(2)
    replay_error, metric_error, receipts, role_checks = 0., 0., 0, 0
    for task in tasks():
        directory = root/"fits"/task["id"]
        config = configuration(protocol, task)
        if selection["fit_receipts"][task["id"]] != sha256_file(directory/"selection_complete.json"):
            raise ValueError("fit changed after source selection")
        verify(directory, "selection_complete.json", config)
        verify(directory, "test_complete.json", config)
        receipts += 2
        model, info = load_model(directory, task, protocol)
        if sum(p.numel() for p in model.parameters()) != protocol["parameters"][task["recipe"]]:
            raise ValueError("incorrect candidate capacity")
        ctx = context(root, task, data)
        norm = json.loads((directory/"normalization.json").read_text())
        if norm != ctx["inputs"]["normalization"]:
            raise ValueError("receiving/source normalization mismatch")
        logs = np.log1p(np.asarray(data["y"]).ravel()[ctx["split"]["train"]])
        if abs(float(logs.mean())-info["target_normalization"]["mean"]) > 1e-5:
            raise ValueError("target normalization not source-only")
        for role, name in (("val", "validation"), ("test", "source")):
            part = pd.read_parquet(directory/f"{name}_predictions.parquet")
            if not np.array_equal(part.cell, ctx["split"][role]) or not np.array_equal(part.y_true, np.asarray(data["y"]).ravel()[part.cell]):
                raise ValueError("source role labels changed")
            values = cell_prediction(model, ctx, role, info["target_normalization"], protocol["settings"]["batch_months"])
            replay_error = max(replay_error, float(np.abs(values-part.y_pred.to_numpy()).max()))
            role_checks += 1
        for population in ("development", "confirmation"):
            if fields["fields"][f"{task['id']}:{population}"] != sha256_file(directory/f"{population}_field_complete.json"):
                raise ValueError("receiving field changed after blind gate")
            verify(directory, f"{population}_field_complete.json")
            receipts += 1
            raw = receiving[population]
            inputs = coordinate_policy(portable_inputs(raw, norm), task["recipe"])
            graph = controlled_graph(raw["edge_index"], inputs["order"], np.arange(len(raw["site_no"])), "direction_heads")
            grid = native_prediction(predict(model, inputs, graph, np.arange(len(raw["months"])),
                protocol["settings"]["batch_months"]), info["target_normalization"]).T
            with np.load(directory/f"{population}_field.npz") as saved:
                replay_error = max(replay_error, float(np.abs(grid-saved["y_pred"]).max()))
            part = pd.read_parquet(directory/f"{population}_predictions.parquet")
            cells = np.flatnonzero(np.asarray(labels[population]["y_mask"]).ravel())
            if not np.array_equal(part.cell, cells) or not np.array_equal(part.y_true, np.asarray(labels[population]["y"]).ravel()[cells]):
                raise ValueError("receiving query cohort changed")
            replay_error = max(replay_error, float(np.abs(grid.ravel()[cells]-part.y_pred.to_numpy()).max()))
        for population in ("source", "development", "confirmation"):
            part = pd.read_parquet(directory/f"{population}_predictions.parquet")
            row = detail[(detail.id == task["id"]) & (detail.population == population)].iloc[0]
            for key, value in independent_metrics(part).items():
                metric_error = max(metric_error, abs(value-float(row[key])))
            metadata = json.loads((directory/f"{population}_predictions.meta.json").read_text())
            if sha256_file(directory/f"{population}_predictions.parquet") != metadata["prediction_sha256"]:
                raise ValueError("prediction provenance mismatch")
        print(json.dumps({"replayed": task["id"]}), flush=True)
    if replay_error > 1e-6 or metric_error > 1e-10:
        raise ValueError(f"prediction/metric mismatch {replay_error}/{metric_error}")
    independently_grouped = detail.groupby(["population", "recipe"], as_index=False).mean(numeric_only=True)
    summary = pd.read_csv(root/"analysis/summary.csv")
    joined = independently_grouped.merge(summary, on=["population", "recipe"], suffixes=("_own", "_table"), validate="one_to_one")
    for key in ("mae", "log1p_mae", "rmse", "r2", "bias", "station_macro_mae"):
        if float(np.abs(joined[key+"_own"]-joined[key+"_table"]).max()) > 1e-10:
            raise ValueError("summary aggregation mismatch")
    write_json(root/"analysis/independent_audit.json", {"passed": True, "protocol_hash": digest(protocol),
        "fits": 63, "receipts_verified": receipts, "source_role_checks": role_checks,
        "prediction_replay_max_abs": replay_error, "metric_recompute_max_abs": metric_error,
        "confirmation_doc_used_for_fit_or_selection": False,
        "files": {name: sha256_file(root/name) for name in ("analysis/by_fit.csv", "analysis/summary.csv", "selected_model.json",
            "all_selections.json", "all_fields.json")}})
    print(json.dumps({"passed": True, "replay_error": replay_error, "metric_error": metric_error}))


if __name__ == "__main__":
    main()
