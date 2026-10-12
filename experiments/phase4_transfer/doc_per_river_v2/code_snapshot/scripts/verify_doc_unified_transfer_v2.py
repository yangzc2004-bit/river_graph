"""Independent all-checkpoint replay, metrics and paired station resampling."""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
import torch
from run_doc_unified_transfer_v2 import (
    DATA,
    ROOT,
    cell_prediction,
    configuration,
    context,
    controlled_graph,
    digest,
    external_archive,
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
    y, p = (part[name].to_numpy(dtype=np.float64) for name in ("y_true", "y_pred"))
    if not np.isfinite(y).all() or not np.isfinite(p).all() or (y < 0).any() or (p < 0).any():
        raise ValueError("invalid DOC predictions/labels")
    error = p-y
    return {"mae": float(np.abs(error).mean()), "log1p_mae": float(np.abs(np.log1p(p)-np.log1p(y)).mean()),
        "rmse": float(np.sqrt(np.square(error).mean())),
        "r2": float(1-np.square(error).sum()/np.square(y-y.mean()).sum())}


def paired_interval(root, domain, arm, reference, metric, draws=3000):
    frames = {}
    for split in (142, 143, 144):
        for seed in (42, 43, 44):
            parts = [pd.read_parquet(root/"fits"/f"split{split}_{model}_s{seed}"/f"{domain}_predictions.parquet")
                     for model in (arm, reference)]
            if not np.array_equal(parts[0].cell, parts[1].cell) or not np.array_equal(parts[0].y_true, parts[1].y_true):
                raise ValueError("unmatched paired queries")
            errors = []
            for part in parts:
                p, y = part.y_pred.to_numpy(dtype=float), part.y_true.to_numpy(dtype=float)
                errors.append(np.abs(p-y) if metric == "mae" else np.abs(np.log1p(p)-np.log1p(y)))
            frames[(split, seed)] = pd.DataFrame({"station": parts[0].station,
                "difference": errors[0]-errors[1], "one": 1}).groupby("station").agg({"difference": "sum", "one": "sum"})
    stations = sorted(set().union(*(set(frame.index) for frame in frames.values())))
    differences = np.stack([f.reindex(stations, fill_value=0).difference.to_numpy() for f in frames.values()])
    counts = np.stack([f.reindex(stations, fill_value=0).one.to_numpy() for f in frames.values()])
    rng = np.random.default_rng(71551)
    samples = []
    for _ in range(draws):
        ids = rng.integers(len(stations), size=len(stations))
        denominator = counts[:, ids].sum(1)
        if np.any(denominator == 0):
            continue
        samples.append(float((differences[:, ids].sum(1)/denominator).mean()))
    return {"domain": domain, "arm": arm, "reference": reference, "metric": metric,
        "delta": float((differences.sum(1)/counts.sum(1)).mean()),
        "ci95": np.quantile(samples, [.025, .975]).tolist(), "station_blocks": len(stations),
        "draws": len(samples), "scope": "exploratory paired station blocks in fixed datasets; not independent river replication or multiplicity-adjusted inference"}


def main():
    root = ROOT
    protocol = json.loads((root/"protocol.json").read_text())
    manifest = json.loads((root/"manifest.json").read_text())
    if digest(protocol) != manifest["protocol_hash"]:
        raise ValueError("manifest/protocol mismatch")
    for name, expected in protocol["runtime_snapshot"].items():
        if sha256_file(root/"code_snapshot"/name) != expected:
            raise ValueError("snapshot changed")
        if name != "scripts/run_ladder.py" and sha256_file(name) != expected:
            raise ValueError("live replay code changed")
    data = torch.load(DATA, weights_only=False, map_location="cpu")
    external = external_archive(root)
    gates = json.loads((root/"all_selections.json").read_text())
    fields = json.loads((root/"all_fields.json").read_text())
    if gates["selected_model_sha256"] != sha256_file(root/"selected_model.json") or fields["selection_gate_sha256"] != sha256_file(root/"all_selections.json"):
        raise ValueError("choice/selections were altered after prediction")
    external_labels = torch.load(protocol["external_labels"], weights_only=False, map_location="cpu")
    if sha256_file(DATA) != protocol["dataset_sha256"] or sha256_file(protocol["external_labels"]) != protocol["external_labels_sha256"]:
        raise ValueError("datasets changed")
    torch.set_num_threads(protocol["settings"]["torch_threads"])
    detail = pd.read_csv(root/"analysis/by_fit.csv")
    replay_error, metric_error, receipts, role_checks = 0., 0., 0, 0
    for task in tasks():
        directory = root/"fits"/task["id"]
        config = configuration(protocol, task)
        if gates["fit_receipts"][task["id"]] != sha256_file(directory/"selection_complete.json") or fields["fields"][task["id"]] != sha256_file(directory/"field_complete.json"):
            raise ValueError("fit/field gate changed")
        for receipt in ("selection_complete.json", "field_complete.json", "test_complete.json"):
            verify(directory, receipt, config)
            receipts += 1
        ctx = context(root, task, data)
        normalization = json.loads((directory/"normalization.json").read_text())
        if normalization != ctx["inputs"]["normalization"]:
            raise ValueError("normalization/source mismatch")
        model, info = load_model(directory, task, protocol)
        for role, name in (("val", "validation"), ("test", "source")):
            part = pd.read_parquet(directory/f"{name}_predictions.parquet")
            if not np.array_equal(part.cell, ctx["split"][role]) or not np.array_equal(part.y_true, np.asarray(data["y"]).ravel()[part.cell]):
                raise ValueError("source query/label mismatch")
            replay = cell_prediction(model, ctx, role, info["target_normalization"], protocol["settings"]["batch_months"])
            replay_error = max(replay_error, float(np.abs(replay-part.y_pred.to_numpy()).max()))
            role_checks += 1
        inputs = portable_inputs(external, normalization)
        graph = controlled_graph(external["edge_index"], inputs["order"], np.arange(len(external["site_no"])), task["arm"])
        replay_grid = native_prediction(predict(model, inputs, graph, np.arange(len(external["months"])),
            protocol["settings"]["batch_months"]), info["target_normalization"]).T
        with np.load(directory/"external_field.npz") as archive:
            replay_error = max(replay_error, float(np.abs(replay_grid-archive["y_pred"]).max()))
        for name, truth in (("source", data), ("external", external_labels)):
            part = pd.read_parquet(directory/f"{name}_predictions.parquet")
            cells = np.flatnonzero(np.asarray(truth["y_mask"]).ravel()) if name == "external" else ctx["split"]["test"]
            if not np.array_equal(part.cell, cells) or not np.array_equal(part.y_true, np.asarray(truth["y"]).ravel()[cells]):
                raise ValueError("evaluation queries mismatch")
            metadata = json.loads((directory/f"{name}_predictions.meta.json").read_text())
            if metadata["prediction_sha256"] != sha256_file(directory/f"{name}_predictions.parquet"):
                raise ValueError("prediction provenance mismatch")
            expected_dataset = protocol["dataset_sha256"] if name == "source" else protocol["external_labels_sha256"]
            if metadata["dataset_sha256"] != expected_dataset:
                raise ValueError("wrong prediction dataset")
            if name == "external":
                replay_error = max(replay_error, float(np.abs(replay_grid.ravel()[cells]-part.y_pred.to_numpy()).max()))
            row = detail[(detail.id == task["id"]) & (detail.domain == part.domain.iloc[0])].iloc[0]
            for key, score in independent_metrics(part).items():
                metric_error = max(metric_error, abs(score-float(row[key])))
        print(json.dumps({"replayed": task["id"]}), flush=True)
    if replay_error > 1e-6 or metric_error > 1e-10:
        raise ValueError(f"replay/metrics diverged: {replay_error}/{metric_error}")
    intervals = [paired_interval(root, domain, "direction_heads", reference, metric)
        for domain in ("source", "external") for reference in ("self_only", "generic_dense", "permuted_river")
        for metric in ("mae", "log1p_mae")]
    write_json(root/"analysis/paired_intervals.json", intervals)
    diagnostics = {"source_stations": len(data["site_no"]), "receiving_stations": len(external["site_no"]),
        "receiving_doc_cells": int(np.asarray(external_labels["y_mask"]).sum()), "receiving_graph_edges": external["edge_index"].shape[1],
        "receiving_doc_mean": float(np.asarray(external_labels["y"])[np.asarray(external_labels["y_mask"], bool)].mean()),
        "receiving_doc_median": float(np.median(np.asarray(external_labels["y"])[np.asarray(external_labels["y_mask"], bool)])),
        "station_overlap": int(np.intersect1d(np.asarray(data["site_no"], str), external["site_no"]).size),
        "receiving_hydro_at_doc": {key: float(external["x_mask"][np.asarray(external_labels["y_mask"], bool), i].mean())
            for i, key in enumerate(("temperature", "discharge"))},
        "primary_checkpoint": json.loads((root/"selected_model.json").read_text())["primary_checkpoint_id"]}
    write_json(root/"analysis/diagnostics.json", diagnostics)
    report = {"passed": True, "protocol_hash": digest(protocol), "fit_count": 36, "receipts_verified": receipts,
        "source_label_role_checks": role_checks, "prediction_replay_max_abs": replay_error, "metric_recompute_max_abs": metric_error,
        "source_only_selection_and_zero_external_support": True,
        "files": {name: sha256_file(root/name) for name in ("analysis/by_fit.csv", "analysis/summary.csv", "analysis/paired_intervals.json",
            "analysis/diagnostics.json", "selected_model.json", "all_selections.json", "all_fields.json")}}
    write_json(root/"analysis/independent_audit.json", report)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
