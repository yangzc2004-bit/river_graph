"""Generate a complete monthly station DOC field using the source-selected model.

Input NPZ keys: site_no[N], months[T] (consecutive YYYY-MM), static[N,2]
(latitude, longitude), regime[N,13] (the canonical source schema), x[N,T,2]
(water temperature in degC, discharge in ft3/s), x_mask[N,T,2],
edge_index[2,E] (upstream->downstream).
DOC observations are deliberately not accepted as reconstruction inputs.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from run_doc_unified_transfer_v2 import (
    ROOT,
    controlled_graph,
    digest,
    load_model,
    native_prediction,
    portable_inputs,
    predict,
    verify,
    write_json,
)

from river_graph.experiments.provenance import sha256_file


def reconstruct(root, archive_path, output):
    protocol = json.loads((root/"protocol.json").read_text())
    choice = json.loads((root/"selected_model.json").read_text())
    gate = json.loads((root/"all_selections.json").read_text())
    if gate["protocol_hash"] != digest(protocol) or gate["selected_model_sha256"] != sha256_file(root/"selected_model.json"):
        raise ValueError("architecture choice changed after freeze")
    task = next(t for t in protocol["tasks"] if t["id"] == choice["primary_checkpoint_id"])
    directory = root/"fits"/task["id"]
    if gate["fit_receipts"][task["id"]] != sha256_file(directory/"selection_complete.json"):
        raise ValueError("fit altered after selection freeze")
    verify(directory, "selection_complete.json")
    core = ("src/river_graph/models/unified_river_doc.py", "src/river_graph/models/river_relation_comparison.py",
            "src/river_graph/models/river_architecture_comparison.py", "scripts/run_doc_unified_transfer_v2.py",
            "scripts/run_doc_river_architecture_comparison_v1.py", "scripts/run_doc_fusion_benchmark_development_v1.py")
    for name in core:
        if sha256_file(name) != protocol["runtime_snapshot"][name]:
            raise ValueError(f"trained model execution code changed: {name}")
    config = {"protocol_hash": digest(protocol), "selected_model_sha256": sha256_file(root/"selected_model.json"),
        "checkpoint": task["id"], "model_sha256": sha256_file(directory/"model.pt"),
        "fit_receipt_sha256": sha256_file(directory/"selection_complete.json"), "info_sha256": sha256_file(directory/"info.json"),
        "runtime_snapshot_hash": digest(protocol["runtime_snapshot"]),
        "normalization_sha256": sha256_file(directory/"normalization.json"), "inputs_sha256": sha256_file(archive_path),
        "inference_code_sha256": sha256_file(__file__)}
    metadata_path = output.with_suffix(".meta.json")
    if output.exists() or metadata_path.exists():
        if not output.exists() or not metadata_path.exists():
            raise ValueError("incomplete existing reconstruction; use a fresh output path")
        previous = json.loads(metadata_path.read_text())
        if previous["config_hash"] != digest(config) or previous["prediction_sha256"] != sha256_file(output):
            raise ValueError("existing reconstruction differs; use a fresh output path")
        return previous
    required = {"site_no", "months", "static", "regime", "x", "x_mask", "edge_index"}
    with np.load(archive_path) as saved:
        if set(saved.files) != required:
            raise ValueError(f"exact label-free input schema required: {sorted(required)}")
        data = {k: saved[k].copy() for k in required}
    n, t = len(data["site_no"]), len(data["months"])
    if data["x"].shape != (n, t, 2) or data["x_mask"].shape != (n, t, 2) or data["regime"].shape != (n, 13) or data["static"].shape != (n, 2):
        raise ValueError("inconsistent receiving input dimensions")
    if len(np.unique(data["site_no"])) != n or not n or not t:
        raise ValueError("nonempty unique station identities and calendar required")
    months = pd.PeriodIndex(data["months"].astype(str), freq="M")
    if np.any(np.diff(months.asi8) != 1):
        raise ValueError("include every calendar month; missing covariates use x_mask")
    data["feature_channels"] = ["temperature", "discharge"]
    normalization = json.loads((directory/"normalization.json").read_text())
    inputs = portable_inputs(data, normalization)
    graph = controlled_graph(data["edge_index"], inputs["order"], np.arange(n), task["arm"])
    model, info = load_model(directory, task, protocol)
    torch.set_num_threads(protocol["settings"]["torch_threads"])
    grid = native_prediction(predict(model, inputs, graph, np.arange(t),
        protocol["settings"]["batch_months"]), info["target_normalization"]).T
    if not np.isfinite(grid).all():
        raise ValueError("nonfinite DOC field")
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output, y_pred=grid, site_no=data["site_no"], months=data["months"])
    report = {"config": config, "config_hash": digest(config), "prediction_sha256": sha256_file(output),
        "shape": list(grid.shape), "units": "mg/L", "receiving_doc_used": False,
        "resolution": "monthly at provided stations, not a validated arbitrary continuous-space/time field",
        "architecture": choice["selected_arm"], "source_selected": True, "ensemble": False}
    write_json(metadata_path, report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--inputs", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.suffix != ".npz":
        parser.error("--output must end in .npz")
    print(json.dumps(reconstruct(args.root, args.inputs, args.output), indent=2))


if __name__ == "__main__":
    main()
