"""Serial CPU inference timing at the same 357-station by 654-month shape."""
from __future__ import annotations

import json
import time

import numpy as np
import torch
from run_doc_river_architecture_comparison_v1 import DATASET, ROOT, predict, write_json

from river_graph.experiments.provenance import sha256_file
from river_graph.models.river_architecture_comparison import (
    ARMS,
    RiverArchitectureModel,
    covariate_inputs,
    graph_view,
)


def main():
    if not (ROOT / "batch_complete.json").exists():
        raise ValueError("finish all training before the serial timing benchmark")
    protocol = json.loads((ROOT / "protocol.json").read_text())
    torch.set_num_threads(protocol["settings"]["torch_threads"])
    data = torch.load(DATASET, weights_only=False, map_location="cpu")
    n, t = data["y"].shape
    run = ROOT / "runs/huc4_1013_seed42"
    normalization = json.loads((run / "normalization.json").read_text())
    inputs = covariate_inputs({key: value for key, value in data.items() if key not in ("y", "y_mask")},
                              normalization["source_rows"])
    graph = graph_view(data["edge_index"], inputs["order"], np.arange(n))
    records = []
    for arm in ARMS:
        checkpoint = run / f"{arm}.pt"
        payload = torch.load(checkpoint, weights_only=False, map_location="cpu")
        model = RiverArchitectureModel(arm, **payload["architecture"])
        model.load_state_dict(payload["state"])
        predict(model, inputs, graph, np.arange(t), protocol["settings"]["batch_months"])
        durations = []
        for _ in range(5):
            started = time.monotonic()
            predict(model, inputs, graph, np.arange(t), protocol["settings"]["batch_months"])
            durations.append(time.monotonic() - started)
        records.append({"model_name": arm, "seconds": durations, "median_seconds": float(np.median(durations)),
                        "nodes": n, "months": t, "predicted_cells": n * t,
                        "checkpoint_sha256": sha256_file(checkpoint)})
    write_json(ROOT / "analysis/serial_inference_timing.json", {"records": records,
        "device": "CPU", "threads": torch.get_num_threads(), "repeats": 5, "warmup_passes": 1,
        "script_sha256": sha256_file(__file__), "dataset_sha256": sha256_file(DATASET),
        "purpose": "equal-shape computation timing only; includes known station nodes, not all physical reaches"})
    for row in records:
        print(json.dumps(row), flush=True)


if __name__ == "__main__":
    main()
