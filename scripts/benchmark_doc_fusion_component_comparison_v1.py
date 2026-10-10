"""Serial equal-shape inference of one region's frozen component checkpoints."""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from run_doc_fusion_component_comparison_v1 import (
    DATASET,
    ROOT,
    load_model,
    verify_package,
)
from run_doc_river_architecture_comparison_v1 import predict, write_json

from river_graph.experiments.provenance import sha256_file
from river_graph.models.river_architecture_comparison import (
    covariate_inputs,
    graph_view,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--region", default="1013")
    args = parser.parse_args()
    if any(not (args.root / "regions" / f"huc4_{region}" / "complete.json").exists()
           for region in json.loads((args.root / "protocol.json").read_text())["regions"]):
        raise ValueError("finish all training before serial benchmarking")
    torch.set_num_threads(1)
    run = args.root / "regions" / f"huc4_{args.region}"
    selection = json.loads((run / "selections.json").read_text())
    roles = json.loads((run / "node_roles.json").read_text())
    data = torch.load(DATASET, weights_only=False, map_location="cpu")
    n, t = data["y"].shape
    inputs = covariate_inputs({key: value for key, value in data.items() if key not in ("y", "y_mask")},
                              roles["train"], lookback=12)
    graph = graph_view(data["edge_index"], inputs["order"], np.arange(n))
    records, cached = [], {}
    for alias, key in selection["aliases"].items():
        if key not in cached:
            directory = run / "fits" / key / "seed42"
            verify_package(directory, "fit_complete.json")
            model = load_model(directory)
            predict(model, inputs, graph, np.arange(t), 32)
            durations = []
            for _ in range(3):
                started = time.perf_counter()
                predict(model, inputs, graph, np.arange(t), 32)
                durations.append(time.perf_counter() - started)
            cached[key] = {"median_seconds": float(np.median(durations)), "passes_seconds": durations,
                "checkpoint_sha256": sha256_file(directory / "checkpoint.pt"),
                "parameters": sum(value.numel() for value in model.parameters())}
            del model
        records.append({"model_name": alias, "fit_specification": key, **cached[key]})
        print(json.dumps(records[-1]), flush=True)
    write_json(args.root / "analysis/serial_inference_timing.json", {
        "region": args.region, "seed": 42, "station_nodes": n, "months": t, "grid_outputs": n * t,
        "torch_threads": 1, "timed_passes": 3, "warmups": 1, "records": records,
        "scope": "equal-shape all-station inference only; feature assembly, graph construction and I/O excluded; "
                 "one representative region's selected parent modules; station grid is not an all-reach field",
        "script_sha256": sha256_file(__file__)})


if __name__ == "__main__":
    main()
