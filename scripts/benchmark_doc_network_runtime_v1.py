"""Source-covariate-only CPU timing; no DOC labels or target evaluation."""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import torch
from run_doc_fusion_benchmark_development_v1 import context

from river_graph.experiments.doc_network_work import PHYST_RECIPES, PhySTArchitecture
from river_graph.models.river_architecture_comparison import graph_view


def main():
    dataset = torch.load("data/processed/mississippi_graph_graphfix_st357.pt", weights_only=False, map_location="cpu")
    ctx = context(dataset, "1013")
    rows, inputs = ctx["rows"]["train"], ctx["inputs"]
    graph = graph_view(dataset["edge_index"], inputs["order"], rows)
    windows = inputs["windows"][100:132][:, rows]
    environment, season = inputs["environment"][rows], inputs["season"][100:132]
    results = []
    for threads in (1, 2, 4):
        torch.set_num_threads(threads)
        model = PhySTArchitecture("cache/doc_network_work_v1/sources/PhyST-HC/scripts/Functions_DOC.py", PHYST_RECIPES["compact"])
        times = []
        for _ in range(4):
            started = time.monotonic()
            model.train()
            model.zero_grad(set_to_none=True)
            model(windows, environment, season, graph).square().mean().backward()
            model.eval()
            with torch.no_grad():
                model(windows, environment, season, graph)
            times.append(time.monotonic()-started)
        results.append({"threads": threads, "median_train_plus_eval_batch_seconds": float(np.median(times[1:]))})
    destination = Path("cache/doc_network_work_v1/source_only_runtime_benchmark.json")
    destination.write_text(json.dumps({"source_only": True, "doc_labels_read": False, "results": results}, indent=2)+"\n")
    print(json.dumps(results))


if __name__ == "__main__":
    main()
