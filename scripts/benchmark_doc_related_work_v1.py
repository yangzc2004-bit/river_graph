"""Synthetic timing and official-source verification; never open DOC test labels."""
from __future__ import annotations

import argparse
import json
import time

import numpy as np
import torch

from river_graph.models.doc_related_work import (
    RECIPES,
    build_related_model,
    official_file_hashes,
)
from river_graph.models.river_architecture_comparison import graph_view


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--nodes", type=int, default=250)
    parser.add_argument("--threads", type=int, default=2)
    args = parser.parse_args()
    torch.set_num_threads(args.threads)
    n = args.nodes
    graph = graph_view(np.array([np.arange(n - 1), np.arange(1, n)]), np.ones(n), np.arange(n))
    graph["adjacency"] = (graph["relation"] == 1).float()
    windows = torch.randn(8, n, 12, 8)
    windows[..., 2:4] = 1.
    env, target = torch.randn(n, 15), torch.randn(8, n)
    visible = torch.ones_like(target, dtype=torch.bool)
    print(json.dumps({"sources": official_file_hashes()}), flush=True)
    for method, recipes in RECIPES.items():
        for name, recipe in recipes.items():
            model = build_related_model(method, recipe)
            started = time.monotonic()
            loss = model.objective(windows, env, target, visible, graph)
            loss.backward()
            print(json.dumps({"method": method, "recipe": name,
                "synthetic_nodes": n, "batch_months": 8, "step_seconds": time.monotonic() - started,
                "parameters": sum(p.numel() for p in model.parameters()), "finite_loss": bool(torch.isfinite(loss))}), flush=True)


if __name__ == "__main__":
    main()
