"""Matched-tensor information removals for the fixed DOC fusion model."""
from __future__ import annotations

import torch

PRIMARY = ("no_environment", "no_time", "no_river")
CONTROLS = ("local_no_messages", "current_only", "no_relations")
ARMS = ("fixed_fusion", *PRIMARY, *CONTROLS)
RIVER_COLUMNS = (0, 1, 2, 3)
DEFINITIONS = {
    "fixed_fusion": "Unchanged environmental MLP + 12-month GRU + dense graph Transformer",
    "no_environment": "Zero all 15 normalized static inputs; retain temporal inputs and river relations",
    "no_time": "Zero all 12x8 temporal inputs and the separate current seasonal pair; retain environment/network",
    "no_river": "Self-only attention and zero stream order/log drainage area/slope/monitoring-headwater inputs; retain ecological/climate/coordinate inputs",
    "local_no_messages": "Self-only attention; retain all 15 static descriptors and the full temporal window",
    "current_only": "Zero the first 11 temporal steps and both current age channels; retain current values/masks/season and separate seasonal pair",
    "no_relations": "Keep dense cross-node attention and all covariates; replace river connectivity categories with generic self/unconnected categories",
}


def model_arm(arm):
    if arm not in ARMS:
        raise ValueError("undeclared information ablation")
    return "local" if arm in ("no_river", "local_no_messages") else "graph_transformer"


def ablate_inputs(inputs, arm):
    model_arm(arm)
    out = dict(inputs)
    for name in ("windows", "environment", "season"):
        out[name] = inputs[name].clone()
    if out["environment"].shape[1] != 15 or out["windows"].shape[-2:] != (12, 8):
        raise ValueError("frozen 15-static/12x8-time input schema required")
    if arm == "no_environment":
        out["environment"].zero_()
    elif arm == "no_time":
        out["windows"].zero_()
        out["season"].zero_()
    elif arm == "no_river":
        out["environment"][:, RIVER_COLUMNS] = 0.
    elif arm == "current_only":
        out["windows"][:, :, :-1].zero_()
        out["windows"][:, :, -1, 4:6] = 0.
    return out


def ablate_graph(graph, arm):
    model_arm(arm)
    if arm not in ("no_river", "local_no_messages", "no_relations"):
        return graph
    out = dict(graph)
    n = len(graph["rows"])
    out["relation"] = torch.full((n, n), 5, dtype=torch.long)
    out["relation"].fill_diagonal_(0)
    out["edge_count"] = 0
    out["neighbours"] = torch.arange(n)[:, None]
    out["valid"] = torch.ones((n, 1), dtype=torch.bool)
    out["groups"] = [torch.arange(n)]
    return out
