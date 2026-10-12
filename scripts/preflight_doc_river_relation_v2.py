"""One-epoch source/validation-only integration check, never target scoring."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from run_doc_river_relation_v2 import (
    ARMS,
    DATASET,
    SPACE_PARENT,
    role_labels,
    task_context,
    train_one,
    write_json,
)
from threadpoolctl import threadpool_limits


def main():
    data = torch.load(DATASET, weights_only=False, map_location="cpu")
    root = Path("experiments/phase4_transfer/doc_information_ablation_v1")
    settings = json.loads((SPACE_PARENT/"protocol.json").read_text())["settings"]
    settings = {**settings, "max_epochs": 1, "min_epochs": 1, "patience": 1}
    torch.set_num_threads(settings["torch_threads"])
    rows = []
    with threadpool_limits(limits=settings["torch_threads"]):
        for arm in ARMS:
            task = {"track": "space", "region": "0708", "mask": "space_0708", "seed": 42, "arm": arm}
            ctx = task_context(root, task, data)
            train_y = role_labels(data, ctx["split"], "train", ctx["graphs"]["train"]["rows"].numpy())
            val_y = role_labels(data, ctx["split"], "val", ctx["graphs"]["val"]["rows"].numpy())
            model, info, trace = train_one(arm, 42, ctx["inputs"], ctx["graphs"]["train"],
                ctx["graphs"]["val"], train_y, val_y, settings, lambda *_: None)
            assert info["parameters"] == 12565 and np.isfinite(info["validation_mae"])
            assert all(torch.isfinite(parameter).all() for parameter in model.parameters())
            rows.append({"arm": arm, "parameters": info["parameters"], "finite_loss_and_validation": True,
                "restored_training_ancestor_pairs": ctx["graphs"]["train"].get("restored_ancestor_pairs", 0),
                "epoch_count": len(trace)})
    result = {"status": "passed", "test_used": False, "preflight_only": True, "recipes": rows}
    write_json(Path("cache/doc_river_relation_v2/preflight.json"), result)
    print(json.dumps(result))


if __name__ == "__main__":
    main()
