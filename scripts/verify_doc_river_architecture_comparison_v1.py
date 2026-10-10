"""Independently recompute primary arithmetic against the original DOC grid."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from river_graph.experiments.provenance import sha256_file

ROOT = Path("experiments/phase4_transfer/doc_river_architecture_comparison_v1")


def main():
    protocol = json.loads((ROOT / "protocol.json").read_text())
    truth = np.asarray(torch.load(protocol["dataset"], weights_only=False, map_location="cpu")["y"],
                       dtype=np.float64).ravel()
    errors, receipts, cells_by_region = {}, {}, {}
    for region in protocol["regions"]:
        for seed in protocol["seeds"]:
            path = ROOT / "runs" / f"huc4_{region}_seed{seed}" / "predictions.parquet"
            data = pd.read_parquet(path)
            with np.load(f"experiments/phase4_transfer/doc_unmonitored_tasks_v1/geographical_masks/huc4_{region}.npz") as saved:
                expected_cells = saved["test"]
            cells_by_region[region] = expected_cells
            for arm in protocol["arms"]:
                selected = data[data.model_name == arm].sort_values("cell")
                np.testing.assert_array_equal(selected.cell, expected_cells)
                np.testing.assert_array_equal(selected.y_true, truth[expected_cells])
                residual = selected.y_pred.to_numpy(dtype=np.float64) - truth[expected_cells]
                errors.setdefault(arm, []).append(np.abs(residual))
            receipts[str(path)] = sha256_file(path)
    means = {arm: float(np.concatenate(values).sum() / sum(len(v) for v in values))
             for arm, values in errors.items()}
    summary_path = ROOT / "analysis/summary.csv"
    summary = pd.read_csv(summary_path).set_index("model_name")
    differences = {arm: abs(value - float(summary.loc[arm, "mae"])) for arm, value in means.items()}
    if max(differences.values()) > 1e-6:
        raise ValueError("independent source-grid calculation disagrees with summary")
    for arm, value in means.items():
        gain = 100 * (means["local"] - value) / means["local"]
        if abs(gain - summary.loc[arm, "gain_vs_local_percent"]) > 1e-4:
            raise ValueError("independent percentage disagrees with summary")
    all_cells = np.concatenate(list(cells_by_region.values()))
    if len(np.unique(all_cells)) != len(all_cells):
        raise ValueError("held-region observation populations overlap")
    record = {"independent_primary_mae": means, "maximum_difference": max(differences.values()),
              "unique_query_cells": len(all_cells), "seed_repeats_are_not_new_observations": True,
              "prediction_hashes": receipts, "dataset_sha256": sha256_file(protocol["dataset"]),
              "summary_sha256": sha256_file(summary_path), "verification_script_sha256": sha256_file(__file__)}
    (ROOT / "analysis/independent_metrics_audit.json").write_text(json.dumps(record, indent=2) + "\n")
    print(json.dumps({"independent_mae": means, "maximum_difference": max(differences.values())}), flush=True)


if __name__ == "__main__":
    main()
