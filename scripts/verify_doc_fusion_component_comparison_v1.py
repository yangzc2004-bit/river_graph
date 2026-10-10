"""Independent source-grid arithmetic; no shared aggregation helpers."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from river_graph.experiments.provenance import sha256_file

ROOT = Path("experiments/phase4_transfer/doc_fusion_component_comparison_v1")


def main():
    protocol = json.loads((ROOT / "protocol.json").read_text())
    truth = np.asarray(torch.load(protocol["dataset"], weights_only=False, map_location="cpu")["y"],
                       dtype=np.float64).ravel()
    sums, counts, receipts, populations = {}, {}, {}, []
    for region in protocol["regions"]:
        run = ROOT / "regions" / f"huc4_{region}"
        frame = pd.read_parquet(run / "predictions.parquet")
        with np.load(Path("experiments/phase4_transfer/doc_unmonitored_tasks_v1/geographical_masks") /
                     f"huc4_{region}.npz") as split:
            cells = split["test"]
        populations.append(cells)
        for (alias, seed), part in frame.groupby(["model_name", "seed"]):
            selected = part.sort_values("cell")
            np.testing.assert_array_equal(selected.cell, np.sort(cells))
            np.testing.assert_array_equal(selected.y_true, truth[selected.cell.to_numpy()])
            residual = np.abs(selected.y_pred.to_numpy(dtype=np.float64) - truth[selected.cell.to_numpy()])
            sums[alias] = sums.get(alias, 0.) + float(residual.sum())
            counts[alias] = counts.get(alias, 0) + len(residual)
        receipts[str(run / "predictions.parquet")] = sha256_file(run / "predictions.parquet")
    mae = {alias: sums[alias] / counts[alias] for alias in sums}
    table = pd.read_csv(ROOT / "analysis/summary.csv").set_index("model_name")
    differences = {alias: abs(value - table.loc[alias, "mae"]) for alias, value in mae.items()}
    if max(differences.values()) > 1e-10:
        raise ValueError("independent source-grid MAE disagrees")
    for alias, value in mae.items():
        gain = 100 * (mae["selected__reference"] - value) / mae["selected__reference"]
        if abs(gain - table.loc[alias, "gain_vs_reference_percent"]) > 1e-9:
            raise ValueError("gain arithmetic disagrees")
    cells = np.concatenate(populations)
    if len(cells) != len(np.unique(cells)) or len(mae) != 17 or len(set(counts.values())) != 1:
        raise ValueError("overlapping outer populations or inconsistent candidate coverage")
    record = {"primary_mae": mae, "maximum_difference": max(differences.values()),
        "query_cells": len(cells), "seed_repeats_are_not_independent_observations": True,
        "prediction_hashes": receipts, "dataset_sha256": sha256_file(protocol["dataset"]),
        "summary_sha256": sha256_file(ROOT / "analysis/summary.csv"),
        "verification_script_sha256": sha256_file(__file__)}
    (ROOT / "analysis/independent_metrics_audit.json").write_text(json.dumps(record, indent=2) + "\n")
    print(json.dumps({"independent_mae": mae, "maximum_difference": max(differences.values())}), flush=True)


if __name__ == "__main__":
    main()
