"""Hydrological history availability, reading DOC cell indices but no DOC values."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from river_graph.experiments.provenance import sha256_file

ROOT = Path("experiments/phase4_transfer/doc_fusion_component_comparison_v1")
MASKS = Path("experiments/phase4_transfer/doc_unmonitored_tasks_v1/geographical_masks")


def describe(windows, cells, months):
    history = windows[cells // months, cells % months]
    result = {"cells": len(cells), "stations": len(np.unique(cells // months)),
        "no_hydrology_in_12_months_fraction": float((~history.any(axis=(1, 2))).mean()),
        "no_older_hydrology_fraction": float((~history[:, :-1].any(axis=(1, 2))).mean())}
    for channel, name in enumerate(("temperature", "discharge")):
        visible = history[:, :, channel]
        result[f"{name}_current_fraction"] = float(visible[:, -1].mean())
        result[f"{name}_mean_observed_months_in_12"] = float(visible.sum(1).mean())
        result[f"{name}_older_any_fraction"] = float(visible[:, :-1].any(1).mean())
    return result


def main():
    protocol = json.loads((ROOT / "protocol.json").read_text())
    data = torch.load(protocol["dataset"], weights_only=False, map_location="cpu")
    values = np.asarray(data["x"])
    mask = np.asarray(data["x_mask"], dtype=bool) & np.isfinite(values)
    mask[..., 1] &= values[..., 1] >= 0
    n, t, _ = mask.shape
    lookback = protocol["settings"]["lookback"]
    padded = np.pad(mask, ((0, 0), (lookback - 1, 0), (0, 0)))
    windows = np.lib.stride_tricks.sliding_window_view(padded, lookback, axis=1).transpose(0, 1, 3, 2)
    records, pooled_query, pooled_gap, hashes = [], [], [], {}
    for region in protocol["regions"]:
        path = MASKS / f"huc4_{region}.npz"
        with np.load(path, allow_pickle=False) as split:
            cells = split["test"].copy()
        rows = np.unique(cells // t)
        observed_rows, observed_months = np.where(np.asarray(data["y_mask"], dtype=bool)[rows])
        np.testing.assert_array_equal(np.sort(cells), np.sort(rows[observed_rows] * t + observed_months))
        grid = (rows[:, None] * t + np.arange(t)[None]).ravel()
        nonquery = np.setdiff1d(grid, cells)
        interior = []
        for row in rows:
            month = cells[cells // t == row] % t
            interior.extend((row * t + np.arange(month.min(), month.max() + 1)).tolist())
        gap = np.setdiff1d(np.asarray(interior), cells)
        populations = {"observed_query": cells, "nonquery_full_target_grid": nonquery,
                       "nonquery_within_station_query_span": gap}
        for name, population in populations.items():
            records.append({"target_huc4": region, "population": name, **describe(windows, population, t)})
        pooled_query.append(cells)
        pooled_gap.append(gap)
        hashes[str(path)] = sha256_file(path)
    for name, populations in (("observed_query", pooled_query),
                              ("nonquery_within_station_query_span", pooled_gap)):
        records.append({"target_huc4": "pooled", "population": name,
                        **describe(windows, np.concatenate(populations), t)})
    out = ROOT / "analysis"
    out.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(records)
    frame.to_csv(out / "input_availability.csv", index=False)
    metadata = {"DOC_values_read_or_scored": False, "all_observed_target_cells_are_queries": True,
        "channels": ["temperature", "discharge"],
        "lookback": lookback, "grid_stations": n, "grid_months": t,
        "scope": "measurement availability only; nonquery cells are not evaluated DOC predictions; "
                 "interior gaps lie between each target station's first and last query months",
        "dataset_sha256": sha256_file(protocol["dataset"]), "mask_sha256": hashes,
        "script_sha256": sha256_file(__file__), "csv_sha256": sha256_file(out / "input_availability.csv")}
    (out / "input_availability.meta.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(frame[frame.target_huc4 == "pooled"].to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
