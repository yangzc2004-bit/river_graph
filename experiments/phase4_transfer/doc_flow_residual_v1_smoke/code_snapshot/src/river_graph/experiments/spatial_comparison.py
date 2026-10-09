"""Paired cell-weighted errors with whole-station bootstrap resampling."""
from __future__ import annotations

import numpy as np
import pandas as pd


def paired_station_comparison(candidate: pd.DataFrame, reference: pd.DataFrame,
                              *, draws: int = 5000, seed: int = 1729) -> dict:
    """Average seed losses, then bootstrap stations and recompute both MAEs.

    Stations are sampling clusters, not the estimand weights. Each draw keeps
    all cells of each sampled station, including multiplicity. Relative changes
    use the reference MAE of that same draw as their denominator.
    """
    keys = ["seed", "cell", "station"]
    for frame in (candidate, reference):
        if frame.empty or frame.duplicated(keys).any():
            raise ValueError("empty or duplicate seed/cell/station predictions")
        if not np.isfinite(frame[["y_true", "y_pred"]]).all().all():
            raise ValueError("non-finite predictions or truth")
    joined = candidate[keys + ["y_true", "y_pred"]].merge(
        reference[keys + ["y_true", "y_pred"]], on=keys, how="outer",
        suffixes=("_new", "_ref"), validate="one_to_one", indicator=True,
    )
    if not (joined._merge == "both").all():
        raise ValueError("candidate and reference query/seed sets differ")
    if not np.array_equal(joined.y_true_new, joined.y_true_ref):
        raise ValueError("paired truth differs")
    seeds_per_cell = joined.groupby("cell").seed.nunique()
    if not (seeds_per_cell == joined.seed.nunique()).all():
        raise ValueError("incomplete seed coverage")
    joined["new_error"] = np.abs(joined.y_true_new - joined.y_pred_new)
    joined["ref_error"] = np.abs(joined.y_true_ref - joined.y_pred_ref)
    cells = joined.groupby(["station", "cell"], as_index=False)[
        ["new_error", "ref_error"]
    ].mean()
    groups = cells.groupby("station").agg(
        new_sum=("new_error", "sum"), ref_sum=("ref_error", "sum"),
        count=("cell", "size"),
    )
    new_mae, ref_mae = cells[["new_error", "ref_error"]].mean()
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(groups), (draws, len(groups)))
    sums = groups.to_numpy()[indices].sum(axis=1)
    delta_samples = (sums[:, 0] - sums[:, 1]) / sums[:, 2]
    pct_samples = 100 * (sums[:, 1] - sums[:, 0]) / sums[:, 1]
    delta_lo, delta_hi = np.quantile(delta_samples, [0.025, 0.975])
    pct_lo, pct_hi = np.quantile(pct_samples, [0.025, 0.975])
    return {
        "mae": float(new_mae), "reference_mae": float(ref_mae),
        "delta_mae": float(new_mae - ref_mae),
        "delta_lo": float(delta_lo), "delta_hi": float(delta_hi),
        "reduction_pct": float(100 * (ref_mae - new_mae) / ref_mae),
        "reduction_lo": float(pct_lo), "reduction_hi": float(pct_hi),
        "station_macro_mae": float((groups.new_sum / groups["count"]).mean()),
        "reference_station_macro_mae": float((groups.ref_sum / groups["count"]).mean()),
        "improved_stations": int((groups.new_sum < groups.ref_sum).sum()),
        "n_stations": len(groups), "n_cells": len(cells),
        "n_seeds": int(joined.seed.nunique()),
    }
