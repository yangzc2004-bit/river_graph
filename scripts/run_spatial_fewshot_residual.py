"""Paired fixed-query spatial few-shot transfer with support residual calibration.

The source regional expert is fitted once per seed with every target station
hidden. For each K, target support labels are used only to estimate a
station-specific residual in log1p space:

    delta_i = mean(log1p(y_support) - log1p(base_support_prediction))

The residual is shrunk by an alpha selected on a nested internal E3 split and
added to the predictions for the fixed outer query set. This experiment is
independent of the previous few-shot curve and leaves that result untouched.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from run_spatial_fewshot_curve import (
    MASK,
    T,
    fit_source_models,
    load_split,
    predict_source_models,
)
from run_spatial_source_selection import station_descriptors

from river_graph.experiments.evaluate import metrics
from river_graph.experiments.h3_masks import e3_internal_split
from river_graph.experiments.spatial_fewshot import (
    ALPHA_VALUES,
    K_VALUES,
    support_schedule,
    support_view,
)
from river_graph.experiments.transfer import DATASETS


def residual_correction(
    base_query: np.ndarray,
    base_support: np.ndarray,
    values: np.ndarray,
    support: np.ndarray,
    query: np.ndarray,
    *,
    n_months: int,
    alpha: float,
) -> tuple[np.ndarray, dict[int, float]]:
    """Apply support residual calibration without reading query labels."""
    base_query = np.asarray(base_query, dtype=np.float64)
    base_support = np.asarray(base_support, dtype=np.float64)
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    support = np.asarray(support, dtype=np.int64)
    query = np.asarray(query, dtype=np.int64)
    if base_query.shape != query.shape or not np.isfinite(base_query).all():
        raise ValueError("base query predictions must be finite and aligned")
    if base_support.shape != support.shape or not np.isfinite(base_support).all():
        raise ValueError("base support predictions must be finite and aligned")
    if np.intersect1d(support, query).size:
        raise ValueError("support and query must be disjoint")
    if not 0 <= alpha <= 1:
        raise ValueError("alpha must lie in [0, 1]")
    z_query = np.log1p(np.maximum(base_query, 0.0))
    z_support = np.log1p(np.maximum(base_support, 0.0))
    y_support = np.log1p(values[support])
    out = z_query.copy()
    deltas: dict[int, float] = {}
    for station in np.unique(query // n_months):
        q = query // n_months == station
        s = support // n_months == station
        if not s.any():
            deltas[int(station)] = 0.0
            continue
        delta = float(np.mean(y_support[s] - z_support[s]))
        deltas[int(station)] = delta
        out[q] += alpha * delta
    return np.maximum(np.expm1(out), 0.0), deltas


def run_stage(
    data: dict,
    split: dict,
    target_role: str,
    descriptors: np.ndarray,
    seeds: list[int],
    *,
    n_estimators: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Run nested validation or outer test with one fixed query per K."""
    values = np.asarray(data["y"], dtype=np.float64).reshape(-1)
    metric_rows: list[dict] = []
    query_rows: list[pd.DataFrame] = []
    support_rows: list[pd.DataFrame] = []
    for seed in seeds:
        base_view, _support5, query = support_view(
            split, target_role=target_role, k=0, n_months=T,
        )
        models = fit_source_models(
            data, base_view, descriptors, seed=seed,
            n_estimators=n_estimators, leaf=4,
        )
        base_query = predict_source_models(data, base_view, models, query)
        truth = values[query]
        for k in K_VALUES:
            _view, support, paired_query = support_view(
                split, target_role=target_role, k=k, n_months=T,
            )
            if not np.array_equal(query, paired_query):
                raise ValueError("paired query set changed across K")
            base_support = predict_source_models(data, base_view, models, support)
            support_truth = values[support]
            support_residual = (
                np.log1p(support_truth) - np.log1p(np.maximum(base_support, 0.0))
            )
            for alpha in ALPHA_VALUES:
                corrected, _deltas = residual_correction(
                    base_query, base_support, values, support, query,
                    n_months=T, alpha=alpha,
                )
                metric_rows.append({
                    "stage": target_role,
                    "seed": seed,
                    "k": k,
                    "alpha": alpha,
                    "variant": "residual_corrected",
                    **metrics(truth, corrected),
                    "support_cells": len(support),
                    "query_cells": len(query),
                })
                if alpha == 0.0:
                    metric_rows.append({
                        "stage": target_role,
                        "seed": seed,
                        "k": k,
                        "alpha": 0.0,
                        "variant": "raw",
                        **metrics(truth, base_query),
                        "support_cells": len(support),
                        "query_cells": len(query),
                    })
                if target_role == "test":
                    query_rows.append(pd.DataFrame({
                        "cell": query,
                        "station": query // T,
                        "month": query % T,
                        "y_true": truth,
                        "base_pred": base_query,
                        "y_pred": corrected,
                        "seed": seed,
                        "k": k,
                        "alpha": alpha,
                        "variant": "residual_corrected",
                        "support_cells": len(support),
                    }))
            if target_role == "test":
                support_rows.append(pd.DataFrame({
                    "cell": support,
                    "station": support // T,
                    "month": support % T,
                    "y_true": support_truth,
                    "base_pred": base_support,
                    "support_residual": support_residual,
                    "seed": seed,
                    "k": k,
                }))
    predictions = (
        pd.concat(query_rows, ignore_index=True)
        if query_rows else pd.DataFrame()
    )
    support_predictions = (
        pd.concat(support_rows, ignore_index=True)
        if support_rows else pd.DataFrame()
    )
    return pd.DataFrame(metric_rows), predictions, support_predictions


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44, 45, 46])
    parser.add_argument("--n-estimators", type=int, default=300)
    args = parser.parse_args()

    data = torch.load(DATASETS["doc"], map_location="cpu", weights_only=False)
    outer = load_split(MASK)
    internal, _val_rows, _ = e3_internal_split(
        np.asarray(data["y_mask"]), outer["test"],
    )
    descriptors = station_descriptors(data)

    _schedule_outer, query5 = support_schedule(outer["test"], T)
    del _schedule_outer
    reserved_outer = np.setdiff1d(outer["test"], query5, assume_unique=True)
    query_sets: list[np.ndarray] = []
    support_sizes: dict[int, int] = {}
    support_cells_by_k: dict[int, list[int]] = {}
    for k in K_VALUES:
        _view, support, query = support_view(
            outer, target_role="test", k=k, n_months=T,
        )
        query_sets.append(query)
        support_sizes[k] = len(support)
        support_cells_by_k[k] = support.tolist()
        if np.intersect1d(support, query).size:
            raise ValueError("support/query overlap in outer E3")
        if not np.isin(support, outer["test"]).all():
            raise ValueError("support contains a non-test E3 cell")
    if support_sizes != {0: 0, 1: 43, 3: 129, 5: 215}:
        raise ValueError(f"unexpected support sizes: {support_sizes}")
    if not set(support_cells_by_k[1]).issubset(support_cells_by_k[3]):
        raise ValueError("K=1 support is not nested in K=3")
    if not set(support_cells_by_k[3]).issubset(support_cells_by_k[5]):
        raise ValueError("K=3 support is not nested in K=5")
    if not all(np.array_equal(query_sets[0], query) for query in query_sets[1:]):
        raise ValueError("K query sets are not identical")
    if not np.array_equal(query_sets[0], query5):
        raise ValueError("paired query set does not equal K=5 reserved-query set")
    if len(query_sets[0]) != 2316:
        raise ValueError(f"unexpected paired query size: {len(query_sets[0])}")
    if not np.array_equal(np.sort(support_cells_by_k[5]), reserved_outer):
        raise ValueError("K=5 support does not match reserved support set")

    validation, _validation_predictions, _validation_support = run_stage(
        data, internal, "val", descriptors, args.seeds,
        n_estimators=args.n_estimators,
    )
    selected = (
        validation[validation.variant == "residual_corrected"]
        .groupby(["k", "alpha"], as_index=False).mae.mean()
        .sort_values(["k", "mae", "alpha"])
        .groupby("k", as_index=False).first()
    )
    selected_map = {int(row.k): float(row.alpha) for row in selected.itertuples()}
    test, predictions, support_predictions = run_stage(
        data, outer, "test", descriptors, args.seeds,
        n_estimators=args.n_estimators,
    )
    test["selected"] = test.apply(
        lambda row: row.alpha == selected_map[int(row.k)], axis=1,
    )

    out = args.out_dir
    out.mkdir(parents=True, exist_ok=True)
    validation.to_csv(out / "validation.csv", index=False)
    test.to_csv(out / "test_all_alphas.csv", index=False)
    test[test.selected].to_csv(out / "test_selected.csv", index=False)
    predictions.to_parquet(out / "test_predictions.parquet", index=False)
    support_predictions.to_parquet(out / "support_predictions.parquet", index=False)
    selected.to_csv(out / "selected_alpha.csv", index=False)
    query_manifest = {
        "experiment": "fewshot_residual_paired_v1",
        "k_values": list(K_VALUES),
        "alpha_values": list(ALPHA_VALUES),
        "selected_alpha": selected_map,
        "seeds": args.seeds,
        "n_estimators": args.n_estimators,
        "target_role": "outer E3 test",
        "query_cells_by_k": test.groupby("k").query_cells.first().to_dict(),
        "support_cells_by_k": test.groupby("k").support_cells.first().to_dict(),
        "paired_query_cells": len(query_sets[0]),
        "reserved_support_cells": len(reserved_outer),
        "support_schedule": (
            "first, middle, last, first-quarter, third-quarter; nested prefixes"
        ),
        "selection": "mean MAE over nested internal station-heldout validation",
        "calibration": (
            "station mean of log1p support label minus log1p source-model "
            "prediction, multiplied by K-specific alpha"
        ),
        "source_fit": (
            "one K=0 source regional model per target station; support labels "
            "are hidden when generating support predictions"
        ),
    }
    (out / "query_manifest.json").write_text(
        json.dumps(query_manifest, indent=2) + "\n",
    )
    print("selected alpha", selected_map)
    print(test[test.selected].groupby("k").mae.agg(["mean", "std"]))


if __name__ == "__main__":
    main()
