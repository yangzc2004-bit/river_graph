"""Cache K=0 source-model predictions for residual-method development.

The cache contains only base predictions and cell identifiers for validation
and outer test query/support cells. Target labels are intentionally excluded;
the analysis script reads labels only when scoring a method on the appropriate
split. The outer-test method candidates are never used for selection.
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

from river_graph.experiments.h3_masks import e3_internal_split
from river_graph.experiments.spatial_fewshot import K_VALUES, support_view
from river_graph.experiments.transfer import DATASETS


def cache_stage(
    data: dict,
    split: dict,
    target_role: str,
    descriptors: np.ndarray,
    seeds: list[int],
    *,
    n_estimators: int,
) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    for seed in seeds:
        base_view, _support5, query = support_view(
            split, target_role=target_role, k=0, n_months=T,
        )
        models = fit_source_models(
            data, base_view, descriptors, seed=seed,
            n_estimators=n_estimators, leaf=4,
        )
        for k in K_VALUES:
            _view, support, paired_query = support_view(
                split, target_role=target_role, k=k, n_months=T,
            )
            if not np.array_equal(query, paired_query):
                raise ValueError("paired query changed across K")
            query_pred = predict_source_models(data, base_view, models, query)
            support_pred = predict_source_models(data, base_view, models, support)
            rows.extend([
                pd.DataFrame({
                    "stage": target_role,
                    "seed": seed,
                    "k": k,
                    "role": "query",
                    "cell": query,
                    "station": query // T,
                    "month": query % T,
                    "base_pred": query_pred,
                }),
                pd.DataFrame({
                    "stage": target_role,
                    "seed": seed,
                    "k": k,
                    "role": "support",
                    "cell": support,
                    "station": support // T,
                    "month": support % T,
                    "base_pred": support_pred,
                }),
            ])
    cache = pd.concat(rows, ignore_index=True)
    if cache.base_pred.isna().any() or not np.isfinite(cache.base_pred).all():
        raise ValueError("base prediction cache contains non-finite values")
    return cache


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
    cache = pd.concat([
        cache_stage(
            data, internal, "val", descriptors, args.seeds,
            n_estimators=args.n_estimators,
        ),
        cache_stage(
            data, outer, "test", descriptors, args.seeds,
            n_estimators=args.n_estimators,
        ),
    ], ignore_index=True)
    expected = {
        "val": {"0": 1187, "1": 1187, "3": 1187, "5": 1187},
        "test": {"0": 2316, "1": 2316, "3": 2316, "5": 2316},
    }
    for stage, by_k in expected.items():
        for k, query_size in by_k.items():
            query = cache[
                (cache.stage == stage) & (cache.k == int(k))
                & (cache.role == "query")
            ]
            if len(query) != query_size * len(args.seeds):
                raise ValueError(f"unexpected query count for {stage} K={k}")
    if set(cache.columns) != {
        "stage", "seed", "k", "role", "cell", "station", "month", "base_pred",
    }:
        raise ValueError("cache includes unexpected or label-bearing columns")
    out = args.out_dir
    out.mkdir(parents=True, exist_ok=True)
    cache.to_parquet(out / "base_predictions.parquet", index=False)
    manifest = {
        "experiment": "fewshot_residual_methods_v1",
        "seeds": args.seeds,
        "n_estimators": args.n_estimators,
        "min_samples_leaf": 4,
        "k_values": list(K_VALUES),
        "outer_query_cells": 2316,
        "outer_support_cells_by_k": {"0": 0, "1": 43, "3": 129, "5": 215},
        "cache_columns": list(cache.columns),
        "label_policy": (
            "cache contains no target labels; support labels are read only by "
            "the validation scoring and final selected-method evaluation"
        ),
        "source_fit": (
            "one K=0 source regional model per target station; target support "
            "is hidden while generating query and support predictions"
        ),
        "selection_policy": (
            "candidate and regularization selection uses internal validation "
            "only; outer test is scored only for the selected candidate and "
            "the pre-existing mean-residual control"
        ),
        "development_extension": (
            "candidate methods were specified after the previous outer mean "
            "result was observed; this is a development extension, not a new "
            "primary endpoint test"
        ),
    }
    (out / "cache_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n",
    )
    print(cache.groupby(["stage", "role", "k"]).size().to_string())


if __name__ == "__main__":
    main()
