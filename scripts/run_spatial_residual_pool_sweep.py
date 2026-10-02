"""Sweep source-station pool size for the K=5 residual adapter.

The pool size and residual alpha are selected on nested internal E3
validation. The outer split is scored only for the selected pair and the
pre-existing k=40 mean-residual control.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from run_spatial_fewshot_curve import MASK, T, load_split
from run_spatial_source_selection import station_descriptors
from sklearn.ensemble import ExtraTreesRegressor

from river_graph.experiments.evaluate import metrics
from river_graph.experiments.h3_masks import e3_internal_split
from river_graph.experiments.spatial_fewshot import support_view
from river_graph.experiments.transfer import DATASETS
from river_graph.models.kgml_local_transport import (
    FIT_ROLES,
    TEST_ROLES,
    build_rf_features,
)

POOL_VALUES = (20, 40, 80, 160)
ALPHA_VALUES = (0.25, 0.5, 0.75, 1.0)


def fit_source_models_pool(
    data: dict,
    split: dict,
    descriptors: np.ndarray,
    *,
    seed: int,
    pool_k: int,
    n_estimators: int,
) -> dict[int, ExtraTreesRegressor]:
    y = np.log1p(np.asarray(data["y"], dtype=np.float64)).reshape(-1)
    fit_x = build_rf_features(
        data, split, FIT_ROLES, target_transform="log1p", include_network=True,
    )
    train = np.asarray(split["train"], dtype=np.int64)
    source = np.unique(train // T)
    target = np.unique(np.asarray(split["test"], dtype=np.int64) // T)
    by_station = {int(s): train[train // T == s] for s in source}
    models: dict[int, ExtraTreesRegressor] = {}
    for station in target:
        distance = np.square(
            descriptors[source] - descriptors[int(station)],
        ).sum(axis=1)
        neighbours = source[np.argsort(distance)[: min(pool_k, len(source))]]
        cells = np.concatenate([by_station[int(s)] for s in neighbours])
        model = ExtraTreesRegressor(
            n_estimators=n_estimators,
            min_samples_leaf=4,
            max_features=1.0,
            random_state=seed,
            n_jobs=4,
        )
        model.fit(fit_x[cells], y[cells])
        models[int(station)] = model
    return models


def predict_source_models_pool(
    data: dict,
    split: dict,
    models: dict[int, ExtraTreesRegressor],
    cells: np.ndarray,
) -> np.ndarray:
    eval_x = build_rf_features(
        data, split, TEST_ROLES, target_transform="log1p", include_network=True,
    )
    out = np.empty(len(cells), dtype=np.float64)
    for station in np.unique(cells // T):
        take = cells // T == station
        out[take] = np.expm1(models[int(station)].predict(eval_x[cells[take]]))
    return out


def residual_prediction(
    base_query: np.ndarray,
    base_support: np.ndarray,
    values: np.ndarray,
    support: np.ndarray,
    query: np.ndarray,
    alpha: float,
) -> np.ndarray:
    zq = np.log1p(np.maximum(base_query, 0.0))
    zs = np.log1p(np.maximum(base_support, 0.0))
    ys = np.log1p(np.asarray(values).reshape(-1)[support])
    out = zq.copy()
    for station in np.unique(query // T):
        q = query // T == station
        s = support // T == station
        out[q] += alpha * float(np.mean(ys[s] - zs[s]))
    return np.maximum(np.expm1(out), 0.0)


def run_stage(
    data: dict,
    split: dict,
    target_role: str,
    descriptors: np.ndarray,
    seeds: list[int],
    *,
    n_estimators: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    values = np.asarray(data["y"], dtype=np.float64).reshape(-1)
    metric_rows: list[dict] = []
    pred_rows: list[pd.DataFrame] = []
    for pool_k in POOL_VALUES:
        for seed in seeds:
            base_view, _support5, query = support_view(
                split, target_role=target_role, k=0, n_months=T,
            )
            _view, support, paired_query = support_view(
                split, target_role=target_role, k=5, n_months=T,
            )
            if not np.array_equal(query, paired_query):
                raise ValueError("K=5 query changed relative to K=0")
            models = fit_source_models_pool(
                data, base_view, descriptors, seed=seed,
                pool_k=pool_k, n_estimators=n_estimators,
            )
            base_query = predict_source_models_pool(data, base_view, models, query)
            base_support = predict_source_models_pool(data, base_view, models, support)
            for alpha in ALPHA_VALUES:
                corrected = residual_prediction(
                    base_query, base_support, values, support, query, alpha,
                )
                metric_rows.append({
                    "stage": target_role,
                    "pool_k": pool_k,
                    "seed": seed,
                    "alpha": alpha,
                    "variant": "residual",
                    **metrics(values[query], corrected),
                })
                if target_role == "test":
                    pred_rows.append(pd.DataFrame({
                        "cell": query,
                        "station": query // T,
                        "month": query % T,
                        "y_true": values[query],
                        "y_pred": corrected,
                        "pool_k": pool_k,
                        "seed": seed,
                        "alpha": alpha,
                    }))
    return pd.DataFrame(metric_rows), (
        pd.concat(pred_rows, ignore_index=True) if pred_rows else pd.DataFrame()
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44])
    parser.add_argument("--n-estimators", type=int, default=300)
    args = parser.parse_args()
    data = torch.load(DATASETS["doc"], map_location="cpu", weights_only=False)
    outer = load_split(MASK)
    internal, _val_rows, _ = e3_internal_split(
        np.asarray(data["y_mask"]), outer["test"],
    )
    descriptors = station_descriptors(data)
    validation, _ = run_stage(
        data, internal, "val", descriptors, args.seeds,
        n_estimators=args.n_estimators,
    )
    selected = (
        validation.groupby(["pool_k", "alpha"], as_index=False).mae.mean()
        .sort_values(["mae", "pool_k", "alpha"]).iloc[0]
    )
    selected_pair = {
        "pool_k": int(selected.pool_k),
        "alpha": float(selected.alpha),
        "validation_mae": float(selected.mae),
    }
    test_all, predictions = run_stage(
        data, outer, "test", descriptors, args.seeds,
        n_estimators=args.n_estimators,
    )
    selected_mask = (
        (test_all.pool_k == selected_pair["pool_k"])
        & (test_all.alpha == selected_pair["alpha"])
    )
    control_mask = (test_all.pool_k == 40) & (test_all.alpha == 0.75)
    test_selected = test_all[selected_mask | control_mask].copy()
    test_selected["report_role"] = np.where(
        (test_selected.pool_k == selected_pair["pool_k"])
        & (test_selected.alpha == selected_pair["alpha"]),
        "selected_pool",
        "mean_k40_control",
    )
    prediction_mask = (
        ((predictions.pool_k == selected_pair["pool_k"])
         & (predictions.alpha == selected_pair["alpha"]))
        | ((predictions.pool_k == 40) & (predictions.alpha == 0.75))
    )
    predictions = predictions[prediction_mask].copy()
    out = args.out_dir
    out.mkdir(parents=True, exist_ok=True)
    validation.to_csv(out / "validation_pool_candidates.csv", index=False)
    test_selected.to_csv(out / "test_selected_and_control.csv", index=False)
    predictions.to_parquet(out / "test_selected_and_control_predictions.parquet", index=False)
    (out / "selected_pool.json").write_text(
        json.dumps(selected_pair, indent=2) + "\n",
    )
    manifest = {
        "experiment": "fewshot_residual_pool_sweep_v1",
        "pool_values": list(POOL_VALUES),
        "alpha_values": list(ALPHA_VALUES),
        "seeds": args.seeds,
        "n_estimators": args.n_estimators,
        "selection": "K=5 mean MAE on internal E3 validation only",
        "outer_scored": "selected pool/alpha and pre-existing k=40 alpha=.75 control",
        "development_extension": (
            "pool sweep was specified after the previous outer residual result "
            "was observed"
        ),
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print("selected", selected_pair)
    print(test_selected.groupby(["pool_k", "alpha"]).mae.agg(["mean", "std"]).to_string())


if __name__ == "__main__":
    main()
