"""Paired 0/1/3/5-support curve for spatial DOC transfer.

Every K is evaluated on one fixed query set: the outer E3 cells minus the
five reserved support candidates at each target station. Source regional
forests are fitted once without target support; K only changes visible target
support features and the separately selected station-level correction.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from run_spatial_source_selection import load_split, station_descriptors
from sklearn.ensemble import ExtraTreesRegressor

from river_graph.experiments.evaluate import metrics
from river_graph.experiments.h3_masks import e3_internal_split
from river_graph.experiments.spatial_fewshot import (
    ALPHA_VALUES,
    K_VALUES,
    station_level_correction,
    support_schedule,
    support_view,
)
from river_graph.experiments.temporal_h2x import _as_tensor
from river_graph.experiments.transfer import DATASETS
from river_graph.models.kgml_local_transport import (
    FIT_ROLES,
    TEST_ROLES,
    build_rf_features,
)

T = 654
MASK = Path("experiments/masks_stcore_v1/e3_spatial_seed42.npz")


def fit_source_models(data: dict, split: dict, descriptors: np.ndarray, *, seed: int,
                      n_estimators: int, leaf: int) -> dict[int, ExtraTreesRegressor]:
    """Fit one source regional expert per target station from K=0 inputs."""
    y = np.log1p(np.asarray(data["y"], dtype=np.float64)).reshape(-1)
    fit_x = build_rf_features(
        data, split, FIT_ROLES, target_transform="log1p", include_network=True,
    )
    train = np.asarray(split["train"], dtype=np.int64)
    source = np.unique(train // T)
    models: dict[int, ExtraTreesRegressor] = {}
    target = np.unique(np.asarray(split.get("test", []), dtype=np.int64) // T)
    for station in target:
        distance = np.square(descriptors[source] - descriptors[int(station)]).sum(axis=1)
        neighbours = source[np.argsort(distance)[: min(40, len(source))]]
        cells = np.concatenate([train[train // T == int(s)] for s in neighbours])
        model = ExtraTreesRegressor(
            n_estimators=n_estimators,
            min_samples_leaf=leaf,
            max_features=1.0,
            random_state=seed,
            n_jobs=4,
        )
        model.fit(fit_x[cells], y[cells])
        models[int(station)] = model
    return models


def predict_source_models(
    data: dict,
    split: dict,
    models: dict[int, ExtraTreesRegressor],
    query: np.ndarray,
) -> np.ndarray:
    eval_x = build_rf_features(
        data, split, TEST_ROLES, target_transform="log1p", include_network=True,
    )
    output = np.empty(len(query), dtype=np.float64)
    for station in np.unique(query // T):
        take = query // T == station
        output[take] = np.expm1(models[int(station)].predict(eval_x[query[take]]))
    return output


def run_stage(
    data: dict,
    split: dict,
    target_role: str,
    descriptors: np.ndarray,
    seeds: list[int],
    *,
    n_estimators: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows: list[dict] = []
    predictions: list[pd.DataFrame] = []
    for seed in seeds:
        base_view, _support5, query = support_view(
            split, target_role=target_role, k=0, n_months=T,
        )
        models = fit_source_models(
            data, base_view, descriptors, seed=seed,
            n_estimators=n_estimators, leaf=4,
        )
        for k in K_VALUES:
            view, support, paired_query = support_view(
                split, target_role=target_role, k=k, n_months=T,
            )
            if not np.array_equal(query, paired_query):
                raise ValueError("paired query set changed with K")
            raw = predict_source_models(data, view, models, query)
            truth = np.asarray(data["y"], dtype=np.float64).reshape(-1)[query]
            for alpha in ALPHA_VALUES:
                corrected = station_level_correction(
                    data["y"], support, query, raw, n_months=T, alpha=alpha,
                )
                score = metrics(truth, corrected)
                rows.append({
                    "stage": target_role,
                    "seed": seed,
                    "k": k,
                    "alpha": alpha,
                    "variant": "corrected",
                    **score,
                    "support_cells": len(support),
                    "query_cells": len(query),
                })
                if alpha == 0.0:
                    raw_score = metrics(truth, raw)
                    rows.append({
                        "stage": target_role,
                        "seed": seed,
                        "k": k,
                        "alpha": 0.0,
                        "variant": "raw",
                        **raw_score,
                        "support_cells": len(support),
                        "query_cells": len(query),
                    })
                if target_role == "test":
                    predictions.append(pd.DataFrame({
                        "cell": query,
                        "station": query // T,
                        "month": query % T,
                        "y_true": truth,
                        "y_pred": corrected,
                        "seed": seed,
                        "k": k,
                        "alpha": alpha,
                        "variant": "corrected",
                        "support_cells": len(support),
                    }))
    prediction_frame = (
        pd.concat(predictions, ignore_index=True)
        if predictions else pd.DataFrame()
    )
    return pd.DataFrame(rows), prediction_frame


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44, 45, 46])
    parser.add_argument("--n-estimators", type=int, default=300)
    args = parser.parse_args()
    data = torch.load(DATASETS["doc"], map_location="cpu", weights_only=False)
    outer = load_split(MASK)
    internal, _val_rows, _ = e3_internal_split(
        _as_tensor(data["y_mask"]).numpy(), outer["test"],
    )
    descriptors = station_descriptors(data)
    _schedule_outer, query5 = support_schedule(outer["test"], T)
    reserved_outer = np.setdiff1d(outer["test"], query5, assume_unique=True)
    query_sets = []
    support_sizes = {}
    support_cells_by_k = {}
    for k in K_VALUES:
        _view, support, query = support_view(outer, target_role="test", k=k, n_months=T)
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
    validation, _ = run_stage(
        data, internal, "val", descriptors, args.seeds,
        n_estimators=args.n_estimators,
    )
    selected = (
        validation[validation.variant == "corrected"]
        .groupby(["k", "alpha"], as_index=False).mae.mean()
        .sort_values(["k", "mae", "alpha"])
        .groupby("k", as_index=False).first()
    )
    selected_map = {int(row.k): float(row.alpha) for row in selected.itertuples()}
    test, predictions = run_stage(
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
    query_manifest = {
        "k_values": list(K_VALUES),
        "selected_alpha": selected_map,
        "n_seeds": len(args.seeds),
        "n_estimators": args.n_estimators,
        "target_role": "outer E3 test",
        "query_cells_by_k": test.groupby("k").query_cells.first().to_dict(),
        "support_cells_by_k": test.groupby("k").support_cells.first().to_dict(),
        "paired_query_cells": len(query_sets[0]),
        "reserved_support_cells": len(reserved_outer),
        "support_schedule": "first, middle, last, first-quarter, third-quarter; nested prefixes",
        "selection": "mean MAE over nested internal station-heldout validation",
        "source_fit": (
            "one K=0 source regional model per target station; target support "
            "only in evaluation features"
        ),
    }
    (out / "query_manifest.json").write_text(
        json.dumps(query_manifest, indent=2) + "\n",
    )
    selected.to_csv(out / "selected_alpha.csv", index=False)
    print("selected alpha", selected_map)
    print(test[test.selected].groupby("k").mae.agg(["mean", "std"]))


if __name__ == "__main__":
    main()
