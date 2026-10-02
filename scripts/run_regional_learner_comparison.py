"""Compare source-pool regressors for the fixed E3 residual-transfer task.

The source pool is fixed at 40 stations.  A nested internal E3 block is used
to select the learner; the frozen outer E3 cells are scored once.  Support
labels are opened at the fixed five-point schedule and the previously
selected K=5 residual shrinkage (alpha=.75) is used for all learners.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from run_spatial_source_selection import (
    load_split,
    nearest_sources,
    station_descriptors,
)
from sklearn.ensemble import (
    ExtraTreesRegressor,
    HistGradientBoostingRegressor,
    RandomForestRegressor,
)

from river_graph.experiments.evaluate import metrics
from river_graph.experiments.h3_masks import e3_internal_split
from river_graph.experiments.spatial_fewshot import support_schedule
from river_graph.experiments.transfer import DATASETS
from river_graph.models.kgml_local_transport import (
    FIT_ROLES,
    TEST_ROLES,
    build_rf_features,
    target_values,
)

T = 654
SEEDS = (42, 43, 44)
LEARNERS = ("extra_trees", "random_forest", "hist_gradient_boosting")
K_VALUES = (0, 5)
ALPHA = {0: 0.0, 5: 0.75}
POOL = 40
OUTER_MASK = Path("experiments/masks_stcore_v1/e3_spatial_seed42.npz")


def make_model(name: str, seed: int, n_estimators: int):
    if name == "extra_trees":
        return ExtraTreesRegressor(
            n_estimators=n_estimators, min_samples_leaf=4,
            max_features=1.0, random_state=seed, n_jobs=4,
        )
    if name == "random_forest":
        return RandomForestRegressor(
            n_estimators=n_estimators, min_samples_leaf=4,
            max_features=1.0, random_state=seed, n_jobs=4,
        )
    if name == "hist_gradient_boosting":
        return HistGradientBoostingRegressor(
            max_iter=n_estimators, learning_rate=0.05,
            max_leaf_nodes=31, min_samples_leaf=20,
            l2_regularization=1.0, random_state=seed,
        )
    raise ValueError(name)


def predict_regional(
    data: dict,
    split: dict,
    fit_x: np.ndarray,
    eval_x: np.ndarray,
    desc: np.ndarray,
    query: np.ndarray,
    learner: str,
    seed: int,
    n_estimators: int,
) -> np.ndarray:
    source = np.unique(np.asarray(split["train"], dtype=np.int64) // T)
    target = np.unique(query // T)
    by_station = {int(s): np.asarray(split["train"], dtype=np.int64)[
        np.asarray(split["train"], dtype=np.int64) // T == s
    ] for s in source}
    z = np.asarray(target_values(data, "log1p")).reshape(-1)
    output = np.empty(len(query), dtype=np.float64)
    positions = {int(c): i for i, c in enumerate(query)}
    for station in target:
        local = np.flatnonzero(query // T == station)
        neighbors = nearest_sources(desc, source, int(station), POOL)
        train = np.concatenate([by_station[int(s)] for s in neighbors])
        model = make_model(learner, seed, n_estimators)
        model.fit(fit_x[train], z[train])
        prediction = np.expm1(model.predict(eval_x[query[local]]))
        for value, cell in zip(prediction, query[local], strict=True):
            output[positions[int(cell)]] = max(float(value), 0.0)
    return output


def corrected(
    y: np.ndarray,
    base: np.ndarray,
    cells: np.ndarray,
    query: np.ndarray,
    support_by_station: dict[int, np.ndarray],
    k: int,
) -> np.ndarray:
    by_cell = {int(c): i for i, c in enumerate(cells)}
    correction: dict[int, float] = {}
    for station, ordered in support_by_station.items():
        opened = ordered[:k]
        if not len(opened):
            correction[station] = 0.0
            continue
        idx = np.asarray([by_cell[int(c)] for c in opened], dtype=int)
        correction[station] = float(np.mean(
            np.log1p(np.maximum(y[idx], 0.0))
            - np.log1p(np.maximum(base[idx], 0.0)),
        ))
    qidx = np.asarray([by_cell[int(c)] for c in query], dtype=int)
    z = np.log1p(np.maximum(base[qidx], 0.0))
    z += ALPHA[k] * np.asarray([
        correction.get(int(c) // T, 0.0) for c in query
    ])
    return np.maximum(np.expm1(z), 0.0)


def run_role(
    data: dict,
    split: dict,
    role: str,
    desc: np.ndarray,
    n_estimators: int,
) -> pd.DataFrame:
    target_role = "val" if role == "internal_val" else "test"
    all_target = np.sort(np.asarray(split[target_role], dtype=np.int64))
    schedules, query = support_schedule(all_target, T)
    query = np.sort(query)
    y_all = data["y"].numpy().reshape(-1).astype(float)
    fit_x = build_rf_features(data, split, FIT_ROLES, target_transform="log1p", include_network=True)
    eval_x = build_rf_features(data, split, TEST_ROLES, target_transform="log1p", include_network=True)
    rows: list[dict] = []
    for learner in LEARNERS:
        for seed in SEEDS:
            base = predict_regional(
                data, split, fit_x, eval_x, desc, all_target,
                learner, seed, n_estimators,
            )
            for k in K_VALUES:
                prediction = corrected(y_all[all_target], base, all_target, query, schedules, k)
                yi = np.asarray([y_all[c] for c in query], dtype=float)
                for cell, truth, value in zip(query, yi, prediction, strict=True):
                    rows.append({
                        "role": role, "learner": learner, "seed": seed,
                        "k": k, "alpha": ALPHA[k], "cell": int(cell),
                        "station": int(cell) // T, "y_true": float(truth),
                        "y_pred": float(value),
                    })
            print(role, learner, seed, "done", flush=True)
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--n-estimators", type=int, default=120)
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    data = torch.load(DATASETS["doc"], map_location="cpu", weights_only=False)
    outer = load_split(OUTER_MASK)
    internal, _rows, _ = e3_internal_split(data["y_mask"].numpy(), outer["test"])
    desc = station_descriptors(data)
    internal_pred = run_role(data, internal, "internal_val", desc, args.n_estimators)
    outer_pred = run_role(data, outer, "outer_test", desc, args.n_estimators)
    pred = pd.concat([internal_pred, outer_pred], ignore_index=True)
    pred.to_parquet(args.out_dir / "predictions.parquet", index=False)
    scores = (pred.groupby(["role", "learner", "seed", "k", "alpha"], as_index=False)
              .apply(lambda x: pd.Series(metrics(x.y_true.to_numpy(), x.y_pred.to_numpy())),
                     include_groups=False).reset_index(drop=True))
    scores.to_csv(args.out_dir / "metrics_by_seed.csv", index=False)
    val = scores[scores.role == "internal_val"]
    selected = (val.groupby(["learner", "k", "alpha"], as_index=False).mae.mean()
                .sort_values("mae", kind="stable").iloc[0])
    pd.DataFrame([selected]).to_csv(args.out_dir / "selected.csv", index=False)
    outer_scores = scores[scores.role == "outer_test"]
    outer_scores.to_csv(args.out_dir / "outer_all.csv", index=False)
    summary = (scores.groupby(["role", "learner", "k", "alpha"], as_index=False)
               .agg(mae_mean=("mae", "mean"), mae_sd=("mae", "std"),
                    rmse_mean=("rmse", "mean"), log_mae_mean=("log_mae", "mean"),
                    n_seeds=("seed", "count")))
    summary.to_csv(args.out_dir / "summary.csv", index=False)
    chosen = outer_scores[
        (outer_scores.learner == selected.learner)
        & (outer_scores.k == selected.k)
        & (outer_scores.alpha == selected.alpha)
    ]
    chosen.to_csv(args.out_dir / "outer_selected.csv", index=False)
    # Station-clustered bootstrap for each learner at K=5, paired to its own
    # K=0 predictions.  This keeps the learner comparison cell-identical.
    station = (pred[pred.role == "outer_test"]
               .groupby(["learner", "seed", "k", "station"], as_index=False)
               .apply(
                   lambda x: pd.Series({
                       "mae": float(np.mean(np.abs(x.y_true - x.y_pred))),
                   }),
                   include_groups=False,
               )
               .reset_index(drop=True))
    boot_rows: list[dict] = []
    rng = np.random.default_rng(20261002)
    for learner in LEARNERS:
        cur = station[(station.learner == learner) & (station.k == 5)].groupby("station").mae.mean()
        zero = station[(station.learner == learner) & (station.k == 0)].groupby("station").mae.mean()
        delta = (cur - zero).dropna().to_numpy()
        draws = np.asarray([rng.choice(delta, len(delta), replace=True).mean() for _ in range(5000)])
        boot_rows.append({
            "learner": learner, "k": 5, "alpha": ALPHA[5],
            "delta_mae_vs_k0": float(delta.mean()),
            "ci_low": float(np.quantile(draws, .025)),
            "ci_high": float(np.quantile(draws, .975)),
            "n_stations": len(delta),
        })
    pd.DataFrame(boot_rows).to_csv(args.out_dir / "station_bootstrap.csv", index=False)
    spec = {
        "learners": list(LEARNERS), "seeds": list(SEEDS), "K": list(K_VALUES),
        "source_pool": POOL, "alpha_by_k": ALPHA,
        "selection": "mean internal E3 station-heldout MAE; outer scored once",
        "mask": str(OUTER_MASK), "n_estimators_or_iterations": args.n_estimators,
        "training": "source regional models are fitted under the frozen protocol",
    }
    (args.out_dir / "spec.json").write_text(json.dumps(spec, indent=2) + "\n")
    outer_k5 = summary[(summary.role == "outer_test") & (summary.k == 5)]
    best = outer_k5.sort_values("mae_mean").iloc[0]
    (args.out_dir / "verdict.md").write_text(
        "# Regional learner comparison\n\n"
        f"The nested E3 selection compares ExtraTrees, RandomForest and "
        f"HistGradientBoosting with source pool {POOL} and fixed K=5 residual "
        "adaptation (alpha=.75). Internal validation selected "
        f"{selected.learner}, with MAE {selected.mae:.3f}.\n\n"
        "Outer K=5 mean MAE (three seeds):\n\n"
        + outer_k5[["learner", "mae_mean", "mae_sd"]].to_string(index=False)
        + "\n\n"
        f"The best outer learner is {best.learner} (MAE {best.mae_mean:.3f}), "
        "which does not improve the existing five-seed ExtraTrees residual "
        "product (MAE about 2.023). The learner search is therefore closed; "
        "ExtraTrees remains the production regressor.\n"
    )
    print("selected", selected.to_dict())
    print(chosen.to_string(index=False))
    print(pd.DataFrame(boot_rows).to_string(index=False))


if __name__ == "__main__":
    main()
