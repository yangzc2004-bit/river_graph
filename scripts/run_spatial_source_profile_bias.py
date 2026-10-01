"""Profile-bias correction on top of the source-station regional expert.

The regional expert (k nearest label-free source stations) is fitted for each
query station.  A station-level residual profile is estimated with a
station-blocked OOF regional expert and added in log-DOC space.  The shrinkage
alpha is selected on the support-matched spatial validation block; the frozen
E3 stations are scored only once afterwards.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from run_spatial_source_selection import (
    OUTER_MASK,
    load_split,
    nearest_sources,
    predict_query_models,
    station_descriptors,
)
from sklearn.ensemble import ExtraTreesRegressor

from river_graph.experiments.evaluate import metrics
from river_graph.experiments.transfer import DATASETS
from river_graph.models.kgml_local_transport import (
    FIT_ROLES,
    TEST_ROLES,
    build_rf_features,
    target_values,
)

T = 654
SUPPORT_MASK = Path(
    "experiments/phase4_transfer/kgml_local_transport_v1/"
    "spatial_validation_e3_supportmatched/masks/"
    "e3_spatial_validation_supportmatched.npz"
)
ALPHAS = (-0.5, -0.25, 0.0, 0.25, 0.5, 0.75, 1.0, 1.25)


def station_cells(split: dict, role: str) -> dict[int, np.ndarray]:
    cells = np.asarray(split[role], dtype=np.int64)
    return {int(s): cells[cells // T == s] for s in np.unique(cells // T)}


def oof_source_residuals(
    data: dict,
    split: dict,
    desc: np.ndarray,
    *,
    seed: int,
    k: int,
    n_estimators: int,
    leaf: int,
    n_folds: int = 5,
    jobs: int = 4,
) -> dict[int, float]:
    """Station-blocked OOF residual profile in log space.

    A fold model is fit on all source stations outside the held-out fold, then
    predicts the held-out stations. This gives an inexpensive, leakage-free
    station profile which is applied to the query-specific k-nearest expert.
    """
    z = target_values(data, "log1p").reshape(-1)
    fit_x = build_rf_features(data, split, FIT_ROLES, target_transform="log1p", include_network=True)
    source_cells = np.asarray(split["train"], dtype=np.int64)
    source_stations = np.unique(source_cells // T)
    rng = np.random.default_rng(202609)
    perm = rng.permutation(source_stations)
    folds = np.array_split(perm, n_folds)
    out: dict[int, float] = {}
    for heldout in folds:
        held = set(map(int, heldout))
        train_stations = np.asarray([s for s in source_stations if int(s) not in held], dtype=np.int64)
        fit_cells = np.asarray([c for c in source_cells if int(c // T) in set(map(int, train_stations))], dtype=np.int64)
        model = ExtraTreesRegressor(
            n_estimators=n_estimators,
            min_samples_leaf=leaf,
            max_features=1.0,
            random_state=seed,
            n_jobs=jobs,
        )
        model.fit(fit_x[fit_cells], z[fit_cells])
        for station in heldout:
            qcells = source_cells[source_cells // T == int(station)]
            pred = model.predict(fit_x[qcells])
            out[int(station)] = float(np.mean(z[qcells] - pred))
    return out


def base_prediction(
    data: dict,
    split: dict,
    desc: np.ndarray,
    role: str,
    *,
    seed: int,
    k: int,
    n_estimators: int,
    leaf: int,
    jobs: int,
) -> tuple[np.ndarray, np.ndarray]:
    fit_x = build_rf_features(data, split, FIT_ROLES, target_transform="log1p", include_network=True)
    eval_x = build_rf_features(data, split, TEST_ROLES, target_transform="log1p", include_network=True)
    q = np.asarray(split[role], dtype=np.int64)
    pred_raw = predict_query_models(
        data,
        split,
        np.asarray(split["train"], dtype=np.int64),
        q,
        fit_x,
        eval_x,
        desc,
        k,
        seed,
        n_estimators,
        leaf,
        jobs,
    )
    z = target_values(data, "log1p").reshape(-1)[q]
    return np.log1p(np.maximum(pred_raw, 0.0)), z


def apply_profile(
    q: np.ndarray,
    base_z: np.ndarray,
    desc: np.ndarray,
    source: np.ndarray,
    residual: dict[int, float],
    *,
    k: int,
    alpha: float,
) -> np.ndarray:
    out = base_z.copy()
    for station in np.unique(q // T):
        take = q // T == station
        neigh = nearest_sources(desc, source, int(station), k)
        profile = float(np.mean([residual.get(int(s), 0.0) for s in neigh]))
        out[take] += alpha * profile
    return np.expm1(out)


def run(args: argparse.Namespace) -> None:
    data = torch.load(DATASETS["doc"], map_location="cpu", weights_only=False)
    desc = station_descriptors(data)
    support = load_split(SUPPORT_MASK)
    outer = load_split(OUTER_MASK)
    rows = []
    # Support-matched validation: select alpha using only its held-out stations.
    for seed in args.seeds:
        residual = oof_source_residuals(
            data, support, desc, seed=seed, k=args.k, n_estimators=args.n_estimators,
            leaf=args.leaf, jobs=args.jobs,
        )
        q = np.asarray(support["val"], dtype=np.int64)
        base_z, true_z = base_prediction(
            data, support, desc, "val", seed=seed, k=args.k,
            n_estimators=args.n_estimators, leaf=args.leaf, jobs=args.jobs,
        )
        source = np.unique(np.asarray(support["train"], dtype=np.int64) // T)
        for alpha in ALPHAS:
            pred = apply_profile(q, base_z, desc, source, residual, k=args.k, alpha=alpha)
            rows.append({"seed": seed, "alpha": alpha, "role": "supportmatched_val", **metrics(np.expm1(true_z), pred)})
    validation = pd.DataFrame(rows)
    selected = float(validation.groupby("alpha").mae.mean().sort_values().index[0])

    test_rows = []
    for seed in args.seeds:
        residual = oof_source_residuals(
            data, outer, desc, seed=seed, k=args.k, n_estimators=args.n_estimators,
            leaf=args.leaf, jobs=args.jobs,
        )
        q = np.asarray(outer["test"], dtype=np.int64)
        base_z, true_z = base_prediction(
            data, outer, desc, "test", seed=seed, k=args.k,
            n_estimators=args.n_estimators, leaf=args.leaf, jobs=args.jobs,
        )
        source = np.unique(np.asarray(outer["train"], dtype=np.int64) // T)
        pred = apply_profile(q, base_z, desc, source, residual, k=args.k, alpha=selected)
        test_rows.append({"seed": seed, "k": args.k, "alpha": selected, "role": "outer_e3_test", **metrics(np.expm1(true_z), pred)})
    test = pd.DataFrame(test_rows)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    validation.to_csv(args.out_dir / "supportmatched_validation.csv", index=False)
    test.to_csv(args.out_dir / "outer_test.csv", index=False)
    (args.out_dir / "spec.json").write_text(json.dumps({
        "model": "source_station_regional_expert_plus_oof_profile_bias",
        "k": args.k,
        "alpha_grid": list(ALPHAS),
        "selected_alpha": selected,
        "selection": "support-matched spatial validation only",
        "oof_profile": "5-fold station-blocked regional expert residuals in log1p space",
        "n_estimators": args.n_estimators,
        "leaf": args.leaf,
        "seeds": args.seeds,
    }, indent=2) + "\n")
    print(validation.groupby("alpha").mae.mean().sort_values().to_string())
    print("selected_alpha", selected)
    print(test.to_string(index=False))


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44])
    p.add_argument("--k", type=int, default=40)
    p.add_argument("--n-estimators", type=int, default=120)
    p.add_argument("--leaf", type=int, default=4)
    p.add_argument("--jobs", type=int, default=4)
    run(p.parse_args())


if __name__ == "__main__":
    main()
