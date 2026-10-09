"""Station-profile mean correction for DOC spatial transfer.

The spatial test contains whole stations with no target labels.  This script
fits a label-free station-profile model on source station means and uses it to
correct the level of a global ExtraTrees context forecast.  Shrinkage is
chosen on the nested station-held-out validation split before the frozen E3
test is evaluated.
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
from river_graph.experiments.transfer import DATASETS
from river_graph.models.kgml_local_transport import (
    FIT_ROLES,
    TEST_ROLES,
    build_rf_features,
    target_values,
)

T = 654
OUTER_MASK = Path("experiments/masks_stcore_v1/e3_spatial_seed42.npz")
ALPHAS = (0.0, 0.25, 0.5, 0.75, 1.0, 1.25)
LEAVES = (2, 4, 8)


def station_fit_cells(split: dict, roles: tuple[str, ...]) -> np.ndarray:
    return np.unique(np.concatenate([np.asarray(split.get(r, []), dtype=np.int64)
                                      for r in roles]))


def profile_model(data: dict, split: dict, desc: np.ndarray, *, seed: int,
                  leaf: int, n_estimators: int) -> ExtraTreesRegressor:
    z = target_values(data, "log1p").reshape(-1)
    cells = station_fit_cells(split, FIT_ROLES)
    stations = np.unique(cells // T)
    means = np.asarray([z[cells[cells // T == s]].mean() for s in stations])
    model = ExtraTreesRegressor(
        n_estimators=n_estimators, min_samples_leaf=leaf, max_features=1.0,
        random_state=seed, n_jobs=4,
    )
    model.fit(desc[stations], means)
    return model


def fit_global(data: dict, split: dict, *, seed: int, n_estimators: int) -> tuple:
    z = target_values(data, "log1p").reshape(-1)
    x_fit = build_rf_features(data, split, FIT_ROLES, target_transform="log1p",
                              include_network=True)
    x_eval = build_rf_features(data, split, TEST_ROLES, target_transform="log1p",
                               include_network=True)
    model = ExtraTreesRegressor(
        n_estimators=n_estimators, min_samples_leaf=4, max_features=1.0,
        random_state=seed, n_jobs=4,
    )
    cells = station_fit_cells(split, FIT_ROLES)
    model.fit(x_fit[cells], z[cells])
    return model, x_eval, z


def corrected_prediction(data: dict, split: dict, desc: np.ndarray, *, seed: int,
                         alpha: float, leaf: int, n_estimators: int,
                         eval_role: str) -> tuple[np.ndarray, np.ndarray]:
    model, x_eval, z = fit_global(data, split, seed=seed, n_estimators=n_estimators)
    profile = profile_model(data, split, desc, seed=seed, leaf=leaf,
                            n_estimators=max(100, n_estimators // 2))
    cells = np.asarray(split[eval_role], dtype=np.int64)
    base_z = model.predict(x_eval[cells])
    target_stations = cells // T
    prior = profile.predict(desc[target_stations])
    adjusted = base_z.copy()
    for station in np.unique(target_stations):
        take = target_stations == station
        adjusted[take] += alpha * (prior[take][0] - np.mean(base_z[take]))
    return np.expm1(adjusted), z[cells]


def run(args: argparse.Namespace) -> None:
    data = torch.load(DATASETS["doc"], map_location="cpu", weights_only=False)
    outer = load_split(OUTER_MASK)
    desc = station_descriptors(data)
    internal, _val_rows, _ = e3_internal_split(data["y_mask"].numpy(), outer["test"])
    rows: list[dict] = []
    for seed in args.seeds:
        for leaf in LEAVES:
            for alpha in ALPHAS:
                pred, z_true = corrected_prediction(
                    data, internal, desc, seed=seed, alpha=alpha, leaf=leaf,
                    n_estimators=args.n_estimators, eval_role="val",
                )
                truth = np.expm1(z_true)
                rows.append({"seed": seed, "leaf": leaf, "alpha": alpha,
                             "role": "internal_val", **metrics(truth, pred)})
    validation = pd.DataFrame(rows)
    selected = (validation.groupby(["leaf", "alpha"], as_index=False)["mae"]
                .mean().sort_values("mae").iloc[0])
    test_rows: list[dict] = []
    for seed in args.seeds:
        pred, z_true = corrected_prediction(
            data, outer, desc, seed=seed, alpha=float(selected["alpha"]),
            leaf=int(selected["leaf"]), n_estimators=args.n_estimators,
            eval_role="test",
        )
        test_rows.append({"seed": seed, "leaf": int(selected["leaf"]),
                          "alpha": float(selected["alpha"]),
                          "role": "outer_e3_test",
                          **metrics(np.expm1(z_true), pred)})
    test = pd.DataFrame(test_rows)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    validation.to_csv(args.out_dir / "internal_validation.csv", index=False)
    test.to_csv(args.out_dir / "outer_test.csv", index=False)
    (args.out_dir / "spec.json").write_text(json.dumps({
        "selection": "mean MAE on nested station-heldout validation",
        "alphas": list(ALPHAS), "leaves": list(LEAVES), "seeds": args.seeds,
        "n_estimators": args.n_estimators,
        "correction": "log-space station prior minus predicted station mean",
    }, indent=2) + "\n")
    print(validation.groupby(["leaf", "alpha"]).mae.mean().sort_values().head(10))
    print("selected", selected.to_dict())
    print(test.to_string(index=False))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44])
    parser.add_argument("--n-estimators", type=int, default=200)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
