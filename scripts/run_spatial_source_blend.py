"""Validation-selected blend of global and source-similarity forests."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from run_spatial_source_selection import (
    load_split,
    predict_query_models,
    station_descriptors,
)
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
KS = (20, 40, 80, 160)
ALPHAS = (0.0, 0.25, 0.5, 0.75, 1.0)


def fit_global(data: dict, split: dict, cells: np.ndarray, *, seed: int,
               n_estimators: int) -> np.ndarray:
    z = target_values(data, "log1p").reshape(-1)
    x_fit = build_rf_features(data, split, FIT_ROLES, target_transform="log1p",
                              include_network=True)
    x_eval = build_rf_features(data, split, TEST_ROLES, target_transform="log1p",
                               include_network=True)
    model = ExtraTreesRegressor(n_estimators=n_estimators, min_samples_leaf=4,
                                max_features=1.0, random_state=seed, n_jobs=4)
    source = np.unique(np.asarray(split["train"], dtype=np.int64))
    model.fit(x_fit[source], z[source])
    return np.expm1(model.predict(x_eval[cells]))


def run(args: argparse.Namespace) -> None:
    data = torch.load(DATASETS["doc"], map_location="cpu", weights_only=False)
    outer = load_split(OUTER_MASK)
    desc = station_descriptors(data)
    internal, _rows, _ = e3_internal_split(data["y_mask"].numpy(), outer["test"])
    y = data["y"].numpy().reshape(-1)
    validation: list[dict] = []
    test: list[dict] = []
    for seed in args.seeds:
        for split, role, out in ((internal, "val", validation), (outer, "test", test)):
            cells = np.asarray(split[role], dtype=np.int64)
            global_pred = fit_global(data, split, cells, seed=seed,
                                     n_estimators=args.n_estimators)
            fit_x = build_rf_features(data, split, FIT_ROLES, target_transform="log1p",
                                      include_network=True)
            eval_x = build_rf_features(data, split, TEST_ROLES, target_transform="log1p",
                                       include_network=True)
            for k in KS:
                source_pred = predict_query_models(
                    data, split, np.asarray(split["train"], dtype=np.int64), cells,
                    fit_x, eval_x, desc, k, seed, args.n_estimators, 4, 4,
                )
                for alpha in ALPHAS:
                    pred = (1.0 - alpha) * global_pred + alpha * source_pred
                    out.append({"seed": seed, "k": k, "alpha": alpha, "role": role,
                                **metrics(y[cells], pred)})
    val = pd.DataFrame(validation)
    selected = (val.groupby(["k", "alpha"], as_index=False)["mae"]
                .mean().sort_values("mae").iloc[0])
    test_df = pd.DataFrame(test)
    test_df = test_df[(test_df.k == selected.k) & (test_df.alpha == selected.alpha)]
    args.out_dir.mkdir(parents=True, exist_ok=True)
    val.to_csv(args.out_dir / "internal_validation.csv", index=False)
    test_df.to_csv(args.out_dir / "outer_test.csv", index=False)
    (args.out_dir / "spec.json").write_text(json.dumps({
        "ks": list(KS), "alphas": list(ALPHAS), "seeds": args.seeds,
        "selection": "mean MAE on nested station-heldout validation",
        "n_estimators": args.n_estimators,
    }, indent=2) + "\n")
    print(val.groupby(["k", "alpha"]).mae.mean().sort_values().head(10))
    print("selected", selected.to_dict())
    print(test_df.to_string(index=False))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44])
    parser.add_argument("--n-estimators", type=int, default=120)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
