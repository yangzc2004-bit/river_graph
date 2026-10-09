"""Analysis-only expanded tree baseline search for DOC RF-context features."""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor

from river_graph.experiments.evaluate import metrics
from river_graph.experiments.transfer import DATASETS
from river_graph.models.kgml_local_transport import (
    FIT_ROLES,
    TEST_ROLES,
    build_rf_features,
    target_values,
)

MASKS = {
    "e1_r20_seed42": Path("experiments/masks_stcore_v1/e1_r20_seed42.npz"),
    "e2a_strict": Path("experiments/masks_stcore_v1/e2a_strict.npz"),
    "e2b_partial": Path("experiments/masks_stcore_v1/e2b_partial.npz"),
    "e3_spatial_seed42": Path("experiments/masks_stcore_v1/e3_spatial_seed42.npz"),
}


def models(seed: int, n_estimators: int, n_jobs: int) -> dict[str, object]:
    common = {"n_estimators": n_estimators, "random_state": seed, "n_jobs": n_jobs}
    out: dict[str, object] = {}
    for backend, cls in (("rf", RandomForestRegressor), ("et", ExtraTreesRegressor)):
        for mf in (0.25, 0.5):
            for leaf in (2, 4, 8):
                out[f"{backend}_mf{mf:g}_leaf{leaf}"] = cls(
                    **common, max_features=mf, min_samples_leaf=leaf
                )
    return out


def run(mask: str, seed: int, n_estimators: int, n_jobs: int) -> pd.DataFrame:
    data = torch.load(DATASETS["doc"], map_location="cpu", weights_only=False)
    with np.load(MASKS[mask], allow_pickle=False) as z:
        split = {k: np.asarray(z[k], dtype=np.int64) for k in z.files if k in {"train", "val", "test", "context"}}
    target = target_values(data, "log1p").reshape(-1)
    y = data["y"].numpy().reshape(-1)
    train, val, test = split["train"], split["val"], split["test"]
    fit_x = build_rf_features(data, split, FIT_ROLES, target_transform="log1p", include_network=True)
    test_x = build_rf_features(data, split, TEST_ROLES, target_transform="log1p", include_network=True)
    rows: list[dict] = []
    for name, model in models(seed, n_estimators, n_jobs).items():
        model.fit(fit_x[train], target[train])
        vp, tp = np.expm1(model.predict(fit_x[val])), np.expm1(model.predict(test_x[test]))
        rows += [{"mask": mask, "seed": seed, "model": name, "role": "val", **metrics(y[val], vp)},
                 {"mask": mask, "seed": seed, "model": name, "role": "test", **metrics(y[test], tp)}]
    return pd.DataFrame(rows)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--masks", nargs="+", choices=tuple(MASKS), default=list(MASKS))
    p.add_argument("--seeds", nargs="+", type=int, default=[42])
    p.add_argument("--n-estimators", type=int, default=300)
    p.add_argument("--n-jobs", type=int, default=4)
    a = p.parse_args()
    table = pd.concat([run(m, s, a.n_estimators, a.n_jobs) for s in a.seeds for m in a.masks], ignore_index=True)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(a.out, index=False)
    val = table[table.role.eq("val")].sort_values(["seed", "mask", "mae"])
    selected = val.groupby(["seed", "mask"], as_index=False).first()[["seed", "mask", "model", "mae"]]
    test = table[table.role.eq("test")].merge(selected, on=["seed", "mask", "model"], how="inner")
    selected.rename(columns={"model": "selected_model", "mae": "selected_val_mae"}).to_csv(a.out.with_name("selection.csv"), index=False)
    test.rename(columns={"mae": "selected_test_mae"}).to_csv(a.out.with_name("selected_test.csv"), index=False)
    print(selected.rename(columns={"model": "selected_model", "mae": "selected_val_mae"}).to_string(index=False))
    print(test[["seed", "mask", "model", "selected_test_mae"]].to_string(index=False))


if __name__ == "__main__":
    main()
