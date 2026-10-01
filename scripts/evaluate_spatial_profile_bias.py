"""Nested profile-bias correction for the spatial source-selection model.

A global ExtraTrees context prediction is corrected by a descriptor-nearest,
station-OOF residual profile.  k and alpha are selected on a station-held-out
split derived from E3; the outer E3 stations are evaluated once afterwards.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from river_graph.experiments.evaluate import metrics
from river_graph.experiments.h3_masks import e3_internal_split
from river_graph.experiments.transfer import DATASETS
from river_graph.models.kgml_local_transport import (
    FIT_ROLES,
    TEST_ROLES,
    fit_rf_artifacts,
    target_values,
)
from scripts.run_spatial_source_selection import (
    OUTER_MASK,
    load_split,
    nearest_sources,
    station_descriptors,
)


def correction(data: dict, split: dict, desc: np.ndarray, rf, query: np.ndarray,
               base_z: np.ndarray, source: np.ndarray, k: int, alpha: float) -> np.ndarray:
    """Return raw predictions after descriptor-nearest OOF profile correction."""
    t = data["y"].shape[1]
    oof = np.asarray(rf.context_oof_z).reshape(-1)
    y_z = target_values(data, "log1p").reshape(-1)
    train = np.asarray(split["train"], dtype=np.int64)
    station_resid = {}
    for station in source:
        cells = train[train // t == station]
        valid = np.isfinite(oof[cells])
        station_resid[int(station)] = float(np.mean(y_z[cells][valid] - oof[cells][valid])) if valid.any() else 0.0
    out_z = base_z.copy()
    for station in np.unique(query // t):
        q = query // t == station
        neigh = nearest_sources(desc, source, int(station), k)
        # Equal weights avoid a distance scale hyperparameter; alpha controls
        # the strength and is selected only on internal station validation.
        profile = float(np.mean([station_resid[int(s)] for s in neigh]))
        out_z[q] += alpha * profile
    return rf.inverse_std(out_z)


def fit_eval(data: dict, split: dict, desc: np.ndarray, query_role: str,
             seeds: list[int], ks: list[int], alphas: list[float], n_estimators: int,
             leaf: int, jobs: int) -> pd.DataFrame:
    rows = []
    target = np.asarray(split[query_role], dtype=np.int64)
    roles = FIT_ROLES if query_role == "val" else TEST_ROLES
    for seed in seeds:
        rf = fit_rf_artifacts(data, split, target_transform="log1p", seed=seed,
                              n_estimators=n_estimators, n_jobs=jobs,
                              forest_backend="extra_trees", min_samples_leaf=leaf,
                              max_features=1.0)
        _local, base_z = rf.predict_std(data, split, roles)
        base_z = base_z.reshape(-1)[target]
        source = np.unique(np.asarray(split["train"], dtype=np.int64) // data["y"].shape[1])
        y = data["y"].numpy().reshape(-1)[target]
        for k in ks:
            for alpha in alphas:
                pred = correction(data, split, desc, rf, target, base_z, source, k, alpha)
                rows.append({"seed": seed, "k": k, "alpha": alpha,
                             "role": query_role, **metrics(y, pred)})
    return pd.DataFrame(rows)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44])
    p.add_argument("--ks", nargs="+", type=int, default=[10, 20, 40, 80])
    p.add_argument("--alphas", nargs="+", type=float, default=[-0.5, -0.25, 0.0, 0.25, 0.5, 0.75, 1.0])
    p.add_argument("--n-estimators", type=int, default=200)
    p.add_argument("--leaf", type=int, default=4)
    p.add_argument("--jobs", type=int, default=4)
    a = p.parse_args()
    data = torch.load(DATASETS["doc"], map_location="cpu", weights_only=False)
    outer = load_split(OUTER_MASK)
    internal, _rows, _cand = e3_internal_split(data["y_mask"].numpy(), outer["test"])
    desc = station_descriptors(data)
    val = fit_eval(data, internal, desc, "val", a.seeds, a.ks, a.alphas,
                   a.n_estimators, a.leaf, a.jobs)
    selected = (val.groupby(["k", "alpha"], as_index=False).mae.mean()
                   .sort_values("mae").iloc[0])
    k, alpha = int(selected.k), float(selected.alpha)
    # Evaluate the selected pair on outer E3 only after internal selection.
    test = fit_eval(data, outer, desc, "test", a.seeds, [k], [alpha],
                    a.n_estimators, a.leaf, a.jobs)
    a.out_dir.mkdir(parents=True, exist_ok=True)
    val.to_csv(a.out_dir / "internal_profile_grid.csv", index=False)
    test.to_csv(a.out_dir / "outer_profile_test.csv", index=False)
    (a.out_dir / "spec.json").write_text(json.dumps({"selected_k": k, "selected_alpha": alpha,
        "selection": "internal station heldout only", "n_estimators": a.n_estimators,
        "leaf": a.leaf}, indent=2) + "\n")
    print(val.groupby(["k", "alpha"]).mae.mean().sort_values().head(12).to_string())
    print("selected", k, alpha)
    print(test.to_string(index=False))


if __name__ == "__main__":
    main()
