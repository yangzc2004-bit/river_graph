"""Run the U3 temporal random-forest baseline on the frozen target masks."""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from run_temporal_h2x import _split, _target_mask_path
from sklearn.ensemble import RandomForestRegressor

from river_graph.experiments.evaluate import metrics
from river_graph.experiments.provenance import (
    config_hash,
    file_identity,
    run_identity_sha256,
    runtime_code_snapshot_sha256,
)
from river_graph.experiments.temporal_h2x import _as_tensor
from river_graph.experiments.transfer import ANALYTES, DATASETS

FIT_VISIBILITY = ("train", "context")
TEST_VISIBILITY = ("train", "val", "context")


def _visible(split: dict[str, np.ndarray], roles: tuple[str, ...], n: int, t: int):
    out = np.zeros(n * t, dtype=bool)
    for role in roles:
        out[np.asarray(split.get(role, []), dtype=np.int64)] = True
    return out.reshape(n, t)


def _features(dataset: dict, split: dict[str, np.ndarray], roles: tuple[str, ...]):
    y = _as_tensor(dataset["y"]).numpy()
    x = _as_tensor(dataset["x"]).numpy()
    xm = _as_tensor(dataset["x_mask"]).numpy()
    regime = _as_tensor(dataset["regime"]).numpy()
    static = _as_tensor(dataset["static"]).numpy()
    n, t = y.shape
    visible = _visible(split, roles, n, t)
    y_log = np.log1p(np.maximum(y, 0.0))
    values = np.where(visible, y_log, 0.0)
    counts = visible.astype(float)
    month_idx = np.asarray(dataset["months"], dtype="datetime64[M]").astype(int) % 12
    phase = np.column_stack((np.sin(2 * np.pi * month_idx / 12),
                             np.cos(2 * np.pi * month_idx / 12)))
    edge = dataset["edge_index"].numpy()
    upstream = np.zeros((n, n), dtype=float)
    upstream[edge[1], edge[0]] = 1.0
    downstream = upstream.T
    rows = []
    for i in range(n):
        for j in range(t):
            own = float(visible[i, j])
            total_count = counts[:, j].sum() - own
            total_sum = values[:, j].sum() - values[i, j]
            current_global = total_sum / max(total_count, 1.0)
            up_count = float((upstream[i] * counts[:, j]).sum())
            up_sum = float((upstream[i] * values[:, j]).sum())
            down_count = float((downstream[i] * counts[:, j]).sum())
            down_sum = float((downstream[i] * values[:, j]).sum())
            row = [*x[i, j], *xm[i, j], *phase[j], *static[i], *regime[i],
                   current_global, total_count / max(n - 1, 1),
                   up_sum / max(up_count, 1.0), up_count / max(upstream[i].sum(), 1.0),
                   down_sum / max(down_count, 1.0),
                   down_count / max(downstream[i].sum(), 1.0)]
            for lag in (1, 3, 6, 12):
                k = j - lag
                if k >= 0 and visible[i, k]:
                    row.extend([y_log[i, k], 1.0, float(lag)])
                else:
                    row.extend([0.0, 0.0, float(lag)])
            rows.append(row)
    return np.asarray(rows, dtype=np.float32)


def run_one(analyte: str, mask_name: str, seed: int, out_root: Path,
            n_estimators: int = 200):
    dataset_path = DATASETS[analyte]
    dataset = torch.load(dataset_path, map_location="cpu", weights_only=False)
    mask_path = _target_mask_path(analyte, mask_name, dataset)
    with np.load(mask_path, allow_pickle=False) as archive:
        split = _split({k: archive[k] for k in archive.files})
    n, t = dataset["y"].shape
    train = split["train"]
    test = split["test"]
    fit_x = _features(dataset, split, FIT_VISIBILITY)
    test_x = _features(dataset, split, TEST_VISIBILITY)
    y = np.log1p(np.maximum(dataset["y"].numpy(), 0.0)).reshape(-1)
    rf = RandomForestRegressor(n_estimators=n_estimators, n_jobs=-1,
                               random_state=seed)
    rf.fit(fit_x[train], y[train])
    pred = np.full(n * t, np.nan, dtype=float)
    pred[test] = np.expm1(rf.predict(test_x[test]))
    truth = dataset["y"].numpy().reshape(-1)
    metric = metrics(truth[test], pred[test])
    name = f"temporal_rf__{analyte}__{mask_name}__seed{seed}"
    run_dir = out_root / "runs" / name
    run_dir.mkdir(parents=True, exist_ok=True)
    roles = np.full(n * t, "", dtype=object)
    for role, cells in split.items():
        roles[cells] = role
    full = pd.DataFrame({
        "analyte": analyte,
        "station": np.repeat([str(s) for s in dataset["site_no"]], t),
        "month": np.tile([str(m) for m in dataset["months"]], n),
        "y_true": truth,
        "y_pred": pred,
        "split": roles,
        "model_name": "temporal_rf",
        "mask": mask_name,
        "seed": seed,
    })
    full.to_parquet(run_dir / "full_grid.parquet", index=False)
    started_at = datetime.now(timezone.utc).isoformat()
    runtime_hash = runtime_code_snapshot_sha256()
    params = {
        "model_name": "temporal_rf", "analyte": analyte, "mask": mask_name,
        "seed": seed, "n_estimators": n_estimators,
        "features": "hydro+ecology+current_visible_network_context+target_lags_1_3_6_12",
        "fit_visibility": FIT_VISIBILITY, "test_visibility": TEST_VISIBILITY,
        "dataset_sha256": file_identity(dataset_path)["sha256"],
        "mask_sha256": file_identity(mask_path)["sha256"],
        "runtime_snapshot_hash": runtime_hash,
    }
    params["config_hash_version"] = 3
    params["config_hash"] = config_hash(params, version=3)
    meta = {"run": name, "config": params, "metrics": metric,
            "started_at": started_at,
            "runtime_code_snapshot_sha256": runtime_hash,
            "run_identity_sha256": run_identity_sha256(
                params["config_hash"], started_at, runtime_hash
            ),
            "full_grid_sha256": file_identity(run_dir / "full_grid.parquet")["sha256"],
            "rows_full_grid": len(full)}
    (run_dir / "meta.json").write_text(json.dumps(meta, indent=2) + "\n")
    (run_dir / "metrics.json").write_text(json.dumps(metric, indent=2) + "\n")
    return {"run": name, "analyte": analyte, "mask": mask_name, "seed": seed,
            **metric}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default="experiments/phase4_transfer/model_upgrade_v1/u3_temporal_rf")
    ap.add_argument("--analytes", nargs="+", choices=ANALYTES, default=ANALYTES)
    ap.add_argument("--masks", nargs="+", default=["e2a_strict", "e3_spatial_seed42"])
    ap.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44])
    args = ap.parse_args()
    out = Path(args.out_dir)
    rows = [run_one(a, m, s, out) for a in args.analytes for m in args.masks for s in args.seeds]
    pd.DataFrame(rows).to_csv(out / "metrics.csv", index=False)
    print(pd.DataFrame(rows).groupby(["analyte", "mask"], as_index=False).mae.mean().to_string(index=False))


if __name__ == "__main__":
    main()
