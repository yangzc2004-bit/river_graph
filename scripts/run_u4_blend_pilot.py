"""Validation-safe blend pilot: temporal GNN versus temporal RF.

The RF and GNN are fitted with train/context visibility.  The validation
weight is selected without exposing validation labels to either model input;
the test view is evaluated once after the weight is frozen.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from run_temporal_h2x import _split, _target_mask_path
from run_temporal_rf_upgrade import _features
from sklearn.ensemble import RandomForestRegressor

from river_graph.experiments.evaluate import metrics
from river_graph.experiments.provenance import (
    config_hash,
    file_identity,
    runtime_code_snapshot_sha256,
)
from river_graph.experiments.temporal_h2x import (
    TARGET_TRANSFORMS,
    H2XTemporalModel,
    _as_tensor,
)
from river_graph.experiments.transfer import DATASETS

FIT_ROLES = ("train", "context")
TEST_ROLES = ("train", "val", "context")
ALPHAS = np.linspace(0.0, 1.0, 21)
PROTOCOL = Path("experiments/phase4_transfer/model_upgrade_v1/u4_blend_protocol.json")


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _cells(split: dict[str, np.ndarray], roles: tuple[str, ...]) -> np.ndarray:
    return np.concatenate([np.asarray(split.get(r, []), dtype=np.int64) for r in roles])


def run_one(analyte: str, mask_name: str, seed: int, out_root: Path,
            protocol: Path) -> dict:
    dataset_path = Path(DATASETS[analyte])
    dataset = torch.load(dataset_path, map_location="cpu", weights_only=False)
    mask_path = _target_mask_path(analyte, mask_name, dataset)
    with np.load(mask_path, allow_pickle=False) as archive:
        split = _split({k: archive[k] for k in archive.files})
    y = _as_tensor(dataset["y"]).numpy().reshape(-1)
    val = np.asarray(split["val"], dtype=np.int64)
    test = np.asarray(split["test"], dtype=np.int64)

    # RF uses exactly the U3 feature builder and the two visibility views.
    fit_x = _features(dataset, split, FIT_ROLES)
    test_x = _features(dataset, split, TEST_ROLES)
    target_transform = TARGET_TRANSFORMS[analyte]
    y_fit = np.log1p(np.maximum(y, 0.0))
    rf = RandomForestRegressor(n_estimators=200, n_jobs=-1, random_state=seed)
    train = np.asarray(split["train"], dtype=np.int64)
    rf.fit(fit_x[train], y_fit[train])
    rf_val = np.expm1(rf.predict(fit_x[val]))
    rf_test = np.expm1(rf.predict(test_x[test]))

    # GNN fit is unchanged; only the prediction visibility differs.
    gnn = H2XTemporalModel(seed=seed, lookback=12, temporal_hidden=64,
                           hidden=64, layers=2, dropout=0.1, lr=1e-3,
                           max_epochs=30, patience=6,
                           target_transform=target_transform)
    gnn.fit(dataset, split)
    fit_visible = _cells(split, FIT_ROLES)
    gnn_val = gnn.predict(only_visible=fit_visible).reshape(-1)[val]
    gnn_test = gnn.predict().reshape(-1)[test]

    val_scores = []
    for alpha in ALPHAS:
        pred = alpha * rf_val + (1.0 - alpha) * gnn_val
        val_scores.append({"alpha_rf": float(alpha), "val_mae": metrics(y[val], pred)["mae"]})
    best = min(val_scores, key=lambda row: (row["val_mae"], -row["alpha_rf"]))
    alpha = float(best["alpha_rf"])
    rf_metric = metrics(y[test], rf_test)
    gnn_metric = metrics(y[test], gnn_test)
    blend_test = alpha * rf_test + (1.0 - alpha) * gnn_test
    blend_metric = metrics(y[test], blend_test)

    name = f"blend__{analyte}__{mask_name}__seed{seed}"
    run_dir = out_root / "runs" / name
    run_dir.mkdir(parents=True, exist_ok=True)
    predictions = pd.DataFrame({
        "analyte": analyte, "cell": test, "y_true": y[test],
        "rf_pred": rf_test, "gnn_pred": gnn_test, "blend_pred": blend_test,
        "alpha_rf": alpha, "mask": mask_name, "seed": seed,
    })
    predictions.to_parquet(run_dir / "test_predictions.parquet", index=False)
    validation = pd.DataFrame(val_scores)
    validation.to_csv(run_dir / "validation_alpha_grid.csv", index=False)
    runtime_hash = runtime_code_snapshot_sha256()
    params = {
        "script": "scripts/run_u4_blend_pilot.py", "model_name": "validation_selected_blend",
        "analyte": analyte, "mask": mask_name, "seed": seed,
        "rf_estimators": 200, "gnn_max_epochs": 30, "gnn_patience": 6,
        "fit_roles": FIT_ROLES, "test_roles": TEST_ROLES,
        "target_transform": target_transform,
        "dataset_sha256": file_identity(dataset_path)["sha256"],
        "mask_sha256": file_identity(mask_path)["sha256"],
        "protocol_sha256": _sha(protocol), "runtime_snapshot_hash": runtime_hash,
    }
    params["config_hash_version"] = 3
    params["config_hash"] = config_hash(params, version=3)
    meta = {
        "run": name, "config": params, "alpha_rf": alpha,
        "validation_best": best, "validation_grid": val_scores,
        "metrics": {"temporal_rf": rf_metric, "h2x_t": gnn_metric, "blend": blend_metric},
        "started_at": datetime.now(timezone.utc).isoformat(),
        "runtime_code_snapshot_sha256": runtime_hash,
        "prediction_sha256": file_identity(run_dir / "test_predictions.parquet")["sha256"],
    }
    (run_dir / "meta.json").write_text(json.dumps(meta, indent=2) + "\n")
    return {"run": name, "analyte": analyte, "mask": mask_name, "seed": seed,
            "alpha_rf": alpha, "rf_mae": rf_metric["mae"],
            "gnn_mae": gnn_metric["mae"], "blend_mae": blend_metric["mae"],
            "blend_minus_rf_mae": blend_metric["mae"] - rf_metric["mae"],
            "val_mae": best["val_mae"], "status": "completed"}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default="experiments/phase4_transfer/model_upgrade_v1/u4_blend_pilot")
    ap.add_argument("--analytes", nargs="+", default=["doc", "spec_conductance"])
    ap.add_argument("--masks", nargs="+", default=["e2a_strict"])
    ap.add_argument("--seeds", nargs="+", type=int, default=[42])
    ap.add_argument("--protocol", type=Path, default=PROTOCOL)
    args = ap.parse_args()
    out = Path(args.out_dir)
    rows = []
    for analyte in args.analytes:
        for mask in args.masks:
            for seed in args.seeds:
                print(f"[u4 blend] {analyte} {mask} seed={seed}", flush=True)
                rows.append(run_one(analyte, mask, seed, out, args.protocol))
    pd.DataFrame(rows).to_csv(out / "metrics.csv", index=False)
    print(pd.DataFrame(rows).to_string(index=False))


if __name__ == "__main__":
    main()
