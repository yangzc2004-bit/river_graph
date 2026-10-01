"""Pilot model selection for the DOC RF-context baseline.

The comparison keeps the feature construction and visibility protocol fixed and
only changes the tree ensemble.  Validation predictions use ``train+context``
features; final test predictions use ``train+val+context`` features, matching
the existing temporal RF protocol.  A model is selected per mask by validation
MAE and the test result is reported once for that selected model.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor

from river_graph.experiments.evaluate import metrics
from river_graph.experiments.provenance import sha256_file
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


def _digest(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), default=str).encode()
    ).hexdigest()


def _load(mask_name: str) -> tuple[dict, dict, Path]:
    dataset_path = Path(DATASETS["doc"])
    data = torch.load(dataset_path, map_location="cpu", weights_only=False)
    mask_path = MASKS[mask_name]
    with np.load(mask_path, allow_pickle=False) as saved:
        split = {k: np.asarray(saved[k], dtype=np.int64)
                 for k in ("train", "val", "test", "context") if k in saved.files}
    y_mask = data["y_mask"].numpy().reshape(-1)
    for role, cells in split.items():
        if not np.asarray(y_mask[cells], dtype=bool).all():
            raise ValueError(f"{mask_name}/{role} contains an unobserved label")
    return data, split, mask_path


def _estimators(seed: int, n_estimators: int, n_jobs: int) -> dict[str, object]:
    common = {"n_estimators": n_estimators, "random_state": seed, "n_jobs": n_jobs}
    return {
        "rf_default": RandomForestRegressor(**common, max_features=1.0, min_samples_leaf=1),
        "rf_leaf2": RandomForestRegressor(**common, max_features=1.0, min_samples_leaf=2),
        "et_default": ExtraTreesRegressor(**common, max_features=1.0, min_samples_leaf=1),
        "et_sqrt": ExtraTreesRegressor(**common, max_features="sqrt", min_samples_leaf=1),
        "et_leaf2": ExtraTreesRegressor(**common, max_features=1.0, min_samples_leaf=2),
        "et_leaf4": ExtraTreesRegressor(**common, max_features=1.0, min_samples_leaf=4),
    }


def run_one(mask_name: str, seed: int, out_dir: Path, *, n_estimators: int, n_jobs: int) -> pd.DataFrame:
    data, split, mask_path = _load(mask_name)
    transform = "log1p"
    target = target_values(data, transform).reshape(-1)
    train = np.asarray(split["train"], dtype=np.int64)
    val = np.asarray(split.get("val", []), dtype=np.int64)
    test = np.asarray(split["test"], dtype=np.int64)
    fit_x = build_rf_features(data, split, FIT_ROLES, target_transform=transform, include_network=True)
    val_x = build_rf_features(data, split, FIT_ROLES, target_transform=transform, include_network=True)
    test_x = build_rf_features(data, split, TEST_ROLES, target_transform=transform, include_network=True)
    if not np.array_equal(fit_x, val_x):
        raise AssertionError("fit and validation feature views unexpectedly differ")
    rows: list[dict] = []
    predictions: list[pd.DataFrame] = []
    for model_name, model in _estimators(seed, n_estimators, n_jobs).items():
        model.fit(fit_x[train], target[train])
        val_pred = np.expm1(model.predict(val_x[val]))
        test_pred = np.expm1(model.predict(test_x[test]))
        y_val = data["y"].numpy().reshape(-1)[val]
        y_test = data["y"].numpy().reshape(-1)[test]
        val_metric = metrics(y_val, val_pred)
        test_metric = metrics(y_test, test_pred)
        rows.extend([
            {"mask": mask_name, "seed": seed, "model": model_name, "role": "val", **val_metric},
            {"mask": mask_name, "seed": seed, "model": model_name, "role": "test", **test_metric},
        ])
        predictions.append(pd.DataFrame({
            "cell": np.concatenate([val, test]),
            "role": ["val"] * len(val) + ["test"] * len(test),
            "y_true": np.concatenate([y_val, y_test]),
            "y_pred": np.concatenate([val_pred, test_pred]),
            "mask": mask_name, "seed": seed, "model": model_name,
        }))
    run = out_dir / "runs" / f"{mask_name}__seed{seed}"
    run.mkdir(parents=True, exist_ok=True)
    pd.concat(predictions, ignore_index=True).to_parquet(run / "predictions.parquet", index=False)
    config = {
        "analyte": "doc", "mask": mask_name, "seed": seed,
        "models": list(_estimators(seed, n_estimators, n_jobs)),
        "n_estimators": n_estimators, "n_jobs": n_jobs,
        "features": "RF-context: hydro+ecology+calendar+current-month network context+target lags",
        "fit_visibility": list(FIT_ROLES), "val_visibility": list(FIT_ROLES),
        "test_visibility": list(TEST_ROLES), "target_transform": transform,
        "dataset_path": str(DATASETS["doc"]), "dataset_sha256": sha256_file(DATASETS["doc"]),
        "mask_path": str(mask_path), "mask_sha256": sha256_file(mask_path),
    }
    meta = {"config": config, "config_hash": _digest(config),
            "prediction_sha256": sha256_file(run / "predictions.parquet")}
    (run / "meta.json").write_text(json.dumps(meta, indent=2) + "\n")
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--masks", nargs="+", choices=tuple(MASKS), default=list(MASKS))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-estimators", type=int, default=200)
    parser.add_argument("--n-jobs", type=int, default=4)
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    frames = [run_one(m, args.seed, args.out_dir,
                      n_estimators=args.n_estimators, n_jobs=args.n_jobs) for m in args.masks]
    table = pd.concat(frames, ignore_index=True)
    table.to_csv(args.out_dir / "metrics.csv", index=False)
    val = table[table.role.eq("val")].sort_values(["mask", "mae"])
    selected = val.groupby("mask", as_index=False).first()[["mask", "model", "mae"]]
    selected = selected.rename(columns={"model": "selected_model", "mae": "selected_val_mae"})
    test = table[table.role.eq("test")].merge(selected[["mask", "selected_model"]],
                                                left_on=["mask", "model"],
                                                right_on=["mask", "selected_model"])
    test = test.rename(columns={"mae": "selected_test_mae"})
    selected.to_csv(args.out_dir / "validation_selection.csv", index=False)
    test.to_csv(args.out_dir / "selected_test_metrics.csv", index=False)
    spec = {"models": list(_estimators(args.seed, args.n_estimators, args.n_jobs)),
            "selection": "minimum validation MAE per mask; test evaluated once",
            "seed": args.seed, "masks": args.masks,
            "fit_visibility": list(FIT_ROLES), "test_visibility": list(TEST_ROLES)}
    (args.out_dir / "spec.json").write_text(json.dumps(spec, indent=2) + "\n")
    print("\nValidation ranking:")
    print(val[["mask", "model", "mae", "r2"]].to_string(index=False))
    print("\nSelected test metrics:")
    print(test[["mask", "selected_model", "selected_test_mae", "r2"]].to_string(index=False))


if __name__ == "__main__":
    main()
