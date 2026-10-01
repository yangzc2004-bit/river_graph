"""Run the minimal station-blocked spatial residual transfer pilot.

The model is a fixed RF-context base plus a forest trained only on
station-blocked OOF residuals from source train stations.  The residual forest
uses hydro, ecology, static and calendar features, so held-out station DOC
labels never enter its input or target.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from river_graph.experiments.evaluate import metrics
from river_graph.experiments.provenance import sha256_file
from river_graph.experiments.temporal_h2x import TARGET_TRANSFORMS
from river_graph.experiments.transfer import DATASETS
from river_graph.models.kgml_local_transport import (
    FIT_ROLES,
    TEST_ROLES,
    fit_rf_artifacts,
)
from river_graph.models.spatial_residual_transfer import (
    SIMILARITY_FEATURE_NAMES,
    SpatialResidualTransfer,
)

DEFAULT_MASKS = {
    "e3_spatial_seed42": Path("experiments/masks_stcore_v1/e3_spatial_seed42.npz"),
    "e3_spatial_validation_supportmatched": Path(
        "experiments/phase4_transfer/kgml_local_transport_v1/"
        "spatial_validation_e3_supportmatched/masks/"
        "e3_spatial_validation_supportmatched.npz"
    ),
}


def digest(value: object) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, default=str).encode()).hexdigest()


def load_task(mask_name: str) -> tuple[dict, dict, Path]:
    data = torch.load(DATASETS["doc"], map_location="cpu", weights_only=False)
    mask_path = DEFAULT_MASKS.get(mask_name, Path(mask_name))
    if not mask_path.exists():
        raise FileNotFoundError(mask_path)
    with np.load(mask_path, allow_pickle=False) as saved:
        observed = data["y_mask"].numpy().reshape(-1)
        split = {
            role: np.asarray(saved[role], dtype=np.int64)
            for role in ("train", "val", "test", "context") if role in saved
        }
    for role, cells in split.items():
        if len(cells) and not observed[cells].all():
            raise ValueError(f"{role} contains an unobserved DOC cell")
    return data, split, mask_path


def product_frame(data: dict, split: dict, comp: dict, roles: tuple[str, ...]) -> pd.DataFrame:
    _n, t = data["y"].shape
    cells = np.asarray(split["test"], dtype=np.int64)
    station = np.asarray(data["site_no"], dtype=str)[cells // t]
    month = np.asarray(data["months"], dtype=str)[cells % t]
    y_true = data["y"].numpy().reshape(-1)[cells]
    return pd.DataFrame({
        "cell": cells,
        "station": station,
        "month": month,
        "y_true": y_true,
        "base_pred": comp["base_pred"].reshape(-1)[cells],
        "final_pred": comp["final_pred"].reshape(-1)[cells],
        "graph_delta": comp["graph_delta"].reshape(-1)[cells],
        "graph_delta_abs": comp["graph_delta_abs"].reshape(-1)[cells],
        "visibility_role": "test",
        "input_roles": "+".join(roles),
        "model_name": "spatial_residual_transfer",
    })


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mask", default="e3_spatial_seed42", choices=tuple(DEFAULT_MASKS) )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-estimators", type=int, default=200)
    parser.add_argument("--n-jobs", type=int, default=4)
    parser.add_argument(
        "--out-dir", type=Path,
        default=Path("experiments/phase4_transfer/spatial_residual_transfer_v1/smoke"),
    )
    args = parser.parse_args()
    data, split, mask_path = load_task(args.mask)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    rf = fit_rf_artifacts(
        data, split, target_transform=TARGET_TRANSFORMS["doc"], seed=args.seed,
        n_estimators=args.n_estimators, n_jobs=args.n_jobs,
    )
    model = SpatialResidualTransfer(
        seed=args.seed, n_estimators=args.n_estimators, n_jobs=args.n_jobs,
    )
    model.fit(data, split, rf=rf)
    comp = model.predict_components(TEST_ROLES)
    frame = product_frame(data, split, comp, TEST_ROLES)
    base_metrics = metrics(frame.y_true.to_numpy(), frame.base_pred.to_numpy())
    final_metrics = metrics(frame.y_true.to_numpy(), frame.final_pred.to_numpy())
    meta = {
        "model": "station_blocked_spatial_residual_transfer",
        "analyte": "doc",
        "mask": args.mask,
        "seed": args.seed,
        "dataset_path": DATASETS["doc"],
        "mask_path": str(mask_path),
        "dataset_sha256": sha256_file(DATASETS["doc"]),
        "mask_sha256": sha256_file(mask_path),
        "target_transform": TARGET_TRANSFORMS["doc"],
        "fit_roles": list(FIT_ROLES),
        "test_roles": list(TEST_ROLES),
        "residual_target": "DOC_z - RF_context_OOF_z, source train stations only",
        "similarity_features": list(SIMILARITY_FEATURE_NAMES),
        "source_cells": len(model.source_cells),
        "source_stations": len(model.source_stations),
        "test_stations": int(frame.station.nunique()),
        "rf_context_metrics": base_metrics,
        "transfer_metrics": final_metrics,
        "config_hash": digest({"mask": args.mask, "seed": args.seed,
                                "n_estimators": args.n_estimators, "n_jobs": args.n_jobs}),
    }
    frame.to_parquet(args.out_dir / "test_predictions.parquet", index=False)
    (args.out_dir / "metrics.json").write_text(json.dumps(meta, indent=2) + "\n")
    print(pd.DataFrame([
        {"arm": "rf_context", **base_metrics},
        {"arm": "spatial_residual_transfer", **final_metrics},
    ]).to_string(index=False))
    print(f"saved {args.out_dir}")


if __name__ == "__main__":
    main()
