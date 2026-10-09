"""Unlabelled target-station adaptation for DOC spatial transfer.

The predictor remains the current ExtraTrees RF-context model.  A logistic
source-vs-target domain classifier estimates a covariate density ratio from
label-free hydro, ecology, calendar, static and graph-structure features.
Source labels are then reweighted before fitting the target regressor.  The
support-matched spatial block chooses the adaptation strength; the frozen E3
test is evaluated only after that choice.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from river_graph.experiments.evaluate import metrics
from river_graph.experiments.provenance import sha256_file
from river_graph.experiments.temporal_h2x import _as_tensor
from river_graph.experiments.transfer import DATASETS
from river_graph.models.kgml_local_transport import (
    FIT_ROLES,
    TEST_ROLES,
    build_rf_features,
    target_values,
)

DATASET = Path(DATASETS["doc"])
MASKS = {
    "e3_supportmatched": Path(
        "experiments/phase4_transfer/kgml_local_transport_v1/"
        "spatial_validation_e3_supportmatched/masks/"
        "e3_spatial_validation_supportmatched.npz"
    ),
    "e3_test": Path("experiments/masks_stcore_v1/e3_spatial_seed42.npz"),
}
TARGET_TRANSFORM = "log1p"
ALPHAS = (0.0, 0.25, 0.5, 0.75, 1.0)
CLIPS = (2.0, 4.0, 8.0)


def _load(mask_name: str) -> tuple[dict, dict, Path]:
    data = torch.load(DATASET, map_location="cpu", weights_only=False)
    path = MASKS[mask_name]
    with np.load(path, allow_pickle=False) as saved:
        split = {
            role: np.asarray(saved[role], dtype=np.int64)
            for role in ("train", "val", "test", "context") if role in saved.files
        }
    observed = _as_tensor(data["y_mask"]).numpy().reshape(-1).astype(bool)
    for role, cells in split.items():
        if not observed[cells].all():
            raise ValueError(f"{mask_name}/{role} contains an unobserved cell")
    return data, split, path


def label_free_features(data: dict) -> np.ndarray:
    """Cell features that do not use target-analyte values or masks."""
    x = _as_tensor(data["x"]).numpy().astype(np.float32)
    x_mask = _as_tensor(data["x_mask"]).numpy().astype(np.float32)
    n, t, _ = x.shape
    phase = 2 * np.pi * (
        np.asarray(data["months"], dtype="datetime64[M]").astype(int) % 12
    ) / 12
    calendar = np.broadcast_to(
        np.stack([np.sin(phase), np.cos(phase)], axis=-1), (n, t, 2)
    ).astype(np.float32)
    static = _as_tensor(data["static"]).numpy().astype(np.float32)
    regime = _as_tensor(data["regime"]).numpy().astype(np.float32)
    edge = np.asarray(data["edge_index"], dtype=np.int64)
    indegree = np.bincount(edge[1], minlength=n).astype(np.float32)
    outdegree = np.bincount(edge[0], minlength=n).astype(np.float32)
    graph = np.broadcast_to(
        np.stack([indegree, outdegree, indegree + outdegree], axis=-1)[:, None, :],
        (n, t, 3),
    )
    blocks = [x, x_mask, calendar,
              np.broadcast_to(static[:, None, :], (n, t, static.shape[1])),
              np.broadcast_to(regime[:, None, :], (n, t, regime.shape[1])), graph]
    features = np.concatenate(blocks, axis=-1).reshape(n * t, -1)
    if not np.isfinite(features).all():
        raise ValueError("domain features contain nonfinite values")
    return features.astype(np.float32)


def density_ratio(source: np.ndarray, target: np.ndarray, *, seed: int, clip: float) -> np.ndarray:
    x = np.concatenate([source, target], axis=0)
    y = np.concatenate([np.zeros(len(source), dtype=int), np.ones(len(target), dtype=int)])
    classifier = make_pipeline(
        StandardScaler(),
        LogisticRegression(C=1.0, class_weight="balanced", max_iter=500, random_state=seed),
    )
    classifier.fit(x, y)
    p = np.clip(classifier.predict_proba(source)[:, 1], 1e-4, 1 - 1e-4)
    ratio = p / (1.0 - p)
    ratio /= np.mean(ratio)
    return np.clip(ratio, 1.0 / clip, clip)


def fit_predict(data: dict, split: dict, target_cells: np.ndarray, eval_role: str,
                *, seed: int, alpha: float, clip: float, n_estimators: int) -> dict:
    target = target_values(data, TARGET_TRANSFORM).reshape(-1)
    source = np.asarray(split["train"], dtype=np.int64)
    source_domain = label_free_features(data)[source]
    target_domain = label_free_features(data)[target_cells]
    ratio = density_ratio(source_domain, target_domain, seed=seed, clip=clip)
    weights = 1.0 + alpha * (ratio - 1.0)
    x_fit = build_rf_features(data, split, FIT_ROLES, target_transform=TARGET_TRANSFORM,
                              include_network=True)
    x_eval = build_rf_features(data, split, TEST_ROLES, target_transform=TARGET_TRANSFORM,
                               include_network=True)
    model = ExtraTreesRegressor(
        n_estimators=n_estimators, min_samples_leaf=4, max_features=1.0,
        n_jobs=4, random_state=seed,
    )
    model.fit(x_fit[source], target[source], sample_weight=weights)
    eval_cells = np.asarray(split[eval_role], dtype=np.int64)
    prediction = np.expm1(model.predict(x_eval[eval_cells]))
    truth = _as_tensor(data["y"]).numpy().reshape(-1)[eval_cells]
    return {
        "metrics": metrics(truth, prediction),
        "n_source": len(source), "n_target_domain": len(target_cells),
        "mean_ratio": float(np.mean(ratio)), "sd_ratio": float(np.std(ratio)),
    }


def run_validation(args: argparse.Namespace) -> pd.DataFrame:
    data, split, _ = _load("e3_supportmatched")
    rows = []
    target_cells = np.asarray(split["val"], dtype=np.int64)
    for seed in args.seeds:
        for alpha in ALPHAS:
            for clip in CLIPS:
                result = fit_predict(data, split, target_cells, "val", seed=seed,
                                     alpha=alpha, clip=clip, n_estimators=args.n_estimators)
                rows.append({"seed": seed, "alpha": alpha, "clip": clip,
                             **result["metrics"], "mean_ratio": result["mean_ratio"],
                             "sd_ratio": result["sd_ratio"]})
    return pd.DataFrame(rows)


def run_test(args: argparse.Namespace, selected: pd.Series) -> pd.DataFrame:
    data, split, _ = _load("e3_test")
    target_cells = np.asarray(split["test"], dtype=np.int64)
    rows = []
    for seed in args.seeds:
        result = fit_predict(
            data, split, target_cells, "test", seed=seed,
            alpha=float(selected["alpha"]), clip=float(selected["clip"]),
            n_estimators=args.n_estimators,
        )
        rows.append({"seed": seed, "alpha": selected["alpha"], "clip": selected["clip"],
                     **result["metrics"], "mean_ratio": result["mean_ratio"],
                     "sd_ratio": result["sd_ratio"]})
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44])
    parser.add_argument("--n-estimators", type=int, default=300)
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    validation = run_validation(args)
    selected = (validation.groupby(["alpha", "clip"], as_index=False)["mae"].mean()
                .sort_values("mae").iloc[0])
    test = run_test(args, selected)
    validation.to_csv(args.out_dir / "validation_candidates.csv", index=False)
    test.to_csv(args.out_dir / "selected_test_metrics.csv", index=False)
    config = {
        "dataset": str(DATASET), "dataset_sha256": sha256_file(DATASET),
        "validation_mask": str(MASKS["e3_supportmatched"]),
        "test_mask": str(MASKS["e3_test"]), "target_transform": TARGET_TRANSFORM,
        "selection": "minimum mean validation MAE over seeds; test evaluated once",
        "alphas": list(ALPHAS), "clips": list(CLIPS), "seeds": args.seeds,
        "n_estimators": args.n_estimators,
        "selected_alpha": float(selected["alpha"]), "selected_clip": float(selected["clip"]),
    }
    config["config_hash"] = hashlib.sha256(
        json.dumps(config, sort_keys=True).encode()
    ).hexdigest()
    (args.out_dir / "spec.json").write_text(json.dumps(config, indent=2) + "\n")
    print("selected:", selected.to_dict())
    print("test:")
    print(test.to_string(index=False))


if __name__ == "__main__":
    main()
