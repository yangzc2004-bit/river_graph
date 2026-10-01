"""Station-profile mean/bias correction for DOC spatial transfer.

This is a small, label-free-at-target post-processor for the current
ExtraTrees context model.  A profile model learns a station-level transformed
DOC mean from source stations using only static/ecological/hydro/graph
attributes.  At an unseen station its predicted mean supplies a log-space
shrinkage correction to the context prediction.  The shrinkage is selected on
the support-matched spatial validation split and evaluated once on E3.

The script deliberately keeps the context model fixed.  It is a diagnostic
spatial-transfer component, not a replacement for the existing DOC product.
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
VALIDATION_MASK = Path(
    "experiments/phase4_transfer/kgml_local_transport_v1/"
    "spatial_validation_e3_supportmatched/masks/"
    "e3_spatial_validation_supportmatched.npz"
)
TEST_MASK = Path("experiments/masks_stcore_v1/e3_spatial_seed42.npz")
SELECTION_SEEDS = (42, 43, 44)
TEST_SEEDS = (42, 43, 44, 45, 46)
ALPHAS = (0.0, 0.25, 0.5, 0.75, 1.0, 1.25)

# These are the validation-selected E3 context forests already used by the
# current ET-context comparison.  The profile correction does not re-select
# the base model after seeing E3 test labels.
BASE_CONFIG = {
    42: ("extra_trees", 2),
    43: ("extra_trees", 2),
    44: ("extra_trees", 1),
    45: ("extra_trees", 1),
    46: ("extra_trees", 1),
}


def _load_split(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as saved:
        return {
            role: np.asarray(saved[role], dtype=np.int64)
            for role in ("train", "val", "test", "context")
            if role in saved.files
        }


def _profile_features(dataset: dict) -> np.ndarray:
    """Station-level covariates available at an unseen target station."""

    x = _as_tensor(dataset["x"]).numpy().astype(np.float32)
    x_mask = _as_tensor(dataset["x_mask"]).numpy().astype(np.float32)
    static = _as_tensor(dataset["static"]).numpy().astype(np.float32)
    regime = _as_tensor(dataset["regime"]).numpy().astype(np.float32)
    edge_index = _as_tensor(dataset["edge_index"]).numpy().astype(np.int64)
    edge_attr = _as_tensor(dataset["edge_attr"]).numpy().astype(np.float32)
    n = x.shape[0]

    indegree = np.bincount(edge_index[1], minlength=n).astype(np.float32)
    outdegree = np.bincount(edge_index[0], minlength=n).astype(np.float32)
    incoming = np.zeros((n, edge_attr.shape[1]), dtype=np.float32)
    outgoing = np.zeros_like(incoming)
    for edge_id in range(edge_attr.shape[0]):
        incoming[edge_index[1, edge_id]] += edge_attr[edge_id]
        outgoing[edge_index[0, edge_id]] += edge_attr[edge_id]
    incoming /= np.maximum(indegree[:, None], 1.0)
    outgoing /= np.maximum(outdegree[:, None], 1.0)

    features = np.concatenate(
        [
            static,
            regime,
            x.mean(axis=1),
            x.std(axis=1),
            x_mask.mean(axis=1),
            incoming,
            outgoing,
            indegree[:, None],
            outdegree[:, None],
        ],
        axis=1,
    )
    if not np.isfinite(features).all():
        raise ValueError("station profile features contain non-finite values")
    return features.astype(np.float32)


def _fit_base(
    dataset: dict,
    split: dict[str, np.ndarray],
    seed: int,
) -> tuple[ExtraTreesRegressor, np.ndarray, np.ndarray, np.ndarray]:
    """Fit the fixed context forest and return train/eval transformed arrays."""

    backend, leaf = BASE_CONFIG[seed]
    if backend != "extra_trees":
        raise ValueError(f"unsupported base backend: {backend}")
    target_z = target_values(dataset, "log1p").reshape(-1)
    fit_x = build_rf_features(
        dataset, split, FIT_ROLES, target_transform="log1p", include_network=True
    )
    test_x = build_rf_features(
        dataset, split, TEST_ROLES, target_transform="log1p", include_network=True
    )
    train = split["train"]
    model = ExtraTreesRegressor(
        n_estimators=300,
        min_samples_leaf=leaf,
        max_features=1.0,
        n_jobs=4,
        random_state=seed,
    )
    model.fit(fit_x[train], target_z[train])
    return model, target_z, fit_x, test_x


def _station_correction(
    dataset: dict,
    split: dict[str, np.ndarray],
    seed: int,
    eval_role: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return base z predictions, station profile correction, and raw truth."""

    model, target_z, _fit_x, test_x = _fit_base(dataset, split, seed)
    train = np.asarray(split["train"], dtype=np.int64)
    cells = np.asarray(split[eval_role], dtype=np.int64)
    _, t = dataset["y"].shape
    stations = np.unique(cells // t)
    source_stations = np.unique(train // t)
    profile = _profile_features(dataset)

    source_mean = np.asarray(
        [np.mean(target_z[train[train // t == station]]) for station in source_stations]
    )
    profile_model = ExtraTreesRegressor(
        n_estimators=500,
        min_samples_leaf=3,
        max_features=1.0,
        n_jobs=4,
        random_state=seed,
    )
    profile_model.fit(profile[source_stations], source_mean)
    target_mean = profile_model.predict(profile[stations])

    base_z = model.predict(test_x[cells])
    base_station_mean = np.asarray(
        [np.mean(base_z[cells // t == station]) for station in stations]
    )
    station_delta = target_mean - base_station_mean
    lookup = {int(station): delta for station, delta in zip(stations, station_delta)}
    delta = np.asarray([lookup[int(station)] for station in cells // t])
    truth = _as_tensor(dataset["y"]).numpy().reshape(-1)[cells]
    return base_z, delta, truth


def _evaluate_split(
    dataset: dict,
    split: dict[str, np.ndarray],
    seed: int,
    eval_role: str,
    role_name: str,
) -> list[dict]:
    base_z, delta, truth = _station_correction(dataset, split, seed, eval_role)
    rows = []
    for alpha in ALPHAS:
        prediction = np.expm1(base_z + alpha * delta)
        rows.append(
            {
                "role": role_name,
                "seed": seed,
                "alpha": alpha,
                **metrics(truth, prediction),
            }
        )
    return rows


def _read_density_adaptation(out_dir: Path) -> pd.DataFrame:
    paths = [
        out_dir.parent / "spatial_domain_adaptation_v1/selected_test_metrics.csv",
        out_dir.parent / "spatial_domain_adaptation_seed45_46/selected_test_metrics.csv",
    ]
    frames = []
    for path in paths:
        if path.exists():
            frame = pd.read_csv(path)
            frame["model"] = "density_ratio_reweighting"
            frames.append(frame[["seed", "mae", "model"]])
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    dataset = torch.load(DATASET, map_location="cpu", weights_only=False)
    validation_split = _load_split(VALIDATION_MASK)
    test_split = _load_split(TEST_MASK)

    validation_rows = []
    for seed in SELECTION_SEEDS:
        validation_rows.extend(
            _evaluate_split(dataset, validation_split, seed, "val", "supportmatched_val")
        )
    validation = pd.DataFrame(validation_rows)
    validation.to_csv(args.out_dir / "validation_candidates.csv", index=False)
    selected_alpha = float(
        validation.groupby("alpha")["mae"].mean().sort_values().index[0]
    )

    test_rows = []
    for seed in TEST_SEEDS:
        test_rows.extend(_evaluate_split(dataset, test_split, seed, "test", "e3_test"))
    test = pd.DataFrame(test_rows)
    selected = test[test["alpha"].eq(selected_alpha)].copy()
    selected.to_csv(args.out_dir / "selected_test_metrics.csv", index=False)

    summary_rows = []
    for name, frame in {
        "station_profile_bias": selected,
        "et_context": test[test["alpha"].eq(0.0)],
    }.items():
        summary_rows.append(
            {
                "model": name,
                "mean_mae": frame["mae"].mean(),
                "sd_mae": frame["mae"].std(ddof=1),
                "mean_rmse": frame["rmse"].mean(),
                "mean_log_mae": frame["log_mae"].mean(),
                "seeds": len(frame),
            }
        )
    density = _read_density_adaptation(args.out_dir)
    if not density.empty:
        summary_rows.append(
            {
                "model": "density_ratio_reweighting",
                "mean_mae": density["mae"].mean(),
                "sd_mae": density["mae"].std(ddof=1),
                "mean_rmse": np.nan,
                "mean_log_mae": np.nan,
                "seeds": len(density),
            }
        )
    summary = pd.DataFrame(summary_rows)
    summary.to_csv(args.out_dir / "summary.csv", index=False)

    config = {
        "dataset": str(DATASET),
        "dataset_sha256": sha256_file(DATASET),
        "validation_mask": str(VALIDATION_MASK),
        "validation_mask_sha256": sha256_file(VALIDATION_MASK),
        "test_mask": str(TEST_MASK),
        "test_mask_sha256": sha256_file(TEST_MASK),
        "selection": "minimum mean support-matched validation MAE over seeds",
        "selection_seeds": list(SELECTION_SEEDS),
        "test_seeds": list(TEST_SEEDS),
        "alphas": list(ALPHAS),
        "base_config": BASE_CONFIG,
        "selected_alpha": selected_alpha,
    }
    config["config_hash"] = hashlib.sha256(
        json.dumps(config, sort_keys=True).encode()
    ).hexdigest()
    (args.out_dir / "spec.json").write_text(json.dumps(config, indent=2) + "\n")

    et_mean = summary.loc[summary.model.eq("et_context"), "mean_mae"].iloc[0]
    profile_mean = summary.loc[
        summary.model.eq("station_profile_bias"), "mean_mae"
    ].iloc[0]
    lines = [
        "# Station-profile mean/bias correction",
        "",
        (
            f"Selected shrinkage alpha: **{selected_alpha:g}**, chosen by the "
            "support-matched E3 validation split."
        ),
        "",
        f"Five-seed E3 ET-context MAE: **{et_mean:.4f}**.",
        (
            f"Five-seed E3 profile-corrected MAE: **{profile_mean:.4f}** "
            f"({100 * (et_mean - profile_mean) / et_mean:.2f}% lower)."
        ),
        "",
        (
            "The correction uses only source-station DOC means and target-station "
            "static/ecological/hydro/graph profile features. Target E3 DOC labels "
            "are used only for the final metric."
        ),
        "",
        (
            "See `summary.csv` for comparison with density-ratio reweighting when "
            "that analysis is present."
        ),
    ]
    (args.out_dir / "verdict.md").write_text("\n".join(lines) + "\n")
    print(summary.to_string(index=False))
    print(f"selected_alpha={selected_alpha:g}")


if __name__ == "__main__":
    main()
