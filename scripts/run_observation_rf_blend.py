"""Fit and evaluate the R7 validation-selected Temporal RF / M1 blend.

Run ``--analyze-only`` to verify saved products and regenerate summaries.
New executions are kept separate from the historical M1 and U4/U5 fits.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
from analyze_u4_blend import cluster_interval
from run_temporal_h2x import _split, _target_mask_path
from run_temporal_rf_upgrade import FIT_VISIBILITY, TEST_VISIBILITY, _features
from sklearn.ensemble import RandomForestRegressor

from river_graph.experiments.evaluate import metrics
from river_graph.experiments.graph_upgrade_v2 import (
    GraphUpgradeModel,
    observation_statistics,
)
from river_graph.experiments.provenance import (
    run_identity_sha256,
    runtime_code_snapshot,
    sha256_file,
)
from river_graph.experiments.transfer import DATASETS

ROOT = Path("experiments/phase4_transfer/graph_upgrade_v2/continuation_r7_rf_m1_blend")
MASKS = ("e2a_strict", "e3_spatial_seed42")
SEEDS = (42, 43, 44)
ALPHAS = np.linspace(0.0, 1.0, 21)


def digest(payload: dict) -> str:
    """Hash every experiment setting, including the blend selection rule."""
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode()).hexdigest()


def write_json(path: Path, value: dict | list) -> None:
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def visible_cells(split: dict, roles=FIT_VISIBILITY) -> np.ndarray:
    return np.concatenate([np.asarray(split.get(r, []), dtype=np.int64) for r in roles])


def select_alpha(y_val, rf_val, gnn_val) -> tuple[float, list[dict]]:
    """No test input is accepted here; exact ties prefer the RF endpoint."""
    y, rf, gnn = (np.asarray(v, dtype=np.float64) for v in (y_val, rf_val, gnn_val))
    if not (y.ndim == 1 and len(y) and y.shape == rf.shape == gnn.shape):
        raise ValueError("nonempty aligned validation vectors required")
    if not all(np.isfinite(v).all() for v in (y, rf, gnn)):
        raise ValueError("validation values must be finite")
    grid = [{"alpha_rf": float(a), "val_mae": float(np.abs(y - (a * rf + (1-a) * gnn)).mean())}
            for a in ALPHAS]
    best = min(grid, key=lambda row: (row["val_mae"], -row["alpha_rf"]))
    return best["alpha_rf"], grid


def model_settings(seed: int, *, edge_set: str = "river") -> dict:
    return {"mechanism": "m1", "spatial_variant": "baseline", "seed": seed,
            "lookback": 12, "hidden": 64, "temporal_hidden": 64, "layers": 2,
            "dropout": 0.1, "lr": 0.001, "max_epochs": 30, "patience": 5,
            "target_transform": "log1p", "edge_direction": "both", "edge_set": edge_set,
            "feature_edge_set": "river",
            "env_encoder": True, "context_mode": "none", "masking": "mixed",
            "chunk_months": 256}


def build_frame(data, split, cells, role, rf, gnn, alpha, mask, seed):
    _, t = data["y"].shape
    cells = np.asarray(cells, dtype=np.int64)
    return pd.DataFrame({
        "cell": cells, "station": np.asarray(data["site_no"], dtype=str)[cells // t],
        "month": np.asarray(data["months"], dtype=str)[cells % t],
        "analyte": "doc", "mask": mask, "seed": seed, "visibility_role": role,
        "y_true": data["y"].numpy().reshape(-1)[cells], "rf_pred": rf, "gnn_pred": gnn,
        "blend_pred": alpha * rf + (1-alpha) * gnn, "alpha_rf": alpha,
    })


def add_strata(frame, gnn):
    bundle = gnn._bundle
    visible, feed = gnn._visible_input()
    feature_edges = getattr(bundle.model, "_feature_edge_index", bundle.model._edge_index)
    stats = observation_statistics(feed, visible, feature_edges)
    cells = frame.cell.to_numpy()
    age = np.rint(np.expm1(stats[1].T.numpy().ravel()[cells] * np.log1p(12)))
    known = stats[2].T.numpy().ravel()[cells].astype(bool)
    frame["observation_age_group"] = np.select(
        [~known, age == 0, age <= 3, age <= 12],
        ["never_observed", "fresh", "recent_1_3", "seasonal_4_12"], default="old_over_12",
    )
    fraction = stats[-2].T.numpy().ravel()[cells]
    frame["upstream_support_group"] = np.where(fraction > 0, "visible_upstream", "none")


def verify_run(run: Path) -> tuple[dict, pd.DataFrame, pd.DataFrame]:
    meta = json.loads((run / "meta.json").read_text())
    config = meta["config"]
    if digest(config) != meta["config_hash"]:
        raise ValueError(f"config mismatch: {run}")
    for name, expected in meta["artifacts"].items():
        if sha256_file(run / name) != expected:
            raise ValueError(f"changed product: {run / name}")
    for name in ("dataset", "mask"):
        if sha256_file(config[f"{name}_path"]) != config[f"{name}_sha256"]:
            raise ValueError(f"changed {name}: {run}")
    expected_identity = run_identity_sha256(meta["config_hash"], meta["started_at"],
                                             config["runtime_snapshot_hash"])
    if expected_identity != meta["run_identity_sha256"]:
        raise ValueError(f"run identity mismatch: {run}")
    snapshot = json.loads((run.parent.parent / "runtime_snapshot.json").read_text())
    if digest(snapshot) != config["runtime_snapshot_hash"]:
        raise ValueError(f"runtime snapshot mismatch: {run}")
    if sha256_file(config["plan_path"]) != config["plan_sha256"]:
        raise ValueError(f"plan mismatch: {run}")
    val, test = (pd.read_parquet(run / f"{role}_predictions.parquet") for role in ("val", "test"))
    data = torch.load(config["dataset_path"], map_location="cpu", weights_only=False)
    with np.load(config["mask_path"], allow_pickle=False) as mask:
        for role, frame in (("val", val), ("test", test)):
            cells = frame.cell.to_numpy(dtype=int)
            np.testing.assert_array_equal(cells, mask[role])
            if frame.cell.duplicated().any():
                raise ValueError("duplicated query cell")
            np.testing.assert_array_equal(frame.y_true, data["y"].numpy().ravel()[cells])
            np.testing.assert_array_equal(frame.station.astype(str),
                                          np.asarray(data["site_no"], dtype=str)[cells // data["y"].shape[1]])
            np.testing.assert_array_equal(frame.month,
                                          np.asarray(data["months"], dtype=str)[cells % data["y"].shape[1]])
            if not np.isfinite(frame[["y_true", "rf_pred", "gnn_pred", "blend_pred"]]).all().all():
                raise ValueError("nonfinite predictions")
            if not (frame.seed.eq(config["seed"]).all() and frame["mask"].eq(config["mask"]).all()
                    and frame.visibility_role.eq(role).all() and frame.analyte.eq("doc").all()):
                raise ValueError("product identity mismatch")
            np.testing.assert_allclose(frame.blend_pred, meta["alpha_rf"] * frame.rf_pred
                                       + (1-meta["alpha_rf"]) * frame.gnn_pred)
            np.testing.assert_array_equal(frame.alpha_rf, np.full(len(frame), meta["alpha_rf"]))
        train = mask["train"]
    alpha, grid = select_alpha(val.y_true, val.rf_pred, val.gnn_pred)
    selection = json.loads((run / "selection.json").read_text())
    if alpha != meta["alpha_rf"] or selection["alpha_rf"] != alpha:
        raise ValueError("validation selection mismatch")
    np.testing.assert_allclose([r["val_mae"] for r in grid],
                               [r["val_mae"] for r in selection["grid"]])
    threshold = float(np.quantile(data["y"].numpy().ravel()[train], 0.9))
    np.testing.assert_allclose(threshold, meta["q90_threshold_train"])
    return meta, val, test


def run_one(data, split, fit_x, test_x, *, mask_path, mask, seed, root, snapshot_hash,
            edge_set: str):
    run = root / "runs" / f"rf_m1_blend__doc__{mask}__seed{seed}"
    config = {
        "model_name": "validation_selected_rf_m1_blend", "analyte": "doc", "mask": mask,
        "seed": seed, "dataset_path": str(DATASETS["doc"]), "mask_path": str(mask_path),
        "dataset_sha256": sha256_file(DATASETS["doc"]), "mask_sha256": sha256_file(mask_path),
        "gnn": model_settings(seed, edge_set=edge_set),
        "rf": {"n_estimators": 200, "n_jobs": 4, "random_state": seed},
        "fit_roles": list(FIT_VISIBILITY), "test_roles": list(TEST_VISIBILITY),
        "alpha_rf_grid": ALPHAS.tolist(), "selection": "raw_validation_mae; ties_prefer_rf",
        "torch_threads": torch.get_num_threads(), "runtime_snapshot_hash": snapshot_hash,
        "plan_path": str(root / "plan.md"), "plan_sha256": sha256_file(root / "plan.md"),
        "config_hash_schema": "canonical_json_all_fields_v1",
    }
    if (run / "meta.json").exists():
        old, _, _ = verify_run(run)
        if old["config_hash"] != digest(config):
            raise ValueError(f"changed configuration; use a new directory: {run}")
        print(f"cached {mask} seed={seed}", flush=True)
        return
    if run.exists() and any(run.iterdir()):
        raise ValueError(f"unfinished run retained at {run}; choose a new directory")
    run.mkdir(parents=True, exist_ok=True)
    started = datetime.now(timezone.utc).isoformat()
    start = time.perf_counter()
    train, val, test = (np.asarray(split[r], dtype=np.int64) for r in ("train", "val", "test"))
    y = data["y"].numpy().ravel()

    def epoch(row):
        row = {**row, "elapsed_s": time.perf_counter() - start}
        with (run / "training_trace.jsonl").open("a") as stream:
            stream.write(json.dumps(row) + "\n")
        if row["epoch"] == 1 or row["epoch"] % 5 == 0:
            print(f"  epoch {row['epoch']}: val={row['val_loss']:.4f}, {row['elapsed_s']:.0f}s", flush=True)

    label = "R7" if edge_set == "river" else "R8"
    print(f"[{label}] {mask} seed={seed}: fitting M1 + RF", flush=True)
    gnn = GraphUpgradeModel(**config["gnn"], epoch_callback=epoch)
    gnn.fit(data, split)
    rf = RandomForestRegressor(**config["rf"])
    rf.fit(fit_x[train], np.log1p(y[train]))
    rf_val = np.expm1(rf.predict(fit_x[val]))
    gnn_val = gnn.predict(only_visible=visible_cells(split)).ravel()[val]
    alpha, grid = select_alpha(y[val], rf_val, gnn_val)
    write_json(run / "selection.json", {"alpha_rf": alpha, "grid": grid,
                                        "selected_at": datetime.now(timezone.utc).isoformat()})
    val_frame = build_frame(data, split, val, "val", rf_val, gnn_val, alpha, mask, seed)
    val_frame.to_parquet(run / "val_predictions.parquet", index=False)
    # Test prediction and evaluation start after validation selection is saved.
    rf_test = np.expm1(rf.predict(test_x[test]))
    gnn_test = gnn.predict().ravel()[test]
    test_frame = build_frame(data, split, test, "test", rf_test, gnn_test, alpha, mask, seed)
    add_strata(test_frame, gnn)
    test_frame.to_parquet(run / "test_predictions.parquet", index=False)
    inputs = gnn._bundle.inputs
    torch.save({"state_dict": gnn._bundle.model.state_dict(), "config": config["gnn"],
                "target_mu": inputs.target_mu, "target_sd": inputs.target_sd}, run / "gnn_checkpoint.pt")
    joblib.dump(rf, run / "rf.joblib", compress=3)
    meta = {"config": config, "config_hash": digest(config), "started_at": started,
            "finished_at": datetime.now(timezone.utc).isoformat(), "elapsed_s": time.perf_counter()-start,
            "run_identity_sha256": run_identity_sha256(digest(config), started, snapshot_hash),
            "alpha_rf": alpha, "q90_threshold_train": float(np.quantile(y[train], .9)),
            "training": {"epochs_run": gnn._bundle.epochs_run, "best_val_loss": gnn._bundle.best_val_loss},
            "artifacts": {p.name: sha256_file(p) for p in sorted(run.iterdir()) if p.is_file()}}
    write_json(run / "meta.json", meta)
    print(f"  completed; validation alpha_RF={alpha:.2f}; elapsed={meta['elapsed_s']:.0f}s", flush=True)


def summarize(frames: list[pd.DataFrame], *, tail: bool = False) -> dict:
    sizes = [len(f) for f in frames]
    if len(set(sizes)) != 1:
        raise ValueError("unmatched query across seeds")
    for frame in frames[1:]:
        np.testing.assert_array_equal(frame.cell, frames[0].cell)
        np.testing.assert_array_equal(frame.y_true, frames[0].y_true)
    errors = []
    result = {"seeds": len(frames), "unique_cells": sizes[0], "q90_unstable": tail and sizes[0] < 20}
    for name in ("rf", "gnn", "blend"):
        values = [metrics(f.y_true.to_numpy(), f[f"{name}_pred"].to_numpy()) for f in frames]
        for key in ("mae", "rmse", "r2"):
            result[f"{name}_{key}"] = float(np.mean([v[key] for v in values]))
    gains = []
    for frame in frames:
        delta = abs(frame.y_true-frame.rf_pred) - abs(frame.y_true-frame.blend_pred)
        errors.append(pd.DataFrame({"station": frame.station, "cell": frame.cell, "delta": delta}))
        gains.append(delta.mean())
    error = pd.concat(errors, ignore_index=True)
    result.update({"rf_minus_blend_mae": float(np.mean(gains)),
                   "relative_improvement_pct": 100*(1-result["blend_mae"]/result["rf_mae"]),
                   "improved_seeds": int(np.sum(np.asarray(gains) > 1e-12)),
                   "tied_seeds": int(np.sum(np.abs(gains) <= 1e-12))})
    result["ci_low"], result["ci_high"] = (cluster_interval(error) if error.station.nunique() > 1
                                            else (float("nan"), float("nan")))
    return result


def analyze(root: Path):
    groups, runs, audit = {}, [], []
    for path in sorted((root / "runs").glob("*/meta.json")):
        meta, val, test = verify_run(path.parent)
        key, seed = meta["config"]["mask"], meta["config"]["seed"]
        if any(r["mask"] == key and r["seed"] == seed for r in runs):
            raise ValueError("duplicate run identity")
        groups.setdefault(key, []).append((meta, test))
        row = {"mask": key, "seed": seed, "alpha_rf": meta["alpha_rf"],
               "epochs": meta["training"]["epochs_run"], "elapsed_s": meta["elapsed_s"]}
        for role, frame in (("val", val), ("test", test)):
            for model in ("rf", "gnn", "blend"):
                row[f"{role}_{model}_mae"] = float(abs(frame.y_true-frame[f"{model}_pred"]).mean())
        runs.append(row)
        audit.append({"run": str(path.parent), "status": "verified", "meta_sha256": sha256_file(path)})
    output = root / "analysis"
    output.mkdir(exist_ok=True)
    pd.DataFrame(runs).to_csv(output / "runs.csv", index=False)
    summary, strata = [], []
    for mask, group in groups.items():
        frames = [item[1] for item in group]
        summary.append({"mask": mask, "subset": "all", **summarize(frames)})
        tail_frames = [frame[frame.y_true >= meta["q90_threshold_train"]] for meta, frame in group]
        if all(len(f) for f in tail_frames):
            summary.append({"mask": mask, "subset": "q90", **summarize(tail_frames, tail=True)})
        for column in ("observation_age_group", "upstream_support_group"):
            for label in sorted(frames[0][column].unique()):
                subset = [f[f[column].eq(label)] for f in frames]
                strata.append({"mask": mask, "stratum": column, "group": label, **summarize(subset)})
    table = pd.DataFrame(summary)
    table.to_csv(output / "summary.csv", index=False)
    pd.DataFrame(strata).to_csv(output / "strata.csv", index=False)
    complete = {(r["mask"], r["seed"]) for r in runs} == {(m, s) for m in MASKS for s in SEEDS}
    write_json(output / "audit.json", {"complete": complete, "expected": 6, "verified": len(runs),
                                       "runs": audit, "bootstrap": "2000 station draws; seed-mean cell errors"})
    print(table.to_string(index=False))
    return table


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--analyze-only", action="store_true")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--edge-set", choices=("river", "empty"), default="river")
    args = parser.parse_args()
    torch.set_num_threads(args.threads)
    if args.analyze_only:
        analyze(args.root)
        return
    snapshot = runtime_code_snapshot()
    for filename in (__file__, "scripts/run_temporal_rf_upgrade.py", "scripts/run_temporal_h2x.py",
                     "scripts/analyze_u4_blend.py"):
        path = Path(filename).resolve()
        snapshot[str(path.relative_to(Path.cwd()))] = sha256_file(path)
    snapshot_path = args.root / "runtime_snapshot.json"
    if snapshot_path.exists() and json.loads(snapshot_path.read_text()) != snapshot:
        raise ValueError("runtime changed; use a new experiment directory")
    write_json(snapshot_path, snapshot)
    data = torch.load(DATASETS["doc"], map_location="cpu", weights_only=False)
    for mask in MASKS:
        mask_path = _target_mask_path("doc", mask, data)
        with np.load(mask_path, allow_pickle=False) as archive:
            split = _split({k: archive[k] for k in archive.files})
        label = "R7" if args.edge_set == "river" else "R8"
        print(f"[{label}] building paired visibility features for {mask}", flush=True)
        fit_x, test_x = _features(data, split, FIT_VISIBILITY), _features(data, split, TEST_VISIBILITY)
        for seed in SEEDS:
            run_one(data, split, fit_x, test_x, mask_path=mask_path, mask=mask, seed=seed,
                    root=args.root, snapshot_hash=digest(snapshot), edge_set=args.edge_set)
            gc.collect()
        del fit_x, test_x
    analyze(args.root)


if __name__ == "__main__":
    main()
