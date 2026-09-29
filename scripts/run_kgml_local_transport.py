"""Local--Transport KGML: paired forests, M1 and OOF graph residuals.

Use run_ladder.py --experiment kgml for training. --verify-only checks saved
products without fitting; --dry-run materializes nothing.
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

from river_graph.experiments.evaluate import metrics
from river_graph.experiments.graph_upgrade_v2 import GraphUpgradeModel
from river_graph.experiments.provenance import (
    run_identity_sha256,
    runtime_code_snapshot,
    sha256_file,
)
from river_graph.experiments.temporal_h2x import TARGET_TRANSFORMS
from river_graph.experiments.transfer import ANALYTES, DATASETS
from river_graph.models.kgml_local_transport import (
    FIT_ROLES,
    TEST_ROLES,
    LocalTransportKGML,
    fit_rf_artifacts,
    local_history,
    network_context,
    rf_feature_names,
    role_visible,
    target_values,
)

ROOT = Path("experiments/phase4_transfer/kgml_local_transport_v1")
MASKS = ("e2a_strict", "e3_spatial_seed42")
ARMS = ("rf_local", "rf_context", "h2x_t", "residual_upstream", "residual_both", "residual_nomsg")
EXTRA_RUNTIME = ("scripts/run_ladder.py", "scripts/run_kgml_local_transport.py",
                 "scripts/analyze_kgml_local_transport.py", str(ROOT / "plan.md"))


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def clean_json(value):
    if isinstance(value, dict):
        return {k: clean_json(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean_json(v) for v in value]
    if isinstance(value, (float, np.floating)):
        return float(value) if np.isfinite(value) else None
    if isinstance(value, np.integer):
        return int(value)
    return value


def write_json(path, value):
    path.write_text(json.dumps(clean_json(value), indent=2, allow_nan=False) + "\n")


def load_task(root, analyte, mask):
    dataset = torch.load(DATASETS[analyte], map_location="cpu", weights_only=False)
    source = Path("experiments/masks_stcore_v1") / f"{mask}.npz"
    with np.load(source, allow_pickle=False) as saved:
        observed = dataset["y_mask"].numpy().ravel()
        split = {r: saved[r][observed[saved[r]]] for r in ("train", "val", "test", "context") if r in saved}
    path = root / "masks" / f"{analyte}__{mask}.npz"
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        with np.load(path, allow_pickle=False) as saved:
            if set(saved.files) != set(split):
                raise ValueError("mask roles changed")
            for role, cells in split.items():
                np.testing.assert_array_equal(saved[role], cells)
    else:
        np.savez_compressed(path, **split)
    return dataset, split, path, source


def freeze_runtime(root):
    snapshot = runtime_code_snapshot()
    snapshot.update({p: sha256_file(p) for p in EXTRA_RUNTIME})
    root.mkdir(parents=True, exist_ok=True)
    path = root / "runtime_snapshot.json"
    if path.exists():
        if json.loads(path.read_text()) != snapshot:
            raise ValueError("code changed since this batch began; use a new output directory")
    else:
        write_json(path, snapshot)
        for name in snapshot:
            destination = root / "code_snapshot" / name
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes(Path(name).read_bytes())
    return digest(snapshot)


def rf_bundle(root, dataset, split, config):
    key = {k: config[k] for k in ("analyte", "mask", "seed", "dataset_sha256", "mask_sha256",
                                 "target_transform", "n_estimators", "n_jobs", "runtime_snapshot_hash")}
    path = root / "rf_bundles" / f"{config['analyte']}__{config['mask']}__seed{config['seed']}"
    path.mkdir(parents=True, exist_ok=True)
    artifact = path / "forests.joblib"
    meta_path = path / "meta.json"
    if meta_path.exists():
        meta = json.loads(meta_path.read_text())
        if meta["config_hash"] != digest(key) or sha256_file(artifact) != meta["sha256"]:
            raise ValueError("RF bundle mismatch")
        return joblib.load(artifact), artifact
    rf = fit_rf_artifacts(dataset, split, target_transform=config["target_transform"],
                          seed=config["seed"], n_estimators=config["n_estimators"], n_jobs=config["n_jobs"])
    joblib.dump(rf, artifact, compress=3)
    train = split["train"]
    folds = np.full(dataset["y"].shape[0], -1)
    for f, stations in enumerate(rf.folds):
        folds[stations] = f
    np.savez_compressed(path / "oof.npz", cells=train, stations_by_fold=folds,
                        prediction_z=rf.local_oof_z.ravel()[train])
    write_json(meta_path, {"config": key, "config_hash": digest(key), "sha256": sha256_file(artifact),
                           "oof_sha256": sha256_file(path / "oof.npz"),
                           "local_features": rf_feature_names(False), "context_features": rf_feature_names(True)})
    return rf, artifact


def component_frame(dataset, split, comp, *, config, roles, cells=None):
    y = dataset["y"].numpy()
    n, t = y.shape
    idx = np.arange(n * t) if cells is None else np.asarray(cells)
    visible = role_visible(split, roles, (n, t))
    names = np.full(n * t, "unobserved", dtype=object)
    names[dataset["y_mask"].numpy().ravel()] = "observed_unassigned"
    for role, selected in split.items():
        names[selected] = role
    values = target_values(dataset, config["target_transform"])
    hist = local_history(values, visible)
    net = network_context(values, visible, dataset["edge_index"].numpy())
    y_obs = np.where(dataset["y_mask"].numpy(), y, np.nan).ravel()[idx]
    local = comp["local_pred"].ravel()[idx]
    if config["target_transform"] == "log1p":
        residual_true = np.log1p(y_obs) - np.log1p(local)
    else:
        train_sd = max(float(y.ravel()[split["train"]].std()), 1e-8)
        residual_true = (y_obs - local) / train_sd
    known = hist[..., 2].ravel()[idx].astype(bool)
    age = np.rint(np.expm1(hist[..., 1].ravel()[idx]))
    frame = pd.DataFrame({
        "cell": idx, "station": np.asarray(dataset["site_no"], str)[idx // t],
        "month": np.asarray(dataset["months"], str)[idx % t], "month_index": idx % t,
        "analyte": config["analyte"], "mask": config["mask"], "seed": config["seed"],
        "model_name": config["arm"], "visibility_role": names[idx],
        "visible_input": visible.ravel()[idx], "input_roles": "+".join(roles),
        "y_true": y_obs, "residual_true": residual_true,
        "component_applicable": config["arm"].startswith("residual"),
        "age_group": np.select([~known, age <= 3, age <= 12],
                                ["never_observed", "recent_1_3", "seasonal_4_12"], default="older_12"),
        "upstream_support_group": np.where(net[..., 3].ravel()[idx] > 0, "visible_upstream", "none"),
        "flow": dataset["x"][..., 1].numpy().ravel()[idx],
        "flow_visible": dataset["x_mask"][..., 1].numpy().ravel()[idx].astype(bool),
    })
    for name, value in comp.items():
        frame[name] = value.ravel()[idx]
    frame["y_pred"] = frame.final_pred
    return frame


def metric_summary(frame, threshold):
    if not len(frame) or not np.isfinite(frame[["y_true", "final_pred"]]).all().all():
        raise ValueError("missing/nonfinite query predictions")
    result = metrics(frame.y_true.to_numpy(), frame.final_pred.to_numpy())
    tail = frame[frame.y_true >= threshold]
    result.update({"q90_threshold_train": threshold, "q90_n": len(tail), "q90_unstable": len(tail) < 20,
                   "q90_mae": float(np.abs(tail.y_true - tail.final_pred).mean()),
                   "graph_delta_mean": float(frame.graph_delta.mean()),
                   "graph_delta_sd": float(frame.graph_delta.std(ddof=0))})
    return clean_json(result)


def verify_run(run, expected_config=None):
    meta = json.loads((run / "meta.json").read_text())
    cfg = meta["config"]
    if digest(cfg) != meta["config_hash"] or (expected_config is not None and cfg != expected_config):
        raise ValueError(f"config mismatch: {run}")
    if run_identity_sha256(meta["config_hash"], meta["started_at"], cfg["runtime_snapshot_hash"]) != meta["run_identity_sha256"]:
        raise ValueError("run identity mismatch")
    root = run.parent.parent
    snapshot = json.loads((root / "runtime_snapshot.json").read_text())
    if digest(snapshot) != cfg["runtime_snapshot_hash"]:
        raise ValueError("runtime identity mismatch")
    for name, expected in snapshot.items():
        if sha256_file(root / "code_snapshot" / name) != expected:
            raise ValueError("saved runtime code changed")
    for key in ("dataset", "mask", "source_mask", "rf_bundle"):
        if sha256_file(cfg[f"{key}_path"]) != cfg[f"{key}_sha256"]:
            raise ValueError(f"changed {key}")
    for name, expected in meta["artifacts"].items():
        if sha256_file(run / name) != expected:
            raise ValueError(f"changed artifact: {name}")
    data = torch.load(cfg["dataset_path"], map_location="cpu", weights_only=False)
    frames = {}
    t = data["y"].shape[1]
    with np.load(cfg["mask_path"], allow_pickle=False) as mask:
        for role in ("val", "test"):
            frame = pd.read_parquet(run / f"{role}_predictions.parquet")
            cells = frame.cell.to_numpy()
            np.testing.assert_array_equal(cells, mask[role])
            if frame.cell.duplicated().any():
                raise ValueError("duplicate query")
            np.testing.assert_array_equal(frame.y_true, data["y"].numpy().ravel()[cells])
            np.testing.assert_array_equal(frame.station, np.asarray(data["site_no"], str)[cells // t])
            np.testing.assert_array_equal(frame.month, np.asarray(data["months"], str)[cells % t])
            if not (frame.analyte.eq(cfg["analyte"]).all() and frame.seed.eq(cfg["seed"]).all()
                    and frame.model_name.eq(cfg["arm"]).all() and frame["mask"].eq(cfg["mask"]).all()
                    and frame.visibility_role.eq(role).all() and not frame.visible_input.any()):
                raise ValueError("prediction role or identity mismatch")
            roles = FIT_ROLES if role == "val" else TEST_ROLES
            if not frame.input_roles.eq("+".join(roles)).all():
                raise ValueError("input visibility mismatch")
            recomputed = metric_summary(frame, meta["summary"]["q90_threshold_train"])
            if recomputed != meta[f"{role}_metrics"]:
                raise ValueError("saved metrics differ from query predictions")
            frames[role] = frame
    full = pd.read_parquet(run / "full_grid.parquet")
    np.testing.assert_array_equal(full.cell, np.arange(data["y"].numel()))
    if not np.isfinite(full[["local_pred", "context_pred", "final_pred", "graph_delta"]]).all().all():
        raise ValueError("invalid full-grid product")
    test = frames["test"]
    np.testing.assert_array_equal(full.final_pred.to_numpy()[test.cell], test.final_pred)
    return meta, frames["test"]


def run_one(root, data, split, cfg, rf):
    run = root / "runs" / f"{cfg['arm']}__{cfg['analyte']}__{cfg['mask']}__seed{cfg['seed']}"
    if (run / "meta.json").exists():
        return verify_run(run, cfg)[0]["summary"]
    if run.exists():
        raise ValueError(f"incomplete execution preserved: {run}")
    run.mkdir(parents=True)
    started = datetime.now(timezone.utc).isoformat()
    start = time.perf_counter()
    arm = cfg["arm"]
    print(f"[{arm}] {cfg['analyte']} {cfg['mask']} seed={cfg['seed']}", flush=True)

    def trace(row):
        with (run / "training_trace.jsonl").open("a") as stream:
            stream.write(json.dumps({**row, "elapsed_s": time.perf_counter() - start}) + "\n")
        if row["epoch"] == 1 or row["epoch"] % 5 == 0:
            print(f"  epoch {row['epoch']}: val={row['val_loss']:.4f}", flush=True)

    settings = {k: cfg[k] for k in ("seed", "hidden", "max_epochs", "patience", "chunk_months", "lr", "dropout")}
    if arm.startswith("residual"):
        model = LocalTransportKGML(analyte=cfg["analyte"], edge_direction=cfg["edge_direction"],
            edge_set=cfg["edge_set"], n_estimators=cfg["n_estimators"], n_jobs=cfg["n_jobs"],
            epoch_callback=trace, **settings)
        model.fit(data, split, rf=rf)
        epochs = model.epochs_run
        checkpoint = {"state_dict": model.model.state_dict(), "target_mu": rf.target_mu,
                      "target_sd": rf.target_sd, "best_epoch": model.best_epoch}
    elif arm == "h2x_t":
        model = GraphUpgradeModel(mechanism="m1", lookback=12, temporal_hidden=cfg["hidden"],
            target_transform=cfg["target_transform"], edge_direction="both", edge_set="river",
            feature_edge_set="river", env_encoder=True, masking="mixed", epoch_callback=trace, **settings)
        model.fit(data, split)
        epochs = model._bundle.epochs_run
        checkpoint = {"state_dict": model._bundle.model.state_dict(),
                      "target_mu": model._bundle.inputs.target_mu, "target_sd": model._bundle.inputs.target_sd}
    else:
        epochs, checkpoint = 0, None
    # Export validation under its held-out view, full grid/test under terminal view.
    threshold = float(np.quantile(data["y"].numpy().ravel()[split["train"]], .9))
    reports = {}
    for role, roles in (("val", FIT_ROLES), ("test", TEST_ROLES)):
        if arm.startswith("residual"):
            comp = model.predict_components(roles)
        else:
            local_std, context_std = rf.predict_std(data, split, roles)
            local, context = rf.inverse_std(local_std), rf.inverse_std(context_std)
            final = local if arm == "rf_local" else context
            if arm == "h2x_t":
                visible = np.flatnonzero(role_visible(split, roles, data["y"].shape))
                final = model.predict(only_visible=visible)
            zero = np.zeros_like(final)
            comp = {"local_pred": local, "context_pred": context, "final_pred": final,
                    "graph_delta": zero, "graph_delta_std": zero, "graph_delta_raw": zero, "graph_delta_abs": zero}
        frame = component_frame(data, split, comp, config=cfg, roles=roles, cells=split[role])
        frame.to_parquet(run / f"{role}_predictions.parquet", index=False)
        reports[role] = metric_summary(frame, threshold)
        if role == "test":
            full = component_frame(data, split, comp, config=cfg, roles=roles)
            full.to_parquet(run / "full_grid.parquet", index=False)
    if checkpoint is not None:
        torch.save({**checkpoint, "config": cfg}, run / "checkpoint.pt")
    summary = {"arm": arm, "analyte": cfg["analyte"], "mask": cfg["mask"], "seed": cfg["seed"],
               "epochs_run": epochs, "elapsed_s": time.perf_counter() - start, **reports["test"]}
    write_json(run / "metrics.json", summary)
    write_json(run / "meta.json", {"config": cfg, "config_hash": digest(cfg),
        "config_hash_schema": "canonical_all_fields_v1", "started_at": started,
        "run_identity_sha256": run_identity_sha256(digest(cfg), started, cfg["runtime_snapshot_hash"]),
        "summary": summary, "val_metrics": reports["val"], "test_metrics": reports["test"],
        "artifacts": {p.name: sha256_file(p) for p in sorted(run.iterdir()) if p.is_file()}})
    verify_run(run, cfg)
    print(f"  saved: MAE={summary['mae']:.4f}, {summary['elapsed_s']:.0f}s", flush=True)
    gc.collect()
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", type=Path, default=ROOT / "k1")
    parser.add_argument("--analytes", nargs="+", choices=ANALYTES, default=["doc"])
    parser.add_argument("--masks", nargs="+", choices=MASKS, default=list(MASKS))
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44])
    parser.add_argument("--arms", nargs="+", choices=ARMS, default=list(ARMS))
    parser.add_argument("--max-epochs", type=int, default=30)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--n-estimators", type=int, default=200)
    parser.add_argument("--n-jobs", type=int, default=4)
    parser.add_argument("--hidden", type=int, default=64)
    parser.add_argument("--chunk-months", type=int, default=64)
    parser.add_argument("--torch-threads", type=int, default=4)
    parser.add_argument("--verify-only", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    expected = [{"analyte": a, "mask": m, "seed": s, "arm": arm}
                for a in args.analytes for m in args.masks for s in args.seeds for arm in args.arms]
    if args.dry_run:
        print(json.dumps({"runs": len(expected), "matrix": expected}, indent=2))
        return
    if args.verify_only:
        rows = [verify_run(p.parent)[0]["summary"] for p in sorted((args.out_dir / "runs").glob("*/meta.json"))]
        if not rows:
            raise ValueError("no completed runs")
        print(f"Verified {len(rows)} completed runs")
        return
    torch.set_num_threads(args.torch_threads)
    snapshot = freeze_runtime(args.out_dir)
    manifest = {"matrix": expected, "runtime_snapshot_hash": snapshot,
                "max_epochs": args.max_epochs, "patience": args.patience, "n_estimators": args.n_estimators,
                "n_jobs": args.n_jobs, "torch_threads": args.torch_threads, "chunk_months": args.chunk_months,
                "hidden": args.hidden}
    manifest_path = args.out_dir / "execution_manifest.json"
    if manifest_path.exists() and json.loads(manifest_path.read_text()) != manifest:
        raise ValueError("batch configuration changed")
    write_json(manifest_path, manifest)
    rows = []
    for analyte in args.analytes:
        for mask in args.masks:
            data, split, path, source = load_task(args.out_dir, analyte, mask)
            for seed in args.seeds:
                base = {"analyte": analyte, "mask": mask, "seed": seed, "target_transform": TARGET_TRANSFORMS[analyte],
                    "dataset_path": DATASETS[analyte], "dataset_sha256": sha256_file(DATASETS[analyte]),
                    "mask_path": str(path), "mask_sha256": sha256_file(path),
                    "source_mask_path": str(source), "source_mask_sha256": sha256_file(source),
                    "runtime_snapshot_hash": snapshot, "n_estimators": args.n_estimators, "n_jobs": args.n_jobs,
                    "max_epochs": args.max_epochs, "patience": args.patience, "hidden": args.hidden,
                    "layers": 2, "dropout": .1, "lr": .001, "lookback": 12,
                    "chunk_months": args.chunk_months, "torch_threads": args.torch_threads,
                    "fit_roles": list(FIT_ROLES), "test_roles": list(TEST_ROLES),
                    "oof": "station_blocked_5_fold_all_station_targets_hidden",
                    "residual_updates_per_epoch": 1, "downstream_support_in_residual": False,
                    "local_feature_names": rf_feature_names(False), "context_feature_names": rf_feature_names(True)}
                rf, rf_path = rf_bundle(args.out_dir, data, split, base)
                for arm in args.arms:
                    cfg = {**base, "arm": arm, "rf_bundle_path": str(rf_path), "rf_bundle_sha256": sha256_file(rf_path),
                           "edge_direction": "both" if arm in ("h2x_t", "residual_both") else "upstream",
                           "edge_set": "empty" if arm == "residual_nomsg" else "river"}
                    rows.append(run_one(args.out_dir, data, split, cfg, rf))
                    pd.DataFrame(rows).to_csv(args.out_dir / "metrics.csv", index=False)
                    write_json(args.out_dir / "progress.json", {"completed": len(rows), "expected": len(expected),
                                                                "latest": rows[-1]})
    from analyze_kgml_local_transport import analyze
    analyze(args.out_dir)


if __name__ == "__main__":
    main()
