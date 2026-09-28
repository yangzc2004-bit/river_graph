"""Run the observation-driven graph upgrade mechanisms.

The default ``pilot`` is M1 (observation-aware memory) on DOC, pH and
specific conductance, the temporal and spatial holdout families, and three
seeds.  M2 and M3 are available with ``--mechanism`` after the M1 pilot is
inspected.  Results are written only under ``graph_upgrade_v2``.
"""

from __future__ import annotations

import argparse
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from run_temporal_h2x import MASKS_DIR, _split, _target_mask_path

from river_graph.experiments.evaluate import load_mask, metrics
from river_graph.experiments.graph_upgrade_v2 import (
    MECHANISMS,
    OBS_FEATURE_NAMES,
    SPATIAL_VARIANTS,
    GraphUpgradeModel,
    observation_statistics,
)
from river_graph.experiments.provenance import (
    config_hash,
    file_identity,
    run_identity_sha256,
    runtime_code_snapshot,
    runtime_code_snapshot_sha256,
)
from river_graph.experiments.temporal_h2x import TARGET_TRANSFORMS
from river_graph.experiments.transfer import ANALYTES, DATASETS

DEFAULT_MASKS = ("e2a_strict", "e3_spatial_seed42")
DEFAULT_SEEDS = (42, 43, 44)
OUT_ROOT = Path("experiments/phase4_transfer/graph_upgrade_v2")


def _load(path: str | Path) -> dict:
    return torch.load(path, map_location="cpu", weights_only=False)


def _full_grid(dataset: dict, pred: np.ndarray, split: dict, *, model_name: str,
               analyte: str, mask: str, seed: int) -> pd.DataFrame:
    y = np.asarray(dataset["y"])
    observed = np.asarray(dataset["y_mask"]).astype(bool)
    n, t = y.shape
    roles = np.full(n * t, "", dtype=object)
    for role in ("train", "val", "test", "context"):
        roles[np.asarray(split.get(role, []), dtype=np.int64)] = role
    return pd.DataFrame({
        "analyte": analyte,
        "station": np.repeat([str(s) for s in dataset["site_no"]], t),
        "month": np.tile([str(m) for m in dataset["months"]], n),
        "month_index": np.tile(np.arange(t), n),
        "y_true": np.where(observed.reshape(-1), y.reshape(-1), np.nan),
        "y_pred": pred.reshape(-1),
        "observed": observed.reshape(-1),
        "split": roles,
        "model_name": model_name,
        "mask": mask,
        "seed": int(seed),
    })


def _diagnostics(model: GraphUpgradeModel, dataset: dict, split: dict,
                 pred: np.ndarray) -> dict:
    """Train-defined Q90 and fixed observation-age/network-support strata."""
    inputs = model._bundle.inputs
    n, t = inputs.y_model.shape
    visible = torch.zeros(n * t, dtype=torch.bool)
    for role in ("train", "val", "context"):
        cells = np.asarray(split.get(role, []), dtype=np.int64)
        if cells.size:
            visible[cells] = True
    visible = visible.reshape(n, t)
    stats = observation_statistics(inputs.y_model, visible, model._bundle.model._edge_index)
    # observation_statistics is [time, nodes], while prediction/data cells
    # are flattened in [nodes, time] order.  Transpose before indexing so
    # strata refer to the same station-month cells as the test split.
    age = stats[1].T.reshape(-1).numpy()
    up = stats[-2].T.reshape(-1).numpy()
    down = stats[-1].T.reshape(-1).numpy()
    last_valid = stats[2].T.reshape(-1).numpy().astype(bool)
    test = np.asarray(split["test"], dtype=np.int64)
    y = np.asarray(dataset["y"]).reshape(-1)
    err = np.abs(y - pred.reshape(-1))
    train = np.asarray(split["train"], dtype=np.int64)
    threshold = float(np.quantile(y[train], .9))
    tail = test[y[test] >= threshold]
    result = {
        "q90_threshold_train": threshold,
        "q90_n": len(tail),
        "q90_unstable": len(tail) < 20,
        "q90_mae": float(err[tail].mean()) if len(tail) else float("nan"),
    }
    age_months = np.rint(np.expm1(age * np.log1p(12.0)))
    strata = {
        "age": (("never_observed", ~last_valid),
                ("fresh", last_valid & (age_months == 0)),
                ("recent", last_valid & (age_months >= 1) & (age_months <= 3)),
                ("seasonal", last_valid & (age_months > 3) & (age_months <= 12)),
                ("old", last_valid & (age_months > 12))),
        "support": (("none", (up == 0) & (down == 0)),
                    ("one_side", (up > 0) ^ (down > 0)),
                    ("both_sides", (up > 0) & (down > 0))),
    }
    for label, groups in strata.items():
        for group, select in groups:
            select = select[test]
            cells = test[select]
            result[f"{label}_{group}_n"] = len(cells)
            result[f"{label}_{group}_mae"] = float(err[cells].mean()) if len(cells) else float("nan")
    return result


def _run_one(*, mechanism: str, analyte: str, mask_name: str, seed: int,
             max_epochs: int, patience: int, out_root: Path,
             force: bool = False, lag_mode: str = "learned",
             stage: str = 'pilot', runtime_snapshot: str | None = None,
             spatial_variant: str = "baseline") -> dict:
    dataset_path = DATASETS[analyte]
    dataset = _load(dataset_path)
    mask_path = _target_mask_path(analyte, mask_name, dataset)
    with np.load(mask_path, allow_pickle=False) as archive:
        split = _split({key: archive[key] for key in archive.files})
    base_model_name = {"m1": "observation_memory", "m2": "lagged_transport",
                       "m3": "multiscale_temporal", "m13": "observation_multiscale"}[mechanism]
    model_name = (base_model_name if spatial_variant == "baseline"
                  else f"{base_model_name}_{spatial_variant}")
    lag_suffix = f"__lag-{lag_mode}" if mechanism == "m2" else ""
    run_name = f"{model_name}__{analyte}__{mask_name}__seed{seed}{lag_suffix}"
    run_dir = out_root / mechanism / "runs" / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    # M13 keeps M1's local 12-month memory while its M3 paths see the full
    # causal sequence.  Passing the full chronology to the rolling GRU would
    # turn the combination pilot into an unnecessary quadratic computation.
    lookback = dataset['y'].shape[1] if mechanism == 'm3' else 12
    variant_layers = {"baseline": 2, "res2": 2, "res3": 3,
                      "res4": 4, "res3_jk": 3}
    spatial_layers = variant_layers[spatial_variant]
    params = {
        "script": "scripts/run_graph_upgrade_v2.py",
        "model_name": model_name,
        "mechanism": mechanism,
        "tag": f"graph_upgrade_v2_{mechanism}",
        "architecture": "observation_driven_transport_temporal",
        "variant": "river",
        "seed": int(seed),
        "lr": 1e-3,
        "weight_decay": 0.0,
        "edge_dropout": 0.0,
        "share_weights": False,
        "env_groups": None,
        "env_encoder": True,
        "edge_set": "river",
        "edge_direction": "upstream" if mechanism == "m2" else "both",
        "hidden": 64,
        "layers": spatial_layers,
        "dropout": .1,
        "max_epochs": int(max_epochs),
        "patience": int(patience),
        "env_emb": 32,
        "dataset_path": str(dataset_path),
        "mask_path": str(mask_path),
        "temporal": "gru_d" if mechanism == "m1" else ("lagged_gru" if mechanism == "m2" else ("multiscale" if mechanism == "m3" else "observation_multiscale")),
        "lookback": lookback,
        "history_scope": 'full_causal_sequence' if mechanism == 'm3' else ('mixed_causal_sequence' if mechanism == 'm13' else 'rolling_12_months'),
        "chunk_months": 256,
        "torch_threads": torch.get_num_threads(),
        "execution_stage": stage,
        "causal": True,
        "temporal_hidden": 64,
        "target_analyte": analyte,
        "target_transform": TARGET_TRANSFORMS[analyte],
        "upgrade_stage": mechanism,
        "spatial_variant": spatial_variant,
        "residual": spatial_variant != "baseline",
        "jumping_knowledge": spatial_variant == "res3_jk",
        "training_masking": "point_temporal_block_station_block",
        "lag_buckets": [0, 1, 3, 6, 12] if mechanism == "m2" else None,
        "lag_mode": lag_mode if mechanism == "m2" else None,
        "observation_features": list(OBS_FEATURE_NAMES),
        "training_protocol": "graph_upgrade_v2_mechanism_pilot",
        "config_hash_version": 6,
    }
    params["dataset_sha256"] = file_identity(dataset_path).get("sha256")
    params["mask_sha256"] = file_identity(mask_path).get("sha256")
    params["runtime_snapshot_hash"] = runtime_snapshot or runtime_code_snapshot_sha256()
    params["config_hash"] = config_hash(params)
    meta_path, metrics_path = run_dir / "meta.json", run_dir / "metrics.json"
    full_path = run_dir / "full_grid.parquet"
    if not force and all(p.is_file() for p in (meta_path, metrics_path, full_path)):
        old = json.loads(meta_path.read_text(encoding="utf-8"))
        cached_metrics = json.loads(metrics_path.read_text())
        if (old.get("config_hash") == params["config_hash"]
                and config_hash(old['config']) == params['config_hash']
                and old['full_grid_sha256'] == file_identity(full_path)['sha256']
                and all(np.isfinite(cached_metrics[k]) for k in ('mae', 'rmse', 'r2'))):
            return {"run": run_name, "status": "cached", **cached_metrics}
        if not force:
            raise RuntimeError(
                f"existing run has a different configuration: {run_dir}; "
                "use a new output directory or pass --force explicitly"
            )

    started_at = datetime.now(timezone.utc).isoformat()
    clock_start = time.perf_counter()
    trace_path = run_dir / 'training_trace.jsonl'
    trace_path.write_text('')

    def record_epoch(row):
        row = {**row, 'elapsed_s': time.perf_counter() - clock_start}
        with trace_path.open('a') as stream:
            stream.write(json.dumps(row) + '\n')
        if row['epoch'] == 1 or row['epoch'] % 5 == 0:
            print(f"  epoch {row['epoch']}: val={row['val_loss']:.5f}, "
                  f"elapsed={row['elapsed_s']:.0f}s", flush=True)

    model = GraphUpgradeModel(
        mechanism=mechanism, seed=seed,
        lookback=lookback, hidden=64,
        temporal_hidden=64, max_epochs=max_epochs, patience=patience,
        target_transform=TARGET_TRANSFORMS[analyte],
        edge_direction="upstream" if mechanism == "m2" else "both",
        chunk_months=256,
        lag_mode=lag_mode,
        epoch_callback=record_epoch,
        spatial_variant=spatial_variant,
    )
    pred = model.fit_predict(dataset, split)
    test = np.asarray(split["test"], dtype=np.int64)
    y = np.asarray(dataset["y"])
    metric = metrics(y.reshape(-1)[test], pred.reshape(-1)[test])
    metric.update(_diagnostics(model, dataset, split, pred))
    metric.update({"run": run_name, "model_name": model_name,
                   "mechanism": mechanism, "analyte": analyte,
                   "mask": mask_name, "seed": int(seed),
                   "config_hash": params["config_hash"]})
    full = _full_grid(dataset, pred, split, model_name=model_name,
                      analyte=analyte, mask=mask_name, seed=seed)
    full.to_parquet(full_path, index=False)
    full[full["observed"]].to_parquet(run_dir / "observed_predictions.parquet", index=False)
    # Save selected weights and source-train statistics so the mechanism can
    # be inspected without another fit. Target labels are not in this file.
    inputs = model._bundle.inputs
    torch.save({
        'state_dict': model._bundle.model.state_dict(), 'config': params,
        'target_mu': inputs.target_mu, 'target_sd': inputs.target_sd,
        'edge_index': model._bundle.model._edge_index,
        'edge_attr': model._bundle.model._edge_attr,
    }, run_dir / 'checkpoint.pt')
    if mechanism == 'm2':
        with torch.no_grad():
            visible, feed = model._visible_input()
            x, _ = model._make_input(inputs, visible, model._bundle.model, feed)
            weights = model._bundle.model.lag_weights(
                x, model._bundle.model._edge_index, model._bundle.model._edge_attr
            )
        np.savez_compressed(run_dir / 'lag_weights.npz',
                            weights=weights.numpy(), lags=model._bundle.model.LAGS,
                            edge_index=model._bundle.model._edge_index.numpy())
    meta = {
        "run": run_name, "config": params,
        "config_hash_version": params["config_hash_version"],
        "started_at": started_at,
        "finished_at": datetime.now(timezone.utc).isoformat(),
        "config_hash": params["config_hash"], "created_at": started_at,
        "runtime_code_snapshot_sha256": params["runtime_snapshot_hash"],
        "run_identity_sha256": run_identity_sha256(
            params["config_hash"], started_at, params["runtime_snapshot_hash"]
        ),
        "dataset": file_identity(dataset_path), "mask": file_identity(mask_path),
        "mechanism": mechanism, "model_name": model_name,
        "target_analyte": analyte, "target_transform": TARGET_TRANSFORMS[analyte],
        "causal": True, "lookback": params["lookback"],
        "full_grid_sha256": file_identity(full_path)["sha256"],
        "checkpoint_sha256": file_identity(run_dir / 'checkpoint.pt')['sha256'],
        "rows_full_grid": len(full), "rows_observed": int(full["observed"].sum()),
        "training": {"epochs_run": model._bundle.epochs_run,
                     "best_val_loss": model._bundle.best_val_loss,
                     "masking_history": model._bundle.masking_history},
    }
    meta_path.write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    metrics_path.write_text(json.dumps(metric, indent=2) + "\n", encoding="utf-8")
    return {"run": run_name, "status": "completed", **metric}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=("smoke", "pilot"), default="smoke")
    ap.add_argument("--mechanism", choices=MECHANISMS, default="m1")
    ap.add_argument("--spatial-variant", choices=SPATIAL_VARIANTS, default="baseline")
    ap.add_argument("--lag-mode", choices=("learned", "static", "fixed", "none"), default="learned")
    ap.add_argument("--analytes", nargs="+", choices=ANALYTES, default=None)
    ap.add_argument("--masks", nargs="+", default=None)
    ap.add_argument("--seeds", nargs="+", type=int, default=None)
    ap.add_argument("--max-epochs", type=int, default=None)
    ap.add_argument("--patience", type=int, default=None)
    ap.add_argument("--out-dir", default=str(OUT_ROOT))
    ap.add_argument("--force", action="store_true")
    ap.add_argument('--threads', type=int, default=torch.get_num_threads())
    args = ap.parse_args()
    torch.set_num_threads(args.threads)
    if args.stage == "smoke":
        analytes = args.analytes or ["doc"]
        masks, seeds = args.masks or ["e2a_strict"], args.seeds or [42]
        max_epochs, patience = args.max_epochs or 2, args.patience or 1
    else:
        analytes = args.analytes or list(ANALYTES)
        masks, seeds = args.masks or list(DEFAULT_MASKS), args.seeds or list(DEFAULT_SEEDS)
        max_epochs, patience = args.max_epochs or 30, args.patience or 5
    out = Path(args.out_dir) / args.stage
    variant_root = out / args.spatial_variant
    variant_root.mkdir(parents=True, exist_ok=True)
    snapshot = runtime_code_snapshot_sha256()
    (variant_root / 'runtime_snapshot.json').write_text(json.dumps({
        'hash': snapshot, 'files': runtime_code_snapshot(),
    }, indent=2) + '\n')
    plan = {
        "version": "graph_upgrade_v2", "mechanism": args.mechanism,
        "spatial_variant": args.spatial_variant,
        "analytes": analytes, "masks": masks, "seeds": seeds,
        "max_epochs": max_epochs, "patience": patience,
        "history_scope": 'full_causal_sequence' if args.mechanism == 'm3' else ('mixed_causal_sequence' if args.mechanism == 'm13' else 'rolling_12_months'),
        "causal": True,
        "lag_buckets": [0, 1, 3, 6, 12] if args.mechanism == "m2" else None,
        "lag_mode": args.lag_mode if args.mechanism == "m2" else None,
        "purpose": "mechanism exploration; results are compared to H2X-T and Temporal RF",
    }
    (variant_root / args.mechanism).mkdir(parents=True, exist_ok=True)
    (variant_root / args.mechanism / "run_plan.json").write_text(json.dumps(plan, indent=2) + "\n")
    rows = []
    for analyte in analytes:
        for mask in masks:
            load_mask(mask, MASKS_DIR)
            for seed in seeds:
                print(f"[{args.mechanism}] {analyte} {mask} seed={seed}", flush=True)
                rows.append(_run_one(
                    mechanism=args.mechanism, analyte=analyte,
                    mask_name=mask, seed=seed, max_epochs=max_epochs,
                    patience=patience, out_root=variant_root, force=args.force,
                    lag_mode=args.lag_mode,
                    stage=args.stage, runtime_snapshot=snapshot,
                    spatial_variant=args.spatial_variant,
                ))
                # Keep an incremental ledger so an interrupted batch still
                # exposes the completed configurations for review.
                pd.DataFrame(rows).to_csv(variant_root / args.mechanism / "metrics.csv", index=False)
    pd.DataFrame(rows).to_csv(variant_root / args.mechanism / "metrics.csv", index=False)
    print(pd.DataFrame(rows).to_string(index=False))


if __name__ == "__main__":
    main()
