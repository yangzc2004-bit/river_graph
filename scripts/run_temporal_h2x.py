"""Train and evaluate the H2X-T temporal extension.

The script keeps the existing H2X snapshot as the matched baseline and fits
the new causal GRU extension per analyte.  A run writes a full station-month
grid, observed-cell predictions, metrics, and a provenance sidecar under a
versioned experiment directory.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from river_graph.experiments.evaluate import load_mask, metrics
from river_graph.experiments.provenance import (
    build_meta,
    config_hash,
    file_identity,
    run_identity_sha256,
    runtime_code_snapshot_sha256,
)
from river_graph.experiments.temporal_h2x import (
    TARGET_TRANSFORMS,
    H2XTemporalModel,
)
from river_graph.experiments.transfer import ANALYTES, DATASETS
from river_graph.models.gcn import GCNDocModel

DEFAULT_MASKS = (
    "e1_r20_seed42",
    "e2a_strict",
    "e2b_partial",
    "e3_spatial_seed42",
)
DEFAULT_SEEDS = (42, 43, 44)
FULL_SEEDS = (42, 43, 44, 45, 46)
OUT_ROOT = Path("experiments/phase4_transfer/temporal_h2x_v1")
MASKS_DIR = Path("experiments/masks_stcore_v1")
TARGET_MASKS_DIR = OUT_ROOT / "target_masks"


def _dataset(path: str | Path) -> dict:
    return torch.load(path, map_location="cpu", weights_only=False)


def _split(mask: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    # Mask archives may carry non-cell metadata (e.g. held-out station IDs for
    # E3).  Only the four flat-cell roles enter the model contract.
    roles = ("train", "val", "test", "context")
    return {
        key: np.asarray(mask[key], dtype=np.int64)
        for key in roles
        if key in mask
    }


def _target_mask_path(analyte: str, mask_name: str, dataset: dict) -> Path:
    """Materialize the frozen missingness family on an analyte's observed grid.

    The primary masks were frozen on DOC.  Other analytes have a small number
    of additional missing labels, so their role cells are intersected with
    that analyte's observed mask.  The resulting file is deterministic and
    becomes the mask identity for both matched models.
    """
    TARGET_MASKS_DIR.mkdir(parents=True, exist_ok=True)
    out = TARGET_MASKS_DIR / f"{analyte}__{mask_name}.npz"
    observed = np.asarray(
        dataset["y_mask"].cpu() if isinstance(dataset["y_mask"], torch.Tensor)
        else dataset["y_mask"]
    ).reshape(-1).astype(bool)
    source = load_mask(mask_name, MASKS_DIR)
    roles = ("train", "val", "test", "context")
    filtered = {
        role: np.asarray(source[role], dtype=np.int64)
        for role in roles if role in source
    }
    filtered = {role: cells[observed[cells]] for role, cells in filtered.items()}
    if not out.is_file():
        np.savez_compressed(out, **filtered)
    else:
        with np.load(out, allow_pickle=False) as saved:
            for role, cells in filtered.items():
                if not np.array_equal(saved[role], cells):
                    raise ValueError(f"target mask drift for {analyte}/{mask_name}")
    return out


def _model_name(kind: str, analyte: str, mask: str, seed: int) -> str:
    return f"{kind}__{analyte}__{mask}__seed{seed}"


def _history_ablation(kind: str) -> str:
    return {
        "h2x_t": "none",
        "h2x_t_current_only": "none",
        "h2x_t_no_history": "shuffle",
        "h2x_t_hydro_only": "hydro_only",
    }.get(kind, "none")


def _lookback_for_kind(kind: str) -> int:
    """Return the frozen temporal window for each diagnostic arm."""
    return {
        "h2x_t_current_only": 1,
        "h2x_t_lb3": 3,
        "h2x_t_lb6": 6,
    }.get(kind, 12)


def _full_grid(dataset: dict, pred: np.ndarray, split: dict, *, kind: str,
               analyte: str, mask: str, seed: int) -> pd.DataFrame:
    y = np.asarray(dataset["y"].cpu() if isinstance(dataset["y"], torch.Tensor) else dataset["y"])
    observed = np.asarray(
        dataset["y_mask"].cpu()
        if isinstance(dataset["y_mask"], torch.Tensor)
        else dataset["y_mask"]
    ).astype(bool)
    n, t = y.shape
    roles = np.full(n * t, "", dtype=object)
    for role in ("train", "val", "test", "context"):
        cells = np.asarray(split.get(role, np.array([], dtype=np.int64)), dtype=np.int64)
        roles[cells] = role
    rows = []
    for i, station in enumerate(dataset["site_no"]):
        for j, month in enumerate(dataset["months"]):
            rows.append({
                "analyte": analyte,
                "station": str(station),
                "month": str(month),
                "month_index": j,
                "y_true": float(y[i, j]) if observed[i, j] else np.nan,
                "y_pred": float(pred[i, j]),
                "observed": bool(observed[i, j]),
                "split": roles[i * t + j],
                "model_name": kind,
                "mask": mask,
                "seed": int(seed),
            })
    return pd.DataFrame(rows)


def _params(*, kind: str, analyte: str, seed: int, dataset_path: str,
            mask_name: str, mask_path: str, max_epochs: int, patience: int,
            lookback: int = 12) -> dict:
    temporal = "gru" if kind.startswith("h2x_t") else "none"
    return {
        "script": "scripts/run_temporal_h2x.py",
        "model_name": kind,
        "tag": f"temporal_h2x_v1_{kind}",
        "architecture": "transport_enc",
        "variant": "river",
        "seed": int(seed),
        "lr": 1e-3,
        "weight_decay": 0.0,
        "edge_dropout": 0.0,
        "share_weights": False,
        "env_groups": None,
        "env_encoder": True,
        "edge_set": "river",
        "edge_direction": "both",
        "hidden": 64,
        "layers": 2,
        "dropout": 0.1,
        "max_epochs": int(max_epochs),
        "patience": int(patience),
        "env_emb": 32,
        "dataset_path": str(dataset_path),
        "mask_path": str(mask_path),
        "temporal": temporal,
        "lookback": int(lookback if kind == "h2x_t" else _lookback_for_kind(kind)),
        "causal": True,
        "temporal_hidden": 64 if kind == "h2x_t" else None,
        "target_analyte": analyte,
        "target_transform": TARGET_TRANSFORMS[analyte],
        # Both arms use one optimizer update per full monthly sequence.  This
        # makes the snapshot comparison commensurate with the GRU wrapper.
        "training_protocol": "matched_full_grid",
        "history_ablation": _history_ablation(kind),
    }


def _run_one(*, kind: str, analyte: str, mask_name: str, seed: int,
             max_epochs: int, patience: int, out_root: Path,
             force: bool = False) -> dict:
    dataset_path = DATASETS[analyte]
    dataset = _dataset(dataset_path)
    target_mask_path = _target_mask_path(analyte, mask_name, dataset)
    with np.load(target_mask_path, allow_pickle=False) as target_mask:
        split = _split({key: target_mask[key] for key in target_mask.files})
    run_name = _model_name(kind, analyte, mask_name, seed)
    run_dir = out_root / "runs" / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    params = _params(
        kind=kind, analyte=analyte, seed=seed, dataset_path=dataset_path,
        mask_name=mask_name, mask_path=str(target_mask_path),
        max_epochs=max_epochs, patience=patience,
    )
    ds_identity = file_identity(dataset_path)
    mask_identity = file_identity(target_mask_path)
    params["dataset_sha256"] = ds_identity.get("sha256")
    params["mask_sha256"] = mask_identity.get("sha256")
    params["config_hash_version"] = 3
    runtime_sha = runtime_code_snapshot_sha256()
    params["runtime_snapshot_hash"] = runtime_sha
    expected_hash = config_hash(params, version=3)
    meta_path = run_dir / "meta.json"
    metrics_path = run_dir / "metrics.json"
    full_path = run_dir / "full_grid.parquet"
    if not force and meta_path.is_file() and metrics_path.is_file() and full_path.is_file():
        old = json.loads(meta_path.read_text(encoding="utf-8"))
        if old.get("config_hash") == expected_hash:
            return {"run": run_name, "status": "cached", **json.loads(metrics_path.read_text())}

    from datetime import datetime, timezone

    started_at = datetime.now(timezone.utc).isoformat()
    if kind == "h2x":
        model = GCNDocModel(
            architecture="transport_enc",
            edge_set="river",
            edge_direction="both",
            env_encoder=True,
            variant="river",
            hidden=64,
            layers=2,
            dropout=0.1,
            lr=1e-3,
            max_epochs=max_epochs,
            patience=patience,
            seed=seed,
            target_transform=TARGET_TRANSFORMS[analyte],
            training_protocol="matched_full_grid",
        )
    elif kind.startswith("h2x_t"):
        model = H2XTemporalModel(
            seed=seed,
            lookback=_lookback_for_kind(kind),
            temporal_hidden=64,
            hidden=64,
            layers=2,
            dropout=0.1,
            lr=1e-3,
            max_epochs=max_epochs,
            patience=patience,
            target_transform=TARGET_TRANSFORMS[analyte],
            history_ablation=_history_ablation(kind),
        )
    else:
        raise ValueError(f"unknown model kind: {kind}")

    pred = model.fit_predict(dataset, split)
    y = np.asarray(dataset["y"].cpu() if isinstance(dataset["y"], torch.Tensor) else dataset["y"])
    test = split["test"]
    metric = metrics(y.ravel()[test], pred.ravel()[test])
    metric.update({
        "run": run_name,
        "model_name": kind,
        "analyte": analyte,
        "mask": mask_name,
        "seed": int(seed),
        "config_hash": expected_hash,
    })
    full = _full_grid(dataset, pred, split, kind=kind, analyte=analyte,
                      mask=mask_name, seed=seed)
    full.to_parquet(full_path, index=False)
    observed = full[full["observed"]].copy()
    observed.to_parquet(run_dir / "observed_predictions.parquet", index=False)
    meta = build_meta(
        model_name=kind,
        mask_name=mask_name,
        dataset_path=dataset_path,
        split=split,
        params=params,
        results_path=full_path,
        masks_dir=MASKS_DIR,
        mask_path=target_mask_path,
        caller="scripts/run_temporal_h2x.py",
    )
    meta["config_hash"] = expected_hash
    meta["started_at"] = started_at
    meta["runtime_code_snapshot_sha256"] = runtime_sha
    meta["run_identity_sha256"] = run_identity_sha256(
        expected_hash, started_at, runtime_sha
    )
    meta["target_analyte"] = analyte
    meta["target_transform"] = TARGET_TRANSFORMS[analyte]
    meta["parent_mask_name"] = mask_name
    meta["target_mask_path"] = str(target_mask_path)
    meta["target_mask_sha256"] = mask_identity.get("sha256")
    meta["temporal"] = params["temporal"]
    meta["lookback"] = params["lookback"]
    meta["causal"] = params["causal"]
    meta["temporal_hidden"] = params["temporal_hidden"]
    meta["history_ablation"] = params["history_ablation"]
    meta["temporal_product"] = kind.startswith("h2x_t")
    meta["export_scope"] = (
        "full station-month grid; observed labels are retained only for final "
        "evaluation and hidden cells have y_true=NaN"
    )
    bundle = getattr(model, "_bundle", {})
    meta["training"] = {
        "epochs_run": bundle.get("epochs_run"),
        "best_val_loss": bundle.get("best_val_loss"),
    }
    meta["full_grid_path"] = str(full_path)
    meta["full_grid_sha256"] = file_identity(full_path)["sha256"]
    meta["rows_full_grid"] = len(full)
    meta["rows_observed"] = len(observed)
    meta_path.write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    metrics_path.write_text(json.dumps(metric, indent=2) + "\n", encoding="utf-8")
    return {"run": run_name, "status": "completed", **metric}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=("smoke", "pilot", "full"), default="smoke")
    ap.add_argument("--out-dir", default=str(OUT_ROOT))
    ap.add_argument("--analytes", nargs="+", choices=ANALYTES, default=None)
    ap.add_argument("--masks", nargs="+", default=None)
    ap.add_argument("--seeds", nargs="+", type=int, default=None)
    ap.add_argument(
        "--models", nargs="+",
        choices=("h2x", "h2x_t", "h2x_t_current_only", "h2x_t_lb3",
                 "h2x_t_lb6",
                 "h2x_t_no_history", "h2x_t_hydro_only"),
        default=None,
    )
    ap.add_argument("--max-epochs", type=int, default=None)
    ap.add_argument("--patience", type=int, default=None)
    ap.add_argument("--force", action="store_true")
    ap.add_argument(
        "--no-plan-write",
        action="store_true",
        help="use an already frozen run_plan.json (for parallel workers)",
    )
    args = ap.parse_args()

    if args.stage == "smoke":
        analytes = args.analytes or ["doc"]
        masks = args.masks or ["e2a_strict"]
        seeds = args.seeds or [42]
        models = args.models or ["h2x_t"]
        max_epochs, patience = args.max_epochs or 3, args.patience or 1
    elif args.stage == "pilot":
        analytes = args.analytes or list(ANALYTES)
        masks = args.masks or list(DEFAULT_MASKS)
        seeds = args.seeds or list(DEFAULT_SEEDS)
        models = args.models or ["h2x", "h2x_t"]
        # Pilot is a feasibility screen.  The full sequence is expensive
        # because each epoch encodes all 654 monthly graphs; keep both arms
        # on the same bounded budget and reserve the longer formal budget for
        # T3.
        max_epochs, patience = args.max_epochs or 10, args.patience or 3
    else:
        analytes = args.analytes or list(ANALYTES)
        masks = args.masks or list(DEFAULT_MASKS)
        seeds = args.seeds or list(FULL_SEEDS)
        models = args.models or ["h2x_t"]
        max_epochs, patience = args.max_epochs or 50, args.patience or 10

    out_root = Path(args.out_dir)
    out_root.mkdir(parents=True, exist_ok=True)
    plan = {
        "version": "temporal_h2x_v1",
        "stage": args.stage,
        "analytes": analytes,
        "masks": masks,
        "seeds": seeds,
        "models": models,
        "max_epochs": max_epochs,
        "patience": patience,
        "lookback": (
            1 if models and all(_lookback_for_kind(kind) == 1 for kind in models)
            else 12
        ),
        "temporal": "gru",
        "causal": True,
        "target_mask_policy": (
            "parent frozen mask intersected with analyte y_mask; "
            "same missingness family, analyte-valid cells only"
        ),
        "target_masks_dir": str(TARGET_MASKS_DIR),
        "training_started": True,
    }
    plan_path = out_root / "run_plan.json"
    if args.no_plan_write:
        if not plan_path.is_file():
            raise FileNotFoundError(
                f"--no-plan-write requires an existing frozen plan: {plan_path}"
            )
    else:
        plan_path.write_text(json.dumps(plan, indent=2) + "\n", encoding="utf-8")
    results = []
    for analyte in analytes:
        for mask_name in masks:
            load_mask(mask_name, MASKS_DIR)  # fail before training on a bad name
            for seed in seeds:
                for kind in models:
                    print(f"[{kind}] {analyte} {mask_name} seed={seed}", flush=True)
                    results.append(_run_one(
                        kind=kind,
                        analyte=analyte,
                        mask_name=mask_name,
                        seed=seed,
                        max_epochs=max_epochs,
                        patience=patience,
                        out_root=out_root,
                        force=args.force,
                    ))
    frame = pd.DataFrame(results)
    metrics_path = out_root / "metrics.csv"
    if metrics_path.is_file():
        previous = pd.read_csv(metrics_path)
        frame = pd.concat([previous, frame], ignore_index=True)
        frame = frame.drop_duplicates(subset=["run"], keep="last")
    frame.sort_values("run").to_csv(metrics_path, index=False)
    print(json.dumps({"stage": args.stage, "runs": len(results), "out": str(out_root)}))


if __name__ == "__main__":
    main()
