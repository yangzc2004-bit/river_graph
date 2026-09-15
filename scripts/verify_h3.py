"""T11: independently recompute an H3 run from the weights stored on disk.

    python scripts/verify_h3.py --stage pilot
    python scripts/verify_h3.py --stage pilot --arm h3a --seed 0 --mask e2b_partial

For every run record this reloads the best checkpoint into a freshly built
model, re-runs the forward pass over the *stored* configuration, and compares
the result with the saved branch outputs.  It then recomputes the headline
metrics from the saved predictions and refuses any disagreement.

It also refuses to let a frozen H3A checkpoint stand in for the independently
trained ENV arm: the base-only output of an H3A checkpoint is computed and
required to differ from that run's actually trained ENV counterpart.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import pandas as pd

from river_graph.experiments.evaluate import metrics
from river_graph.experiments.h3_masks import load_mask
from river_graph.experiments.h3_runs import audit_run
from river_graph.experiments.h3_training import (
    H3Trainer,
    load_protocol,
    restrict_to_stations,
)

TOLERANCE = 1e-6


def restore_inputs(record: dict, masks_dir: Path) -> tuple[dict, dict]:
    """Rebuild the exact data a run read, from the record's own identity."""
    config = record["config"]
    dataset = torch.load(config["dataset"]["dataset_path"], weights_only=False)
    mask = config["mask"]
    split = load_mask(Path(masks_dir) / (mask + ".npz"))
    subset = config["dataset"]["subset_stations"]
    if subset:
        dataset, split = restrict_to_stations(dataset, split, int(subset))
    return dataset, split


def forward_from_checkpoint(
    record: dict, dataset: dict, split: dict, protocol: dict
) -> dict:
    """Re-run the stored configuration and return the raw branch outputs."""
    config = record["config"]
    arm = config["arm"]
    trainer = H3Trainer(arm, int(config["seed"]), protocol)
    trainer.prepare(dataset, split)
    trainer.build_model()
    checkpoint = torch.load(
        Path(record["_root"]) / record["artifacts"]["checkpoint"]["path"],
        weights_only=False,
    )
    state = checkpoint["state_dict"]
    model_keys = set(trainer.model.state_dict())
    if set(state) != model_keys:
        raise ValueError("checkpoint state_dict keys disagree with the model")
    trainer.model.load_state_dict(state)
    trainer.model.eval()

    bounds = checkpoint["clip_log_bounds"]
    cells = torch.as_tensor(np.asarray(split["val"], dtype=np.int64))
    with torch.no_grad():
        trainer._fill_doc_channel(trainer.eval_visible)
        ci, cj = cells // trainer.t, cells % trainer.t
        base = torch.full((len(cells),), float("nan"))
        correction = torch.full((len(cells),), float("nan"))
        total = torch.empty(len(cells))
        for month in np.unique(cj.numpy()):
            selector = cj == month
            b, c, t = trainer.components_month(int(month))
            base[selector] = b[ci[selector]]
            correction[selector] = c[ci[selector]]
            total[selector] = t[ci[selector]]
        clipped = total.clamp(min=float(bounds[0]), max=float(bounds[1]))
    return {
        "cells": cells.numpy(),
        "base_log": base.numpy(),
        "correction_log": correction.numpy(),
        "total_log": total.numpy(),
        "pred_log_clipped": clipped.numpy(),
        "y_pred": np.expm1(clipped.numpy()),
        "clip_log_bounds": [float(bounds[0]), float(bounds[1])],
        "base_only_log": base.numpy(),
    }


def compare(stored, rebuilt, key: str) -> None:
    a = np.asarray(stored, dtype=np.float64)
    b = np.asarray(rebuilt, dtype=np.float64)
    same_nan = np.array_equal(np.isnan(a), np.isnan(b))
    if not same_nan:
        raise ValueError(key + ": NaN pattern differs from the stored values")
    mask = ~np.isnan(a)
    if not np.allclose(a[mask], b[mask], rtol=TOLERANCE, atol=TOLERANCE):
        worst = float(np.max(np.abs(a[mask] - b[mask]))) if mask.any() else 0.0
        raise ValueError(
            key + ": rebuilt prediction disagrees with the stored value (max "
            + format(worst, ".3e") + " > " + format(TOLERANCE, ".1e") + ")"
        )


def verify_record(record: dict, root: Path, masks_dir: Path, protocol: dict) -> dict:
    record["_root"] = str(root)
    dataset, split = restore_inputs(record, masks_dir)
    stored = pd.read_parquet(
        root / record["artifacts"]["validation"]["path"]
    )
    rebuilt = forward_from_checkpoint(record, dataset, split, protocol)

    if not np.array_equal(stored["cell"].to_numpy(), rebuilt["cells"]):
        raise ValueError("validation cell order disagrees with the split")
    for key in ("base_log", "correction_log", "total_log", "pred_log_clipped",
                "y_pred"):
        compare(stored[key].to_numpy(), rebuilt[key], key)
    np.testing.assert_allclose(
        stored["y_pred"].to_numpy(), np.expm1(stored["pred_log_clipped"].to_numpy()),
        rtol=1e-12, atol=0,
    )
    np.testing.assert_array_equal(
        stored["clipped"].to_numpy(),
        stored["pred_log_clipped"].to_numpy() != stored["total_log"].to_numpy(),
    )
    bounds = record["training"]["clip_log_bounds"]
    np.testing.assert_allclose(rebuilt["clip_log_bounds"], bounds, rtol=0, atol=0)

    # no NaN may reach a prediction, and the branch columns must match the arm:
    # ENV has no graph branch, H2X has no environmental base, H3A carries both.
    for key in ("total_log", "pred_log_clipped", "y_pred"):
        if not np.isfinite(stored[key].to_numpy()).all():
            raise ValueError(key + " contains non-finite values")
    expectation = {
        "env": {"base_log": "finite", "correction_log": "nan"},
        "h2x": {"base_log": "nan", "correction_log": "finite"},
        "h3a": {"base_log": "finite", "correction_log": "finite"},
        "h3a_no_message": {"base_log": "finite", "correction_log": "finite"},
    }[record["config"]["arm"]]
    for key, want in expectation.items():
        values = stored[key].to_numpy()
        if want == "nan" and not np.isnan(values).all():
            raise ValueError(key + " must be absent for arm " + record["config"]["arm"])
        if want == "finite" and not np.isfinite(values).all():
            raise ValueError(key + " must be finite for arm " + record["config"]["arm"])

    recomputed = metrics(stored["y_true"].to_numpy(), stored["y_pred"].to_numpy())
    for key, value in record["validation_metrics"].items():
        if not np.isfinite(value) or abs(value - recomputed[key]) > TOLERANCE:
            raise ValueError("metric " + key + " does not recompute")
    residual = stored["y_true_log"].to_numpy() - stored["total_log"].to_numpy()
    raw_rmse = float(np.sqrt(np.mean(residual**2)))
    if abs(raw_rmse - record["raw_log_metrics"]["raw_log_rmse"]) > TOLERANCE:
        raise ValueError("raw log RMSE does not recompute")
    if float(np.mean(residual**2)) != record["training"]["best_val_mse_raw"] and abs(
        float(np.mean(residual**2)) - record["training"]["best_val_mse_raw"]
    ) > TOLERANCE:
        raise ValueError("selected checkpoint does not reproduce the raw loss")
    return {
        "stem": Path(record["_manifest"]).stem,
        "arm": record["config"]["arm"],
        "seed": record["config"]["seed"],
        "mask": record["config"]["mask"],
        "scenario": record["config"]["scenario"],
        "log_rmse": recomputed["log_rmse"],
        "mae": recomputed["mae"],
        "raw_log_rmse": raw_rmse,
        "cells": len(stored),
        "finite": True,
        "clipped_fraction": float(stored["clipped"].to_numpy().mean()),
        "epochs": int(record["training"]["epochs"]),
        "best_epoch": int(record["training"]["best_epoch"]),
        "parameters": int(record["training"]["parameter_count"]),
        "seconds": float(record["training"]["elapsed_seconds"]),
    }


def base_only_versus_env(records: list[dict]) -> list[dict]:
    """H3A's frozen base path must not be reported as an independently trained ENV."""
    by_key = {
        (r["config"]["arm"], r["config"]["seed"], r["config"]["mask"]): r
        for r in records
    }
    findings = []
    for record in records:
        config = record["config"]
        if config["arm"] != "h3a":
            continue
        key = ("env", config["seed"], config["mask"])
        if key not in by_key:
            findings.append(
                {"mask": config["mask"], "seed": config["seed"],
                 "status": "no independently trained ENV run to compare"}
            )
            continue
        root = Path(record["_root"])
        stored = pd.read_parquet(root / record["artifacts"]["validation"]["path"])
        env_record = by_key[key]
        env_stored = pd.read_parquet(
            Path(env_record["_root"]) / env_record["artifacts"]["validation"]["path"]
        )
        if not np.array_equal(stored["cell"].to_numpy(), env_stored["cell"].to_numpy()):
            raise ValueError("ENV and H3A validation cells are not aligned")
        base_only = stored["base_log"].to_numpy()
        env_pred = env_stored["total_log"].to_numpy()
        identical = bool(np.allclose(base_only, env_pred, rtol=0, atol=0))
        findings.append(
            {
                "mask": config["mask"],
                "seed": config["seed"],
                "status": "distinct",
                "identical_base_only_and_env": identical,
            }
        )
        if identical:
            raise ValueError(
                "an H3A base-only output equals the independently trained ENV "
                "run; the two must not be treated as interchangeable"
            )
    return findings


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", default="experiments/h3a_v1")
    ap.add_argument("--masks-dir", default=None)
    ap.add_argument("--protocol", default="configs/h3a_v1.json")
    ap.add_argument("--stage", default=None, choices=("smoke", "pilot", "expand"))
    ap.add_argument("--arm", default=None)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--mask", default=None)
    ap.add_argument("--json-out", default=None)
    args = ap.parse_args()

    root = ROOT / args.root
    protocol = load_protocol(ROOT / args.protocol)
    masks_dir = Path(args.masks_dir) if args.masks_dir else (
        root / "masks" if (root / "masks").is_dir() else ROOT / "experiments/masks"
    )
    if not masks_dir.is_absolute():
        masks_dir = ROOT / masks_dir

    manifests = sorted((root / "runs").glob("*.json"))
    manifests = [p for p in manifests if not p.name.endswith((".intent.json",
                                                             ".pending.json"))]
    records = []
    for path in manifests:
        record = json.loads(path.read_text(encoding="utf-8"))
        if record.get("status") != "complete":
            continue
        record["_manifest"] = str(path)
        config = record["config"]
        if args.stage and record.get("stage") != args.stage:
            continue
        if args.arm and config["arm"] != args.arm:
            continue
        if args.seed is not None and config["seed"] != args.seed:
            continue
        if args.mask and config["mask"] != args.mask:
            continue
        records.append(record)
    if not records:
        print("no run records matched")
        return 1

    verified = []
    for record in records:
        dataset, split = restore_inputs(record, masks_dir)
        audit_run(root, Path(record["_manifest"]), dataset, split,
                  record["config_hash"], expected_config=record["config"])
        verified.append(verify_record(record, root, masks_dir, protocol))
        print(
            "verified " + verified[-1]["stem"] + " cells="
            + str(verified[-1]["cells"]) + " log_rmse="
            + format(verified[-1]["log_rmse"], ".5f"),
            flush=True,
        )

    findings = base_only_versus_env(records)
    summary = {
        "verified_runs": len(verified),
        "root": str(root),
        "tolerance": TOLERANCE,
        "runs": verified,
        "base_only_vs_env": findings,
    }
    out = ROOT / args.json_out if args.json_out else root / "verification_summary.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
                   encoding="utf-8")
    print("verified " + str(len(verified)) + " runs; wrote " + str(out))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
