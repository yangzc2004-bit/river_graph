"""A0 evidence inventory: identify, verify and classify every stored artifact.

Zero retraining. This script only reads what is already on disk
(predictions, metric JSONs, masks, datasets, frozen tables) and answers:

* which prediction files exist, and do they belong to their mask?
* do the stored predictions reproduce the frozen benchmark table?
* which (model, mask) pairs carry metrics but no predictions at all?
* which files are conflicting, zero-coverage, or of unknown origin?

Usage:
    python scripts/audit_artifacts.py                 # inventory only
    python scripts/audit_artifacts.py --verify        # + recompute vs frozen
    python scripts/audit_artifacts.py --verify --out experiments/analysis

Outputs (under --out):
    artifact_inventory_<date>.json   machine-readable inventory
    artifact_verification_<date>.csv per-prediction verification table
"""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from river_graph.experiments.evaluate import load_dataset, load_mask, metrics

PRED_DIR = Path("experiments/predictions")
RESULTS_DIR = Path("experiments/results")
MASKS_DIR = Path("experiments/masks")
FROZEN = Path("experiments/frozen_results")

# Which dataset each frozen model family was trained on (scripts/run_freeze.py).
MODEL_DATASETS = {
    "B0_station_mean": "data/processed/mississippi_graph_v02.pt",
    "B1_kriging": "data/processed/mississippi_graph_v02.pt",
    "B2_random_forest": "data/processed/mississippi_graph_v02.pt",
    "B3_mlp": "data/processed/mississippi_graph_v02.pt",
    "G0_gcn_none": "data/processed/mississippi_graph_v02.pt",
    "G0_gcn_random": "data/processed/mississippi_graph_v02.pt",
    "G0_gcn_river": "data/processed/mississippi_graph_v02.pt",
    "H1_directed_river": "data/processed/mississippi_graph_v02.pt",
    "H2_transport_river": "data/processed/mississippi_graph_v03.pt",
    "H2E_transport_river": "data/processed/mississippi_graph_v04.pt",
    "H2X_transport_enc_river": "data/processed/mississippi_graph_v04.pt",
}
# Multiseed runs (H1f/H2f/H2Xf + _s1.._s4) all come from the transport family.
DATASET_BY_PREFIX = [
    ("H2X", "data/processed/mississippi_graph_v04.pt"),
    ("H2E", "data/processed/mississippi_graph_v04.pt"),
    ("H2", "data/processed/mississippi_graph_v03.pt"),
    ("H1", "data/processed/mississippi_graph_v02.pt"),
    ("H15", "data/processed/mississippi_graph_v02.pt"),
    ("G0", "data/processed/mississippi_graph_v02.pt"),
]

TOL = 1e-4


def sha256_file(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        while block := fh.read(chunk):
            h.update(block)
    return h.hexdigest()


def dataset_for_model(model: str) -> str:
    if model in MODEL_DATASETS:
        return MODEL_DATASETS[model]
    for prefix, path in DATASET_BY_PREFIX:
        if model.startswith(prefix):
            return path
    return "data/processed/mississippi_graph_v02.pt"


def verify_one(
    parquet: Path,
    datasets: dict[str, dict],
    masks: dict[str, dict],
) -> dict:
    """Recompute test metrics straight from the stored predictions.

    Test cells are taken from the MASK (ground truth boundaries), then matched
    to parquet rows by (station, month). A file whose rows do not line up with
    the mask is reported, never silently averaged.
    """
    df = pd.read_parquet(parquet)
    model = str(df["model"].iloc[0])
    mask_name = str(df["mask"].iloc[0])
    dpath = dataset_for_model(model)
    if dpath not in datasets:
        datasets[dpath] = load_dataset(dpath)
    ds = datasets[dpath]
    if mask_name not in masks:
        masks[mask_name] = load_mask(mask_name, MASKS_DIR)
    split = masks[mask_name]

    y = ds["y"].numpy()
    _n, t = y.shape
    sites = list(ds["site_no"])
    months = list(ds["months"])

    test_idx = np.asarray(split["test"], dtype=np.int64)
    # mask flat index -> (station, month) key
    want = [(sites[i // t], months[i % t]) for i in test_idx]
    lut = dict(zip(zip(df["station"].tolist(), df["month"].tolist()),
                   zip(df["y_true"].tolist(), df["y_pred"].tolist())))
    yt = np.array([lut.get(k, (np.nan, np.nan))[0] for k in want], dtype=float)
    yp = np.array([lut.get(k, (np.nan, np.nan))[1] for k in want], dtype=float)
    # A cell is "missing from parquet" only when neither label nor prediction
    # was stored for it; NaN predictions are a legitimate coverage gap.
    present = np.array([k in lut for k in want], dtype=bool)
    unmatched = int((~present).sum())

    m = metrics(yt, yp)
    stored_test = df[df["split"] == "test"]
    return {
        "file": parquet.name,
        "model": model,
        "mask": mask_name,
        "dataset": dpath,
        "rows": len(df),
        "split_counts": {str(k): int(v) for k, v in
                         df["split"].value_counts().items()},
        "test_stored_rows": len(stored_test),
        "test_mask_cells": len(test_idx),
        "test_unmatched": unmatched,
        "test_cells_nonfinite_pred": int(np.isnan(yp).sum()),
        "coverage": float(np.isfinite(yp).mean()) if len(yp) else 0.0,
        "recomputed": {"mae": m["mae"], "r2": m["r2"], "rmse": m["rmse"],
                       "n": m["n"]},
        "sha256": sha256_file(parquet),
    }


def classify(rows: list[dict], frozen: pd.DataFrame, multiseed: pd.DataFrame) -> None:
    """Assign a status per prediction file.

    The status keys on whether the recomputed metric agrees with the frozen
    table, because that is what downstream analysis depends on. Coverage gaps
    (NaN predictions) are recorded separately and never silently averaged.
    """
    fmap = {(r["model"], r["mask"]): r for _, r in frozen.iterrows()}
    mmap = {(r["model"], r["mask"]): r for _, r in multiseed.iterrows()}
    for r in rows:
        key = (r["model"], r["mask"])
        fr = fmap.get(key)
        mr = mmap.get(key)
        r["in_frozen_table"] = fr is not None
        r["in_multiseed_table"] = mr is not None
        r["frozen_mae"] = float(fr["mae"]) if fr is not None else None
        r["frozen_r2"] = float(fr["r2"]) if fr is not None else None
        dmae = dr2 = None
        if fr is not None and np.isfinite(r["recomputed"]["mae"]):
            dmae = r["recomputed"]["mae"] - float(fr["mae"])
            dr2 = r["recomputed"]["r2"] - float(fr["r2"])
        r["d_mae"] = dmae
        r["d_r2"] = dr2

        if r["test_unmatched"]:
            # rows that the mask expects but the parquet does not carry
            r["status"] = "unmatched_cells"
        elif not np.isfinite(r["recomputed"]["mae"]):
            r["status"] = "zero_coverage"
        elif dmae is None:
            r["status"] = "not_in_frozen_table"
        elif abs(dmae) > TOL or abs(dr2) > TOL:
            r["status"] = "conflict"
        elif r["coverage"] < 1.0:
            r["status"] = "verified_partial_coverage"
        else:
            r["status"] = "verified"


def metric_gaps() -> list[dict]:
    """(model, mask) pairs that have a metric JSON entry but no prediction."""
    gaps = []
    for jpath in sorted(RESULTS_DIR.glob("*.json")):
        model = jpath.stem
        entry = json.loads(jpath.read_text(encoding="utf-8"))
        have_pred = {p.stem.split("__", 1)[1]
                     for p in PRED_DIR.glob(f"{model}__*.parquet")}
        missing = sorted(set(entry) - have_pred)
        if missing:
            gaps.append({
                "model": model,
                "metrics": len(entry),
                "predictions": len(have_pred),
                "missing_masks": missing,
            })
    return gaps


def today() -> str:
    return datetime.now(timezone.utc).date().strftime("%Y%m%d")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--verify", action="store_true",
                    help="recompute test metrics and compare with frozen table")
    ap.add_argument("--out", default="experiments/analysis")
    ap.add_argument("--date", default=today())
    args = ap.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    frozen = pd.read_csv(FROZEN / "benchmark.csv")
    multiseed = pd.read_csv(FROZEN / "benchmark_multiseed.csv")
    parquets = sorted(PRED_DIR.glob("*.parquet"))

    rows: list[dict] = []
    if args.verify:
        datasets: dict[str, dict] = {}
        masks: dict[str, dict] = {}
        for p in parquets:
            rows.append(verify_one(p, datasets, masks))
        classify(rows, frozen, multiseed)

    inv = {
        "audit_date": args.date,
        "counts": {
            "prediction_files": len(parquets),
            "frozen_rows": len(frozen),
            "multiseed_rows": len(multiseed),
            "masks": len(list(MASKS_DIR.glob("*.npz"))),
            "result_jsons": len(list(RESULTS_DIR.glob("*.json"))),
        },
        "predictions": rows if args.verify else [p.name for p in parquets],
        "metric_gaps": metric_gaps(),
        "frozen_rows_without_prediction": sorted(
            f"{r['model']}__{r['mask']}" for _, r in frozen.iterrows()
            if not (PRED_DIR / f"{r['model']}__{r['mask']}.parquet").exists()
        ),
    }

    jpath = out_dir / f"artifact_inventory_{args.date}.json"
    jpath.write_text(json.dumps(inv, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"inventory -> {jpath}")

    if args.verify:
        vdf = pd.DataFrame(rows)
        vpath = out_dir / f"artifact_verification_{args.date}.csv"
        vdf.to_csv(vpath, index=False)
        print(f"verification -> {vpath}")
        print()
        print("status counts:")
        print(vdf["status"].value_counts().to_string())
        bad = vdf[vdf["status"].isin(["conflict", "unmatched_cells",
                                      "not_in_frozen_table"])]
        if len(bad):
            cols = ["model", "mask", "status", "recomputed", "frozen_mae",
                    "d_mae", "frozen_r2", "d_r2"]
            print()
            print("attention rows:")
            print(bad[cols].to_string(index=False))
    print()
    print(f"prediction files        : {len(parquets)}")
    print(f"frozen table rows       : {len(frozen)}")
    print(f"multiseed table rows    : {len(multiseed)}")
    print(f"frozen rows w/o preds   : {len(inv['frozen_rows_without_prediction'])}")
    print(f"metric-only pairs total : "
          f"{sum(len(g['missing_masks']) for g in inv['metric_gaps'])}")


if __name__ == "__main__":
    main()
