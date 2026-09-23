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
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from river_graph.experiments.evaluate import load_dataset, load_mask, metrics
from river_graph.experiments.provenance import config_hash

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

# Files whose problems are documented and must not fail the --verify gate:
# this parquet was overwritten by a different model run in commit 2b67bf0; the
# frozen table stays authoritative (experiments/analysis/evidence_inventory_20260912.md).
KNOWN_ATTENTION = {"G0_gcn_none__e1_r20_seed42.parquet"}


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


def sidecar_identity(parquet: Path) -> dict:
    """Check the provenance sidecar against its own config and current files.

    ``config_hash`` is recomputed from the stored config fields; dataset and
    mask content hashes are compared with the files on disk when those files
    exist (absent gitignored inputs are reported as unchecked, not as errors).
    """
    mpath = parquet.with_suffix(".meta.json")
    if not mpath.exists():
        return {
            "identity_status": "no_sidecar",
            "identity_problems": ["no provenance sidecar"],
            "config_hash_ok": None,
            "input_hash_status": {},
            "run_identity_sha256": "",
            "runtime_code_snapshot_sha256": "",
        }
    meta = json.loads(mpath.read_text(encoding="utf-8"))
    cfg = meta.get("config") or {}
    problems: list[str] = []
    stored_hash = meta.get("config_hash")
    recomputed = config_hash(cfg)
    config_ok = stored_hash == recomputed
    if not config_ok:
        problems.append(
            f"config_hash mismatch: stored={str(stored_hash)[:12]} "
            f"recomputed={recomputed[:12]}"
        )
    hash_status: dict[str, str] = {}
    for label in ("dataset", "mask"):
        ident = meta.get(label) or {}
        want = ident.get("sha256")
        path = Path(str(ident.get("path") or ""))
        if not want:
            hash_status[label] = "unrecorded"
            problems.append(f"{label} sha256 unrecorded in sidecar")
        elif not path.exists():
            hash_status[label] = "file_absent_not_rechecked"
        else:
            now = sha256_file(path)
            hash_status[label] = "match" if now == want else "changed"
            if now != want:
                problems.append(
                    f"{label} content changed since the run: "
                    f"stored={want[:12]} current={now[:12]}"
                )
    return {
        "identity_status": "ok" if not problems else "stale",
        "identity_problems": problems,
        "config_hash_ok": config_ok,
        "input_hash_status": hash_status,
        "run_identity_sha256": meta.get("run_identity_sha256") or "",
        "runtime_code_snapshot_sha256": meta.get("runtime_code_snapshot_sha256") or "",
    }


def verify_one(
    parquet: Path,
    datasets: dict,
    masks: dict[str, dict],
) -> dict:
    """Recompute test metrics straight from the stored predictions.

    Test cells are taken from the MASK (ground truth boundaries), then matched
    to parquet rows by (station, month). A file whose rows do not line up with
    the mask is reported, never silently averaged. When the run's dataset is
    not on this machine the mask cannot be expanded to (station, month) keys,
    so the check falls back to the parquet's own test rows and is labelled
    ``parquet_rows_only`` — weaker, never silently treated as full verification.
    """
    df = pd.read_parquet(parquet)
    model = str(df["model"].iloc[0])
    mask_name = str(df["mask"].iloc[0])
    dpath = dataset_for_model(model)
    if dpath not in datasets:
        datasets[dpath] = load_dataset(dpath) if Path(dpath).exists() else None
    ds = datasets[dpath]
    stored_test = df[df["split"] == "test"]
    identity = sidecar_identity(parquet)

    if ds is None:
        yt = stored_test["y_true"].to_numpy(dtype=float)
        yp = stored_test["y_pred"].to_numpy(dtype=float)
        m = metrics(yt, yp)
        return {
            "file": parquet.name,
            "model": model,
            "mask": mask_name,
            "dataset": dpath,
            "verify_mode": "parquet_rows_only",
            "dataset_present": False,
            "rows": len(df),
            "split_counts": {str(k): int(v) for k, v in
                             df["split"].value_counts().items()},
            "test_stored_rows": len(stored_test),
            "test_mask_cells": None,
            "test_unmatched": None,
            "test_cells_nonfinite_pred": int(np.isnan(yp).sum()),
            "coverage": float(np.isfinite(yp).mean()) if len(yp) else 0.0,
            "recomputed": {"mae": m["mae"], "r2": m["r2"], "rmse": m["rmse"],
                           "n": m["n"]},
            "sha256": sha256_file(parquet),
            **identity,
        }

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
    return {
        "file": parquet.name,
        "model": model,
        "mask": mask_name,
        "dataset": dpath,
        "verify_mode": "mask_boundaries",
        "dataset_present": True,
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
        **identity,
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
        elif r.get("verify_mode") == "parquet_rows_only":
            # metrics agree, but only via the parquet's own rows: the mask
            # boundary check was impossible on this machine
            r["status"] = "verified_parquet_only"
        elif r["coverage"] < 1.0:
            r["status"] = "verified_partial_coverage"
        else:
            r["status"] = "verified"


def flag_anomalies(rows: list[dict]) -> None:
    """Flag retrained duplicates and names whose arm/seed disagree with the sidecar.

    Same ``config_hash`` with different ``run_identity_sha256`` values means the
    configuration was retrained (a fresh copy, not a reused artifact). A
    ``P2X_<arm>[_river][_s<seed>]__<mask>`` file name whose arm or seed
    disagrees with its sidecar config is marked ``suspicious_name``.
    """
    p2x = re.compile(
        r"^P2X_(?P<arm>H2X_nomsg|eco_RF|eco_MLP|H2X|H2E|H2)"
        r"(?:_river)?(?:_s(?P<seed>\d+))?__"
    )
    expect_arch = {
        "H2": "transport", "H2E": "transport",
        "H2X": "transport_enc", "H2X_nomsg": "transport_enc",
    }
    expect_edge = {"H2X": "river", "H2X_nomsg": "empty"}
    for r in rows:
        r["run_flags"] = []
        mpath = PRED_DIR / Path(r["file"]).with_suffix(".meta.json").name
        meta = (
            json.loads(mpath.read_text(encoding="utf-8"))
            if mpath.exists() else {}
        )
        r["config_hash"] = meta.get("config_hash") or ""
        m = p2x.match(r["file"])
        if not m:
            continue
        cfg = meta.get("config") or {}
        arm = m.group("arm")
        want_seed = int(m.group("seed")) if m.group("seed") else 0
        got_seed = cfg.get("seed")
        if got_seed is not None and int(got_seed) != want_seed:
            r["run_flags"].append(
                f"suspicious_name:seed name={want_seed} sidecar={got_seed}"
            )
        arch = cfg.get("architecture")
        if arm in expect_arch and arch and arch != expect_arch[arm]:
            r["run_flags"].append(
                f"suspicious_name:arm name={arm} architecture={arch}"
            )
        edge = cfg.get("edge_set")
        if arm in expect_edge and edge and edge != expect_edge[arm]:
            r["run_flags"].append(
                f"suspicious_name:edge_set name={arm} edge_set={edge}"
            )
    by_hash: dict[str, list[dict]] = {}
    for r in rows:
        if r.get("config_hash"):
            by_hash.setdefault(r["config_hash"], []).append(r)
    for group in by_hash.values():
        ids = {g.get("run_identity_sha256") for g in group}
        if len(group) > 1 and len(ids) > 1:
            for g in group:
                g["run_flags"].append("retrained_duplicate")


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
    global PRED_DIR, RESULTS_DIR, MASKS_DIR, FROZEN
    ap = argparse.ArgumentParser()
    ap.add_argument("--verify", action="store_true",
                    help="recompute test metrics and compare with frozen table")
    ap.add_argument("--out", default="experiments/analysis")
    ap.add_argument("--date", default=today())
    ap.add_argument("--pred-dir", default=str(PRED_DIR),
                    help="predictions directory (e.g. experiments/phase2_ablation_stcore_v1/predictions)")
    ap.add_argument("--results-dir", default=str(RESULTS_DIR))
    ap.add_argument("--masks-dir", default=str(MASKS_DIR))
    ap.add_argument("--frozen-dir", default=str(FROZEN),
                    help="directory with benchmark.csv / benchmark_multiseed.csv; "
                         "missing tables are tolerated (pre-freeze audits)")
    args = ap.parse_args()

    PRED_DIR = Path(args.pred_dir)
    RESULTS_DIR = Path(args.results_dir)
    MASKS_DIR = Path(args.masks_dir)
    FROZEN = Path(args.frozen_dir)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    bench_path = FROZEN / "benchmark.csv"
    multi_path = FROZEN / "benchmark_multiseed.csv"
    frozen = (pd.read_csv(bench_path) if bench_path.exists()
              else pd.DataFrame(columns=["model", "mask", "mae", "r2"]))
    multiseed = (pd.read_csv(multi_path) if multi_path.exists()
                 else pd.DataFrame(columns=["model", "mask", "mae", "r2"]))
    parquets = sorted(PRED_DIR.glob("*.parquet"))

    rows: list[dict] = []
    if args.verify:
        datasets: dict[str, dict] = {}
        masks: dict[str, dict] = {}
        for p in parquets:
            rows.append(verify_one(p, datasets, masks))
        classify(rows, frozen, multiseed)
        flag_anomalies(rows)

    paper_dir = Path("docs/paper")
    paper_hashes = {
        str(p.as_posix()): sha256_file(p)
        for p in sorted(paper_dir.glob("*")) if p.is_file()
    } if paper_dir.is_dir() else {}

    inv = {
        "audit_date": args.date,
        "counts": {
            "prediction_files": len(parquets),
            "frozen_rows": len(frozen),
            "multiseed_rows": len(multiseed),
            "masks": len(list(MASKS_DIR.glob("*.npz"))),
            "result_jsons": len(list(RESULTS_DIR.glob("*.json"))),
        },
        "paper_artifacts": paper_hashes,
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

    unexpected: list[str] = []
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
        if "identity_status" in vdf.columns:
            print()
            print("identity status:")
            print(vdf["identity_status"].value_counts().to_string())
            stale = vdf[vdf["identity_status"] == "stale"]
            if len(stale):
                print()
                print("identity problems (reported, never changes metric status):")
                for _, r in stale.iterrows():
                    print(f"  {r['file']}: {r['identity_problems']}")
        flagged = [r for r in rows if r.get("run_flags")]
        if flagged:
            print()
            print("run flags (retrained duplicates / suspicious names):")
            for r in flagged:
                print(f"  {r['file']}: {r['run_flags']}")
        fatal = bad[bad["status"].isin(["conflict", "unmatched_cells"])]
        unexpected = sorted(set(fatal["file"]) - KNOWN_ATTENTION)
    print()
    print(f"prediction files        : {len(parquets)}")
    print(f"frozen table rows       : {len(frozen)}")
    print(f"multiseed table rows    : {len(multiseed)}")
    print(f"frozen rows w/o preds   : {len(inv['frozen_rows_without_prediction'])}")
    print(f"metric-only pairs total : "
          f"{sum(len(g['missing_masks']) for g in inv['metric_gaps'])}")
    if unexpected:
        print()
        print("UNEXPECTED attention rows (--verify gate failure):")
        for name in unexpected:
            print(f"  - {name}")
        sys.exit(1)


if __name__ == "__main__":
    main()
