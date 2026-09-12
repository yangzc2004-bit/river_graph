"""A0: recover historical prediction copies from an explicit Git commit.

The current worktree keeps only 85 of the 154 predictions that were frozen in
commit ``256d08e`` ("Phase 0 complete"); a later commit deleted 69 of them and
rewrote one. This script re-extracts the missing/overwritten copies from that
explicit commit into a separate directory, so nothing in
``experiments/predictions/`` is touched and the recovered files stay
distinguishable from the canonical batch.

Recovering a historical copy does NOT make it a new five-seed prediction: the
recovered per-cell predictions belong to the frozen single-run batch.

Usage:
    python scripts/recover_historical_predictions.py --list
    python scripts/recover_historical_predictions.py --commit 256d08e
    python scripts/recover_historical_predictions.py --verify
"""

from __future__ import annotations

import argparse
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from river_graph.experiments.evaluate import load_dataset, load_mask, metrics

DEFAULT_COMMIT = "256d08e"  # "Phase 0 complete: benchmark frozen (11 models x 14 masks)"
GIT_DIR = "experiments/predictions"
OUT_DIR = Path("experiments/predictions_historical")
ANALYSIS = Path("experiments/analysis")

MODEL_DATASETS = {
    "B0": "data/processed/mississippi_graph_v02.pt",
    "B1": "data/processed/mississippi_graph_v02.pt",
    "B2": "data/processed/mississippi_graph_v02.pt",
    "B3": "data/processed/mississippi_graph_v02.pt",
    "G0": "data/processed/mississippi_graph_v02.pt",
    "H1": "data/processed/mississippi_graph_v02.pt",
    "H2": "data/processed/mississippi_graph_v03.pt",
    "H2E": "data/processed/mississippi_graph_v04.pt",
    "H2X": "data/processed/mississippi_graph_v04.pt",
}


def git(*args: str) -> str:
    res = subprocess.run(["git", *args], capture_output=True, text=True,
                         encoding="utf-8", errors="replace", check=False)
    if res.returncode != 0:
        raise RuntimeError(f"git {' '.join(args)} failed: {res.stderr.strip()}")
    return res.stdout


def today() -> str:
    return datetime.now(timezone.utc).date().strftime("%Y%m%d")


def list_commit(commit: str) -> dict[str, str]:
    """{filename: blob_sha} for predictions tracked at ``commit``."""
    out = git("ls-tree", "-r", commit, "--", GIT_DIR)
    files: dict[str, str] = {}
    for line in out.splitlines():
        meta, _, path = line.partition("\t")
        fields = meta.split()
        if len(fields) >= 3 and path.endswith(".parquet"):
            files[Path(path).name] = fields[2]
    return files


def dataset_for(model: str) -> str:
    for prefix, path in MODEL_DATASETS.items():
        if model.startswith(prefix):
            return path
    return "data/processed/mississippi_graph_v02.pt"


def recover(commit: str, names: list[str], out_dir: Path) -> list[dict]:
    out_dir.mkdir(parents=True, exist_ok=True)
    records = []
    for name in names:
        blob = f"{commit}:{GIT_DIR}/{name}"
        data = subprocess.run(["git", "show", blob], capture_output=True,
                              check=False)
        if data.returncode != 0:
            raise RuntimeError(f"cannot read {blob}")
        dest = out_dir / name
        dest.write_bytes(data.stdout)
        records.append({
            "file": name,
            "source_commit": commit,
            "blob": git("rev-parse", f"{commit}:{GIT_DIR}/{name}").strip(),
            "bytes": len(data.stdout),
        })
    return records


def verify(out_dir: Path, frozen: pd.DataFrame) -> pd.DataFrame:
    """Recompute test metrics for every recovered copy and re-check identity.

    Station / month / label / mask correspondence is verified by rebuilding the
    test cells from the mask and matching them to the parquet rows by
    (station, month), exactly like scripts/audit_artifacts.py --verify.
    """
    datasets: dict[str, dict] = {}
    masks: dict[str, dict] = {}
    fmap = {(r["model"], r["mask"]): r for _, r in frozen.iterrows()}
    rows = []
    for p in sorted(out_dir.glob("*.parquet")):
        df = pd.read_parquet(p)
        model = str(df["model"].iloc[0])
        mask_name = str(df["mask"].iloc[0])
        dpath = dataset_for(model)
        if dpath not in datasets:
            datasets[dpath] = load_dataset(dpath)
        if mask_name not in masks:
            masks[mask_name] = load_mask(mask_name, Path("experiments/masks"))
        ds, split = datasets[dpath], masks[mask_name]
        y = ds["y"].numpy()
        _n, t = y.shape
        sites, months = list(ds["site_no"]), list(ds["months"])
        test_idx = np.asarray(split["test"], dtype=np.int64)
        want = [(sites[i // t], months[i % t]) for i in test_idx]
        lut = dict(zip(zip(df["station"].tolist(), df["month"].tolist()),
                       zip(df["y_true"].tolist(), df["y_pred"].tolist())))
        yt = np.array([lut.get(k, (np.nan, np.nan))[0] for k in want], float)
        yp = np.array([lut.get(k, (np.nan, np.nan))[1] for k in want], float)
        present = np.array([k in lut for k in want], bool)
        m = metrics(yt, yp)
        fr = fmap.get((model, mask_name))
        d_mae = d_r2 = None
        if fr is not None and np.isfinite(m["mae"]):
            d_mae = m["mae"] - float(fr["mae"])
            d_r2 = m["r2"] - float(fr["r2"])
        rows.append({
            "file": p.name, "model": model, "mask": mask_name,
            "rows": len(df), "test_mask_cells": len(test_idx),
            "test_unmatched": int((~present).sum()),
            "coverage": float(np.isfinite(yp).mean()) if len(yp) else 0.0,
            "recomputed_mae": m["mae"], "recomputed_r2": m["r2"],
            "recomputed_n": m["n"],
            "frozen_mae": float(fr["mae"]) if fr is not None else np.nan,
            "frozen_r2": float(fr["r2"]) if fr is not None else np.nan,
            "d_mae": d_mae, "d_r2": d_r2,
            "matches_frozen": bool(
                d_mae is not None and abs(d_mae) < 1e-4 and abs(d_r2) < 1e-4),
        })
    return pd.DataFrame(rows)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--commit", default=DEFAULT_COMMIT)
    ap.add_argument("--list", action="store_true", help="show what would be fetched")
    ap.add_argument("--verify", action="store_true",
                    help="recompute recovered copies and compare with frozen")
    ap.add_argument("--overwrite", action="store_true",
                    help="re-extract even if the file already exists")
    ap.add_argument("--out-dir", default=str(OUT_DIR))
    ap.add_argument("--date", default=today())
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    commit_files = list_commit(args.commit)
    current = {p.name for p in Path("experiments/predictions").glob("*.parquet")}

    deleted = sorted(set(commit_files) - current)
    overwritten = sorted(
        name for name in current & set(commit_files)
        if git("rev-parse", f"{args.commit}:{GIT_DIR}/{name}").strip()
        != git("rev-parse", f"HEAD:{GIT_DIR}/{name}").strip()
    )

    print(f"commit {args.commit}: {len(commit_files)} predictions")
    print(f"worktree           : {len(current)} predictions")
    print(f"deleted since then : {len(deleted)}")
    print(f"overwritten since  : {len(overwritten)} {overwritten}")

    targets = sorted(set(deleted) | set(overwritten))
    if args.list:
        for name in targets:
            print("  ", name)
        return

    if not args.overwrite:
        targets = [n for n in targets if not (out_dir / n).exists()]
    if not targets:
        print("nothing to recover (all copies already present)")
    else:
        records = recover(args.commit, targets, out_dir)
        print(f"recovered {len(records)} files -> {out_dir}")

    if args.verify:
        ANALYSIS.mkdir(parents=True, exist_ok=True)
        frozen = pd.read_csv("experiments/frozen_results/benchmark.csv")
        vdf = verify(out_dir, frozen)
        vpath = ANALYSIS / f"historical_verification_{args.date}.csv"
        vdf.to_csv(vpath, index=False)
        print(f"verification -> {vpath}")
        print(f"recovered files verified : {len(vdf)}")
        print(f"matching frozen table    : {int(vdf['matches_frozen'].sum())}")
        print(f"with unmatched test cells: "
              f"{int((vdf['test_unmatched'] > 0).sum())}")
        print(f"partial coverage files   : "
              f"{int((vdf['coverage'] < 1.0).sum())}")
        mismatch = vdf[~vdf["matches_frozen"]]
        if len(mismatch):
            print()
            print("recovered copies NOT matching frozen table:")
            print(mismatch[["file", "recomputed_mae", "frozen_mae",
                            "recomputed_r2", "frozen_r2"]].to_string(index=False))

        manifest = {
            "recovery_date": args.date,
            "source_commit": args.commit,
            "source_commit_subject": git("log", "-1", "--format=%s", args.commit).strip(),
            "note": ("Historical single-run predictions recovered from git. "
                     "These are NOT the five-seed multiseed predictions; "
                     "experiments/predictions/ was never modified."),
            "recovered": sorted(p.name for p in out_dir.glob("*.parquet")),
            "counts": {
                "recovered": len(list(out_dir.glob("*.parquet"))),
                "verified_vs_frozen": int(vdf["matches_frozen"].sum()),
            },
        }
        mpath = ANALYSIS / f"historical_recovery_manifest_{args.date}.json"
        mpath.write_text(json.dumps(manifest, indent=2, ensure_ascii=False),
                         encoding="utf-8")
        print(f"manifest -> {mpath}")


if __name__ == "__main__":
    main()
