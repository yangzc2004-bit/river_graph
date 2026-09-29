"""Audit and summarize Local--Transport KGML pilot products."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from river_graph.experiments.evaluate import metrics
from river_graph.experiments.provenance import config_hash, sha256_file


def verify_run(run: Path) -> tuple[dict, pd.DataFrame]:
    meta = json.loads((run / "meta.json").read_text())
    config = meta["config"]
    expected = config_hash(config, version=int(config["config_hash_version"]))
    if expected != meta["config_hash"]:
        raise ValueError(f"config hash mismatch: {run}")
    for name, identity in meta["artifacts"].items():
        path = run / name
        if not path.is_file() or sha256_file(path) != identity["sha256"]:
            raise ValueError(f"artifact hash mismatch: {run / name}")
    frame = pd.read_parquet(run / "predictions.parquet")
    if frame.empty or not np.isfinite(frame[["y_true", "final_pred"]]).all().all():
        raise ValueError(f"invalid prediction product: {run}")
    if frame.model_name.iloc[0] != config["model_name"]:
        raise ValueError(f"model identity mismatch: {run}")
    return meta, frame


def summarize(root: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    for meta_path in sorted((root / "runs").glob("*/meta.json")):
        meta, frame = verify_run(meta_path.parent)
        test = frame[frame["split"] == "test"]
        row = {"run": meta_path.parent.name, **meta["summary"]}
        for name in ("local_pred", "context_pred", "final_pred"):
            values = metrics(test.y_true.to_numpy(), test[name].to_numpy())
            row.update({f"{name}_{key}": value for key, value in values.items()})
        rows.append(row)
    runs = pd.DataFrame(rows)
    if runs.empty:
        return runs, runs
    grouped = (runs.groupby(["arm", "analyte", "mask"], as_index=False)
               .agg({"final_pred_mae": ["mean", "std"],
                     "context_pred_mae": ["mean", "std"],
                     "local_pred_mae": ["mean", "std"],
                     "graph_delta_sd": ["mean", "std"],
                     "seed": "count"}))
    grouped.columns = [
        "_".join(x).strip("_") if isinstance(x, tuple) else x
        for x in grouped.columns
    ]
    grouped["relative_vs_context_pct"] = (
        100.0 * (1.0 - grouped["final_pred_mae_mean"]
                 / grouped["context_pred_mae_mean"])
    )
    return runs, grouped


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="experiments/phase4_transfer/kgml_local_transport_v1/pilot")
    args = parser.parse_args()
    root = Path(args.root)
    runs, grouped = summarize(root)
    runs.to_csv(root / "audited_runs.csv", index=False)
    grouped.to_csv(root / "summary_by_arm.csv", index=False)
    report = ["# Local--Transport KGML pilot audit", "", f"Runs: {len(runs)}", ""]
    if not grouped.empty:
        report.extend(["## Mean test MAE", "", grouped.to_markdown(index=False), ""])
    (root / "audit_report.md").write_text("\n".join(report) + "\n")
    print(grouped.to_string(index=False))


if __name__ == "__main__":
    main()
