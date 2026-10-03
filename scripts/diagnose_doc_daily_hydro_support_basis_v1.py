"""Summarize source-validation support choices without reading target products."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
from run_unified_doc_spatial import verify_files, write_json

from river_graph.experiments.provenance import sha256_file

ROOT = Path("experiments/phase4_transfer/doc_daily_hydro_support_basis_v1")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    args = parser.parse_args()
    frames, sources = [], {}
    for run in sorted((args.root / "runs").glob("split*_seed*")):
        config = json.loads((run / "config.json").read_text())
        verify_files(run, "complete.json", config)
        path = run / "source_validation.csv"
        frame = pd.read_csv(path)
        frame["split_seed"], frame["seed"] = config["split_seed"], config["seed"]
        if len(frame) != 72 or frame.duplicated(["model_name", "k"]).any():
            raise ValueError("Expected 18 source-validation curves at four K values")
        frames.append(frame)
        for name in ("source_validation.csv", "config.json", "complete.json", "adapters.json", "mixers.json"):
            source = run / name
            sources[str(source)] = sha256_file(source)
    if not frames:
        raise ValueError("No completed support-basis packages")
    frame = pd.concat(frames, ignore_index=True)
    keys = ["model_name", "arm", "basis", "stage", "k"]
    partitions = frame.groupby([*keys, "split_seed"], as_index=False).agg(
        mae=("mae", "mean"), n_seeds=("seed", "nunique"))
    summary = partitions.groupby(keys, as_index=False).agg(
        mae=("mae", "mean"), n_partitions=("split_seed", "nunique"))
    contrasts = summary.pivot(index=["arm", "stage", "k"], columns="basis", values="mae").reset_index()
    contrasts["refreshed_minus_legacy"] = contrasts.refreshed-contrasts.legacy
    output = args.root / "diagnostics"
    output.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output / "source_validation_by_run.csv", index=False)
    partitions.to_csv(output / "source_validation_by_partition.csv", index=False)
    summary.to_csv(output / "source_validation_summary.csv", index=False)
    contrasts.to_csv(output / "source_validation_contrasts.csv", index=False)
    rows = ["# Source-validation support-basis comparison", "",
            "Selected validation scores, seed-averaged within each partition and then partition-equal.",
            "No target prediction file or target DOC outcome is read by this diagnostic.", "",
            "| Expert | Pipeline | K | Legacy MAE | Refreshed MAE | Difference |",
            "| --- | --- | ---: | ---: | ---: | ---: |"]
    for row in contrasts.itertuples():
        rows.append(f"| {row.arm} | {row.stage} | {row.k} | {row.legacy:.6f} | "
                    f"{row.refreshed:.6f} | {row.refreshed_minus_legacy:+.6f} |")
    rows += ["", "K0 is identical by construction. Positive-K choices use the unchanged source-validation",
             "alpha/ridge grids and, for integrated products, positive-K ecological mixing.",
             "Readout and native predictions are fixed; refreshed hidden coordinates may differ from",
             "the old support-episodic coordinates. All three experts and both pipelines are retained.", ""]
    (output / "source_validation.md").write_text("\n".join(rows))
    write_json(output / "source_validation_sources.json", {
        "diagnostic_script_sha256": sha256_file(__file__), "sources": sources,
        "target_outcomes_read": False, "n_runs": len(frames), "source_rows": len(frame),
        "outputs": {path.name: sha256_file(path) for path in output.glob("*")
                    if path.name != "source_validation_sources.json"}})
    print(contrasts.to_string(index=False))
    print(f"Saved source-only comparison for {len(frames)} packages to {output}")


if __name__ == "__main__":
    main()
