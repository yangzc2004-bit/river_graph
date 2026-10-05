"""Describe current K0 failures by source-defined ecology and hydro availability."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from run_unified_doc_spatial import verify_files, write_json

from river_graph.experiments.provenance import sha256_file
from river_graph.models.doc_source_retrieval import SourceResidualBank

ROOT = Path("experiments/phase4_transfer/doc_unmonitored_tasks_v1")
SOURCE = Path("experiments/phase4_transfer/doc_source_retrieval_v1")
MODELS = ("current_model", "matched_daily_trees", "static_memory")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    args = parser.parse_args()
    strata, station_rows, sources = [], [], {}
    for path in sorted((SOURCE / "runs").glob("*/complete.json")):
        run = path.parent
        config = json.loads((run / "config.json").read_text())
        verify_files(run, "complete.json", config)
        sources[str(path)] = sha256_file(path)
        dataset = torch.load(config["dataset_path"], weights_only=False, map_location="cpu")
        months = dataset["y"].shape[1]
        bank = SourceResidualBank.from_dict(json.loads((run / "source_bank.json").read_text()))
        ecological, valid = bank._ecology(np.asarray(dataset["regime"])[:, 4:13])
        distances = np.mean((ecological[:, None]-bank.ecology_[None])**2, axis=-1)
        distances[np.asarray(dataset["site_no"], str)[:, None] == bank.station_names_[None]] = np.inf
        nearest = distances.min(1)
        source_ids = np.flatnonzero(np.isin(np.asarray(dataset["site_no"], str), bank.station_names_))
        bounds = np.quantile(nearest[source_ids], [1/3, 2/3])
        novelty = np.asarray(("familiar", "intermediate", "novel"))[np.searchsorted(bounds, nearest, side="right")]
        novelty[valid.sum(1) < 5] = "unknown"
        frame = pd.read_parquet(run / "predictions.parquet")
        frame = frame[frame.model_name.isin(MODELS)].copy()
        frame["ecological_novelty"] = novelty[frame.cell.to_numpy()//months]
        frame["hydro_information"] = np.asarray(("neither", "one channel", "both channels"))[
            np.round(2*frame.hydro_completeness.to_numpy()).astype(int)]
        frame["doc_value"] = np.where(frame.y_true >= config["q90_threshold_train"], "high DOC", "ordinary DOC")
        frame["error"] = frame.y_pred-frame.y_true
        frame["absolute_error"] = np.abs(frame.error)
        frame["high_detected"] = frame.y_pred >= config["q90_threshold_train"]
        for feature in ("ecological_novelty", "hydro_information", "doc_value"):
            for (model, group), selected in frame.groupby(["model_name", feature]):
                strata.append({"split_seed": config["split_seed"], "seed": config["seed"],
                    "model_name": model, "dimension": feature, "group": group,
                    "n_cells": len(selected), "n_stations": selected.station.nunique(),
                    "mae": selected.absolute_error.mean(), "bias": selected.error.mean(),
                    "station_mae": selected.groupby("station").absolute_error.mean().mean(),
                    "unstable": len(selected) < 20,
                    "q90_recall": selected.high_detected.mean() if group == "high DOC" else np.nan})
        station = frame.groupby(["model_name", "station"], as_index=False).agg(
            mae=("absolute_error", "mean"), bias=("error", "mean"), n_cells=("cell", "size"),
            novelty=("ecological_novelty", "first"), hydro_completeness=("hydro_completeness", "mean"))
        station["split_seed"], station["seed"] = config["split_seed"], config["seed"]
        station_rows.append(station)
    if len(sources) != 9:
        raise ValueError("the task benchmark requires all nine development packages")
    out = args.root / "baseline_analysis"
    out.mkdir(exist_ok=True)
    table = pd.DataFrame(strata)
    table.to_csv(out / "strata_by_run.csv", index=False)
    partition = table.groupby(["split_seed", "model_name", "dimension", "group"], as_index=False)[
        ["mae", "bias", "station_mae", "q90_recall"]].mean()
    partition.to_csv(out / "strata_by_partition.csv", index=False)
    partition.groupby(["model_name", "dimension", "group"], as_index=False)[
        ["mae", "bias", "station_mae", "q90_recall"]].mean().to_csv(out / "strata_summary.csv", index=False)
    pd.concat(station_rows, ignore_index=True).to_csv(out / "station_errors.csv", index=False)
    write_json(out / "sources.json", {"scope": "all-observation source-validation K0 baseline diagnostics",
        "runs": sources, "partition_weighting": "equal present partition means after seed averaging",
        "novelty_threshold": "source-donor leave-self-out nearest-distance thirds; no validation outcomes",
        "hydro_groups": "no/one/both monthly input channels visible", "tail_threshold": "source-training Q90"})
    print(partition.groupby(["model_name", "dimension", "group"])["mae"].mean().to_string())


if __name__ == "__main__":
    main()
