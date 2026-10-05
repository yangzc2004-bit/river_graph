"""Replay the station-hidden residual model and report source-validation gains."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from analyze_doc_chemistry_confirmation_v1 import metric_summary
from analyze_doc_source_retrieval_v1 import load_panel, paired
from analyze_unified_doc_spatial import joint_station_bootstrap
from run_unified_doc_spatial import digest, verify_files, write_json

from river_graph.experiments.provenance import run_identity_sha256, sha256_file
from river_graph.models.ecological_residual_transfer import EcologicalResidualTransfer
from river_graph.models.encoder_native_residual import EncoderNativeResidual

ROOT = Path("experiments/phase4_transfer/doc_unmonitored_residual_v1")
ARMS = ("unmonitored_residual", "unmonitored_integrated")


def verify(run):
    config = json.loads((run / "config.json").read_text())
    verify_files(run, "complete.json", config)
    meta = json.loads((run / "predictions.meta.json").read_text())
    identity = run_identity_sha256(digest(config), config["started_at"], config["runtime_snapshot_hash"])
    if (meta["config_hash"] != digest(config) or meta["run_identity_sha256"] != identity
            or meta["prediction_sha256"] != sha256_file(run / "predictions.parquet")):
        raise ValueError("changed residual prediction identity")
    panel = pd.read_parquet(run / "predictions.parquet")
    with np.load(config["mask_path"], allow_pickle=False) as saved:
        val = saved["val"].copy()
        n_months = int(torch.load(config["dataset_path"], weights_only=False)["y"].shape[1])
    for _, group in panel.groupby("model_name"):
        np.testing.assert_array_equal(group.cell, val)
        if not group.visibility_role.eq("val").all() or not np.isfinite(group.y_pred).all():
            raise ValueError("invalid validation prediction grid")
    records = json.loads((run / "oof_records.json").read_text())
    for fold in records:
        if set(fold["hidden_stations"]) & set(fold["fitted_stations"]):
            raise ValueError("held station entered its OOF tree fit")
    model = EncoderNativeResidual.from_payload(torch.load(run / "native.pt", weights_only=False))
    with np.load(run / "validation_inputs.npz", allow_pickle=False) as saved:
        inputs = {key: saved[key].copy() for key in ("raw", "age", "support", "env", "extra")}
        context = saved["context"].copy()
        cells = saved["cell"].copy()
    np.testing.assert_array_equal(cells, val)
    ids = np.unique(val // n_months)
    local_rows = np.searchsorted(ids, val // n_months)
    dates = val % n_months
    native = np.maximum(0, context + model.selected_scale_ * model.predict_delta(inputs))
    expected = panel[panel.model_name.eq("unmonitored_residual")].y_pred.to_numpy()
    np.testing.assert_array_equal(native[local_rows, dates], expected)
    memory = EcologicalResidualTransfer.from_dict(json.loads((run / "memory.json").read_text()))
    # Only the cached validation slice is needed. Other rows are zero placeholders,
    # not exported full-grid predictions or additional label views.
    context_grid = np.zeros((memory.n_nodes_, memory.n_months_))
    temporal_grid = context_grid.copy()
    context_grid[ids], temporal_grid[ids] = context, native
    integrated = memory.predict(context_grid, temporal_grid).ravel()[val]
    np.testing.assert_array_equal(integrated, panel[panel.model_name.eq("unmonitored_integrated")].y_pred)
    return {"run": run.name, "identity": True, "station_oof": True,
            "native_replay": "bitwise", "integrated_replay": "bitwise", "target_evaluation": False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--bootstrap-draws", type=int, default=5000)
    args = parser.parse_args()
    torch.set_num_threads(2)
    completions = sorted((args.root / "runs").glob("*/complete.json"))
    if len(completions) != 9:
        raise ValueError("complete all nine source-validation packages before analysis")
    write_json(args.root / "verification" / "replay.json", [verify(p.parent) for p in completions])
    panel, thresholds = load_panel(args.root)
    runs, parts, summary = metric_summary(panel, thresholds)
    output = args.root / "analysis"
    output.mkdir(exist_ok=True)
    for name, frame in (("run_metrics", runs), ("partition_metrics", parts), ("summary", summary)):
        frame.to_csv(output / f"{name}.csv", index=False)
    effects, station_tables = [], []
    comparisons = [(arm, ref) for arm in ARMS for ref in
                   ("current_model", "station_hidden_trees", "matched_daily_trees")]
    comparisons.append(("unmonitored_integrated", "unmonitored_residual"))
    for arm, ref in comparisons:
        comparison = paired(panel, arm, ref)
        directions = comparison.groupby(["split_seed", "seed"])[["candidate_error", "reference_error"]].mean()
        partition_directions = directions.groupby("split_seed").mean()
        for region in ("overall", "q90"):
            selected = comparison
            if region == "q90":
                limits = np.array([thresholds[(p, s)] for p, s in
                                   zip(comparison.split_seed, comparison.seed, strict=True)])
                selected = comparison[comparison.y_true_candidate.to_numpy() >= limits]
            effects.append({"candidate": arm, "reference": ref, "region": region,
                **joint_station_bootstrap(selected, draws=args.bootstrap_draws),
                "improved_packages": int((directions.candidate_error < directions.reference_error).sum()),
                "improved_partitions": int((partition_directions.candidate_error < partition_directions.reference_error).sum())})
        station = comparison.groupby(["split_seed", "station"], as_index=False).agg(
            candidate_mae=("candidate_error", "mean"), reference_mae=("reference_error", "mean"),
            n_query=("cell", "nunique"))
        station["candidate"], station["reference"] = arm, ref
        station["delta_mae"] = station.candidate_mae - station.reference_mae
        station_tables.append(station)
    effects = pd.DataFrame(effects)
    effects.to_csv(output / "paired_effects.csv", index=False)
    pd.concat(station_tables, ignore_index=True).to_csv(output / "station_effects.csv", index=False)
    write_json(output / "sources.json", {"evaluation_role": "source-validation development",
        "bootstrap_draws": args.bootstrap_draws, "packages": len(completions),
        "runs": {str(path): sha256_file(path) for path in completions}})
    print(summary[["model_name", "mae", "q90_mae"]].to_string(index=False))
    print(effects[["candidate", "reference", "region", "relative_gain_pct", "gain_ci_low_pct",
                   "gain_ci_high_pct", "improved_partitions"]].to_string(index=False))


if __name__ == "__main__":
    main()
