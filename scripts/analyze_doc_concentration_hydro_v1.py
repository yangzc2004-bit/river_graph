"""Replay and compare concentration/hydro correction on source validation only."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from analyze_doc_chemistry_confirmation_v1 import metric_summary
from analyze_doc_source_retrieval_v1 import load_panel, paired
from analyze_unified_doc_spatial import joint_station_bootstrap
from run_doc_concentration_hydro_v1 import MODES, ROOT
from run_unified_doc_spatial import digest, verify_files, write_json

from river_graph.experiments.provenance import run_identity_sha256, sha256_file
from river_graph.models.concentration_hydro_calibration import (
    ConcentrationHydroCalibration,
)


def verify(run):
    config = json.loads((run/"config.json").read_text())
    verify_files(run, "complete.json", config)
    parent = Path(config["parent_run"])
    if sha256_file(parent/"complete.json") != config["parent_completion_hash"]:
        raise ValueError("saved parent residual changed")
    meta = json.loads((run/"predictions.meta.json").read_text())
    identity = run_identity_sha256(digest(config), config["started_at"], config["runtime_snapshot_hash"])
    if (meta["run_identity_sha256"] != identity or meta["config_hash"] != digest(config)
            or meta["prediction_sha256"] != sha256_file(run/"predictions.parquet")):
        raise ValueError("changed correction prediction identity")
    panel = pd.read_parquet(run/"predictions.parquet")
    with np.load(run/"calibration_inputs.npz", allow_pickle=False) as saved:
        inputs = {key: saved[key].copy() for key in saved.files}
    with np.load(config["mask_path"], allow_pickle=False) as saved:
        np.testing.assert_array_equal(inputs["source_cell"], saved["train"])
        np.testing.assert_array_equal(inputs["query_cell"], saved["val"])
    for _, group in panel.groupby("model_name"):
        np.testing.assert_array_equal(group.cell, inputs["query_cell"])
    if not panel.visibility_role.eq("val").all() or not np.isfinite(panel.y_pred).all():
        raise ValueError("invalid source-validation output")
    prior = pd.read_parquet(parent/"predictions.parquet")
    original = panel[panel.model_name.isin(prior.model_name.unique())]
    pd.testing.assert_frame_equal(original[prior.columns].reset_index(drop=True),
                                  prior.reset_index(drop=True), check_exact=True)
    differences = {}
    for mode in MODES:
        model = ConcentrationHydroCalibration.from_dict(json.loads((run/f"{mode}.json").read_text()))
        if model.n_source_rows_ != len(inputs["source_cell"]):
            raise ValueError("calibration row count differs from source training role")
        calibrated = model.predict(inputs["query_prediction"], inputs["query_hydro"])
        for arm in ("trees", "native", "integrated"):
            residual = np.zeros(len(calibrated)) if arm == "trees" else inputs[f"{arm}_residual"]
            prediction = np.maximum(0., calibrated+residual)
            observed = panel[panel.model_name.eq(f"{arm}_{mode}")].y_pred.to_numpy(float)
            np.testing.assert_array_equal(prediction, observed)
            differences[f"{arm}_{mode}"] = float(np.max(np.abs(prediction-observed)))
    return {"run": run.name, "identity": "passed", "unchanged_parent_predictions": True,
            "correction_replay_max_abs": differences, "target_evaluation": False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--bootstrap-draws", type=int, default=5000)
    args = parser.parse_args()
    complete = sorted((args.root/"runs").glob("*/complete.json"))
    if len(complete) != 9:
        raise ValueError("complete all nine source-development packages before interpretation")
    write_json(args.root/"verification"/"replay.json", [verify(path.parent) for path in complete])
    panel, thresholds = load_panel(args.root)
    if set(map(tuple, panel[["split_seed", "seed"]].drop_duplicates().to_numpy())) != {
            (partition, seed) for partition in (142, 143, 144) for seed in (42, 43, 44)}:
        raise ValueError("source-development partitions/seeds differ")
    run_metrics, partition_metrics, summary = metric_summary(panel, thresholds)
    output = args.root/"analysis"
    output.mkdir(exist_ok=True)
    for name, frame in (("run_metrics", run_metrics), ("partition_metrics", partition_metrics), ("summary", summary)):
        frame.to_csv(output/f"{name}.csv", index=False)
    comparisons = []
    for mode in MODES:
        comparisons.extend([(f"trees_{mode}", "station_hidden_trees"),
            (f"native_{mode}", "unmonitored_residual"), (f"integrated_{mode}", "unmonitored_integrated"),
            (f"integrated_{mode}", f"trees_{mode}"), (f"integrated_{mode}", "current_model")])
    rows = []
    for candidate, reference in comparisons:
        pair = paired(panel, candidate, reference)
        direction = pair.groupby(["split_seed", "seed"])[["candidate_error", "reference_error"]].mean()
        partitions = direction.groupby("split_seed").mean()
        for region in ("overall", "q90"):
            selected = pair
            if region == "q90":
                cutoffs = np.array([thresholds[(p, s)] for p, s in zip(pair.split_seed, pair.seed, strict=True)])
                selected = pair[pair.y_true_candidate.to_numpy() >= cutoffs]
            rows.append({"candidate": candidate, "reference": reference, "region": region,
                **joint_station_bootstrap(selected, draws=args.bootstrap_draws),
                "improved_packages": int((direction.candidate_error < direction.reference_error).sum()),
                "improved_partitions": int((partitions.candidate_error < partitions.reference_error).sum())})
    effects = pd.DataFrame(rows)
    effects.to_csv(output/"paired_effects.csv", index=False)
    panel["absolute_error"] = np.abs(panel.y_pred-panel.y_true)
    panel["bias"] = panel.y_pred-panel.y_true
    station = panel.groupby(["split_seed", "seed", "model_name", "station"], as_index=False).agg(
        mae=("absolute_error", "mean"), bias=("bias", "mean"), n_cells=("cell", "size"))
    station.to_csv(output/"station_metrics.csv", index=False)
    station.groupby(["split_seed", "seed", "model_name"], as_index=False).mae.mean().groupby(
        "model_name", as_index=False).mae.mean().to_csv(output/"station_equal_summary.csv", index=False)
    strata = panel.groupby(["split_seed", "seed", "model_name", "hydro_availability"], as_index=False).agg(
        mae=("absolute_error", "mean"), bias=("bias", "mean"), n_cells=("cell", "size"), n_stations=("station", "nunique"))
    strata.to_csv(output/"hydro_strata.csv", index=False)
    overlap = panel[panel.model_name.eq("integrated_concentration_hydro")].groupby(
        ["split_seed", "seed"], as_index=False).agg(mean_calibration_delta=("calibration_delta", "mean"),
        mean_retained_residual=("retained_residual", "mean"), mean_abs_calibration_delta=("calibration_delta", lambda x: np.abs(x).mean()),
        mean_abs_retained_residual=("retained_residual", lambda x: np.abs(x).mean()))
    overlap.to_csv(output/"correction_overlap.csv", index=False)
    write_json(output/"sources.json", {"evaluation_role": "source_validation_only",
        "bootstrap_draws": args.bootstrap_draws, "analyzer_sha256": sha256_file(__file__),
        "runs": {path.parent.name: sha256_file(path) for path in complete},
        "numerical_outputs": {path.name: sha256_file(path) for path in output.glob("*.csv")}})
    print(summary[["model_name", "mae", "signed_bias", "q90_mae"]].to_string(index=False), flush=True)
    print(effects[effects.region.eq("overall")][["candidate", "reference", "relative_gain_pct",
        "gain_ci_low_pct", "gain_ci_high_pct", "improved_partitions"]].to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
