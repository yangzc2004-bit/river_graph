"""Replay fixed-leaf source distributions and quantify matched DOC effects."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
from analyze_doc_chemistry_confirmation_v1 import metric_summary
from analyze_doc_source_retrieval_v1 import load_panel, paired
from analyze_unified_doc_spatial import joint_station_bootstrap
from run_doc_leaf_distribution_reference_v1 import ARMS, ROOT
from run_unified_doc_spatial import (
    digest,
    verify_files,
    verify_runtime_snapshot,
    write_json,
)
from threadpoolctl import threadpool_limits

from river_graph.experiments.provenance import run_identity_sha256, sha256_file


def verify(run, runtime):
    config = json.loads((run/"config.json").read_text())
    verify_files(run, "complete.json", config)
    verify_files(run, "fitted_complete.json", config)
    if config["runtime_snapshot_hash"] != runtime or config["readouts"] != ARMS or config["quantile"] != .5:
        raise ValueError("changed source distribution execution or readouts")
    parent = Path(config["parent_run"])
    if sha256_file(parent/"complete.json") != config["parent_completion_hash"]:
        raise ValueError("matched retained package changed")
    for kind in ("dataset", "mask"):
        if sha256_file(config[f"{kind}_path"]) != config[f"{kind}_hash"]:
            raise ValueError("source distribution data changed")
    if sha256_file(config["retained_forest_path"]) != config["retained_forest_sha256"]:
        raise ValueError("retained partition geometry changed")
    meta = json.loads((run/"predictions.meta.json").read_text())
    identity = run_identity_sha256(digest(config), config["started_at"], runtime)
    if (meta["config_hash"] != digest(config) or meta["run_identity_sha256"] != identity
            or meta["prediction_sha256"] != sha256_file(run/"predictions.parquet")):
        raise ValueError("source distribution prediction identity changed")
    panel = pd.read_parquet(run/"predictions.parquet")
    prior = pd.read_parquet(parent/"predictions.parquet")
    original = panel[panel.model_name.isin(prior.model_name.unique())]
    pd.testing.assert_frame_equal(original.reset_index(drop=True), prior.reset_index(drop=True), check_exact=True)
    with np.load(config["mask_path"], allow_pickle=False) as saved:
        cells, train = saved["val"].copy(), saved["train"].copy()
    with np.load(run/"validation_inputs.npz", allow_pickle=False) as saved:
        np.testing.assert_array_equal(saved["cell"], cells)
        inputs = saved["features"].copy()
    if inputs.shape != (len(cells), 47) or not np.isfinite(inputs).all():
        raise ValueError("distribution readouts require47 identical finite source features")
    dataset = torch.load(config["dataset_path"], weights_only=False, map_location="cpu")
    months = dataset["y"].shape[1]
    model = joblib.load(run/"distribution.joblib")
    retained = joblib.load(config["retained_forest_path"])
    if model.forest.get_params() != retained.get_params():
        raise ValueError("retained forest parameters changed")
    for new, old in zip(model.forest.estimators_, retained.estimators_, strict=True):
        for field in ("children_left", "children_right", "feature", "threshold", "value"):
            np.testing.assert_array_equal(getattr(new.tree_, field), getattr(old.tree_, field))
    source_y = np.asarray(dataset["y"]).ravel()[train].astype(np.float64)
    np.testing.assert_array_equal(model.source_order_, np.argsort(source_y, kind="stable"))
    np.testing.assert_array_equal(model.source_y_, source_y[model.source_order_])
    record = json.loads((run/"distribution.json").read_text())
    np.testing.assert_array_equal(record["fitted_stations"], np.unique(train//months))
    if (record["source_fit_cells_sha256"] != digest(train.tolist())
            or record["fitted_cell_count"] != len(train) or model.n_features_in_ != 47):
        raise ValueError("source-only concentration population changed")
    populated = np.asarray(model.leaf_weights_.sum(axis=1)).ravel()
    np.testing.assert_allclose(populated[populated > 0], 1./model.n_trees_, atol=1e-12, rtol=1e-12)
    with threadpool_limits(limits=2):
        components = model.predict_components(inputs)
    replay_error = 0.
    for name, key in ARMS.items():
        stored = panel[panel.model_name.eq(name)]
        np.testing.assert_array_equal(stored.cell, cells)
        np.testing.assert_allclose(components[key], stored.y_pred, rtol=1e-12, atol=1e-12)
        replay_error = max(replay_error, float(np.abs(components[key]-stored.y_pred.to_numpy()).max()))
    reference = prior[prior.model_name.eq("unmonitored_integrated")]
    for _, group in panel.groupby("model_name"):
        np.testing.assert_array_equal(group.cell, cells)
        for field in ("station", "month", "y_true"):
            np.testing.assert_array_equal(group[field], reference[field])
    if (set(panel.model_name) != set(config["models"]) or not panel.visibility_role.eq("val").all()
            or not np.isfinite(panel.y_pred).all()):
        raise ValueError("invalid source-only distribution comparison panel")
    return {"run": run.name, "identity": True, "unchanged_parent_predictions": True,
        "same_input_dimensions": 47, "unchanged_tree_states": True,
        "source_concentrations_exact": True, "normalized_leaf_weights": True,
        "max_replay_difference": replay_error, "old_geographical_or_external_evaluation": False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--bootstrap-draws", type=int, default=5000)
    args = parser.parse_args()
    runtime = verify_runtime_snapshot(args.root)
    completions = sorted((args.root/"runs").glob("*/complete.json"))
    expected = {f"split{partition}_seed{seed}" for partition in (142, 143, 144) for seed in (42, 43, 44)}
    if {path.parent.name for path in completions} != expected:
        raise ValueError("complete all nine fixed source packages")
    write_json(args.root/"verification/replay.json", [verify(path.parent, runtime) for path in completions])
    panel, thresholds = load_panel(args.root)
    runs, partitions, summary = metric_summary(panel, thresholds)
    output = args.root/"analysis"
    output.mkdir(exist_ok=True)
    for name, frame in (("run_metrics", runs), ("partition_metrics", partitions), ("summary", summary)):
        frame.to_csv(output/f"{name}.csv", index=False)
    comparisons = [(candidate, reference) for candidate in ARMS for reference in
        ("station_hidden_trees", "unmonitored_integrated")]
    comparisons.append(("leaf_conditional_median", "leaf_native_mean"))
    effects = []
    for candidate, reference in comparisons:
        pair = paired(panel, candidate, reference)
        for region in ("overall", "q90"):
            selected = pair
            if region == "q90":
                cutoff = np.array([thresholds[(p, s)] for p, s in zip(pair.split_seed, pair.seed, strict=True)])
                selected = pair[pair.y_true_candidate.to_numpy() >= cutoff]
            direction = selected.groupby(["split_seed", "seed"])[["candidate_error", "reference_error"]].mean()
            part = direction.groupby("split_seed").mean()
            seed = direction.groupby("seed").mean()
            effects.append({"candidate": candidate, "reference": reference, "region": region,
                **joint_station_bootstrap(selected, draws=args.bootstrap_draws),
                "improved_packages": int((direction.candidate_error < direction.reference_error).sum()),
                "improved_partitions": int((part.candidate_error < part.reference_error).sum()),
                "improved_training_seeds": int((seed.candidate_error < seed.reference_error).sum())})
    effects = pd.DataFrame(effects)
    effects.to_csv(output/"paired_effects.csv", index=False)
    panel["absolute_error"] = np.abs(panel.y_pred-panel.y_true)
    panel["signed_error"] = panel.y_pred-panel.y_true
    station = panel.groupby(["split_seed", "seed", "model_name", "station"], as_index=False).agg(
        mae=("absolute_error", "mean"), bias=("signed_error", "mean"), n_cells=("cell", "size"))
    station.to_csv(output/"station_metrics.csv", index=False)
    station_mean = station.groupby(["split_seed", "model_name"], as_index=False)[["mae", "bias"]].mean()
    station_mean.groupby("model_name", as_index=False)[["mae", "bias"]].mean().to_csv(output/"station_equal_summary.csv", index=False)
    write_json(output/"sources.json", {"evaluation_role": "source_validation_only",
        "bootstrap_draws": args.bootstrap_draws, "analyzer_sha256": sha256_file(__file__),
        "weighting": "seed means within partition; equal three source partitions",
        "runs": {path.parent.name: sha256_file(path) for path in completions},
        "numerical_outputs": {path.name: sha256_file(path) for path in output.glob("*.csv")}})
    print(summary[["model_name", "mae", "signed_bias", "q90_mae"]].to_string(index=False), flush=True)
    print(effects[effects.region.eq("overall")][["candidate", "reference", "relative_gain_pct",
        "gain_ci_low_pct", "gain_ci_high_pct", "improved_partitions"]].to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
