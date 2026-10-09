"""Replay fixed DOC reference objectives and quantify paired source effects."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from analyze_doc_chemistry_confirmation_v1 import metric_summary
from analyze_doc_source_retrieval_v1 import load_panel, paired
from analyze_unified_doc_spatial import joint_station_bootstrap
from run_doc_native_reference_objective_v1 import ROOT
from run_unified_doc_spatial import (
    digest,
    verify_files,
    verify_runtime_snapshot,
    write_json,
)
from threadpoolctl import threadpool_limits

from river_graph.experiments.provenance import run_identity_sha256, sha256_file
from river_graph.models.doc_reference_objectives import (
    ARMS,
    make_reference,
    predict_reference,
)


def verify(run, runtime):
    config = json.loads((run/"config.json").read_text())
    verify_files(run, "complete.json", config)
    if config["runtime_snapshot_hash"] != runtime or config["target_transforms"] != ARMS:
        raise ValueError("changed reference study execution or objectives")
    parent = Path(config["parent_run"])
    if sha256_file(parent/"complete.json") != config["parent_completion_hash"]:
        raise ValueError("matched retained package changed")
    for kind in ("dataset", "mask"):
        if sha256_file(config[f"{kind}_path"]) != config[f"{kind}_hash"]:
            raise ValueError("reference source data changed")
    meta = json.loads((run/"predictions.meta.json").read_text())
    identity = run_identity_sha256(digest(config), config["started_at"], runtime)
    if (meta["config_hash"] != digest(config) or meta["run_identity_sha256"] != identity
            or meta["prediction_sha256"] != sha256_file(run/"predictions.parquet")):
        raise ValueError("reference prediction identity changed")
    panel = pd.read_parquet(run/"predictions.parquet")
    prior = pd.read_parquet(parent/"predictions.parquet")
    original = panel[panel.model_name.isin(prior.model_name.unique())]
    pd.testing.assert_frame_equal(original.reset_index(drop=True), prior.reset_index(drop=True), check_exact=True)
    old = json.loads((parent/"config.json").read_text())
    retained = joblib.load(Path(old["parent_run"])/"station_hidden_trees.joblib")
    with np.load(config["mask_path"], allow_pickle=False) as saved:
        cells, train = saved["val"].copy(), saved["train"].copy()
    with np.load(run/"validation_inputs.npz", allow_pickle=False) as saved:
        np.testing.assert_array_equal(saved["cell"], cells)
        inputs = saved["features"].copy()
    if inputs.shape != (len(cells), 47) or not np.isfinite(inputs).all():
        raise ValueError("all objectives require the same47 retained finite inputs")
    # The dataset month axis is fixed by the frozen parent task, not station IDs.
    import torch
    months = torch.load(config["dataset_path"], weights_only=False, map_location="cpu")["y"].shape[1]
    replay_error = 0.
    for name in ARMS:
        verify_files(run, f"{name}_complete.json", config)
        model = joblib.load(run/f"{name}.joblib")
        expected = make_reference(name, retained, config["seed"])
        if model.get_params() != expected.get_params():
            raise ValueError("saved reference parameters differ from the fixed arm")
        record = json.loads((run/f"{name}.json").read_text())
        np.testing.assert_array_equal(record["fitted_stations"], np.unique(train//months))
        if (record["source_fit_cells_sha256"] != digest(train.tolist())
                or record["fitted_cell_count"] != len(train)
                or record["target_transform"] != ARMS[name] or model.n_features_in_ != 47):
            raise ValueError("source-only fitted roles or target scale changed")
        with threadpool_limits(limits=2):
            values = predict_reference(model, inputs, name)
        stored = panel[panel.model_name.eq(name)]
        np.testing.assert_array_equal(stored.cell, cells)
        np.testing.assert_allclose(values, stored.y_pred, rtol=1e-12, atol=1e-12)
        replay_error = max(replay_error, float(np.abs(values-stored.y_pred.to_numpy()).max()))
    reference = prior[prior.model_name.eq("unmonitored_integrated")]
    for _, group in panel.groupby("model_name"):
        np.testing.assert_array_equal(group.cell, cells)
        for field in ("station", "month", "y_true"):
            np.testing.assert_array_equal(group[field], reference[field])
    if (set(panel.model_name) != set(config["models"]) or not panel.visibility_role.eq("val").all()
            or not np.isfinite(panel.y_pred).all()):
        raise ValueError("invalid source-only comparison panel")
    return {"run": run.name, "identity": True, "unchanged_parent_predictions": True,
        "same_input_dimensions": 47, "fixed_arm_parameters": True,
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
    comparisons.append(("native_l1_boosting", "log_l1_boosting"))
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
