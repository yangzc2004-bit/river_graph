"""Replay antecedent hydro-climate trees and compare matched source validation."""
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
from run_doc_antecedent_hydro_v1 import ROOT
from run_unified_doc_spatial import digest, verify_files, write_json

from river_graph.experiments.provenance import run_identity_sha256, sha256_file

ARMS = ("hydro_availability_trees", "hydro_state_trees")


def verify(run, root):
    config = json.loads((run/"config.json").read_text())
    verify_files(run, "complete.json", config)
    parent = Path(config["parent_run"])
    if sha256_file(parent/"complete.json") != config["parent_completion_hash"]:
        raise ValueError("matched retained package changed")
    for kind in ("dataset", "mask"):
        if sha256_file(config[f"{kind}_path"]) != config[f"{kind}_hash"]:
            raise ValueError("antecedent hydro source data changed")
    meta = json.loads((run/"predictions.meta.json").read_text())
    identity = run_identity_sha256(digest(config), config["started_at"], config["runtime_snapshot_hash"])
    if (meta["config_hash"] != digest(config) or meta["run_identity_sha256"] != identity
            or meta["prediction_sha256"] != sha256_file(run/"predictions.parquet")):
        raise ValueError("antecedent hydro prediction identity changed")
    panel = pd.read_parquet(run/"predictions.parquet")
    prior = pd.read_parquet(parent/"predictions.parquet")
    original = panel[panel.model_name.isin(prior.model_name.unique())]
    pd.testing.assert_frame_equal(original.reset_index(drop=True), prior.reset_index(drop=True), check_exact=True)
    with np.load(config["mask_path"], allow_pickle=False) as saved:
        cells = saved["val"].copy()
    with np.load(run/"validation_inputs.npz", allow_pickle=False) as saved:
        np.testing.assert_array_equal(saved["cell"], cells)
        aggregate, detailed = (saved[name].copy() for name in ARMS)
    if aggregate.shape != (len(cells), 55) or detailed.shape != aggregate.shape:
        raise ValueError("matched trees require47 retained plus8 hydro inputs")
    np.testing.assert_array_equal(aggregate[:, :47], detailed[:, :47])
    np.testing.assert_array_equal(aggregate[:, (48, 50, 52, 54)], detailed[:, (48, 50, 52, 54)])
    if not np.isfinite(detailed).all() or not np.isfinite(aggregate).all():
        raise ValueError("nonfinite tree inputs")
    if ((detailed[:, 47:] < -1) | (detailed[:, 47:] > 1)).any():
        raise ValueError("bounded descriptors must be within-1–1")
    replay_error = 0.
    for name, inputs in zip(ARMS, (aggregate, detailed), strict=True):
        model = joblib.load(run/f"{name}.joblib")
        values = np.maximum(0., np.expm1(model.predict(inputs)))
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
        "matched_input_dimensions": 55, "matched_retained_inputs_and_validity": True,
        "max_replay_difference": replay_error, "old_target_evaluation": False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--bootstrap-draws", type=int, default=5000)
    args = parser.parse_args()
    snapshot = json.loads((args.root/"runtime_snapshot.json").read_text())
    for name, expected in snapshot.items():
        if sha256_file(args.root/"code_snapshot"/name) != expected:
            raise ValueError("saved antecedent hydro execution sources changed")
    completions = sorted((args.root/"runs").glob("*/complete.json"))
    expected = {f"split{partition}_seed{seed}" for partition in (142, 143, 144) for seed in (42, 43, 44)}
    if {path.parent.name for path in completions} != expected:
        raise ValueError("complete all nine fixed source packages")
    write_json(args.root/"verification/replay.json", [verify(path.parent, args.root) for path in completions])
    panel, thresholds = load_panel(args.root)
    runs, partitions, summary = metric_summary(panel, thresholds)
    output = args.root/"analysis"
    output.mkdir(exist_ok=True)
    for name, frame in (("run_metrics", runs), ("partition_metrics", partitions), ("summary", summary)):
        frame.to_csv(output/f"{name}.csv", index=False)
    comparisons = [("hydro_state_trees", "hydro_availability_trees")]
    comparisons += [(candidate, reference) for candidate in ARMS for reference in
        ("station_hidden_trees", "unmonitored_integrated")]
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
            effects.append({"candidate": candidate, "reference": reference, "region": region,
                **joint_station_bootstrap(selected, draws=args.bootstrap_draws),
                "improved_packages": int((direction.candidate_error < direction.reference_error).sum()),
                "improved_partitions": int((part.candidate_error < part.reference_error).sum())})
    effects = pd.DataFrame(effects)
    effects.to_csv(output/"paired_effects.csv", index=False)
    panel["absolute_error"] = np.abs(panel.y_pred-panel.y_true)
    panel["signed_error"] = panel.y_pred-panel.y_true
    station = panel.groupby(["split_seed", "seed", "model_name", "station"], as_index=False).agg(
        mae=("absolute_error", "mean"), bias=("signed_error", "mean"), n_cells=("cell", "size"))
    station.to_csv(output/"station_metrics.csv", index=False)
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
