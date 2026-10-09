"""Replay and compare the small nonlinear DOC readout on source roles."""
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
from run_doc_nonlinear_native_residual_v1 import ROOT
from run_unified_doc_spatial import digest, verify_files, write_json

from river_graph.experiments.provenance import run_identity_sha256, sha256_file
from river_graph.models.ecological_residual_transfer import EcologicalResidualTransfer
from river_graph.models.nonlinear_native_residual import (
    NonlinearNativeResidual,
)

ARMS = ("nonlinear_native_residual", "nonlinear_native_integrated")


def verify(run):
    config = json.loads((run/"config.json").read_text())
    verify_files(run, "complete.json", config)
    parent = Path(config["parent_run"])
    if sha256_file(parent/"complete.json") != config["parent_completion_hash"]:
        raise ValueError("matched parent predictions changed")
    meta = json.loads((run/"predictions.meta.json").read_text())
    identity = run_identity_sha256(digest(config), config["started_at"], config["runtime_snapshot_hash"])
    if (meta["run_identity_sha256"] != identity or meta["config_hash"] != digest(config)
            or meta["prediction_sha256"] != sha256_file(run/"predictions.parquet")):
        raise ValueError("nonlinear residual product identity changed")
    panel = pd.read_parquet(run/"predictions.parquet")
    prior = pd.read_parquet(parent/"predictions.parquet")
    original = panel[panel.model_name.isin(prior.model_name.unique())]
    pd.testing.assert_frame_equal(original.reset_index(drop=True), prior.reset_index(drop=True), check_exact=True)
    model = NonlinearNativeResidual.from_payload(torch.load(run/"nonlinear.pt", weights_only=False, map_location="cpu"))
    with np.load(run/"validation_inputs.npz", allow_pickle=False) as saved:
        inputs = {key: saved[key].copy() for key in ("raw", "env", "age", "support", "extra")}
        context, cells = saved["context"].copy(), saved["cell"].copy()
    with np.load(config["mask_path"], allow_pickle=False) as saved:
        np.testing.assert_array_equal(cells, saved["val"])
    n_months = context.shape[1]
    ids = np.unique(cells//n_months)
    prediction = model.predict(inputs, context)
    selected = prediction[np.searchsorted(ids, cells//n_months), cells % n_months]
    np.testing.assert_array_equal(selected, panel[panel.model_name.eq(ARMS[0])].y_pred)
    memory = EcologicalResidualTransfer.from_dict(json.loads((run/"memory.json").read_text()))
    c = np.zeros((memory.n_nodes_, memory.n_months_))
    p = c.copy()
    c[ids], p[ids] = context, prediction
    integrated = memory.predict(c, p).ravel()[cells]
    np.testing.assert_array_equal(integrated, panel[panel.model_name.eq(ARMS[1])].y_pred)
    for _, group in panel.groupby("model_name"):
        np.testing.assert_array_equal(group.cell, cells)
    if not panel.visibility_role.eq("val").all() or not np.isfinite(panel.y_pred).all():
        raise ValueError("invalid source validation panel")
    return {"run": run.name, "identity": True, "nonlinear_and_integrated_replay": "bitwise",
            "unchanged_parent_predictions": True, "old_target_evaluation": False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--bootstrap-draws", type=int, default=5000)
    args = parser.parse_args()
    torch.set_num_threads(2)
    complete = sorted((args.root/"runs").glob("*/complete.json"))
    if len(complete) != 9:
        raise ValueError("complete all nine matched source packages")
    write_json(args.root/"verification/replay.json", [verify(p.parent) for p in complete])
    panel, thresholds = load_panel(args.root)
    runs, partitions, summary = metric_summary(panel, thresholds)
    output = args.root/"analysis"
    output.mkdir(exist_ok=True)
    for name, frame in (("run_metrics", runs), ("partition_metrics", partitions), ("summary", summary)):
        frame.to_csv(output/f"{name}.csv", index=False)
    comparisons = [(candidate, ref) for candidate in ARMS for ref in
                   ("unmonitored_integrated", "station_hidden_trees", "current_model")]
    comparisons.append((ARMS[0], "unmonitored_residual"))
    effects = []
    for candidate, reference in comparisons:
        pair = paired(panel, candidate, reference)
        direction = pair.groupby(["split_seed", "seed"])[["candidate_error", "reference_error"]].mean()
        part = direction.groupby("split_seed").mean()
        for region in ("overall", "q90"):
            selected = pair
            if region == "q90":
                cutoff = np.array([thresholds[(p, s)] for p, s in zip(pair.split_seed, pair.seed, strict=True)])
                selected = pair[pair.y_true_candidate.to_numpy() >= cutoff]
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
    write_json(output/"sources.json", {"evaluation_role": "source_validation_only", "bootstrap_draws": args.bootstrap_draws,
        "analyzer_sha256": sha256_file(__file__), "runs": {p.parent.name: sha256_file(p) for p in complete},
        "numerical_outputs": {p.name: sha256_file(p) for p in output.glob("*.csv")}})
    print(summary[["model_name", "mae", "signed_bias", "q90_mae"]].to_string(index=False), flush=True)
    print(effects[effects.region.eq("overall")][["candidate", "reference", "relative_gain_pct",
        "gain_ci_low_pct", "gain_ci_high_pct", "improved_partitions"]].to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
