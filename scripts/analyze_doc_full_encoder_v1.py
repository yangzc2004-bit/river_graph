"""Replay and compare full local self-encoder adaptation on the same DOC model."""
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
from run_doc_full_encoder_v1 import ROOT
from run_unified_doc_spatial import (
    digest,
    verify_files,
    verify_runtime_snapshot,
    write_json,
)

from river_graph.experiments.provenance import run_identity_sha256, sha256_file
from river_graph.models.ecological_residual_transfer import EcologicalResidualTransfer
from river_graph.models.encoder_native_residual import EncoderNativeResidual

ARMS = ("full_encoder_residual", "full_encoder_integrated")


def verify(run, runtime):
    config = json.loads((run/"config.json").read_text())
    verify_files(run, "complete.json", config)
    if config["runtime_snapshot_hash"] != runtime:
        raise ValueError("full_encoder execution changed")
    for kind in ("dataset", "mask"):
        if sha256_file(config[f"{kind}_path"]) != config[f"{kind}_hash"]:
            raise ValueError("source inputs changed")
    parent = Path(config["parent_run"])
    old_config = json.loads((parent/"config.json").read_text())
    verify_files(parent, "complete.json", old_config)
    if sha256_file(parent/"complete.json") != config["parent_completion_hash"]:
        raise ValueError("matched parent predictions changed")
    meta = json.loads((run/"predictions.meta.json").read_text())
    identity = run_identity_sha256(digest(config), config["started_at"], config["runtime_snapshot_hash"])
    if (meta["run_identity_sha256"] != identity or meta["config_hash"] != digest(config)
            or meta["prediction_sha256"] != sha256_file(run/"predictions.parquet")):
        raise ValueError("full_encoder residual product identity changed")
    panel = pd.read_parquet(run/"predictions.parquet")
    prior = pd.read_parquet(parent/"predictions.parquet")
    original = panel[panel.model_name.isin(prior.model_name.unique())]
    pd.testing.assert_frame_equal(original.reset_index(drop=True), prior.reset_index(drop=True), check_exact=True)
    model = EncoderNativeResidual.from_payload(torch.load(run/"full_encoder.pt", weights_only=False, map_location="cpu"))
    old_model = EncoderNativeResidual.from_payload(torch.load(parent/"native.pt", weights_only=False, map_location="cpu"))
    expected = {**old_model._config(), "encoder_mode": "all_self_ecology"}
    if (model._config() != expected or config["encoder_mode"] != "all_self_ecology"
            or config["epochs"] != 30 or config["patience"] != 5):
        raise ValueError("changed more than the declared encoder training scope")
    for name in ("spatial", "temporal", "decay"):
        for key, value in getattr(old_model, f"_initial_{name}_state").items():
            torch.testing.assert_close(getattr(model, f"_initial_{name}_state")[key], value, rtol=0, atol=0)
    added = sum(p.numel() for conv in old_model.spatial.convs[:-1] for p in conv.self_lin.parameters())
    if model.trainable_parameter_count_ != old_model.trainable_parameter_count_+added:
        raise ValueError("unexpected full-encoder trainable parameter count")
    for name, value in model.spatial.state_dict().items():
        if ".self_lin." not in name and not name.startswith("env_encoder."):
            torch.testing.assert_close(value, model._initial_spatial_state[name], rtol=0, atol=0)
    with np.load(run/"validation_inputs.npz", allow_pickle=False) as saved:
        inputs = {key: saved[key].copy() for key in ("raw", "env", "age", "support", "extra")}
        context, cells = saved["context"].copy(), saved["cell"].copy()
    with np.load(parent/"validation_inputs.npz", allow_pickle=False) as saved:
        for key, value in inputs.items():
            np.testing.assert_array_equal(value, saved[key])
        np.testing.assert_array_equal(context, saved["context"])
    with np.load(config["mask_path"], allow_pickle=False) as saved:
        np.testing.assert_array_equal(cells, saved["val"])
        train = saved["train"].copy()
    with np.load(parent/"source_oof.npz", allow_pickle=False) as saved:
        source = saved["pred_z"].ravel()
        valid = np.zeros(source.size, dtype=bool)
        valid[train] = True
        if not np.isfinite(source[valid]).all() or not np.isnan(source[~valid]).all():
            raise ValueError("invalid source OOF population")
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
    return {"run": run.name, "identity": True, "full_encoder_and_integrated_replay": "bitwise",
            "unchanged_parent_predictions": True, "old_target_evaluation": False,
            "frozen_message_weights_unchanged": True, "same_initial_weights_and_inputs": True,
            "trainable_parameter_count": model.trainable_parameter_count_,
            "old_trainable_parameter_count": old_model.trainable_parameter_count_,
            "first_self_parameter_distance": model.to_dict()["first_self_parameter_distance"],
            "epochs_run": model.epochs_run_, "best_epoch": model.best_epoch_,
            "old_epochs_run": old_model.epochs_run_, "old_best_epoch": old_model.best_epoch_}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--bootstrap-draws", type=int, default=5000)
    args = parser.parse_args()
    torch.set_num_threads(2)
    runtime = verify_runtime_snapshot(args.root)
    complete = sorted((args.root/"runs").glob("*/complete.json"))
    if {p.parent.name for p in complete} != {f"split{s}_seed{seed}" for s in (142, 143, 144) for seed in (42, 43, 44)}:
        raise ValueError("complete all nine matched source packages")
    verification = [verify(p.parent, runtime) for p in complete]
    write_json(args.root/"verification/replay.json", verification)
    panel, thresholds = load_panel(args.root)
    runs, partitions, summary = metric_summary(panel, thresholds)
    output = args.root/"analysis"
    output.mkdir(exist_ok=True)
    pd.DataFrame(verification).to_csv(output/"training_trajectories.csv", index=False)
    for name, frame in (("run_metrics", runs), ("partition_metrics", partitions), ("summary", summary)):
        frame.to_csv(output/f"{name}.csv", index=False)
    comparisons = [(candidate, ref) for candidate in ARMS for ref in
                   ("unmonitored_integrated", "station_hidden_trees", "current_model")]
    comparisons.append((ARMS[0], "unmonitored_residual"))
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
    write_json(output/"sources.json", {"evaluation_role": "source_validation_only", "bootstrap_draws": args.bootstrap_draws,
        "analyzer_sha256": sha256_file(__file__), "runs": {p.parent.name: sha256_file(p) for p in complete},
        "numerical_outputs": {p.name: sha256_file(p) for p in output.glob("*.csv")}})
    print(summary[["model_name", "mae", "signed_bias", "q90_mae"]].to_string(index=False), flush=True)
    print(effects[effects.region.eq("overall")][["candidate", "reference", "relative_gain_pct",
        "gain_ci_low_pct", "gain_ci_high_pct", "improved_partitions"]].to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
