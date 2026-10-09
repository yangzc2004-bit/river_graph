"""Replay joint source chemistry supervision and compare matched DOC procedures."""
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
from run_doc_joint_source_states_v1 import ROOT
from run_unified_doc_spatial import (
    digest,
    verify_files,
    verify_runtime_snapshot,
    write_json,
)

from river_graph.experiments.provenance import run_identity_sha256, sha256_file
from river_graph.models.ecological_residual_transfer import EcologicalResidualTransfer
from river_graph.models.encoder_native_residual import EncoderNativeResidual
from river_graph.models.joint_source_states import SourceChemistryRegularizer
from river_graph.models.source_auxiliary_pretraining import source_auxiliary_targets

ARMS = tuple(f"{name}_{kind}" for name in ("joint_shuffle", "joint_source") for kind in ("residual", "integrated"))


def verify(run, runtime):
    config = json.loads((run/"config.json").read_text())
    verify_files(run, "complete.json", config)
    if config["runtime_snapshot_hash"] != runtime or config["auxiliary_weight"] != .1:
        raise ValueError("joint source execution or fixed weight changed")
    parent = Path(config["parent_run"])
    verify_files(parent, "complete.json", json.loads((parent/"config.json").read_text()))
    if sha256_file(parent/"complete.json") != config["parent_completion_hash"]:
        raise ValueError("matched parent predictions changed")
    meta = json.loads((run/"predictions.meta.json").read_text())
    identity = run_identity_sha256(digest(config), config["started_at"], config["runtime_snapshot_hash"])
    if (meta["run_identity_sha256"] != identity or meta["config_hash"] != digest(config)
            or meta["prediction_sha256"] != sha256_file(run/"predictions.parquet")):
        raise ValueError("source auxiliary residual product identity changed")
    for kind in ("dataset", "mask"):
        if sha256_file(config[f"{kind}_path"]) != config[f"{kind}_hash"]:
            raise ValueError("source auxiliary inputs changed")
    panel = pd.read_parquet(run/"predictions.parquet")
    prior = pd.read_parquet(parent/"predictions.parquet")
    original = panel[panel.model_name.isin(prior.model_name.unique())]
    pd.testing.assert_frame_equal(original.reset_index(drop=True), prior.reset_index(drop=True), check_exact=True)
    retained = torch.load(parent/"native.pt", weights_only=False, map_location="cpu")
    auxiliary_summaries = []
    for path, expected in config["auxiliary_data_hashes"].items():
        if sha256_file(path) != expected:
            raise ValueError("source auxiliary labels changed")
    with np.load(parent/"validation_inputs.npz", allow_pickle=False) as saved:
        preceding_inputs = {key: saved[key].copy() for key in ("raw", "env", "age", "support", "extra", "context", "cell")}
    with np.load(run/"joint_source_inputs.npz", allow_pickle=False) as saved:
        source_inputs = {key: saved[key].copy() for key in ("raw", "env", "age", "support", "extra")}
        target, valid, source_ids = saved["target"].copy(), saved["valid"].copy(), saved["source_station_ids"].copy()
    dataset = torch.load(config["dataset_path"], weights_only=False, map_location="cpu")
    months = dataset["y"].shape[1]
    with np.load(config["mask_path"], allow_pickle=False) as saved:
        train = saved["train"].copy()
        np.testing.assert_array_equal(source_ids, np.unique(train//months))
        receiving = np.unique(np.r_[saved["val"], saved["test"]]//months)
        if np.intersect1d(source_ids, receiving).size:
            raise ValueError("receiving stations appeared in source auxiliary population")
    auxiliary_data = {name: torch.load(path, weights_only=False, map_location="cpu")
        for name, path in (("ph", "data/processed/mississippi_graph_ph_st357.pt"),
                          ("spec_conductance", "data/processed/mississippi_graph_spec_conductance_st357.pt"))}
    allowed_y, allowed_mask = source_auxiliary_targets(auxiliary_data, source_ids, dataset["site_no"], dataset["months"])
    np.testing.assert_array_equal(target, allowed_y)
    np.testing.assert_array_equal(valid, allowed_mask)
    source_doc_mask = np.zeros(dataset["y"].shape, dtype=bool)
    source_doc_mask.ravel()[train] = True
    auxiliary_only_cells = int(np.count_nonzero(valid.any(-1) & ~source_doc_mask[source_ids]))
    initial_model = EncoderNativeResidual.from_payload(retained)
    for name in ("spatial", "temporal", "decay"):
        getattr(initial_model, name).load_state_dict(retained[f"initial_{name}"])
    for name in ("joint_shuffle", "joint_source"):
        payload = torch.load(run/f"{name}.pt", weights_only=False, map_location="cpu")
        model = EncoderNativeResidual.from_payload(payload)
        head_payload = torch.load(run/f"{name}_auxiliary.pt", weights_only=False, map_location="cpu")
        auxiliary = json.loads((run/f"{name}_auxiliary.json").read_text())
        auxiliary_summaries.append(auxiliary)
        for module in ("spatial", "temporal", "decay"):
            for key, value in retained[f"initial_{module}"].items():
                torch.testing.assert_close(payload[f"initial_{module}"][key], value, rtol=0, atol=0)
        if payload["config"] != retained["config"]:
            raise ValueError("changed DOC architecture, settings or optimization budget")
        if (auxiliary["selection_role"] != "source_validation_doc_mae"
                or auxiliary["receiving_chemistry_used"] or auxiliary["weight"] != .1
                or auxiliary["source_label_shuffle"] != (name == "joint_shuffle")
                or auxiliary["sampled_source_cells"] != model.optimizer_steps_*model.batch_size):
            raise ValueError("invalid simultaneous auxiliary supervision roles")
        np.testing.assert_array_equal(auxiliary["source_station_ids"], source_ids)
        probe = SourceChemistryRegularizer(model, target, valid, weight=.1,
            shuffle=name == "joint_shuffle", seed=config["seed"], batch_size=model.batch_size)
        for key, value in probe.head.state_dict().items():
            torch.testing.assert_close(value, head_payload["initial_head"][key], rtol=0, atol=0)
        for key, value in (("target_mean", probe.mean), ("target_scale", probe.scale),
                           ("fit_label_counts", probe.counts)):
            np.testing.assert_array_equal(value, auxiliary[key])
        data = model._prepare_inputs(source_inputs)
        np.testing.assert_array_equal(probe.diagnostics(initial_model, data), auxiliary["initial_source_mse"])
        probe.head.load_state_dict(head_payload["head"])
        np.testing.assert_array_equal(probe.diagnostics(model, data), auxiliary["selected_source_mse"])
        with np.load(run/f"{name}_validation_inputs.npz", allow_pickle=False) as saved:
            inputs = {key: saved[key].copy() for key in ("raw", "env", "age", "support", "extra")}
            context, cells = saved["context"].copy(), saved["cell"].copy()
        for key in ("raw", "age", "support", "env"):
            np.testing.assert_array_equal(inputs[key], preceding_inputs[key])
        np.testing.assert_array_equal(inputs["extra"], preceding_inputs["extra"])
        np.testing.assert_array_equal(context, preceding_inputs["context"])
        np.testing.assert_array_equal(cells, preceding_inputs["cell"])
        with np.load(config["mask_path"], allow_pickle=False) as saved:
            np.testing.assert_array_equal(cells, saved["val"])
        n_months = context.shape[1]
        ids = np.unique(cells//n_months)
        prediction = model.predict(inputs, context)
        selected = prediction[np.searchsorted(ids, cells//n_months), cells % n_months]
        np.testing.assert_array_equal(selected, panel[panel.model_name.eq(f"{name}_residual")].y_pred)
        memory = EcologicalResidualTransfer.from_dict(json.loads((run/f"{name}_memory.json").read_text()))
        c = np.zeros((memory.n_nodes_, memory.n_months_))
        p = c.copy()
        c[ids], p[ids] = context, prediction
        integrated = memory.predict(c, p).ravel()[cells]
        np.testing.assert_array_equal(integrated, panel[panel.model_name.eq(f"{name}_integrated")].y_pred)
    for key in ("target_mean", "target_scale", "fit_label_counts", "source_station_ids", "union_source_cells"):
        np.testing.assert_array_equal(auxiliary_summaries[0][key], auxiliary_summaries[1][key])
    for _, group in panel.groupby("model_name"):
        np.testing.assert_array_equal(group.cell, cells)
    if not panel.visibility_role.eq("val").all() or not np.isfinite(panel.y_pred).all():
        raise ValueError("invalid source validation panel")
    return {"run": run.name, "identity": True, "native_and_integrated_replay": "bitwise", "retained_initial_backbone": True, "source_normalization_and_head_mse_replay": "bitwise",
            "auxiliary_only_source_cells": auxiliary_only_cells,
            "unchanged_parent_predictions": True, "old_target_evaluation": False}


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
    write_json(args.root/"verification/replay.json", [verify(p.parent, runtime) for p in complete])
    panel, thresholds = load_panel(args.root)
    runs, partitions, summary = metric_summary(panel, thresholds)
    output = args.root/"analysis"
    output.mkdir(exist_ok=True)
    for name, frame in (("run_metrics", runs), ("partition_metrics", partitions), ("summary", summary)):
        frame.to_csv(output/f"{name}.csv", index=False)
    comparisons = [(candidate, ref) for candidate in ARMS for ref in
                   ("unmonitored_integrated", "station_hidden_trees", "current_model")]
    comparisons += [("joint_source_integrated", "joint_shuffle_integrated"),
        ("joint_source_residual", "joint_shuffle_residual"),
        ("joint_source_residual", "unmonitored_residual"),
        ("joint_shuffle_residual", "unmonitored_residual")]
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
    auxiliary_rows = []
    for path in complete:
        for name in ("joint_shuffle", "joint_source"):
            row = json.loads((path.parent/f"{name}_auxiliary.json").read_text())
            auxiliary_rows.append({"run": path.parent.name, "model_name": name,
                "best_epoch": row["doc_best_epoch"], "epochs_run": row["doc_epochs_run"],
                "initial_source_ph_mse": row["initial_source_mse"][0],
                "initial_source_log_ec_mse": row["initial_source_mse"][1],
                "selected_source_ph_mse": row["selected_source_mse"][0],
                "selected_source_log_ec_mse": row["selected_source_mse"][1],
                "fit_ph_labels": row["fit_label_counts"][0], "fit_ec_labels": row["fit_label_counts"][1]})
    pd.DataFrame(auxiliary_rows).to_csv(output/"joint_source_training.csv", index=False)
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
