"""Replay and compare detailed composition in the existing DOC encoder."""
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
from run_doc_composition_encoder_v1 import ROOT
from run_unified_doc_spatial import digest, verify_files, write_json

from river_graph.experiments.provenance import run_identity_sha256, sha256_file
from river_graph.models.ecological_residual_transfer import EcologicalResidualTransfer
from river_graph.models.encoder_native_residual import EncoderNativeResidual

ARMS = tuple(f"{name}_{kind}" for name in ("aggregate_encoder", "composition_encoder") for kind in ("residual", "integrated"))


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
    for kind in ("dataset", "mask", "nodes", "attributes"):
        if sha256_file(config[f"{kind}_path"]) != config[f"{kind}_hash"]:
            raise ValueError("composition encoder source inputs changed")
    panel = pd.read_parquet(run/"predictions.parquet")
    prior = pd.read_parquet(parent/"predictions.parquet")
    original = panel[panel.model_name.isin(prior.model_name.unique())]
    pd.testing.assert_frame_equal(original.reset_index(drop=True), prior.reset_index(drop=True), check_exact=True)
    retained = torch.load(parent/"native.pt", weights_only=False, map_location="cpu")
    old_dim = retained["spatial_architecture"]["env_dim"]
    physical_views = []
    with np.load(parent/"validation_inputs.npz", allow_pickle=False) as saved:
        preceding_inputs = {key: saved[key].copy() for key in ("raw", "env", "age", "support", "extra", "context", "cell")}
    for name in ("aggregate_encoder", "composition_encoder"):
        payload = torch.load(run/f"{name}.pt", weights_only=False, map_location="cpu")
        model = EncoderNativeResidual.from_payload(payload)
        for key, value in retained["initial_spatial"].items():
            expanded = payload["initial_spatial"][key]
            if key == "env_encoder.0.weight":
                torch.testing.assert_close(expanded[:, :old_dim], value, rtol=0, atol=0)
                if torch.count_nonzero(expanded[:, old_dim:]) or expanded.shape[1] != old_dim+22:
                    raise ValueError("new ecological columns must start at zero")
            else:
                torch.testing.assert_close(expanded, value, rtol=0, atol=0)
        for module in ("temporal", "decay"):
            for key, value in retained[f"initial_{module}"].items():
                torch.testing.assert_close(payload[f"initial_{module}"][key], value, rtol=0, atol=0)
        with np.load(run/f"{name}_validation_inputs.npz", allow_pickle=False) as saved:
            inputs = {key: saved[key].copy() for key in ("raw", "env", "age", "support", "extra")}
            context, cells = saved["context"].copy(), saved["cell"].copy()
        for key in ("raw", "age", "support", "extra"):
            np.testing.assert_array_equal(inputs[key], preceding_inputs[key])
        np.testing.assert_array_equal(inputs["env"][:, :old_dim], preceding_inputs["env"])
        np.testing.assert_array_equal(context, preceding_inputs["context"])
        np.testing.assert_array_equal(cells, preceding_inputs["cell"])
        with np.load(config["mask_path"], allow_pickle=False) as saved:
            np.testing.assert_array_equal(cells, saved["val"])
        physical_views.append(inputs["env"][:, old_dim:])
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
    np.testing.assert_array_equal(physical_views[0][:, -11:], physical_views[1][:, -11:])
    for _, group in panel.groupby("model_name"):
        np.testing.assert_array_equal(group.cell, cells)
    if not panel.visibility_role.eq("val").all() or not np.isfinite(panel.y_pred).all():
        raise ValueError("invalid source validation panel")
    return {"run": run.name, "identity": True, "native_and_integrated_replay": "bitwise", "retained_initial_backbone": True,
            "unchanged_parent_predictions": True, "old_target_evaluation": False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--bootstrap-draws", type=int, default=5000)
    args = parser.parse_args()
    torch.set_num_threads(2)
    snapshot = json.loads((args.root/"runtime_snapshot.json").read_text())
    for name, expected in snapshot.items():
        if sha256_file(args.root/"code_snapshot"/name) != expected:
            raise ValueError("saved encoder execution source changed")
    complete = sorted((args.root/"runs").glob("*/complete.json"))
    if {p.parent.name for p in complete} != {f"split{s}_seed{seed}" for s in (142, 143, 144) for seed in (42, 43, 44)}:
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
    comparisons += [("composition_encoder_integrated", "aggregate_encoder_integrated"),
        ("composition_encoder_residual", "aggregate_encoder_residual"),
        ("composition_encoder_residual", "unmonitored_residual"),
        ("aggregate_encoder_residual", "unmonitored_residual")]
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
    write_json(output/"sources.json", {"evaluation_role": "source_validation_only", "bootstrap_draws": args.bootstrap_draws,
        "analyzer_sha256": sha256_file(__file__), "runs": {p.parent.name: sha256_file(p) for p in complete},
        "numerical_outputs": {p.name: sha256_file(p) for p in output.glob("*.csv")}})
    print(summary[["model_name", "mae", "signed_bias", "q90_mae"]].to_string(index=False), flush=True)
    print(effects[effects.region.eq("overall")][["candidate", "reference", "relative_gain_pct",
        "gain_ci_low_pct", "gain_ci_high_pct", "improved_partitions"]].to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
