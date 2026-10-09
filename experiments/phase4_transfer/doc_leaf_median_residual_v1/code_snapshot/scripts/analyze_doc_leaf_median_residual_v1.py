"""Replay conditional-median OOF references and the existing DOC neural fit."""
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
from run_doc_leaf_median_residual_v1 import ROOT
from run_unified_doc_spatial import (
    digest,
    verify_files,
    verify_runtime_snapshot,
    write_json,
)
from threadpoolctl import threadpool_limits

from river_graph.experiments.provenance import run_identity_sha256, sha256_file
from river_graph.models.causal_flow_features import build_causal_flow_features
from river_graph.models.ecological_residual_transfer import EcologicalResidualTransfer
from river_graph.models.encoder_native_residual import EncoderNativeResidual
from river_graph.models.kgml_local_transport import fold_split, station_folds
from river_graph.models.regime_head_features import build_regime_head_features

ARMS = ("leaf_median_reference", "leaf_median_residual", "leaf_median_integrated")


def verify(run, runtime):
    config = json.loads((run/"config.json").read_text())
    verify_files(run, "complete.json", config)
    verify_files(run, "native_complete.json", config)
    if config["runtime_snapshot_hash"] != runtime or config["quantile"] != .5:
        raise ValueError("median neural source execution changed")
    for kind in ("dataset", "mask"):
        if sha256_file(config[f"{kind}_path"]) != config[f"{kind}_hash"]:
            raise ValueError("median neural source inputs changed")
    for path_key, hash_key in (("parent_run", "parent_completion_hash"),
            ("reference_run", "reference_complete_hash"), ("fold_run", "fold_complete_hash")):
        if sha256_file(Path(config[path_key])/"complete.json") != config[hash_key]:
            raise ValueError("median neural source ancestor changed")
    parent = Path(config["parent_run"])
    meta = json.loads((run/"predictions.meta.json").read_text())
    identity = run_identity_sha256(digest(config), config["started_at"], runtime)
    if (meta["run_identity_sha256"] != identity or meta["config_hash"] != digest(config)
            or meta["prediction_sha256"] != sha256_file(run/"predictions.parquet")):
        raise ValueError("median residual prediction identity changed")
    panel, prior = (pd.read_parquet(path/"predictions.parquet") for path in (run, parent))
    pd.testing.assert_frame_equal(panel[panel.model_name.isin(prior.model_name.unique())].reset_index(drop=True),
        prior.reset_index(drop=True), check_exact=True)
    with np.load(config["mask_path"], allow_pickle=False) as saved:
        split = {role: saved[role].copy() for role in ("train", "val", "test", "context")}
    dataset = torch.load(config["dataset_path"], weights_only=False, map_location="cpu")
    truth = np.asarray(dataset["y"], dtype=np.float64)
    months = truth.shape[1]
    with np.load(run/"source_oof.npz", allow_pickle=False) as saved:
        oof = saved["pred_z"].copy()
    source_mask = np.zeros(truth.size, bool)
    source_mask[split["train"]] = True
    if not np.isfinite(oof.ravel()[source_mask]).all() or not np.isnan(oof.ravel()[~source_mask]).all():
        raise ValueError("source OOF population is incorrect")
    for index, stations in enumerate(station_folds(split["train"], months, config["seed"])):
        prefix = f"median_fold{index}"
        verify_files(run, f"{prefix}_complete.json", config)
        record = json.loads((run/f"{prefix}.json").read_text())
        outer = fold_split(split, stations, months)
        np.testing.assert_array_equal(record["hidden_stations"], stations)
        np.testing.assert_array_equal(record["fitted_stations"], np.unique(outer["train"]//months))
        if (record["fitted_cells_sha256"] != digest(outer["train"].tolist())
                or sha256_file(record["cached_forest"]) != record["cached_forest_sha256"]
                or not record["source_distribution_excludes_held_fold"]):
            raise ValueError("nested source concentration/forest roles changed")
        model = joblib.load(run/f"{prefix}.joblib")
        source_y = truth.ravel()[outer["train"]]
        np.testing.assert_array_equal(model.source_y_, source_y[np.argsort(source_y, kind="stable")])
        with np.load(run/f"{prefix}.npz", allow_pickle=False) as saved:
            held = split["train"][np.isin(split["train"]//months, stations)]
            np.testing.assert_array_equal(saved["cell"], held)
            with threadpool_limits(limits=2):
                np.testing.assert_array_equal(np.log1p(model.predict(saved["features"])), saved["pred_z"])
            np.testing.assert_array_equal(oof.ravel()[held], saved["pred_z"])
    with np.load(run/"validation_inputs.npz", allow_pickle=False) as saved:
        inputs = {key: saved[key].copy() for key in ("raw", "env", "age", "support", "extra")}
        context, cells = saved["context"].copy(), saved["cell"].copy()
    np.testing.assert_array_equal(cells, split["val"])
    ids = np.unique(cells//months)
    with np.load(parent/"validation_inputs.npz", allow_pickle=False) as saved:
        for key in ("raw", "env", "age", "support"):
            np.testing.assert_array_equal(inputs[key], saved[key])
        np.testing.assert_array_equal(inputs["extra"][..., :28], saved["extra"][..., :28])
        np.testing.assert_array_equal(inputs["extra"][..., 30:], saved["extra"][..., 30:])
    c = np.zeros(truth.shape)
    c[ids] = context
    extra = build_regime_head_features(dataset["regime"], split["train"], np.expm1(oof.ravel()[split["train"]]),
        c, build_causal_flow_features(dataset)["full"], n_months=months)
    if extra["normalization"] != json.loads((run/"readout_normalization.json").read_text()):
        raise ValueError("readout source-only normalization does not recompute")
    np.testing.assert_array_equal(inputs["extra"][..., :30], extra["full_extra"][ids])
    payload = torch.load(run/"native.pt", weights_only=False, map_location="cpu")
    old_payload = torch.load(parent/"native.pt", weights_only=False, map_location="cpu")
    if payload["config"] != old_payload["config"]:
        raise ValueError("median neural architecture or training settings changed")
    for name in ("spatial", "temporal", "decay"):
        for key, value in old_payload[f"initial_{name}"].items():
            torch.testing.assert_close(payload[f"initial_{name}"][key], value, rtol=0, atol=0)
    model = EncoderNativeResidual.from_payload(payload)
    prediction = model.predict(inputs, context)
    native = prediction[np.searchsorted(ids, cells//months), cells % months]
    np.testing.assert_array_equal(native, panel[panel.model_name.eq("leaf_median_residual")].y_pred)
    np.testing.assert_array_equal(context.ravel()[np.searchsorted(ids, cells//months)*months+cells % months],
        panel[panel.model_name.eq("leaf_median_reference")].y_pred)
    reference_panel = pd.read_parquet(Path(config["reference_run"])/"predictions.parquet")
    np.testing.assert_array_equal(panel[panel.model_name.eq("leaf_median_reference")].y_pred,
        reference_panel[reference_panel.model_name.eq("leaf_conditional_median")].y_pred)
    memory = EcologicalResidualTransfer.from_dict(json.loads((run/"memory.json").read_text()))
    p = c.copy()
    p[ids] = prediction
    integrated = memory.predict(c, p).ravel()[cells]
    np.testing.assert_array_equal(integrated, panel[panel.model_name.eq("leaf_median_integrated")].y_pred)
    for _, group in panel.groupby("model_name"):
        np.testing.assert_array_equal(group.cell, cells)
    if set(panel.model_name) != set(config["models"]) or not panel.visibility_role.eq("val").all() or not np.isfinite(panel.y_pred).all():
        raise ValueError("invalid matched median neural source panel")
    return {"run": run.name, "identity": True, "native_and_integrated_replay": "bitwise",
        "retained_initial_backbone_and_settings": True, "station_hidden_oof_medians": True,
        "source_only_normalization": True, "unchanged_raw_information": True,
        "unchanged_parent_predictions": True, "old_geographical_or_external_evaluation": False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--bootstrap-draws", type=int, default=5000)
    args = parser.parse_args()
    torch.set_num_threads(2)
    runtime = verify_runtime_snapshot(args.root)
    complete = sorted((args.root/"runs").glob("*/complete.json"))
    if {path.parent.name for path in complete} != {f"split{s}_seed{seed}" for s in (142, 143, 144) for seed in (42, 43, 44)}:
        raise ValueError("complete all nine median neural source packages")
    write_json(args.root/"verification/replay.json", [verify(path.parent, runtime) for path in complete])
    panel, thresholds = load_panel(args.root)
    runs, partitions, summary = metric_summary(panel, thresholds)
    output = args.root/"analysis"
    output.mkdir(exist_ok=True)
    for name, frame in (("run_metrics", runs), ("partition_metrics", partitions), ("summary", summary)):
        frame.to_csv(output/f"{name}.csv", index=False)
    comparisons = [(candidate, reference) for candidate in ARMS for reference in
        ("unmonitored_integrated", "station_hidden_trees", "current_model")]
    comparisons += [("leaf_median_residual", "unmonitored_residual"),
        ("leaf_median_residual", "leaf_median_reference"), ("leaf_median_integrated", "leaf_median_reference")]
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
        "analyzer_sha256": sha256_file(__file__), "weighting": "seed means then three equal source partitions",
        "runs": {path.parent.name: sha256_file(path) for path in complete},
        "numerical_outputs": {path.name: sha256_file(path) for path in output.glob("*.csv")}})
    print(summary[["model_name", "mae", "signed_bias", "q90_mae"]].to_string(index=False), flush=True)
    print(effects[effects.region.eq("overall")][["candidate", "reference", "relative_gain_pct",
        "gain_ci_low_pct", "gain_ci_high_pct", "improved_partitions"]].to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
