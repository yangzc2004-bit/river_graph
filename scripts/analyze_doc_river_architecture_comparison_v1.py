"""Audit and aggregate the frozen four-arm geographical architecture study."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from run_doc_river_architecture_comparison_v1 import (
    DATASET,
    MASKS,
    ROOT,
    digest,
    native_prediction,
    predict,
    verify_run,
    write_json,
)

from river_graph.experiments.provenance import run_identity_sha256, sha256_file
from river_graph.models.river_architecture_comparison import (
    ARMS,
    RiverArchitectureModel,
    covariate_inputs,
    graph_view,
)


def verify_package(root, region, seed, protocol, *, replay):
    run = root / "runs" / f"huc4_{region}_seed{seed}"
    config = json.loads((run / "config.json").read_text())
    if config["protocol_hash"] != digest(protocol):
        raise ValueError("run has different frozen protocol")
    verify_run(run, config)
    for kind in ("dataset", "mask"):
        if sha256_file(config[f"{kind}_path"]) != config[f"{kind}_hash"]:
            raise ValueError("run input changed")
    sidecar = json.loads((run / "predictions.meta.json").read_text())
    expected = {"config": config, "config_hash": digest(config),
        "dataset_sha256": config["dataset_hash"], "mask_sha256": config["mask_hash"],
        "runtime_snapshot_hash": config["runtime_snapshot_hash"],
        "prediction_sha256": sha256_file(run / "predictions.parquet"),
        "run_identity_sha256": run_identity_sha256(digest(config), config["started_at"],
                                                   config["runtime_snapshot_hash"])}
    if any(sidecar[key] != value for key, value in expected.items()):
        raise ValueError("prediction sidecar disagrees with identity")
    for name, expected_hash in sidecar["model_files"].items():
        if sha256_file(run / name) != expected_hash:
            raise ValueError("prediction sidecar has changed fitted product")
    frame = pd.read_parquet(run / "predictions.parquet")
    with np.load(MASKS / f"huc4_{region}.npz", allow_pickle=False) as saved:
        split = {key: saved[key].copy() for key in ("train", "val", "test", "context")}
    data = torch.load(DATASET, map_location="cpu", weights_only=False)
    t = data["y"].shape[1]
    if not np.isfinite(frame[["y_true", "y_pred"]].to_numpy()).all() or len(frame) != len(split["test"]) * 4:
        raise ValueError("incorrect prediction coverage")
    names, dates = np.asarray(data["site_no"], str), np.asarray(data["months"], str)
    if frame.model_name.nunique() != 4 or frame.duplicated(["model_name", "cell"]).any():
        raise ValueError("duplicated/missing arm cells")
    for arm in ARMS:
        part = frame[frame.model_name == arm]
        np.testing.assert_array_equal(part.cell, split["test"])
        np.testing.assert_array_equal(part.y_true, np.asarray(data["y"]).ravel()[split["test"]])
        np.testing.assert_array_equal(part.station, names[split["test"] // t])
        np.testing.assert_array_equal(part.month, dates[split["test"] % t])
    maximum_difference = 0.
    if replay:
        source = np.unique(split["train"] // t)
        receiver = np.unique(split["test"] // t)
        inputs = covariate_inputs({key: value for key, value in data.items() if key not in ("y", "y_mask")},
                                  source, lookback=protocol["settings"]["lookback"])
        if inputs["normalization"] != json.loads((run / "normalization.json").read_text()):
            raise ValueError("normalization replay differs")
        graph = graph_view(data["edge_index"], inputs["order"], np.sort(np.r_[source, receiver]))
        cells = split["test"]
        graph_rows = np.searchsorted(graph["rows"].numpy(), cells // t)
        for arm in ARMS:
            payload = torch.load(run / f"{arm}.pt", map_location="cpu", weights_only=False)
            model = RiverArchitectureModel(arm, **payload["architecture"])
            model.load_state_dict(payload["state"])
            info = json.loads((run / f"{arm}.json").read_text())
            grid = native_prediction(predict(model, inputs, graph, np.arange(t),
                                            protocol["settings"]["batch_months"]),
                                     info["target_normalization"])
            replayed = grid[cells % t, graph_rows]
            stored = frame.loc[frame.model_name == arm, "y_pred"].to_numpy()
            np.testing.assert_allclose(replayed, stored, atol=1e-6, rtol=1e-6)
            maximum_difference = max(maximum_difference, float(np.max(np.abs(replayed - stored))))
    return frame, json.loads((run / "timing.json").read_text()), maximum_difference


def metrics(part):
    error = part.y_pred.to_numpy() - part.y_true.to_numpy()
    denominator = ((part.y_true - part.y_true.mean()) ** 2).sum()
    station_error = part.assign(error=np.abs(error)).groupby("station").error.mean()
    return {"mae": float(np.abs(error).mean()), "rmse": float(np.sqrt((error ** 2).mean())),
            "r2": float(1 - (error ** 2).sum() / denominator),
            "bias": float(error.mean()), "station_macro_mae": float(station_error.mean()),
            "q90_mae": float(np.abs(error[part.high_doc.to_numpy()]).mean()),
            "q90_bias": float(error[part.high_doc.to_numpy()].mean()),
            "query_rows": len(part), "unique_cells": int(part.cell.nunique()),
            "stations": int(part.station.nunique())}


def paired_bootstrap(frame, group, baseline, arm, *, repeats=20000):
    """Average seeds first, then resample whole clusters, retaining pairing."""
    cell = frame.assign(error=np.abs(frame.y_pred - frame.y_true)).groupby(
        ["target_huc4", "station", "cell", "model_name"], as_index=False).error.mean()
    wide = cell.pivot(index=["target_huc4", "station", "cell"], columns="model_name", values="error").reset_index()
    clusters = wide.groupby(group).agg(base_sum=(baseline, "sum"), arm_sum=(arm, "sum"),
                                       count=("cell", "count"))
    rng = np.random.default_rng(260610)
    choices = rng.integers(0, len(clusters), size=(repeats, len(clusters)))
    base = clusters.base_sum.to_numpy()[choices].sum(1) / clusters["count"].to_numpy()[choices].sum(1)
    selected = clusters.arm_sum.to_numpy()[choices].sum(1) / clusters["count"].to_numpy()[choices].sum(1)
    gain = 100 * (base - selected) / base
    return {"clusters": len(clusters), "cluster_unit": group,
            "gain_percent_interval": np.quantile(gain, [.025, .975]).tolist()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--replay", action="store_true")
    args = parser.parse_args()
    protocol = json.loads((args.root / "protocol.json").read_text())
    torch.set_num_threads(protocol["settings"]["torch_threads"])
    if digest(protocol) != json.loads((args.root / "manifest.json").read_text())["protocol_hash"]:
        raise ValueError("frozen manifest differs")
    for name, expected in protocol["runtime_snapshot"].items():
        if sha256_file(args.root / "code_snapshot" / name) != expected:
            raise ValueError("execution snapshot differs")
        if args.replay and sha256_file(name) != expected:
            raise ValueError("replay runtime changed")
    all_frames, timing_rows, maximum_difference, receipts = [], [], 0., {}
    for region in protocol["regions"]:
        for seed in protocol["seeds"]:
            frame, timing, difference = verify_package(args.root, region, seed, protocol, replay=args.replay)
            all_frames.append(frame)
            maximum_difference = max(maximum_difference, difference)
            for arm, value in timing.items():
                timing_rows.append({"target_huc4": region, "seed": seed, "model_name": arm, **value})
            path = args.root / "runs" / f"huc4_{region}_seed{seed}" / "complete.json"
            receipts[str(path)] = sha256_file(path)
            print(f"audited {region} seed{seed}", flush=True)
    frame = pd.concat(all_frames, ignore_index=True)
    out = args.root / "analysis"
    out.mkdir(parents=True, exist_ok=True)
    per_run = []
    for (region, seed, arm), part in frame.groupby(["target_huc4", "seed", "model_name"]):
        per_run.append({"target_huc4": region, "seed": seed, "model_name": arm, **metrics(part)})
    pd.DataFrame(per_run).to_csv(out / "per_run_metrics.csv", index=False)
    per_basin = []
    for (region, arm), part in frame.groupby(["target_huc4", "model_name"]):
        per_basin.append({"target_huc4": region, "model_name": arm, **metrics(part)})
    basin = pd.DataFrame(per_basin)
    basin.to_csv(out / "per_basin_metrics.csv", index=False)
    timing = pd.DataFrame(timing_rows)
    timing.to_csv(out / "timing.csv", index=False)
    summary = []
    for arm, part in frame.groupby("model_name"):
        costs = timing[timing.model_name == arm]
        summary.append({"model_name": arm, **metrics(part),
            "basin_macro_mae": float(basin.loc[basin.model_name == arm, "mae"].mean()),
            "median_training_seconds": float(costs.training_seconds.median()),
            "total_training_seconds": float(costs.training_seconds.sum()),
            "median_inference_seconds": float(costs.inference_seconds.median()),
            "median_epochs": float(costs.epochs_run.median()), "max_epoch_best_fits": int(
                (costs.best_epoch == protocol["settings"]["max_epochs"]).sum()),
            "median_selected_epoch": float(costs.best_epoch.median()),
            "epoch1_selected_fits": int((costs.best_epoch == 1).sum()),
            "parameters": int(costs.parameters.iloc[0]), "coverage": 1.})
    summary = pd.DataFrame(summary)
    base_mae = float(summary.loc[summary.model_name == "local", "mae"].iloc[0])
    summary["gain_vs_local_percent"] = 100 * (base_mae - summary.mae) / base_mae
    summary.to_csv(out / "summary.csv", index=False)
    strata = []
    for arm, part in frame.groupby("model_name"):
        for name, lo, hi in (("orders1_3", 1, 3), ("orders4_6", 4, 6), ("orders7_10", 7, 10)):
            selected = part[part.stream_order.between(lo, hi)]
            if len(selected):
                strata.append({"model_name": arm, "stratum": name,
                               "mae": float(np.abs(selected.y_pred - selected.y_true).mean()),
                               "cells": int(selected.cell.nunique()),
                               "stations": int(selected.station.nunique())})
    pd.DataFrame(strata).to_csv(out / "stream_order_metrics.csv", index=False)
    pairs = []
    basin_wide = basin.pivot(index="target_huc4", columns="model_name", values="mae")
    for baseline, arm in (("local", "directed_gnn"), ("local", "hierarchical_gnn"),
                          ("local", "graph_transformer"), ("directed_gnn", "hierarchical_gnn"),
                          ("directed_gnn", "graph_transformer"), ("hierarchical_gnn", "graph_transformer")):
        base = float(summary.loc[summary.model_name == baseline, "mae"].iloc[0])
        current = float(summary.loc[summary.model_name == arm, "mae"].iloc[0])
        pairs.append({"baseline": baseline, "candidate": arm,
            "gain_percent": 100 * (base - current) / base,
            "basins_improved": int((basin_wide[arm] < basin_wide[baseline]).sum()),
            "basin_bootstrap": paired_bootstrap(frame, "target_huc4", baseline, arm),
            "station_bootstrap": paired_bootstrap(frame, "station", baseline, arm)})
    write_json(out / "paired_comparisons.json", pairs)
    winner = summary.loc[summary.mae.idxmin(), "model_name"]
    report = ["DOC river architecture comparison v1", "", f"Numerical primary winner: {winner}.",
              "Primary endpoint: equal-cell native DOC MAE, averaging seed errors, not ensemble predictions.",
              (f"Population: {frame.station.nunique()} stations and {frame.cell.nunique()} unique observed "
               "station-month cells, five whole-HUC4 regions, three seeds, 60 fresh fits."),
              ("All arms use the same causal 12-month GRU, ecological encoder, 12,565 nominal parameters, "
               "initialization, source-only scaling, optimization budget and validation rule."),
              ("Local self-only attention does not use its Q/K comparisons; nominal parameter matching "
               "does not establish identical effective capacity."), "",
              summary[["model_name", "mae", "basin_macro_mae", "rmse", "r2", "q90_mae",
                       "gain_vs_local_percent", "median_training_seconds", "median_selected_epoch"]].to_string(index=False),
              "", "Paired comparisons (positive gain means the candidate improves MAE):"]
    for pair in pairs:
        lo, hi = pair["basin_bootstrap"]["gain_percent_interval"]
        report.append(f"{pair['candidate']} vs {pair['baseline']}: {pair['gain_percent']:.3f}% gain; "
                      f"95% basin-cluster bootstrap [{lo:.3f}, {hi:.3f}]%; "
                      f"improved in {pair['basins_improved']}/5 regions.")
    report.extend(["", "Interpretation limits:",
        ("These are standalone small models on a sampled-station graph, not the released DOC ensemble "
         "or a reconstruction of all physical river reaches."),
        ("The spatial Transformer is a compact graph-biased implementation; the temporal module is a GRU "
         "in every arm. This does not rank all GNNs against all Transformers."),
        ("Target DOC is entirely withheld and never an input. Target temperature/discharge covariates "
         "remain available where observed; this is not a no-hydrology setting."),
        ("The five internal HUC4 tasks were inspected in prior project development. They are not "
         "independent external rivers or fresh blind confirmation."),
        ("Only five regions support the basin bootstrap; uncertainty is exploratory. Station-bootstrap "
         "intervals describe within-region sampling and do not replace basin-level transfer uncertainty."),
        ("Training and initial inference times can include concurrent CPU load. Error estimates are "
         "paired; timing differences also reflect validation-selected stopping epochs."),
        ("Protocol, masks, code copies, checkpoints and predictions have content hashes; audit.json "
         "states whether complete checkpoint replay was performed.")])
    (out / "research_decision.txt").write_text("\n".join(report) + "\n")
    write_json(out / "audit.json", {"protocol_hash": digest(protocol), "runs": len(receipts), "fits": 60,
        "unique_query_cells": int(frame.cell.nunique()), "query_stations": int(frame.station.nunique()),
        "observed_query_coverage": 1., "checkpoint_replay": args.replay,
        "maximum_replay_difference": maximum_difference, "config_and_product_hashes_verified": True,
        "analysis_script_sha256": sha256_file(__file__), "source_completion_hashes": receipts,
        "scope": protocol["scope"],
        "uncertainty": "seed-average per-cell errors; paired percentile cluster bootstrap; "
                       "only five basins, so basin-bootstrap inference is exploratory",
        "local_capacity_note": "identical nominal tensors; self-only attention has inactive Q/K "
                               "inter-node comparisons, so effective capacity is not identical"})
    write_json(out / "sources.json", {"analysis_script": {"path": __file__, "sha256": sha256_file(__file__)},
        "protocol": {"path": str(args.root / "protocol.json"), "sha256": sha256_file(args.root / "protocol.json")},
        "run_completions": receipts, "outputs": {str(path.name): sha256_file(path)
            for path in out.glob("*") if path.is_file() and path.name != "sources.json"}})
    print(summary[["model_name", "mae", "gain_vs_local_percent", "median_training_seconds"]].to_string(index=False))


if __name__ == "__main__":
    main()
