"""Compare relative source departures with native attention and matched controls."""
from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd
import torch
from analyze_doc_chemistry_confirmation_v1 import metric_summary
from analyze_doc_source_retrieval_v1 import load_panel, paired
from analyze_unified_doc_spatial import joint_station_bootstrap
from run_doc_relative_source_attention_v1 import ARMS, ROOT
from run_unified_doc_spatial import verify_runtime_snapshot, write_json
from verify_doc_relative_source_attention_v1 import verify

from river_graph.experiments.provenance import sha256_file


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=type(ROOT), default=ROOT)
    parser.add_argument("--bootstrap-draws", type=int, default=5000)
    args = parser.parse_args()
    torch.set_num_threads(2)
    runtime = verify_runtime_snapshot(args.root)
    complete = sorted((args.root/"runs").glob("*/complete.json"))
    expected = {f"split{s}_seed{seed}" for s in (142, 143, 144) for seed in (42, 43, 44)}
    if {p.parent.name for p in complete} != expected:
        raise ValueError("complete all nine source attention packages before analysis")
    replay = [verify(p.parent, runtime) for p in complete]
    write_json(args.root/"verification/replay.json", replay)
    panel, thresholds = load_panel(args.root)
    runs, partitions, summary = metric_summary(panel, thresholds)
    out = args.root/"analysis"
    out.mkdir(exist_ok=True)
    for name, frame in (("run_metrics", runs), ("partition_metrics", partitions), ("summary", summary)):
        frame.to_csv(out/f"{name}.csv", index=False)
    comparisons = [(candidate, reference) for candidate in ARMS for reference in
                   ("unmonitored_integrated", "station_hidden_trees")]
    comparisons += [("relative_real_integrated", reference) for reference in
                    ("relative_historical_integrated", "relative_fixed_integrated",
                     "innovation_real_integrated", "innovation_real_trees", "attention_real_integrated")]
    comparisons += [("relative_real_native", reference) for reference in
                    ("relative_historical_native", "relative_fixed_native", "innovation_real_native", "unmonitored_residual", "attention_real_native")]
    effects = []
    for candidate, reference in comparisons:
        pair = paired(panel, candidate, reference)
        for region in ("overall", "q90"):
            selected = pair
            if region == "q90":
                q = np.array([thresholds[(s, seed)] for s, seed in zip(pair.split_seed, pair.seed, strict=True)])
                selected = pair[pair.y_true_candidate.to_numpy() >= q]
            directions = selected.groupby(["split_seed", "seed"])[["candidate_error", "reference_error"]].mean()
            partitions_direction, seed_direction = directions.groupby("split_seed").mean(), directions.groupby("seed").mean()
            effects.append({"candidate": candidate, "reference": reference, "region": region,
                **joint_station_bootstrap(selected, draws=args.bootstrap_draws),
                "improved_packages": int((directions.candidate_error < directions.reference_error).sum()),
                "improved_partitions": int((partitions_direction.candidate_error < partitions_direction.reference_error).sum()),
                "improved_training_seeds": int((seed_direction.candidate_error < seed_direction.reference_error).sum())})
    pd.DataFrame(effects).to_csv(out/"paired_effects.csv", index=False)
    panel["absolute_error"], panel["signed_error"] = np.abs(panel.y_pred-panel.y_true), panel.y_pred-panel.y_true
    station_metrics = panel.groupby(["split_seed", "seed", "model_name", "station"], as_index=False).agg(
        mae=("absolute_error", "mean"), bias=("signed_error", "mean"), n_cells=("cell", "size"))
    station_metrics.to_csv(out/"station_metrics.csv", index=False)
    station_metrics.groupby(["split_seed", "seed", "model_name"])[["mae", "bias"]].mean()\
        .groupby(["split_seed", "model_name"]).mean().groupby("model_name").mean()\
        .rename(columns={"mae": "station_equal_mae", "bias": "station_equal_bias"})\
        .to_csv(out/"station_equal_summary.csv")
    donor_rows, strata_rows = [], []
    for path in complete:
        run = path.parent
        config = json.loads((run/"config.json").read_text())
        with np.load(run/"attention_candidates.npz", allow_pickle=False) as candidates:
            with np.load(config["mask_path"], allow_pickle=False) as masks:
                val = masks["val"].copy()
            t = candidates["validation_donor_valid"].shape[1]
            ids = np.unique(val//t)
            index = np.searchsorted(ids, val//t), val % t
            valid = candidates["validation_donor_valid"][index]
            counts = valid[:, :-1].sum(-1)
            owner = candidates["validation_donor_owner"][index[0], :-1]
            bank = candidates["validation_donor_hydro_bank"]
            hydro = bank[owner.clip(min=0), index[1][:, None]]
            # First three daily columns are values; remaining five encode support.
            hydro_available = (hydro[..., 3:] > 0).any(-1) & valid[:, :-1]
            hydrology_support = hydro_available.sum(-1)
        group = panel[panel.split_seed.eq(config["split_seed"]) & panel.seed.eq(config["seed"])]
        descriptor = pd.DataFrame({"cell": val, "source_support_count": counts,
                                   "source_hydro_support_count": hydrology_support})
        descriptor["support_group"] = np.select([counts == 0, counts <= 3], ["none", "1-3"], default="4+")
        descriptor["hydro_group"] = np.where(hydrology_support > 0, "donor_hydro_present", "donor_hydro_absent")
        with np.load(config["parent_run"]+"/validation_inputs.npz", allow_pickle=False) as base_saved:
            reference = base_saved["context"][index]
        descriptor["reference_group"] = np.array(["low", "middle", "high"])[
            np.searchsorted(config["source_reference_tertiles"], reference, side="right")]
        for model_name, model_group in group.groupby("model_name"):
            joined = model_group.drop(columns=["source_support_count"], errors="ignore").merge(descriptor, on="cell", validate="one_to_one")
            for feature in ("support_group", "hydro_group", "reference_group"):
                rows = joined.groupby(feature, as_index=False).agg(mae=("absolute_error", "mean"),
                    bias=("signed_error", "mean"), n_cells=("cell", "size"), n_stations=("station", "nunique"))
                rows["stratum"], rows["model_name"] = feature, model_name
                rows["split_seed"], rows["seed"] = config["split_seed"], config["seed"]
                rows = rows.rename(columns={feature: "group"})
                strata_rows.append(rows)
            if model_name in ARMS:
                donor_rows.append({"run": run.name, "model_name": model_name,
                    "prior_mass": joined.attention_prior_mass.mean(), "entropy": joined.attention_entropy.mean(),
                    "source_support_mean": counts.mean(), "source_hydro_support_mean": hydrology_support.mean(),
                    "source_support_fraction": float((counts > 0).mean())})
    pd.concat(strata_rows, ignore_index=True).to_csv(out/"support_hydro_strata.csv", index=False)
    pd.DataFrame(donor_rows).to_csv(out/"attention_diagnostics.csv", index=False)
    write_json(out/"sources.json", {"evaluation_role": "source_validation_selected_development",
        "bootstrap_draws": args.bootstrap_draws, "analyzer_sha256": sha256_file(__file__),
        "aggregation": "seed then equal source partitions; joint station-cluster resampling",
        "hydro_group": "any of five existing daily support descriptors positive at matched source/current month",
        "runs": {p.parent.name: sha256_file(p) for p in complete},
        "csvs": {p.name: sha256_file(p) for p in out.glob("*.csv")}})
    print(summary[["model_name", "mae", "signed_bias", "q90_mae"]].to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
