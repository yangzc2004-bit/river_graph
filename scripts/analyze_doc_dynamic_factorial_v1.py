"""Paired input/loss effects and exploratory HUC4-blocked temporal uncertainty."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from report_doc_dynamic_reconstruction_v1 import (
    METRICS,
    metrics_from_statistics,
    sufficient_statistics,
)

from river_graph.experiments.provenance import sha256_file

ROOT = Path("experiments/phase4_transfer/doc_dynamic_reconstruction_v1")
ARMS = ("monthly_log", "daily_log", "monthly_joint", "daily_joint")
CONTRASTS = {
    "daily_with_original_loss": {"daily_log": 1, "monthly_log": -1},
    "daily_with_joint_loss": {"daily_joint": 1, "monthly_joint": -1},
    "joint_with_original_inputs": {"monthly_joint": 1, "monthly_log": -1},
    "joint_with_daily_inputs": {"daily_joint": 1, "daily_log": -1},
    "combined_vs_original": {"daily_joint": 1, "monthly_log": -1},
    "input_main_effect": {"daily_log": .5, "daily_joint": .5, "monthly_log": -.5, "monthly_joint": -.5},
    "loss_main_effect": {"monthly_joint": .5, "daily_joint": .5, "monthly_log": -.5, "daily_log": -.5},
    "interaction": {"daily_joint": 1, "daily_log": -1, "monthly_joint": -1, "monthly_log": 1},
}


def main():
    audit = json.loads((ROOT / "analysis/independent_audit.json").read_text())
    assert audit["passed"] and (ROOT / "batch_complete.json").exists()
    protocol = json.loads((ROOT / "protocol.json").read_text())
    by_seed = pd.read_csv(ROOT / "analysis/by_seed.csv")
    differences = []
    for (track, gap, seed), part in by_seed.groupby(["track", "gap_months", "seed"]):
        values = part.set_index("model_name")
        for name, weights in CONTRASTS.items():
            for metric in METRICS:
                differences.append({"track": track, "gap_months": gap, "seed": seed,
                                    "contrast": name, "metric": metric,
                                    "difference": sum(values.loc[arm, metric]*weight for arm, weight in weights.items())})
    differences = pd.DataFrame(differences)
    effect = differences.groupby(["track", "gap_months", "contrast", "metric"], as_index=False).agg(
        difference=("difference", "mean"), positive_repetitions=("difference", lambda x: int((x > 0).sum())),
        negative_repetitions=("difference", lambda x: int((x < 0).sum())), repetitions=("difference", "size"))
    nodes_path = Path("data/processed/graph_nodes_graphfix_st357.csv")
    assert sha256_file(nodes_path) == protocol["nodes_sha256"]
    nodes = pd.read_csv(nodes_path, dtype={"site_no": str, "huc_cd": str})
    assert nodes.huc_cd.str.fullmatch(r"\d+").all()
    widths = nodes.huc_cd.str.len().map({7: 8, 8: 8, 11: 12, 12: 12})
    assert widths.notna().all(), "Expected HUC8 or HUC12, optionally missing a leading zero"
    # The CSV mixes HUC8 and HUC12 and has lost some leading zeroes.
    # Restore each full code before selecting HUC4; never use station IDs.
    nodes["huc4"] = [code.zfill(int(width))[:4] for code, width in zip(nodes.huc_cd, widths)]
    data = torch.load(protocol["dataset"], weights_only=False, map_location="cpu")
    site_huc = nodes.set_index("site_no").huc4
    mapping = dict(enumerate(site_huc.loc[list(data["site_no"])].tolist()))
    frames = []
    for task in protocol["tasks"]:
        if task["track"] != "time" or task["arm"] not in ARMS:
            continue
        frame = pd.read_parquet(ROOT / "fits" / task["id"] / "predictions.parquet")
        frame["huc4"] = frame.station.map(mapping)
        assert frame.huc4.notna().all()
        frames.append(frame)
    all_frames = pd.concat(frames, ignore_index=True)
    basins = sorted(all_frames.huc4.unique())
    draws = np.random.default_rng(20261012).multinomial(len(basins), np.repeat(1/len(basins), len(basins)), size=10000)
    stat_rows = []
    for (gap, seed, arm, basin), part in all_frames.groupby(["gap_months", "seed", "model_name", "huc4"]):
        stat_rows.append({"gap_months": gap, "seed": seed, "arm": arm, "huc4": basin,
                          **sufficient_statistics(part)})
    stats = pd.DataFrame(stat_rows)
    columns = [c for c in stats if c not in ("gap_months", "seed", "arm", "huc4")]
    intervals = []
    gap_estimates = []
    for gap in (3, 6, 12):
        estimated = {}
        for arm in ARMS:
            repetitions = []
            for seed in (42, 43, 44):
                part = stats[(stats.gap_months == gap) & (stats.seed == seed) & (stats.arm == arm)]
                matrix = part.set_index("huc4")[columns].reindex(basins, fill_value=0)
                totals = draws @ matrix.to_numpy(float)
                repetitions.append(metrics_from_statistics({key: totals[:, i] for i, key in enumerate(columns)}))
                exact = metrics_from_statistics(matrix.sum().to_dict())
                reference = by_seed[(by_seed.track == "time") & (by_seed.gap_months == gap)
                                    & (by_seed.seed == seed) & (by_seed.model_name == arm)].iloc[0]
                for metric in METRICS:
                    assert abs(exact[metric]-reference[metric]) < 1e-6, (gap, arm, seed, metric)
            estimated[arm] = {metric: np.mean([r[metric] for r in repetitions], axis=0) for metric in METRICS}
        gap_estimates.append(estimated)
        for name, weights in CONTRASTS.items():
            for metric in METRICS:
                delta = sum(estimated[arm][metric]*weight for arm, weight in weights.items())
                assert np.isfinite(delta).all(), (gap, name, metric)
                lower, upper = np.quantile(delta, [.025, .975])
                point = effect[(effect.track == "time") & (effect.gap_months == gap)
                               & (effect.contrast == name) & (effect.metric == metric)].iloc[0]
                intervals.append({"gap_months": gap, "contrast": name, "metric": metric,
                                  "difference": point.difference, "lower_95": lower,
                                  "upper_95": upper, "huc4_clusters": len(basins), "draws": len(draws)})
    # Gap zero denotes the equal-task mean of three declared gap scores, not
    # pooling or treating the repeated scored cores as new observations.
    average = {arm: {metric: np.mean([e[arm][metric] for e in gap_estimates], axis=0)
                     for metric in METRICS} for arm in ARMS}
    for name, weights in CONTRASTS.items():
        for metric in METRICS:
            delta = sum(average[arm][metric]*weight for arm, weight in weights.items())
            lower, upper = np.quantile(delta, [.025, .975])
            point = effect[(effect.track == "time") & (effect.contrast == name) & (effect.metric == metric)].difference.mean()
            intervals.append({"gap_months": 0, "contrast": name, "metric": metric,
                              "difference": point, "lower_95": lower, "upper_95": upper,
                              "huc4_clusters": len(basins), "draws": len(draws)})
    out = ROOT / "analysis"
    differences.to_csv(out / "factorial_effects_by_seed.csv", index=False)
    effect.to_csv(out / "factorial_effects.csv", index=False)
    stats.to_csv(out / "temporal_huc4_sufficient_statistics.csv", index=False)
    pd.DataFrame(intervals).to_csv(out / "temporal_paired_bootstrap.csv", index=False)
    report = {"script_sha256": sha256_file(__file__),
              "protocol_sha256": sha256_file(ROOT / "protocol.json"),
              "audit_sha256": sha256_file(ROOT / "analysis/independent_audit.json"),
              "nodes_sha256": sha256_file(nodes_path), "huc4_clusters": len(basins),
              "method": "10,000 paired resamples of HUC4 basins, identical cluster weights for every recipe, gap and fixed mask/optimization seed. Pool cell-level sufficient statistics within each repetition and average the three repetitions. Station demeaning is evaluation-only.",
              "limits": "Exploratory conditional uncertainty on a reused development cohort. Fixed and overlapping query cores are preserved, not treated as independent replicates. Basin connectivity and shared training data limit independence. No multiplicity correction or population-wide superiority claim.",
              "selection_modified": False,
              "huc_normalization": "Read huc_cd as text, restore HUC8/HUC12 leading zeroes by code length, then take HUC4. This post-analysis parser was repaired before its first successful run; frozen runs and masks were unchanged.",
              "outputs": {name: sha256_file(out / name) for name in ("factorial_effects_by_seed.csv", "factorial_effects.csv", "temporal_huc4_sufficient_statistics.csv", "temporal_paired_bootstrap.csv")}}
    (out / "factorial_analysis_provenance.json").write_text(json.dumps(report, indent=2)+"\n")
    print(json.dumps({"passed": True, "huc4_clusters": len(basins), "contrasts": len(CONTRASTS)}))


if __name__ == "__main__":
    main()
