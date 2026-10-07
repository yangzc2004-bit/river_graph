"""Controlled routing on actual whole-network and monitored-tributary paths."""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd

from river_graph.analysis.river_form_process import cluster_mean
from river_graph.analysis.river_morphology_effect import (
    group_descriptor_summary,
    paired_response_summary,
)
from river_graph.analysis.river_routing_mechanisms import (
    contract_paths,
    distribute_reach_inputs,
    normalized_inputs,
    route_pulse,
    two_branch_process,
)
from river_graph.experiments.provenance import sha256_file

ROOT = Path("experiments/phase4_transfer/doc_river_routing_mechanisms_v1")
MORPH = Path("experiments/phase4_transfer/doc_river_morphology_effect_v1/analysis")
INVENTORY = Path("experiments/phase4_transfer/doc_river_mechanisms_v1/analysis/confluence_inventory.csv")
REPS = Path("experiments/phase4_transfer/doc_river_planform_typology_v1/analysis/representatives.csv")
ROUTING = Path("data/raw/river_source_placement_v1/routing")
METRICS = ("pulse_peak", "pulse_centroid", "pulse_sd", "travel_delay_sd",
           "anomaly_mass_fraction", "steady_doc", "period_1_gain", "period_4_gain", "period_12_gain")
PROCESSES = {
    "conservative_uniform": (0., 0., 0.),
    "conservative_flow_speed": (1/3, 0., 0.),
    "conservative_flow_speed_half": (.5, 0., 0.),
    "uniform_processing": (0., .1, .1),
    "common_processing": (0., .1, .8),
}
FORCING = {"synchronous": 0., "branch_b_leads": -.3, "branch_b_lags": .3}


def read_context():
    f = pd.read_csv(MORPH/"station_morphology_doc_panel.csv", dtype={"station": str, "huc4": str})
    columns = ["station", "comid", "cluster", "huc4", "basin_area_km2", "n_reaches",
               "mainstem_share", "confluence_position", "tributary_balance", "route_distance_cv"]
    panel = f[columns].copy()
    pairs = pd.read_csv(MORPH/"covariate_selected_pairs.csv", dtype={"station_a": str, "station_b": str, "huc4": str})
    reps = pd.read_csv(REPS, dtype={"station": str})
    reps = reps[reps.station.isin(panel.station)].sort_values(["cluster", "rank"]).groupby("cluster").head(1)
    if reps.cluster.nunique() != 3:
        raise ValueError("fixed representative list must cover three classes")
    inventory = pd.read_csv(INVENTORY, dtype={"source_a": str, "source_b": str, "target": str, "huc4": str})
    candidates = inventory[inventory.eligible_monthly].merge(panel[["station", "basin_area_km2"]],
                        left_on="target", right_on="station", how="left", validate="many_to_one")
    ledger = candidates[["pair_id", "target", "huc4", "cluster", "basin_area_km2"]].copy()
    ledger["included"] = candidates.basin_area_km2.notna()
    candidates = candidates[candidates.basin_area_km2.notna()].copy()
    candidates["path_imbalance"] = abs(candidates.source_a_receiver_km-candidates.source_b_receiver_km)/np.sqrt(candidates.basin_area_km2)
    ordered = candidates.sort_values(["path_imbalance", "pair_id"])
    representative_pair = ordered.pair_id.iloc[len(ordered)//2]
    return panel, pairs, reps, candidates, ledger, representative_pair


def whole_network_experiment(panel, pairs, reps):
    vaa_path = Path("cache/nldplus_vaa.parquet")
    vaa = pd.read_parquet(vaa_path, columns=["comid", "areasqkm", "lengthkm"]).set_index("comid")
    areas = vaa.areasqkm
    rows, traces, convergence, resolution, files = [], [], [], [], [vaa_path]
    tested = set(pairs[pairs.class_a.eq(1) & pairs.class_b.eq(3)].station_a) | set(pairs[pairs.class_a.eq(1) & pairs.class_b.eq(3)].station_b)
    for station in panel.itertuples():
        path = ROUTING/f"comid_{int(station.comid)}.npz"
        files.append(path)
        with np.load(path) as z:
            local_area = areas.reindex(z["comids"]).to_numpy()
            distances = z["distance_km"]
            w, delay, coverage = normalized_inputs(local_area, distances, station.basin_area_km2)
            good = (local_area > 0) & np.isfinite(distances) & (distances >= 0)
            lengths = vaa.lengthkm.reindex(z["comids"]).to_numpy()[good]/np.sqrt(station.basin_area_km2)
        for name, factor in (("actual_paths", 1.), ("half_spread", .5), ("zero_spread", 0.)):
            d = contract_paths(delay, w, factor)
            result, time, pulse = route_pulse(d, w)
            rows.append({"station": station.station, "cluster": station.cluster, "huc4": station.huc4,
                         "scenario": name, "represented_area_fraction": coverage, "n_weighted_sources": len(w), **result})
            if station.station in set(reps.station):
                traces.append(pd.DataFrame({"station": station.station, "cluster": station.cluster,
                                            "scenario": name, "time": time, "outlet_anomaly": pulse}))
            if station.station in tested:
                fine, _, _ = route_pulse(d, w, dt=.0025)
                convergence.append({"station": station.station, "scenario": name, "peak_dt_005": result["pulse_peak"],
                                    "peak_dt_0025": fine["pulse_peak"], "absolute_peak_difference": abs(result["pulse_peak"]-fine["pulse_peak"])})
        distributed, dw = distribute_reach_inputs(delay, w, lengths)
        metrics, _, _ = route_pulse(distributed, dw)
        resolution.append({"station": station.station, "cluster": station.cluster, "huc4": station.huc4,
                           "scenario": "five_points_per_reach", "represented_area_fraction": coverage,
                           "n_weighted_sources": len(dw), **metrics})
    return pd.DataFrame(rows), pd.concat(traces, ignore_index=True), pd.DataFrame(convergence), pd.DataFrame(resolution), files


def tributary_experiment(candidates, representative_pair):
    rows, traces, diagnostics = [], [], []
    for p in candidates.itertuples():
        scale = np.sqrt(p.basin_area_km2)
        paths = np.array([p.source_a_receiver_km, p.source_b_receiver_km])/scale
        observed = p.junction_receiver_km/scale
        if observed > paths.min()+1e-8 or observed < 0:
            raise ValueError(f"invalid monitored path partition: {p.pair_id}")
        observed = min(observed, paths.min())
        junctions = {"observed": observed, "short_common": .2*paths.min(), "long_common": .8*paths.min()}
        actual_weight = p.source_a_area_km2/(p.source_a_area_km2+p.source_b_area_km2)
        for process, (exponent, branch_rate, common_rate) in PROCESSES.items():
            for weighting, weight in (("area_proxy", actual_weight), ("balanced", .5)):
                for forcing, offset in FORCING.items():
                    for junction, common in junctions.items():
                        result, time, pulse = two_branch_process(paths, common, weight, flow_exponent=exponent,
                                            branch_rate=branch_rate, common_rate=common_rate, offset_b=offset)
                        row = {"pair_id": p.pair_id, "target": p.target, "huc4": p.huc4,
                               "cluster": p.cluster, "component": p.component,
                               "process": process, "weighting": weighting, "forcing": forcing,
                               "junction": junction, "source_drainage_coverage": p.source_drainage_coverage, **result}
                        rows.append(row)
                        if p.pair_id == representative_pair and exponent != .5 and forcing == "synchronous":
                            traces.append(pd.DataFrame({"pair_id": p.pair_id, "process": process,
                                "weighting": weighting, "junction": junction, "time": time, "outlet_anomaly": pulse}))
        # The phase diagnostic aligns ARRIVAL times, not the source clock.
        result, _, _ = two_branch_process(paths, observed, actual_weight, offset_b=paths[0]-paths[1])
        diagnostics.append({"pair_id": p.pair_id, "target": p.target, "arrival_aligned_peak": result["pulse_peak"],
                            "arrival_aligned_mass": result["anomaly_mass_fraction"], "source_clock_shift_b": paths[0]-paths[1]})
    return pd.DataFrame(rows), pd.concat(traces, ignore_index=True), pd.DataFrame(diagnostics)


def summarize_whole(frame, panel, pairs, draws):
    class_rows, paired_rows, evidence = [], [], []
    for scenario, f in frame.groupby("scenario"):
        for result in (group_descriptor_summary(f, METRICS, draws=draws),):
            class_rows.append(result.assign(scenario=scenario))
        contrast, detail = paired_response_summary(pairs, f, dict.fromkeys(METRICS, "raw"), draws=draws)
        paired_rows.append(contrast.assign(scenario=scenario))
        evidence.append(detail.assign(scenario=scenario))
    actual = frame[frame.scenario.eq("actual_paths")].set_index("station")
    ablation = []
    for scenario in ("half_spread", "zero_spread"):
        other = frame[frame.scenario.eq(scenario)].set_index("station")
        s = panel[["station", "cluster", "huc4"]].copy().set_index("station")
        for metric in METRICS:
            s[metric] = actual[metric]-other[metric]
        for result in (group_descriptor_summary(s.reset_index(), METRICS, draws=draws),):
            ablation.append(result.assign(comparison="actual_minus_"+scenario))
    return {"whole_class_summary": pd.concat(class_rows, ignore_index=True),
            "whole_matched_contrasts": pd.concat(paired_rows, ignore_index=True),
            "whole_matched_evidence": pd.concat(evidence, ignore_index=True),
            "whole_path_ablation": pd.concat(ablation, ignore_index=True)}


def summarize_branches(frame, draws):
    rows = []
    comparisons = {
        "long_minus_short_common": ("junction", "long_common", "short_common"),
        "balanced_minus_area": ("weighting", "balanced", "area_proxy"),
        "leading_minus_synchronous": ("forcing", "branch_b_leads", "synchronous"),
        "lagging_minus_synchronous": ("forcing", "branch_b_lags", "synchronous"),
    }
    keys = ["pair_id", "target", "huc4", "cluster", "component", "process", "weighting", "forcing", "junction"]
    for name, (column, candidate, reference) in comparisons.items():
        group_keys = [k for k in keys if k not in ("pair_id", "target", "huc4", "cluster", "component", column)]
        # Weight/timing comparisons use the observed junction. Junction
        # comparison preserves each weighting and forcing combination.
        sub = frame if column == "junction" else frame[frame.junction.eq("observed")]
        a = sub[sub[column].eq(candidate)]
        b = sub[sub[column].eq(reference)]
        join = [k for k in keys if k != column]
        merged = a.merge(b, on=join, suffixes=("_a", "_b"), validate="one_to_one")
        for metric in METRICS:
            merged[metric] = merged[metric+"_a"]-merged[metric+"_b"]
        receivers = merged.groupby(["target", "huc4", "cluster", "component", *group_keys], as_index=False)[list(METRICS)].mean()
        for values, r in receivers.groupby(group_keys):
            labels = dict(zip(group_keys, values if isinstance(values, tuple) else (values,), strict=True))
            for unit in ("component", "huc4"):
                for metric in METRICS:
                    result = cluster_mean(r, metric, unit, draws=draws)
                    rows.append({"comparison": name, **labels, **result})
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bootstrap-draws", type=int, default=5000)
    args = parser.parse_args()
    out = ROOT/"analysis"
    out.mkdir(parents=True, exist_ok=True)
    panel, pairs, reps, candidates, ledger, representative = read_context()
    whole, traces, convergence, resolution, files = whole_network_experiment(panel, pairs, reps)
    branch, branch_traces, aligned = tributary_experiment(candidates, representative)
    for name, f in {"network_context": panel, "network_representatives": reps, "tributary_inventory": candidates,
                    "tributary_inclusion": ledger, "whole_network_routing": whole, "time_step_sensitivity": convergence,
                    "tributary_routing": branch, "arrival_alignment": aligned,
                    "reach_resolution_sensitivity": resolution,
                    "reach_resolution_class_summary": group_descriptor_summary(resolution, METRICS, draws=args.bootstrap_draws),
                    "reach_resolution_matched_contrasts": paired_response_summary(pairs, resolution, dict.fromkeys(METRICS, "raw"), draws=args.bootstrap_draws)[0],
                    **summarize_whole(whole, panel, pairs, args.bootstrap_draws),
                    "tributary_contrasts": summarize_branches(branch, args.bootstrap_draws)}.items():
        f.to_csv(out/f"{name}.csv", index=False)
    traces.to_parquet(out/"whole_representative_traces.parquet", index=False)
    branch_traces.to_parquet(out/"tributary_representative_traces.parquet", index=False)
    code = [Path("scripts/analyze_doc_river_routing_mechanisms_v1.py"),
            Path("src/river_graph/analysis/river_routing_mechanisms.py"),
            Path("src/river_graph/analysis/river_form_process.py"),
            Path("src/river_graph/analysis/river_morphology_effect.py")]
    for path in code:
        destination = ROOT/"code_snapshot"/path
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, destination)
    files += [MORPH/"station_morphology_doc_panel.csv", MORPH/"covariate_selected_pairs.csv", INVENTORY, REPS,
              ROOT/"study_plan.md", *code]
    record = {"source_hashes": {str(p): sha256_file(p) for p in files}, "bootstrap_draws": args.bootstrap_draws,
              "whole_network_count": len(panel), "tributary_pairs": len(candidates),
              "tributary_receivers": candidates.target.nunique(), "tributary_components": candidates.component.nunique(),
              "representative_pair": representative, "time_scale": "distance/sqrt(measured basin area), common unit velocity",
              "input": "same Gaussian anomaly SD=.15, same baseline=5, unit total flow; no landscape weights",
              "neural_training": False, "physical_rates_fitted": False,
              "reach_resolution_sensitivity": "five equal-weight midpoint-quadrature inputs per reach; added after initial routing readout; primary results unchanged",
              "max_resolution_peak_difference": convergence.absolute_peak_difference.max()}
    (ROOT/"analysis_sources.json").write_text(json.dumps(record, indent=2)+"\n")
    print(json.dumps({k: v for k, v in record.items() if k != "source_hashes"}, indent=2))


if __name__ == "__main__":
    main()
