"""Classify actual internal river operations before attaching DOC evidence."""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd

from river_graph.analysis.river_form_process import (
    cluster_mean,
    conditional_association,
)
from river_graph.analysis.river_internal_structure import (
    PROFILES,
    assign_profiles,
    classify_geometry,
    path_distribution,
    representatives,
    routing_scenarios,
)
from river_graph.experiments.provenance import sha256_file

ROOT = Path("experiments/phase4_transfer/doc_river_internal_structure_v1")
SHAPE = Path("experiments/phase4_transfer/doc_river_planform_typology_v1/analysis")
SIGNAL = Path("experiments/phase4_transfer/doc_river_signal_mechanisms_v1/analysis")
FLOW = Path("experiments/phase4_transfer/doc_river_form_flow_response_v1/analysis/station_flow_responses.csv")
ROUTES = Path("data/raw/river_source_placement_v1/routing")
VAA = Path("cache/nldplus_vaa.parquet")
CODE = tuple(map(Path, ("scripts/analyze_doc_river_internal_structure_v1.py",
    "src/river_graph/analysis/river_internal_structure.py", "src/river_graph/analysis/river_routing_mechanisms.py",
    "src/river_graph/analysis/river_form_process.py")))
OBS_METRICS = ("mixture_buffer_fraction", "equal_amplitude_buffer_fraction", "source_rho",
    "outlet_mixture_log_sd_ratio", "logscale_outlet_mixture_log_sd_ratio", "source_drainage_coverage")


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def build_geometry():
    frame = pd.read_csv(SHAPE/"classes.csv", dtype={"station": str, "huc_cd": str})
    frame = frame[frame.cluster.notna()].copy().reset_index(drop=True)
    area = pd.read_parquet(VAA, columns=["comid", "areasqkm"]).set_index("comid").areasqkm
    rows, distributions, files = [], {}, []
    for row in frame.itertuples():
        path = ROUTES/f"comid_{int(row.comid)}.npz"
        files.append(path)
        with np.load(path) as z:
            if len(z["comids"]) != row.n_reaches or len(np.unique(z["comids"])) != len(z["comids"]):
                raise ValueError("routing must represent the complete unique mapped network")
            w, d, metrics = path_distribution(area.reindex(z["comids"]).to_numpy(), z["distance_km"])
        rows.append(metrics)
        distributions[int(row.comid)] = (w, d)
    frame = pd.concat([frame, pd.DataFrame(rows)], axis=1)
    frame["huc4"] = frame.huc_cd.str[:4]
    return frame, distributions, files


def map_stations(profiles):
    stations = pd.read_csv(SHAPE/"station_classes.csv", dtype={"station": str, "huc_cd": str})
    return stations[["station", "comid"]].merge(
        profiles[["comid", "profile", "profile_name", "tributary_balance", "path_cv", "basin_area_km2"]],
        on="comid", how="inner", validate="many_to_one")


def observed_tables(profiles, draws):
    mapping = map_stations(profiles)
    source = pd.read_csv(SIGNAL/"mixing_receivers.csv", dtype={"target": str, "huc4": str})
    observed = source[["target", "huc4", "component", "cluster", "n_connections", "branch_balance", *OBS_METRICS]].merge(
        mapping, left_on="target", right_on="station", validate="one_to_one").drop(columns="station")
    if len(observed) != len(source):
        raise ValueError("keep every previously observed receiving station")
    connections = pd.read_csv(SIGNAL/"mixing_connections.csv", dtype={"target": str})
    w = connections.weight_a
    mean_path = w*connections.path_a_km+(1-w)*connections.path_b_km
    connections["gauged_pair_path_cv"] = np.sqrt(w*(1-w))*abs(connections.path_a_km-connections.path_b_km)/mean_path
    connections["gauged_pair_minor_share"] = np.minimum(w, 1-w)
    local = connections.groupby("target", as_index=False)[["gauged_pair_path_cv", "gauged_pair_minor_share"]].mean()
    observed = observed.merge(local, on="target", validate="one_to_one")
    summary = []
    for profile in range(5):
        f = observed[observed.profile.eq(profile)]
        if f.empty:
            for metric in OBS_METRICS:
                summary.append({"profile": profile, "metric": metric, "estimate": np.nan, "ci_low": np.nan,
                    "ci_high": np.nan, "unit": "component", "n_receivers": 0, "n_blocks": 0,
                    "interval_status": "no_observed_receivers"})
            continue
        for metric in OBS_METRICS:
            result = cluster_mean(f, metric, "component", draws=draws)
            summary.append({"profile": profile, **result})
    contrasts = []
    for first in range(1, 5):
        for second in range(first+1, 5):
            pair = observed[observed.profile.isin([first, second])].copy()
            pair["cluster"] = pair.profile.map({first: 1, second: 3})
            for metric in OBS_METRICS:
                result = cluster_mean(pair, metric, "component", draws=draws, contrast=True)
                if result is not None:
                    contrasts.append({"profile_a": first, "profile_b": second, **result})
    source_flow = pd.read_csv(FLOW, dtype={"station": str, "huc4": str})
    mapped_flow = source_flow.merge(mapping, on=["station", "comid"], validate="many_to_one")
    if len(mapped_flow) != len(source_flow):
        raise ValueError("keep the complete previously eligible flow population")
    flow_summary = mapped_flow.groupby(["population", "profile"], as_index=False).agg(
        n_stations=("station", "nunique"), n_huc4=("huc4", "nunique"),
        mean_adjusted_log_response=("adjusted_log_contrast", "mean"),
        median_adjusted_log_response=("adjusted_log_contrast", "median"),
        positive_station_fraction=("adjusted_log_contrast", lambda x: (x > 0).mean()),
        median_basin_area_km2=("basin_area_km2", "median"))
    observed["log_area"] = np.log1p(observed.basin_area_km2)
    associations, omitted = [], []
    for outcome in ("outlet_mixture_log_sd_ratio", "logscale_outlet_mixture_log_sd_ratio"):
        for focal, other in (("tributary_balance", "path_cv"), ("path_cv", "tributary_balance")):
            result, detail = conditional_association(observed, outcome, focal,
                ("log_area", "source_drainage_coverage", other), draws=draws)
            if result is not None:
                associations.append(result)
                detail["outcome"], detail["focal"] = outcome, focal
                omitted.append(detail)
    alignment = []
    for full, pair in (("tributary_balance", "gauged_pair_minor_share"), ("path_cv", "gauged_pair_path_cv")):
        alignment.append({"whole_network_descriptor": full, "gauged_pair_descriptor": pair,
            "spearman": observed[full].corr(observed[pair], method="spearman"),
            "n_receivers": len(observed), "n_components": observed.component.nunique()})
    return {"observed_receivers": observed, "observed_buffer_summary": pd.DataFrame(summary),
            "observed_profile_contrasts": pd.DataFrame(contrasts), "mapped_flow_responses": mapped_flow,
            "descriptive_flow_summary": flow_summary, "whole_pair_alignment": pd.DataFrame(alignment),
            "continuous_observed_associations": pd.DataFrame(associations),
            "association_omitted_systems": pd.concat(omitted, ignore_index=True)}


def summarize_geometry(profiles, routing):
    counts = pd.crosstab(profiles.cluster.astype(int), profiles.profile).reindex(index=[1, 2, 3], columns=range(5), fill_value=0)
    counts.index.name = "cluster"
    counts.columns = [f"profile_{i}" for i in range(5)]
    descriptive = profiles.groupby("profile", as_index=False).agg(
        profile_name=("profile_name", "first"), n_networks=("comid", "size"), n_huc4=("huc4", "nunique"),
        median_balance=("tributary_balance", "median"), median_path_cv=("path_cv", "median"),
        median_area_km2=("basin_area_km2", "median"), median_junctions=("n_junctions", "median"))
    summaries = routing.groupby(["profile", "scenario"], as_index=False).agg(
        n_networks=("comid", "size"), mean_peak=("pulse_peak", "mean"), median_peak=("pulse_peak", "median"),
        mean_pulse_sd=("pulse_sd", "mean"), mean_delay=("mean_travel_delay", "mean"),
        min_mass=("anomaly_mass_fraction", "min"), max_mass=("anomaly_mass_fraction", "max"))
    return {"outline_profile_counts": counts.reset_index(), "geometry_profile_summary": descriptive,
            "normalized_routing_summary": summaries}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bootstrap-draws", type=int, default=5000)
    args = parser.parse_args()
    ROOT.mkdir(parents=True, exist_ok=True)
    frame, distributions, files = build_geometry()
    profiles, thresholds = classify_geometry(frame)
    fixed = assign_profiles(frame, .25, .5)
    reps = representatives(profiles)
    config = {**thresholds, "geometry_networks": len(frame), "profiles": PROFILES,
        "fixed_reference_cut_points": {"balance_cut": .25, "path_cut": .5},
        "pulse_sd": .15, "mean_relative_delay": 1., "path_spread_fractions": [1., .5, 0.],
        "bootstrap_draws": args.bootstrap_draws, "bootstrap_seed": 42,
        "bootstrap_unit": "connected monitoring system", "previous_results_seen": True,
        "profile_selection": "geometry-only median splits; no DOC", "neural_training": False,
        "supplementary_after_profile_results": "scale alignment and continuous area/coverage-adjusted association",
        "association_controls": ["log_area", "source_drainage_coverage", "other structural axis"]}
    # Geometry and profile definitions are written before DOC is read.
    (ROOT/"config.json").write_text(json.dumps(config, indent=2)+"\n")
    out = ROOT/"analysis"
    out.mkdir(parents=True, exist_ok=True)
    profiles.to_csv(out/"network_profiles.csv", index=False)
    fixed.to_csv(out/"fixed_reference_profiles.csv", index=False)
    reps.to_csv(out/"representatives.csv", index=False)
    rows, traces = [], []
    for row in profiles.itertuples():
        result, trace = routing_scenarios(*distributions[int(row.comid)])
        result["comid"], result["profile"], result["cluster"] = row.comid, row.profile, int(row.cluster)
        result["station"], result["huc4"] = row.station, row.huc4
        rows.append(result)
        if row.comid in set(reps.comid):
            trace["comid"], trace["profile"] = row.comid, row.profile
            traces.append(trace)
    routing = pd.concat(rows, ignore_index=True)
    tables = {**summarize_geometry(profiles, routing), **observed_tables(profiles, args.bootstrap_draws)}
    # Fixed cut points remain a sensitivity with the same observation population.
    fixed_observed = observed_tables(fixed, args.bootstrap_draws)
    tables["fixed_reference_counts"] = summarize_geometry(fixed, routing)["outline_profile_counts"]
    tables["fixed_reference_observed_summary"] = fixed_observed["observed_buffer_summary"]
    for name, table in tables.items():
        table.to_csv(out/f"{name}.csv", index=False)
    routing.to_csv(out/"normalized_routing.csv", index=False)
    pd.concat(traces, ignore_index=True).to_parquet(out/"representative_pulses.parquet", index=False)
    sources = [SHAPE/"classes.csv", SHAPE/"station_classes.csv", VAA, SIGNAL/"mixing_receivers.csv", SIGNAL/"mixing_connections.csv", FLOW,
               ROOT/"study_plan.md", *files, *CODE]
    for path in CODE:
        destination = ROOT/"code_snapshot"/path
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, destination)
    record = {"source_hashes": {str(p): sha256_file(p) for p in sources}, "config_hash": digest(config),
              "n_networks": len(profiles), "n_observed_receivers": len(tables["observed_receivers"]),
              "n_flow_stations": tables["mapped_flow_responses"].station.nunique(),
              "previous_results_seen": True, "new_model_training": False}
    (ROOT/"analysis_sources.json").write_text(json.dumps(record, indent=2)+"\n")
    print(json.dumps(config, indent=2))
    print(tables["outline_profile_counts"].to_string(index=False))
    print(tables["normalized_routing_summary"].query("scenario == 'actual_spread'").to_string(index=False))
    print(tables["observed_buffer_summary"].query("metric == 'outlet_mixture_log_sd_ratio'").to_string(index=False))


if __name__ == "__main__":
    main()
