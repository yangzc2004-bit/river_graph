"""Real corridor timing x shared volume/flow x conservative response."""
from __future__ import annotations

import argparse
import json
import shutil
import time
from pathlib import Path

import pandas as pd

from river_graph.analysis.river_junction_dynamics import (
    FLOWS,
    MIXING,
    PHASES,
    VOLUMES,
    field_transit_proxies,
    junction_contrasts,
    junction_response,
)
from river_graph.analysis.river_storage_placement import blocked_summary
from river_graph.experiments.provenance import sha256_file

ROOT = Path("experiments/phase4_transfer/doc_river_junction_dynamics_v1")
OLD = Path("experiments/phase4_transfer/doc_river_storage_placement_v1/analysis")
FIELD = Path("experiments/phase4_transfer/doc_river_confluence_response_v1/analysis")
PAIRS = Path("experiments/phase4_transfer/doc_river_morphology_effect_v1/analysis/covariate_selected_pairs.csv")
IDENTIFIERS = {k: str for k in ("station", "case_id", "huc4", "component", "station_a", "station_b")}
SIGMAS = (.075, .15, .30)
DT = .0025
SETTINGS = ["flow_rule", "input_sd", "source_phase_factor", "volume_factor", "flow_factor", "mixing_fraction"]
METRICS = ["pulse_peak", "duration_80", "pulse_centroid", "joint_peak_reduction_pct",
           "phase_peak_reduction_pct", "mixing_peak_reduction_pct", "mixing_duration_change_pct",
           "volume_flow_peak_reduction_pct", "arrival_peak_reduction_from_aligned_pct"]
SOURCES = (OLD/"selected_footprints.csv", OLD/"junction_inventory.csv", OLD/"representatives.csv",
           PAIRS, FIELD/"fall_geometry_doc.csv", FIELD/"surveyed_transects.csv", ROOT/"study_plan.md")
CODE = tuple(Path(p) for p in (
    "scripts/analyze_doc_river_junction_dynamics_v1.py",
    "src/river_graph/analysis/river_junction_dynamics.py",
    "src/river_graph/analysis/river_storage_placement.py",
    "src/river_graph/analysis/river_storage_transport.py"))


def inputs():
    cases = pd.read_csv(OLD/"selected_footprints.csv", dtype=IDENTIFIERS)
    reps = pd.read_csv(OLD/"representatives.csv", dtype=IDENTIFIERS)
    # Keep three original form representatives; fallback was geometry-only in the parent.
    reps = reps.loc[reps.example_group.isin([1, 2, 3])].copy()
    maximum_common = cases.loc[cases.cohort.eq("whole_confluence")].sort_values(
        ["common_fraction", "case_id"], ascending=[False, True]).iloc[0].to_dict()
    examples = set(reps.case_id) | {maximum_common["case_id"]}
    pairs = pd.read_csv(PAIRS, dtype=IDENTIFIERS)
    pairs = pairs.loc[pairs.class_a.eq(1) & pairs.class_b.eq(3)].copy()
    survey = pd.read_csv(FIELD/"surveyed_transects.csv", dtype={"transect": str})
    chemistry = pd.read_csv(FIELD/"fall_geometry_doc.csv")
    field, proxies = field_transit_proxies(survey, chemistry)
    return cases, reps, examples, pairs, field, proxies


def simulate(cases, examples, *, dt=DT, keep_curves=True):
    rows, curves = [], []
    started = time.monotonic()
    for i, case in enumerate(cases.itertuples()):
        meta = {key: getattr(case, key) for key in ("cohort", "case_id", "station", "huc4", "component", "cluster")}
        for flow_rule, weight, sigma in [("area_proxy", case.weight_a, s) for s in SIGMAS]+[("equal_flow", .5, .15)]:
            cache = {}
            for phase in PHASES:
                for volume in VOLUMES:
                    for flow in FLOWS:
                        for mixing in MIXING:
                            key = (phase, volume/flow, mixing)
                            keep = keep_curves and case.case_id in examples and flow_rule == "area_proxy" and sigma == .15
                            if key not in cache:
                                cache[key] = junction_response(case.branch_a_km, case.branch_b_km,
                                    case.common_km, weight, phase=phase, volume=volume, flow=flow,
                                    mixing=mixing, sigma=sigma, dt=dt, keep_curve=keep)
                            result, trace = cache[key]
                            row = meta | result | {"volume_factor": volume, "flow_factor": flow,
                                                   "flow_rule": flow_rule, "weight_a": weight}
                            rows.append(row)
                            if keep:
                                tagged = trace.copy()
                                for k, v in meta.items():
                                    tagged[k] = v
                                for k in SETTINGS:
                                    tagged[k] = row[k]
                                curves.append(tagged)
        if i % 30 == 0 or i+1 == len(cases):
            print(f"Junctions {i+1}/{len(cases)}; {time.monotonic()-started:.1f}s", flush=True)
    return junction_contrasts(pd.DataFrame(rows)), pd.concat(curves, ignore_index=True) if curves else pd.DataFrame()


def summaries(frame, pairs, *, draws):
    cohort_rows, form_rows = [], []
    for key, selected in frame.groupby(["cohort"]+SETTINGS, sort=True):
        meta = dict(zip(["cohort"]+SETTINGS, key, strict=True))
        unit = selected.groupby(["station", "huc4", "component"], dropna=False)[METRICS].mean().reset_index()
        block = "huc4" if meta["cohort"] == "whole_confluence" else "component"
        report = blocked_summary(unit, METRICS, group=block, unit="station", draws=draws)
        cohort_rows.append(report.assign(**meta))
        for cluster, values in selected.groupby("cluster"):
            unit = values.groupby(["station", "huc4", "component"], dropna=False)[METRICS].mean().reset_index()
            report = blocked_summary(unit, METRICS, group=block, unit="station", draws=draws)
            form_rows.append(report.assign(**meta, cluster=int(cluster)))
    matched = []
    whole = frame.loc[frame.cohort.eq("whole_confluence")]
    for key, selected in whole.groupby(SETTINGS, sort=True):
        indexed = selected.set_index("station")
        valid = pairs.station_a.isin(indexed.index) & pairs.station_b.isin(indexed.index)
        p = pairs.loc[valid].copy().reset_index(drop=True)
        difference = indexed.reindex(p.station_a)[METRICS].to_numpy()-indexed.reindex(p.station_b)[METRICS].to_numpy()
        out = p.copy()
        for i, metric in enumerate(METRICS):
            out[metric] = difference[:, i]
        for setting, value in zip(SETTINGS, key, strict=True):
            out[setting] = value
        matched.append(out)
    matched = pd.concat(matched, ignore_index=True)
    pair_summary = []
    for key, selected in matched.groupby(SETTINGS, sort=True):
        selected = selected.assign(pair_unit=selected.station_a+"__"+selected.station_b)
        report = blocked_summary(selected, METRICS, group="huc4", unit="pair_unit", draws=draws)
        pair_summary.append(report.assign(**dict(zip(SETTINGS, key, strict=True))))
    return {"cohort_summary": pd.concat(cohort_rows, ignore_index=True),
            "form_summary": pd.concat(form_rows, ignore_index=True),
            "matched_pair_contrasts": matched, "matched_pair_summary": pd.concat(pair_summary, ignore_index=True)}


def numerical_summary(frame, refinement):
    pure = frame.loc[frame.mixing_fraction.eq(0.)]
    unit_key = ["cohort", "case_id", "flow_rule", "input_sd", "source_phase_factor"]
    pure_range = pure.groupby(unit_key).pulse_peak.agg(lambda x: x.max()-x.min())
    equivalent = frame.loc[(frame.volume_factor == frame.flow_factor)]
    equiv_range = equivalent.groupby(unit_key+["mixing_fraction"]).pulse_peak.agg(lambda x: x.max()-x.min())
    return {"n_scenarios": len(frame), "n_whole_junctions": frame.loc[frame.cohort.eq("whole_confluence"), "case_id"].nunique(),
        "n_monitored_footprints": frame.loc[frame.cohort.eq("monitored_footprint"), "case_id"].nunique(),
        "max_anomaly_area_error": float(abs(frame.anomaly_area_fraction-1).max()),
        "max_centroid_error": float(abs(frame.pulse_centroid-frame.analytic_centroid).max()),
        "max_sd_error": float(abs(frame.pulse_sd-frame.analytic_sd).max()),
        "max_pure_translation_peak_range": float(pure_range.max()),
        "max_equal_volume_flow_peak_range": float(equiv_range.max()),
        "max_log_decomposition_error": float(abs(frame.joint_log_peak_change-frame.phase_log_peak_change-frame.mixing_log_peak_change).max()),
        "n_refined_scenarios": len(refinement),
        "max_refinement_peak_difference": float(abs(refinement.pulse_peak_difference).max()),
        "max_refinement_duration_difference": float(abs(refinement.duration_80_difference).max()),
        "scope": "constant-flow conservative scenarios on real junction corridors; not measured DOC treatment effects"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bootstrap-draws", type=int, default=5000)
    args = parser.parse_args()
    cases, reps, examples, pairs, field, proxies = inputs()
    frame, curves = simulate(cases, examples)
    reports = summaries(frame, pairs, draws=args.bootstrap_draws)
    fine, _ = simulate(cases.loc[cases.case_id.isin(examples)], set(), dt=DT/2, keep_curves=False)
    keys = ["cohort", "case_id"]+SETTINGS
    joined = frame[keys+["pulse_peak", "duration_80"]].merge(
        fine[keys+["pulse_peak", "duration_80"]], on=keys, suffixes=("_coarse", "_fine"), validate="one_to_one")
    for metric in ("pulse_peak", "duration_80"):
        joined[metric+"_difference"] = joined[metric+"_fine"]-joined[metric+"_coarse"]
    tables = {"geometry": cases, "representatives": reps, "field_transit_proxies": field,
              "field_transect_proxies": proxies, "numerical_refinement": joined} | reports
    folder = ROOT/"analysis"
    folder.mkdir(parents=True, exist_ok=True)
    for name, table in tables.items():
        table.to_csv(folder/f"{name}.csv", index=False)
    shutil.copyfile(OLD/"junction_inventory.csv", folder/"network_inventory.csv")
    frame.to_parquet(folder/"scenario_metrics.parquet", index=False)
    curves.to_parquet(folder/"example_curves.parquet", index=False)
    summary = numerical_summary(frame, joined)
    (folder/"summary.json").write_text(json.dumps(summary, indent=2)+"\n")
    bindings = {"input_hashes": {str(p): sha256_file(p) for p in SOURCES+CODE},
                "output_hashes": {str(p): sha256_file(p) for p in sorted(folder.iterdir())},
                "bootstrap_draws": args.bootstrap_draws, "new_training": False}
    (ROOT/"analysis_sources.json").write_text(json.dumps(bindings, indent=2)+"\n")
    for p in CODE:
        target = ROOT/"code_snapshot"/p
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(p, target)
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
