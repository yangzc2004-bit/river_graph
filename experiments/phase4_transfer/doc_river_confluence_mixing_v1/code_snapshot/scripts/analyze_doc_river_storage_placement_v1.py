"""Early-tributary, late-tributary and common-trunk storage on real geometry."""
from __future__ import annotations

import argparse
import json
import shutil
import time
from pathlib import Path

import numpy as np
import pandas as pd

from river_graph.analysis.river_storage_placement import (
    blocked_summary,
    placement_contrasts,
    placement_responses,
    select_confluence,
    selected_corridor,
)
from river_graph.analysis.river_whole_storage import StoragePaths
from river_graph.experiments.provenance import sha256_file

ROOT = Path("experiments/phase4_transfer/doc_river_storage_placement_v1")
MORPH = Path("experiments/phase4_transfer/doc_river_morphology_effect_v1/analysis")
OLD_REPS = Path("experiments/phase4_transfer/doc_river_routing_mechanisms_v1/analysis/network_representatives.csv")
FOOTPRINT = Path("experiments/phase4_transfer/doc_monitored_river_footprint_v1/analysis/footprints.csv")
FOOT_REPS = FOOTPRINT.with_name("representatives.csv")
ROUTING = Path("data/raw/river_source_placement_v1/routing")
VAA = Path("cache/nldplus_vaa.parquet")
COLS = ["comid", "hydroseq", "dnhydroseq", "dnminorhyd", "lengthkm", "areasqkm", "wbareatype", "wbareacomi"]
DT = .00125
SIGMAS = (.075, .15, .30)
FRACTIONS = (.25, .5, 1.)
IDENTIFIERS = {k: str for k in ("station", "case_id", "target", "source_a", "source_b", "pair_id", "huc4", "component")}
METRICS = ("pulse_peak", "peak_reduction_pct", "duration_80", "duration_change_pct",
    "alignment_ratio", "individual_peak_envelope", "log_peak_change", "log_envelope_change", "log_alignment_change")
CONTRAST_METRICS = ("pulse_peak", "peak_reduction_pct", "duration_80", "duration_change_pct",
    "alignment_ratio", "individual_peak_envelope", "source_peak_gap")
CODE = tuple(Path(p) for p in (
    "src/river_graph/analysis/river_storage_placement.py",
    "src/river_graph/analysis/river_storage_transport.py",
    "src/river_graph/analysis/river_whole_storage.py",
    "scripts/analyze_doc_river_storage_placement_v1.py",
    "scripts/plot_doc_river_storage_placement_v1.py",
    "scripts/verify_doc_river_storage_placement_v1.py",
    "tests/test_river_storage_placement.py"))


def real_paths(row, vaa):
    """Keep lengths in actual km; dimensionless response scaling happens later."""
    with np.load(ROUTING/f"comid_{int(row.comid)}.npz") as cached:
        frame = vaa.loc[cached["comids"]]
        reachable = np.isfinite(cached["distance_km"])
        selected = frame[reachable]
        paths = StoragePaths(selected, int(row.comid), cached["distance_km"][reachable])
    return paths, selected.hydroseq.to_numpy(), selected.areasqkm.sum()/frame.areasqkm.sum()


def geometry():
    fields = ["station", "comid", "cluster", "huc4", "basin_area_km2"]
    panel = pd.read_csv(MORPH/"station_morphology_doc_panel.csv", dtype=IDENTIFIERS)[fields]
    vaa = pd.read_parquet(VAA, columns=COLS).set_index("comid", drop=False)
    inventories, selected, sources = [], [], {}
    started = time.monotonic()
    for i, row in enumerate(panel.itertuples()):
        paths, hydro, coverage = real_paths(row, vaa)
        selection, reason = select_confluence(paths, hydro)
        info = {"station": row.station, "case_id": row.station, "comid": int(row.comid),
                "cluster": int(row.cluster), "huc4": row.huc4, "component": row.huc4,
                "cohort": "whole_confluence", "represented_area_fraction": float(coverage)}
        inventories.append(info | {"inclusion": reason})
        if selection is not None:
            selected.append(info | selection)
        source = ROUTING/f"comid_{int(row.comid)}.npz"
        sources[str(source)] = sha256_file(source)
        if i % 30 == 0 or i+1 == len(panel):
            print(f"Geometry: {i+1}/{len(panel)}; {time.monotonic()-started:.1f}s", flush=True)
    selected = pd.DataFrame(selected)
    if selected.empty:
        raise ValueError("no real positive three-segment confluence found")
    original = pd.read_csv(OLD_REPS, dtype=IDENTIFIERS)
    reps = selected[selected.station.isin(original.station)].copy()
    reps["example_group"] = reps.cluster
    reps["selection"] = "original geometry-selected morphology representative"
    # Explicit geometry-only fallback if a previous shape example has no junction.
    classes = pd.read_csv("experiments/phase4_transfer/doc_river_planform_typology_v1/analysis/classes.csv", dtype=IDENTIFIERS)
    for cluster in sorted(set(selected.cluster)-set(reps.cluster)):
        eligible = selected[selected.cluster.eq(cluster)].merge(classes[["station", "centroid_distance"]], on="station")
        fallback = eligible.sort_values(["centroid_distance", "station"]).iloc[0].to_dict()
        reps = pd.concat([reps, pd.DataFrame([fallback | {"example_group": cluster,
            "selection": "nearest original morphology centroid among eligible real confluences"}])], ignore_index=True)
    choice = selected.assign(score=selected.relative_arrival_cv*selected.variance_capacity_sd).sort_values(
        ["score", "station"], ascending=[False, True]).iloc[0].drop(labels="score").to_dict()
    reps = pd.concat([reps, pd.DataFrame([choice | {"example_group": 4,
        "selection": "maximum arrival-CV times feasible variance capacity; geometry only"}])], ignore_index=True)
    corridors = []
    for row in reps.itertuples():
        p, h, _ = real_paths(row, vaa)
        s, _ = select_confluence(p, h)
        corridor = selected_corridor(p, s)
        corridor["station"] = row.station
        corridor["example_group"] = row.example_group
        corridors.append(corridor)
    f = pd.read_csv(FOOTPRINT, dtype=IDENTIFIERS)
    f["station"], f["case_id"], f["cohort"] = f.target, f.pair_id, "monitored_footprint"
    f["relative_arrival_cv"] = f.total_path_cv
    f["variance_capacity_sd"] = np.minimum.reduce([
        np.sqrt(f.weight_a)*f.branch_a_km,
        np.sqrt(1-f.weight_a)*f.branch_b_km, f.common_km])/f.mean_total_km
    f["selected_pair_area_share"] = f.source_drainage_coverage
    f["represented_area_fraction"] = 1.
    combined = pd.concat([selected, f], ignore_index=True)
    return pd.DataFrame(inventories), combined, reps, pd.concat(corridors, ignore_index=True), sources


def simulate(cases, representatives, *, dt=DT, keep_curves=True):
    outputs, traces = [], []
    examples = set(representatives.case_id)
    foot_examples = set(pd.read_csv(FOOT_REPS, dtype=IDENTIFIERS).pair_id)
    started = time.monotonic()
    for i, row in enumerate(cases.itertuples()):
        for flow, sigma in [("area_proxy", s) for s in SIGMAS]+[("equal_flow", .15)]:
            weight = row.weight_a if flow == "area_proxy" else .5
            keep = keep_curves and sigma == .15 and flow == "area_proxy" and (row.case_id in examples or row.case_id in foot_examples)
            metrics, curves = placement_responses(row.branch_a_km, row.branch_b_km, row.common_km,
                weight, sigma=sigma, fractions=FRACTIONS, dt=dt, keep_curves=keep)
            metrics["case_id"], metrics["cohort"], metrics["flow_rule"] = row.case_id, row.cohort, flow
            outputs.append(metrics)
            if keep:
                curves = curves[curves.strength_match.eq("baseline") | curves.fraction.eq(.5)].copy()
                curves["case_id"], curves["cohort"], curves["flow_rule"] = row.case_id, row.cohort, flow
                traces.append(curves)
        if i % 40 == 0 or i+1 == len(cases):
            print(f"Response: {i+1}/{len(cases)}; {time.monotonic()-started:.1f}s", flush=True)
    return pd.concat(outputs, ignore_index=True).merge(cases, on=["cohort", "case_id"], validate="many_to_one"), \
        pd.concat(traces, ignore_index=True) if traces else pd.DataFrame()


def aggregate(frame, *, contrast=False, draws=5000):
    keys = ["cohort", "flow_rule", "input_sd", "strength_match", "fraction", "contrast" if contrast else "placement"]
    metrics = CONTRAST_METRICS if contrast else METRICS
    rows = []
    for setting, f in frame.groupby(keys, sort=True):
        info = dict(zip(keys, setting))
        if info["cohort"] == "monitored_footprint":
            f = f.groupby(["station", "huc4", "component", "cluster"], as_index=False)[list(metrics)].mean()
            block = "component"
        else:
            block = "huc4"
        for scope, g in [("all", f)]+[(str(c), d) for c, d in f.groupby("cluster")]:
            result = blocked_summary(g, metrics, group=block, unit="station", draws=draws)
            for name, value in info.items():
                result[name] = value
            result["scope"] = scope
            result["aggregation"] = "receiver-equal after pair averaging" if block == "component" else "station-network-instance equal"
            rows.append(result)
    return pd.concat(rows, ignore_index=True)


def save_sources(analysis):
    for p in CODE:
        dest = ROOT/"code_snapshot"/p
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(p, dest)
    source = [MORPH/"station_morphology_doc_panel.csv", OLD_REPS, FOOTPRINT, FOOT_REPS, VAA,
              ROOT/"study_plan.md", ROOT/"config.json", *CODE]
    routes = json.loads((analysis/"geometry_sources.json").read_text())
    payload = {"source_hashes": {str(p): sha256_file(p) for p in source}, "routing_hashes": routes,
               "product_hashes": {str(p): sha256_file(p) for p in sorted(analysis.iterdir()) if p.is_file()}}
    (ROOT/"analysis_sources.json").write_text(json.dumps(payload, indent=2)+"\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--geometry-only", action="store_true")
    parser.add_argument("--report-only", action="store_true")
    parser.add_argument("--bootstrap-draws", type=int, default=5000)
    args = parser.parse_args()
    a = ROOT/"analysis"
    a.mkdir(parents=True, exist_ok=True)
    if not (a/"selected_footprints.csv").exists():
        inventory, cases, reps, corridor, inputs = geometry()
        for name, f in (("junction_inventory", inventory), ("selected_footprints", cases),
                        ("representatives", reps), ("corridor_reaches", corridor)):
            f.to_csv(a/f"{name}.csv", index=False)
        (a/"geometry_sources.json").write_text(json.dumps(inputs, indent=2)+"\n")
    else:
        cases = pd.read_csv(a/"selected_footprints.csv", dtype=IDENTIFIERS)
        reps = pd.read_csv(a/"representatives.csv", dtype=IDENTIFIERS)
    if args.geometry_only:
        print(cases.groupby(["cohort", "cluster"]).size().to_string())
        return
    if args.report_only:
        scenarios = pd.read_parquet(a/"scenario_metrics.parquet")
    else:
        scenarios, curves = simulate(cases, reps)
        scenarios.to_parquet(a/"scenario_metrics.parquet", index=False)
        curves.to_parquet(a/"representative_responses.parquet", index=False)
    contrasts = placement_contrasts(scenarios)
    tables = {"placement_contrasts": contrasts,
        "cohort_summary": aggregate(scenarios, draws=args.bootstrap_draws),
        "contrast_summary": aggregate(contrasts, contrast=True, draws=args.bootstrap_draws)}
    # All fixed examples, chosen without their simulated peak response.
    selected_ids = set(reps.case_id) | set(pd.read_csv(FOOT_REPS, dtype=IDENTIFIERS).pair_id)
    examples = cases[cases.case_id.isin(selected_ids)]
    fine, _ = simulate(examples, reps, dt=DT/2, keep_curves=False)
    keys = ["cohort", "case_id", "flow_rule", "input_sd", "strength_match", "fraction", "placement"]
    fine = fine.set_index(keys).sort_index()
    coarse = scenarios.set_index(keys).loc[fine.index]
    convergence = []
    for metric in ("pulse_peak", "peak_time", "duration_80", "pulse_centroid", "pulse_sd", "alignment_ratio"):
        convergence.append({"metric": metric, "max_absolute_difference": float(abs(coarse[metric]-fine[metric]).max()),
                            "coarse_dt": DT, "fine_dt": DT/2, "n_examples": len(examples)})
    tables["numerical_convergence"] = pd.DataFrame(convergence)
    for name, f in tables.items():
        if name == "placement_contrasts":
            f.to_parquet(a/f"{name}.parquet", index=False)
        else:
            f.to_csv(a/f"{name}.csv", index=False)
    primary = scenarios.query("flow_rule == 'area_proxy' and input_sd == .15 and fraction == .5")
    counts = primary.groupby(["cohort", "strength_match", "placement"]).agg(
        n_cases=("case_id", "size"), reduced_over_1pct=("peak_reduction_pct", lambda x: int((x > 1).sum())),
        increased_over_1pct=("peak_reduction_pct", lambda x: int((x < -1).sum())),
        unchanged_within_1pct=("peak_reduction_pct", lambda x: int((abs(x) <= 1).sum()))).reset_index()
    counts.to_csv(a/"primary_effect_counts.csv", index=False)
    config = {"input_pulse_sds": SIGMAS, "storage_fractions_of_matched_capacity": FRACTIONS,
        "dt": DT, "bootstrap_draws": args.bootstrap_draws, "mean_arrival": 1.,
        "matched_strengths": ["additional variance", "flow-weighted allocated mean time"],
        "flows": "incremental-area proxy; middle-pulse equal-flow sensitivity",
        "forcing": "identical unit-height Gaussian concentration pulse in two selected real branches",
        "storage": "unit-gain causal exponential; displaced deterministic mean time, not chemical loss",
        "time_unit": "relative geometry-derived scenario time, not measured days",
        "storage_locations": "controlled placement, not inferred or relocated real waterbodies",
        "pulse_peaks": "continuous local refinement; earliest maximum within relative1e-8, ambiguity retained",
        "new_model_training": False, "whole_network_results_previously_seen": True}
    (ROOT/"config.json").write_text(json.dumps(config, indent=2)+"\n")
    summary = {"n_original_network_instances": 297,
        "n_eligible_confluences": int(cases.cohort.eq("whole_confluence").sum()),
        "n_monitored_footprints": int(cases.cohort.eq("monitored_footprint").sum()),
        "n_scenarios": len(scenarios), "max_centroid_error": float(abs(scenarios.pulse_centroid-1).max()),
        "max_area_error": float(abs(scenarios.anomaly_area_fraction-1).max()),
        "max_sd_error": float(abs(scenarios.pulse_sd-scenarios.analytic_sd).max()),
        "max_log_decomposition_error": float(abs(scenarios.log_peak_change-scenarios.log_envelope_change-scenarios.log_alignment_change).max()),
        "confluence_class_counts": cases[cases.cohort.eq("whole_confluence")].groupby("cluster").size().to_dict(),
        "confluence_area_share_median": float(cases.loc[cases.cohort.eq("whole_confluence"), "selected_pair_area_share"].median())}
    (a/"summary.json").write_text(json.dumps(summary, indent=2)+"\n")
    save_sources(a)
    print(json.dumps(summary, indent=2))
    print(tables["cohort_summary"].query("scope == 'all' and flow_rule == 'area_proxy' and input_sd == .15 and fraction == .5 and metric == 'peak_reduction_pct'")[["cohort", "strength_match", "placement", "mean", "ci_low", "ci_high", "n_units", "n_blocks"]].to_string(index=False))


if __name__ == "__main__":
    main()
