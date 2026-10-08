"""Separate real arrival paths, shared memory and controlled forcing timescales."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd

from river_graph.analysis.river_form_process import cluster_mean
from river_graph.analysis.river_signal_timescale import (
    harmonic_response,
    source_geometry,
    stationary_response,
)
from river_graph.experiments.provenance import sha256_file

ROOT = Path("experiments/phase4_transfer/doc_river_signal_timescale_v1")
PRIOR = Path("experiments/phase4_transfer/doc_river_pathway_context_v1")
TIMES = (.025, .05, .1, .25, .5, 1., 2., 4., 8.)
COHERENCE = (0., .5, 1.)
MEMORY = (0., .25, .5, 1.)
METRICS = ("sd_ratio", "geometry_log_change", "shared_memory_log_change", "interaction_log_change")
TYPES = {"target": str, "source_station": str, "source_a_station": str,
         "source_b_station": str, "huc4": str}
CODE = tuple(map(Path, (
    "src/river_graph/analysis/river_signal_timescale.py",
    "src/river_graph/analysis/river_form_process.py",
    "scripts/analyze_doc_river_signal_timescale_v1.py",
    "scripts/plot_doc_river_signal_timescale_v1.py",
    "scripts/verify_doc_river_signal_timescale_v1.py",
    "tests/test_river_signal_timescale.py")))


def inputs(shortest=False):
    prior = PRIOR/"shortest_route_sensitivity" if shortest else PRIOR
    files = [prior/"analysis/receiver_pathway_panel.csv", prior/"analysis/source_corridor_reaches.csv",
             prior/"analysis_sources.json"]
    ledger = json.loads(files[2].read_text())
    for path in files[:2]:
        if ledger["output_hashes"].get(str(path)) != sha256_file(path):
            raise ValueError(f"preceding cropped-path product changed: {path}")
    panel = pd.read_csv(files[0], dtype=TYPES)
    reaches = pd.read_csv(files[1], dtype=TYPES)
    if len(panel) != 32 or panel.target.duplicated().any() or panel.comid.duplicated().any():
        raise ValueError("retain the 32 unique physical receivers")
    return panel, reaches, files


def illustrate(sources, panel):
    """Fixed-phase construction; actual geometry, illustrative forcing only."""
    target = "401733105392404"
    f = sources[sources.target.eq(target)].sort_values("normalized_delay")
    d, w = f.normalized_delay.to_numpy(), f.area_weight.to_numpy()
    if len(d) != 2 or d.max()-d.min() < 1e-8:
        raise ValueError("fixed Loch Vale illustration requires its two unequal source paths")
    phase_advance = 3*np.pi/4
    period = 2*np.pi*(d.max()-d.min())/phase_advance
    common = panel.set_index("target").loc[target, "common_fraction"]
    phases = np.array([0., phase_advance])
    time = np.arange(2400)*period*4/2400
    metrics, waves = [], []
    for forcing, phase in (("synchronous", np.zeros(2)), ("path_compensated", phases)):
        for scenario, delays, memory in (("aligned_arrivals", np.full(2, w@d), 0.),
                ("actual_delays", d, 0.), ("actual_plus_shared_memory", d, .5*common)):
            result = harmonic_response(delays, w, phase, period=period, memory_time=memory)
            metrics.append({"target": target, "forcing": forcing, "scenario": scenario,
                "period_relative_mean_path": period, "memory_time": memory,
                "source_a_station": f.source_station.iloc[0], "source_b_station": f.source_station.iloc[1],
                "source_phase_a": phase[0], "source_phase_b": phase[1],
                "mixture_amplitude": result["mixture_amplitude"], "outlet_amplitude": result["outlet_amplitude"],
                "amplitude_ratio": result["amplitude_ratio"], "steady_gain": result["steady_gain"],
                "evidence_type": "constructed periodic source phases on real mapped paths; not observed DOC"})
            values = np.imag(result["outlet_complex"]*np.exp(1j*result["omega"]*time))
            source_a = np.sin(result["omega"]*time+phase[0])
            source_b = np.sin(result["omega"]*time+phase[1])
            waves.append(pd.DataFrame({"target": target, "forcing": forcing, "scenario": scenario,
                "relative_time": time, "cycle": time/period, "source_a": source_a, "source_b": source_b,
                "instantaneous_mixture": w[0]*source_a+w[1]*source_b, "outlet_anomaly": values}))
    return pd.DataFrame(metrics), pd.concat(waves, ignore_index=True)


def calculate(*, draws=5000, shortest=False):
    panel, reaches, files = inputs(shortest)
    sources, inventory, scenarios = [], [], []
    for r in panel.sort_values("target").itertuples():
        f, geometry = source_geometry(reaches[reaches.target.eq(r.target)])
        meta = {"target": r.target, "comid": int(r.comid), "huc4": r.huc4,
                "component": int(r.component), "cluster": int(r.cluster),
                "covered_area_fraction": r.covered_area_fraction}
        np.testing.assert_allclose(geometry["path_mean_km"], r.monitored_path_mean_km, rtol=1e-10)
        np.testing.assert_allclose(geometry["common_km"], r.monitored_common_km, rtol=1e-10)
        np.testing.assert_allclose(geometry["path_cv"], r.monitored_path_cv, atol=1e-10)
        sources.append(f.assign(**meta))
        inventory.append({**meta, **geometry, "station_name": r.station_name,
            "observed_sd_ratio": np.exp(r.outlet_mix_log_sd_ratio),
            "evidence_type": "real cropped source paths and copied observed DOC context"})
        d, w = f.normalized_delay.to_numpy(), f.area_weight.to_numpy()
        for time in TIMES:
            for coherence in COHERENCE:
                for fraction in MEMORY:
                    tau = fraction*geometry["common_fraction"]
                    for arrangement, delay in (("actual_paths", d), ("aligned_arrivals", np.full(len(d), w@d))):
                        result = stationary_response(delay, w, correlation_time=time, coherence=coherence, memory_time=tau)
                        scenarios.append({**meta, "correlation_time": time, "coherence": coherence,
                            "memory_allocation": fraction, "memory_time": tau, "arrangement": arrangement,
                            **result, "evidence_type": "controlled stationary source process; not measured DOC"})
    source, inventory = pd.concat(sources, ignore_index=True), pd.DataFrame(inventory)
    scenario = pd.DataFrame(scenarios)
    summaries, contrasts = [], []
    actual = scenario[scenario.arrangement.eq("actual_paths")]
    for cut in (0., .8):
        pop = actual[actual.covered_area_fraction.ge(cut)]
        for params, group in pop.groupby(["correlation_time", "coherence", "memory_allocation"], sort=True):
            fixed = dict(zip(("correlation_time", "coherence", "memory_allocation"), params, strict=True))
            for name, g in (("all", group), *((f"class_{c}", group[group.cluster.eq(c)]) for c in (1, 2, 3))):
                for metric in METRICS:
                    result = cluster_mean(g, metric, "component", draws=draws)
                    if result:
                        summaries.append({"minimum_coverage": cut, "group": name, **fixed, **result})
            for metric in METRICS:
                result = cluster_mean(group, metric, "component", draws=draws, contrast=True)
                if result:
                    contrasts.append({"minimum_coverage": cut, **fixed, **result,
                        "contrast": "broad_minus_elongated", "n_elongated": int(group.cluster.eq(1).sum()),
                        "n_broad": int(group.cluster.eq(3).sum())})
    working = actual[actual.correlation_time.eq(.5) & actual.coherence.eq(.5) & actual.memory_allocation.eq(.5)]
    context = inventory.merge(working[["target", *METRICS, "geometry_sd_ratio", "aligned_sd_ratio"]],
        on="target", validate="one_to_one")
    context["observed_receiving_amplification"] = context.observed_sd_ratio.gt(1.)
    context["controlled_receiving_amplification"] = context.sd_ratio.gt(1.+1e-10)
    context["comparison_note"] = "Different forcing populations; controlled gain is not a prediction fitted to observed monthly DOC"
    wave_metrics, waves = illustrate(source, inventory)
    tables = {"source_paths": source, "network_inventory": inventory, "scenario_metrics": scenario,
        "scenario_summary": pd.DataFrame(summaries), "form_contrasts": pd.DataFrame(contrasts),
        "observed_and_controlled_context": context, "illustration_metrics": wave_metrics,
        "illustration_waves": waves}
    summary = {"n_receivers": len(inventory), "n_systems": int(inventory.component.nunique()),
        "n_high_coverage_receivers": int(inventory.covered_area_fraction.ge(.8).sum()),
        "n_source_paths": len(source), "n_controlled_scenarios": len(scenario),
        "n_observed_receiving_amplifications": int(context.observed_receiving_amplification.sum()),
        "n_controlled_receiving_amplifications": int(context.controlled_receiving_amplification.sum()),
        "n_memory_scenarios_with_larger_variance_than_delay_only": int((scenario.outlet_variance > scenario.delay_only_variance+1e-10).sum()),
        "max_log_decomposition_error": float(abs(scenario.log_sd_ratio-scenario.geometry_log_change-scenario.shared_memory_log_change).max()),
        "bootstrap_draws": draws, "new_measurements": False, "new_training": False}
    return tables, summary, files


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--bootstrap-draws", type=int, default=5000)
    p.add_argument("--shortest-routes", action="store_true")
    args = p.parse_args()
    root = ROOT/"shortest_route_sensitivity" if args.shortest_routes else ROOT
    out = root/"analysis"
    out.mkdir(parents=True, exist_ok=True)
    tables, summary, files = calculate(draws=args.bootstrap_draws, shortest=args.shortest_routes)
    products = []
    for name, f in tables.items():
        path = out/f"{name}.{'parquet' if name in ('scenario_metrics', 'illustration_waves') else 'csv'}"
        f.to_parquet(path, index=False) if path.suffix == ".parquet" else f.to_csv(path, index=False)
        products.append(path)
    (out/"summary.json").write_text(json.dumps(summary, indent=2)+"\n")
    products.append(out/"summary.json")
    config = {"previous_results_seen": True, "correlation_times": TIMES, "source_coherences": COHERENCE,
        "common_memory_allocations": MEMORY, "coverage_cuts": [0., .8],
        "working_point": {"correlation_time": .5, "coherence": .5, "memory_allocation": .5},
        "time_unit": "network mean cropped gauge path / imposed uniform speed",
        "bootstrap_draws": args.bootstrap_draws, "bootstrap_seed": 42,
        "bootstrap_unit": "complete overlapping-catchment system", "weighting": "receivers equal",
        "routing": "saved shortest" if args.shortest_routes else "original geometric mainstem",
        "source_process": "equal variance exponential autocorrelation; common plus independent components",
        "memory": "unit gain exponential on shared corridor; allocate rather than increase each path mean",
        "illustration": "fixed Loch Vale path-compensated source phases, not measured upstream phases",
        "interpretation": "controlled mechanism; no fitted flow speed, residence time or DOC removal"}
    (root/"config.json").write_text(json.dumps(config, indent=2)+"\n")
    for path in CODE:
        dst = root/"code_snapshot"/path
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, dst)
    (root/"analysis_sources.json").write_text(json.dumps({
        "source_hashes": {str(path): sha256_file(path) for path in [*files, ROOT/"study_plan.md", *CODE]},
        "config_sha256": sha256_file(root/"config.json"),
        "output_hashes": {str(path): sha256_file(path) for path in products}, "new_training": False}, indent=2)+"\n")
    print(json.dumps(summary, indent=2), flush=True)
    print(tables["scenario_summary"].query(
        "group == 'all' and correlation_time == 0.5 and coherence == 0.5 and memory_allocation == 0.5").to_string(index=False))
    print(tables["form_contrasts"].query(
        "correlation_time == 0.5 and coherence == 0.5 and memory_allocation == 0.5").to_string(index=False))
    print(tables["illustration_metrics"].to_string(index=False))


if __name__ == "__main__":
    main()
