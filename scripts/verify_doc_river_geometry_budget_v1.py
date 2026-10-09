"""Independent endpoint budgets, routing nulls and numerical convergence."""

from __future__ import annotations

import json
from itertools import pairwise
from pathlib import Path

import numpy as np
import pandas as pd
from analyze_doc_river_multicatchment_pulses_v1 import load_cases

from river_graph.analysis.river_geometry_budget import route_water_carbon
from river_graph.analysis.river_multicatchment_pulses import hourly_observations
from river_graph.experiments.provenance import sha256_file

ROOT = Path("experiments/phase4_transfer/doc_river_geometry_budget_v1")


def main():
    source = json.loads((ROOT / "analysis_sources.json").read_text())
    mismatches = [str(p) for group in ("inputs", "code", "outputs") for p, digest in source[group].items()
                  if sha256_file(Path(p)) != digest]
    if mismatches:
        raise ValueError(f"Source/result mismatch: {mismatches}")
    budgets = pd.read_csv(ROOT / "analysis/observed_budgets.csv")
    checked = 0
    max_carbon_error, max_water_error = 0., 0.
    for flow, doc, audit, _ in load_cases():
        flow, doc = hourly_observations(flow, doc)
        exact = flow.merge(doc, left_on="flow_timestamp_utc", right_on="timestamp_utc", validate="one_to_one")
        s = budgets.loc[budgets.case.eq(audit["case"]) & budgets.budget_complete]
        for row in s.itertuples():
            start = pd.to_datetime(row.start_clock, utc=audit["case"] == "Kervidy")
            end = pd.to_datetime(row.budget_end_clock, utc=audit["case"] == "Kervidy")
            recorded = exact.loc[exact.flow_timestamp_utc.between(start, end)].sort_values("flow_timestamp_utc")
            water, carbon = 0., 0.
            points = list(recorded.itertuples())
            for a, b in pairwise(points):
                seconds = (b.flow_timestamp_utc - a.flow_timestamp_utc).total_seconds()
                assert seconds == 3600
                water += .5 * (a.q_m3_s + b.q_m3_s) * seconds
                carbon += .5 * (a.q_m3_s * a.doc_mg_l + b.q_m3_s * b.doc_mg_l) * seconds / 1000
            max_carbon_error = max(max_carbon_error, abs(carbon - row.carbon_kg))
            max_water_error = max(max_water_error, abs(water - row.water_volume_m3))
            assert np.isclose(carbon, row.carbon_kg, rtol=1e-12, atol=1e-9)
            assert np.isclose(water, row.water_volume_m3, rtol=1e-12, atol=1e-7)
            assert np.isclose(row.carbon_kg, row.constant_concentration_carbon_kg
                              + row.signed_concentration_extra_kg, rtol=1e-12, atol=1e-9)
            checked += 1
    simulations = pd.read_csv(ROOT / "analysis/geometry_scenarios.csv")
    null_a = simulations.loc[simulations.scenario.eq("short_shared_deterministic")].sort_values(["pair_id", "water_amplitude", "source_sigma"])
    null_b = simulations.loc[simulations.scenario.eq("long_shared_deterministic")].sort_values(["pair_id", "water_amplitude", "source_sigma"])
    metrics = ["concentration_peak_excess", "carbon_excess_centroid", "carbon_excess_sd", "output_extra_carbon"]
    null_error = float(np.max(abs(null_a[metrics].to_numpy() - null_b[metrics].to_numpy())))
    assert null_error < 1e-10
    inventory = pd.read_csv("experiments/phase4_transfer/doc_river_routing_mechanisms_v1/analysis/tributary_inventory.csv")
    # Geometry-order selection, independent of DOC outcomes, spans short/long
    # shared routes and dispersed paths. Check half the primary time step.
    ordered = inventory.sort_values(["junction_receiver_km", "pair_id"])
    sample = ordered.iloc[np.unique(np.linspace(0, len(ordered) - 1, 12).astype(int))]
    resolution = []
    for p in sample.itertuples():
        paths = np.array([p.source_a_receiver_km, p.source_b_receiver_km]) / np.sqrt(p.basin_area_km2)
        weights = np.array([p.source_a_area_km2, p.source_b_area_km2])
        weights /= weights.sum()
        for scenario, common in (("short_shared_distributed", .2 * paths.min()), ("long_shared_distributed", .8 * paths.min())):
            coarse, _ = route_water_carbon(paths, weights, common=common, dispersed=True)
            fine, _ = route_water_carbon(paths, weights, common=common, dispersed=True, dt=.0025)
            resolution.append({"pair_id": p.pair_id, "scenario": scenario,
                "absolute_peak_difference": abs(coarse["concentration_peak_excess"] - fine["concentration_peak_excess"]),
                "absolute_carbon_sd_difference": abs(coarse["carbon_excess_sd"] - fine["carbon_excess_sd"]),
                "absolute_centroid_difference": abs(coarse["carbon_excess_centroid"] - fine["carbon_excess_centroid"])})
    resolution = pd.DataFrame(resolution)
    resolution.to_csv(ROOT / "numerical_resolution.csv", index=False)
    result = {"source_mismatches": mismatches, "checked_complete_bounded_budgets": checked,
        "max_independent_carbon_error_kg": max_carbon_error, "max_independent_water_error_m3": max_water_error,
        "max_deterministic_junction_null_error": null_error,
        "numerical_resolution_cases": len(resolution),
        "max_peak_dt_difference": float(resolution.absolute_peak_difference.max()),
        "max_carbon_sd_dt_difference": float(resolution.absolute_carbon_sd_difference.max()),
        "max_centroid_dt_difference": float(resolution.absolute_centroid_difference.max()),
        "verifier_sha256": sha256_file(Path(__file__)),
        "scope": "All complete field endpoint integrals; all deterministic nulls; 24 geometry-selected resolution cases"}
    (ROOT / "verification.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
