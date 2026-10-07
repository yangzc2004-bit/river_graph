"""Replay routing, bootstrap summaries and physical/scenario invariants."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from analyze_doc_river_routing_mechanisms_v1 import (
    METRICS,
    ROOT,
    read_context,
    summarize_branches,
    summarize_whole,
    tributary_experiment,
    whole_network_experiment,
)

from river_graph.analysis.river_morphology_effect import (
    group_descriptor_summary,
    paired_response_summary,
)
from river_graph.experiments.provenance import sha256_file


def equivalent(rebuilt, path):
    strings = {k: str for k in ("station", "station_a", "station_b", "target", "source_a", "source_b", "huc4")}
    saved = pd.read_csv(path, dtype=strings)
    pd.testing.assert_frame_equal(rebuilt.reset_index(drop=True), saved.reset_index(drop=True),
                                  check_dtype=False, check_exact=False, atol=1e-11, rtol=1e-11)


def main():
    sources = json.loads((ROOT/"analysis_sources.json").read_text())
    for path, expected in sources["source_hashes"].items():
        if sha256_file(Path(path)) != expected:
            raise ValueError(f"source changed: {path}")
    a = ROOT/"analysis"
    panel, pairs, reps, candidates, ledger, representative = read_context()
    assert len(panel) == 297 and panel.station.nunique() == 297
    assert len(candidates) == 121 and candidates.target.nunique() == 38
    assert candidates.component.nunique() == 17 and candidates.eligible_monthly.all()
    assert candidates.branch_status.eq("independent").all()
    whole, traces, convergence, resolution, _ = whole_network_experiment(panel, pairs, reps)
    branch, btraces, aligned = tributary_experiment(candidates, representative)
    frames = {"network_context": panel, "network_representatives": reps, "tributary_inventory": candidates,
              "tributary_inclusion": ledger, "whole_network_routing": whole, "time_step_sensitivity": convergence,
              "tributary_routing": branch, "arrival_alignment": aligned,
              "reach_resolution_sensitivity": resolution,
              "reach_resolution_class_summary": group_descriptor_summary(resolution, METRICS, draws=sources["bootstrap_draws"]),
              "reach_resolution_matched_contrasts": paired_response_summary(pairs, resolution, dict.fromkeys(METRICS, "raw"), draws=sources["bootstrap_draws"])[0],
              **summarize_whole(whole, panel, pairs, sources["bootstrap_draws"]),
              "tributary_contrasts": summarize_branches(branch, sources["bootstrap_draws"])}
    for name, frame in frames.items():
        equivalent(frame, a/f"{name}.csv")
    for frame, name in ((traces, "whole_representative_traces"), (btraces, "tributary_representative_traces")):
        pd.testing.assert_frame_equal(frame, pd.read_parquet(a/f"{name}.parquet"), check_exact=True)
    assert len(whole) == 297*3 and len(branch) == 121*90
    assert not whole.duplicated(["station", "scenario"]).any()
    assert not branch.duplicated(["pair_id", "process", "weighting", "forcing", "junction"]).any()
    assert np.isfinite(whole[list(METRICS)]).all().all()
    assert np.isfinite(branch[list(METRICS)]).all().all()
    np.testing.assert_allclose(whole.input_flow, 1, atol=1e-12, rtol=0)
    np.testing.assert_allclose(whole.outlet_flow, 1, atol=1e-12, rtol=0)
    np.testing.assert_allclose(whole.anomaly_mass_fraction, 1, atol=1e-12, rtol=0)
    np.testing.assert_allclose(whole.steady_doc, 5, atol=1e-12, rtol=0)
    np.testing.assert_allclose(resolution.anomaly_mass_fraction, 1, atol=1e-12, rtol=0)
    midpoint = whole[whole.scenario.eq("actual_paths")].set_index("station")
    np.testing.assert_allclose(resolution.set_index("station").pulse_centroid, midpoint.pulse_centroid, atol=1e-12, rtol=0)
    for metric in ("mean_travel_delay", "pulse_centroid"):
        matrix = whole.pivot(index="station", columns="scenario", values=metric)
        np.testing.assert_allclose(matrix.actual_paths, matrix.half_spread, atol=1e-12, rtol=0)
        np.testing.assert_allclose(matrix.actual_paths, matrix.zero_spread, atol=1e-12, rtol=0)
    sd = whole.pivot(index="station", columns="scenario", values="travel_delay_sd")
    np.testing.assert_allclose(sd.half_spread, sd.actual_paths*.5, atol=1e-12, rtol=0)
    np.testing.assert_allclose(sd.zero_spread, 0, atol=1e-12, rtol=0)
    mass = branch[branch.process.str.startswith("conservative")]
    np.testing.assert_allclose(mass.anomaly_mass_fraction, 1, atol=1e-12, rtol=0)
    for process in ("conservative_uniform", "uniform_processing"):
        f = branch[branch.process.eq(process)]
        for metric in METRICS:
            values = f.pivot(index=["pair_id", "weighting", "forcing"], columns="junction", values=metric)
            np.testing.assert_allclose(values.long_common, values.short_common, atol=1e-12, rtol=0)
            np.testing.assert_allclose(values.observed, values.short_common, atol=1e-12, rtol=0)
    np.testing.assert_allclose(aligned.arrival_aligned_mass, 1, atol=1e-12, rtol=0)
    assert aligned.arrival_aligned_peak.between(.9997, 1.000001).all()
    assert convergence.absolute_peak_difference.max() < .0003
    paired = frames["whole_matched_contrasts"]
    singleton = paired[paired.n_huc4.eq(1)]
    assert singleton[["ci_low", "ci_high"]].isna().all().all()
    for name in ("manifest.json", "manifest_cn.json"):
        manifest = json.loads((ROOT/"figures"/name).read_text())
        for path, expected in {**manifest["source_hashes"], **manifest["figure_hashes"]}.items():
            assert sha256_file(Path(path)) == expected
        assert sha256_file(Path("scripts/plot_doc_river_routing_mechanisms_v1.py")) == manifest["generator_sha256"]
    record = {"status": "passed", "source_files_checked": len(sources["source_hashes"]),
              "networks_replayed": 297, "branch_scenarios_replayed": len(branch),
              "all_bootstrap_tables_recomputed": True, "bootstrap_draws": sources["bootstrap_draws"],
              "constant_mean_and_flow": "preserved in all conservative cases",
              "junction_only_effect": "zero with fixed total paths, uniform speed and uniform processing",
              "arrival_alignment_peak_error_max": float((aligned.arrival_aligned_peak-1).abs().max()),
              "max_time_step_peak_difference": float(convergence.absolute_peak_difference.max()),
              "figure_manifests_checked": True, "neural_training": False, "physical_rates_fitted": False}
    (ROOT/"verification.json").write_text(json.dumps(record, indent=2)+"\n")
    print(json.dumps(record, indent=2))


if __name__ == "__main__":
    main()
