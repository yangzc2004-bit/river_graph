"""Replay geometry, matched-strength identities, decomposition and summaries."""
from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd
from analyze_doc_river_storage_placement_v1 import (
    CODE,
    COLS,
    DT,
    IDENTIFIERS,
    ROOT,
    VAA,
    aggregate,
    geometry,
    real_paths,
    simulate,
)

from river_graph.analysis.river_storage_placement import (
    placement_contrasts,
    select_confluence,
    selected_corridor,
)
from river_graph.experiments.provenance import sha256_file


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--full-replay", action="store_true")
    parser.add_argument("--skip-figures", action="store_true")
    args = parser.parse_args()
    a = ROOT/"analysis"
    manifest = json.loads((ROOT/"analysis_sources.json").read_text())
    for path, expected in {**manifest["source_hashes"], **manifest["product_hashes"], **manifest["routing_hashes"]}.items():
        assert sha256_file(path) == expected, path
    for path in CODE:
        assert sha256_file(path) == sha256_file(ROOT/"code_snapshot"/path), path
    f = pd.read_parquet(a/"scenario_metrics.parquet")
    cases = pd.read_csv(a/"selected_footprints.csv", dtype=IDENTIFIERS)
    reps = pd.read_csv(a/"representatives.csv", dtype=IDENTIFIERS)
    inventory = pd.read_csv(a/"junction_inventory.csv", dtype=IDENTIFIERS)
    assert len(inventory) == 297
    assert len(f) == len(cases)*76
    keys = ["cohort", "case_id", "flow_rule", "input_sd", "strength_match", "fraction", "placement"]
    assert not f.duplicated(keys).any()
    numeric = ["pulse_peak", "duration_80", "pulse_sd", "anomaly_area_fraction", "alignment_ratio"]
    assert np.isfinite(f[numeric]).all().all()
    np.testing.assert_allclose(f.pulse_centroid, 1., atol=1e-8, rtol=0)
    np.testing.assert_allclose(f.anomaly_area_fraction, 1., atol=1e-9, rtol=0)
    np.testing.assert_allclose(f.pulse_sd, f.analytic_sd, atol=1e-8, rtol=0)
    np.testing.assert_allclose(f.log_peak_change, f.log_envelope_change+f.log_alignment_change, atol=1e-12, rtol=0)
    assert f.minimum_remaining_segment.ge(-1e-12).all()
    assert f.alignment_ratio.between(0, 1+1e-12).all()
    sample_tolerance = DT**2/(8*f.input_sd.min()**2)
    assert f.loc[f.placement.eq("shared_trunk"), "peak_reduction_pct"].min() >= -100*sample_tolerance
    assert f.log_envelope_change.max() < 2*sample_tolerance
    grouping = ["cohort", "case_id", "flow_rule", "input_sd", "fraction"]
    for match, field in (("variance_matched", "storage_variance"), ("mean_budget_matched", "allocated_mean_budget")):
        spread = f[f.strength_match.eq(match)].groupby(grouping)[field].agg(["min", "max"])
        np.testing.assert_allclose(spread["min"], spread["max"], atol=1e-12, rtol=0)
    if args.full_replay:
        inv, selected, examples, corridor, _ = geometry()
        for name, actual in (("junction_inventory", inv), ("selected_footprints", selected),
                             ("representatives", examples), ("corridor_reaches", corridor)):
            expected = pd.read_csv(a/f"{name}.csv", dtype=IDENTIFIERS)
            pd.testing.assert_frame_equal(actual, expected, check_dtype=False, atol=1e-9, rtol=1e-9)
        selected = cases
    else:
        selected = cases[cases.case_id.isin(reps.case_id)]
    replay, _ = simulate(selected, reps, keep_curves=False)
    saved = f.set_index(keys).loc[replay.set_index(keys).index].reset_index()
    fields = keys+numeric+["pulse_centroid", "analytic_sd", "log_peak_change"]
    pd.testing.assert_frame_equal(replay[fields].sort_values(keys).reset_index(drop=True),
        saved[fields].sort_values(keys).reset_index(drop=True),
        check_dtype=False, atol=1e-10, rtol=1e-10)
    vaa = pd.read_parquet(VAA, columns=COLS).set_index("comid", drop=False)
    for row in reps.itertuples():
        p, h, _ = real_paths(row, vaa)
        selection, reason = select_confluence(p, h)
        assert reason == "included"
        lengths = selected_corridor(p, selection).groupby("segment").length_km.sum()
        np.testing.assert_allclose(lengths[["branch_a", "branch_b", "common"]],
            [row.branch_a_km, row.branch_b_km, row.common_km], atol=1e-8, rtol=0)
    contrasts = placement_contrasts(f)
    saved_contrasts = pd.read_parquet(a/"placement_contrasts.parquet")
    pd.testing.assert_frame_equal(contrasts, saved_contrasts, check_dtype=False, atol=1e-10, rtol=1e-10)
    draws = json.loads((ROOT/"config.json").read_text())["bootstrap_draws"]
    for name, actual in (("cohort_summary", aggregate(f, draws=draws)),
                          ("contrast_summary", aggregate(contrasts, contrast=True, draws=draws))):
        expected = pd.read_csv(a/f"{name}.csv", dtype={"scope": str})
        pd.testing.assert_frame_equal(actual, expected, check_dtype=False, atol=1e-10, rtol=1e-10)
    convergence = pd.read_csv(a/"numerical_convergence.csv")
    limits = {"pulse_peak": 1e-4, "peak_time": DT, "duration_80": 1e-4,
              "pulse_centroid": 1e-8, "pulse_sd": 1e-8, "alignment_ratio": 1e-4}
    for row in convergence.itertuples():
        assert row.max_absolute_difference <= limits[row.metric], row
    if not args.skip_figures:
        for suffix in ("", "_cn"):
            receipt = json.loads((ROOT/"figures"/f"figure_sources{suffix}.json").read_text())
            for path, expected in {**receipt["inputs"], **receipt["outputs"]}.items():
                assert sha256_file(path) == expected, path
    print(f"Verified {len(cases)} real two-branch geometries, {len(f)} scenarios; {len(selected)} direct response replays.")
    print("Fixed means, unit gain, equal variance / allocated budget, source decomposition, block summaries and figures.")


if __name__ == "__main__":
    main()
