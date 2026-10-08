"""Replay point estimates and verify the preserved observational populations."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from analyze_doc_river_flow_state_mechanisms_v1 import INPUTS, ROOT, point_tables

from river_graph.analysis.river_flow_state_mechanisms import no_processing_feasibility
from river_graph.experiments.provenance import sha256_file


def main():
    metadata = json.loads((ROOT/"analysis_sources.json").read_text())
    for group in ("input_hashes", "code_hashes", "output_hashes"):
        for path, digest in metadata[group].items():
            assert sha256_file(Path(path)) == digest, path
    frame = pd.read_parquet(INPUTS[0])
    assert not frame.duplicated(["population", "receiver", "date_local"]).any()
    joint = frame.loc[frame.population.eq("joint_calendar")]
    assert len(joint) == 188 and joint.date_local.nunique() == 47
    assert joint.groupby("date_local").receiver.nunique().eq(4).all()
    points = point_tables(frame)
    for name, expected, keys in zip(("state_summary", "high_minus_low", "analytic_substitutions"), points,
            (["population", "receiver", "flow_state"], ["population", "receiver"],
             ["population", "receiver", "substituted_high_state_groups"]), strict=True):
        saved = pd.read_csv(ROOT/"analysis"/f"{name}.csv").sort_values(keys).reset_index(drop=True)
        expected = expected.sort_values(keys).reset_index(drop=True)
        assert len(expected) == len(saved), name
        for column in expected:
            if pd.api.types.is_numeric_dtype(expected[column]):
                assert np.allclose(expected[column], saved[column], equal_nan=True), (name, column)
            else:
                assert expected[column].eq(saved[column]).all(), (name, column)
    ledger = pd.read_parquet(ROOT/"analysis/water_budget_ledger.parquet")
    replay = no_processing_feasibility(frame)
    pd.testing.assert_frame_equal(ledger, replay)
    contrast = pd.read_csv(ROOT/"analysis/high_minus_low.csv")
    assert np.allclose(contrast.log_ratio_change,
                       contrast.log_outlet_sd_contribution+contrast.log_mixture_sd_contribution)
    assert np.allclose(contrast.mixing_potential_change_pp,
                       contrast.correlation_pp+contrast.amplitude_balance_pp+contrast.mean_flow_share_pp)
    prior = pd.read_csv(INPUTS[0].parent/"flow_state_comparison.csv")
    current = pd.read_csv(ROOT/"analysis/state_summary.csv")
    pair = current.merge(prior, on=["population", "receiver", "flow_state"], validate="one_to_one")
    assert np.allclose(pair.outlet_mixture_sd_ratio, pair.adjusted_outlet_mix_sd_ratio)
    summary = json.loads((ROOT/"analysis/summary.json").read_text())
    report = {"status": "pass", "state_rows_replayed": len(current),
        "contrasts_replayed": len(contrast), "water_budget_rows_replayed": len(ledger),
        "prior_ratio_estimates_preserved": True, "complete_year_bootstrap_draws": summary["bootstrap_draws"]}
    (ROOT/"verification.json").write_text(json.dumps(report, indent=2)+"\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
