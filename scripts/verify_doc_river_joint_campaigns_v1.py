"""Replay the observed cohort and point estimates, checking saved source bindings."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from analyze_doc_river_joint_campaigns_v1 import (
    RECEIVERS,
    ROOT,
    build_inputs,
    summarize_population,
)

from river_graph.analysis.river_joint_campaigns import joint_calendar
from river_graph.experiments.provenance import sha256_file


def main():
    bindings = json.loads((ROOT/"analysis_sources.json").read_text())
    checks = {}
    for group in ("input_hashes", "code_hashes", "output_hashes"):
        for filename, expected in bindings[group].items():
            if sha256_file(Path(filename)) != expected:
                raise ValueError(f"Changed {group}: {filename}")
        checks[group] = len(bindings[group])
    eligible, ledger, _, flow = build_inputs()
    saved_ledger = pd.read_parquet(ROOT/"analysis/campaign_ledger.parquet")
    pd.testing.assert_frame_equal(ledger.reset_index(drop=True), saved_ledger.reset_index(drop=True))
    for receiver, selected in ledger.groupby("receiver"):
        for column in ("source_a_time_utc", "source_b_time_utc", "receiver_time_utc"):
            if selected[column].duplicated().any():
                raise ValueError(f"Reused sample within {receiver}: {column}")
    timestamps = ledger[["source_a_time_utc", "source_b_time_utc", "receiver_time_utc"]]
    spans = (timestamps.max(axis=1)-timestamps.min(axis=1)).dt.total_seconds()/3600
    if not spans.le(12).all():
        raise ValueError("A matched campaign exceeds the fixed 12-hour span")
    if not eligible.same_flow_calendar_day.all() or eligible.duplicate_receiver_date.any():
        raise ValueError("Calendar/duplicate inclusion error")
    joint = joint_calendar(eligible, RECEIVERS)
    date_sets = [set(joint.loc[joint.receiver.eq(site), "date_local"]) for site in RECEIVERS]
    if not all(dates == date_sets[0] for dates in date_sets):
        raise ValueError("Configurations are not on the same calendar")
    saved = pd.read_csv(ROOT/"analysis/configuration_comparison.csv").set_index(["population", "receiver"])
    for population, frame in (("all_calendar", eligible), ("joint_calendar", joint)):
        table, _, _, _, _ = summarize_population(frame, flow, population=population)
        for row in table.to_dict("records"):
            original = saved.loc[(row["population"], row["receiver"])]
            for name, value in row.items():
                if isinstance(value, (int, float, np.number)) and not np.isclose(
                        value, original[name], atol=1e-10, rtol=1e-10, equal_nan=True):
                    raise ValueError(f"Point estimate differs: {population}/{row['receiver']}/{name}")
    if not np.allclose(saved.fixed_mix_variance,
                       saved.individual_variance_term+saved.covariance_term, atol=1e-9):
        raise ValueError("Saved variance decomposition is incorrect")
    for filename in ("configuration_comparison.csv", "flow_state_comparison.csv",
                     "paired_configuration_contrasts.csv", "flow_state_contrasts.csv"):
        table = pd.read_csv(ROOT/"analysis"/filename)
        if np.isinf(table.select_dtypes(include="number").to_numpy()).any():
            raise ValueError(f"Infinite analysis value: {filename}")
        if not table.n_valid_bootstrap.between(0, 5000).all():
            raise ValueError(f"Invalid resampling count: {filename}")
    for path in (ROOT/"figures").glob("figure_sources*.json"):
        receipt = json.loads(path.read_text())
        for group in ("input_hashes", "output_hashes"):
            for filename, expected in receipt[group].items():
                if sha256_file(Path(filename)) != expected:
                    raise ValueError(f"Changed figure source/product: {filename}")
    checks.update(ledger_replayed=True, matched_samples_used_once_within_configuration=True,
                  joint_calendar_equal=True, point_estimates_replayed=True,
                  exact_variance_identity=True, bootstrap_draws=5000,
                  scope="temporal repetition within one connected catchment, not independent river forms")
    (ROOT/"verification.json").write_text(json.dumps(checks, indent=2)+"\n")
    print(json.dumps(checks, indent=2))


if __name__ == "__main__":
    main()
