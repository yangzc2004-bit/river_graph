"""Replay raw DOC, selection, flow joins and all matched sampling comparisons."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from analyze_doc_river_sampling_resolution_v1 import (
    CODE,
    DTYPES,
    FOOTPRINT,
    INPUT,
    ROOT,
    calendar_examples,
    compare_sampling,
    event_opportunity,
    hydro_summary,
    read_doc,
)

from river_graph.analysis.river_sampling_resolution import (
    attach_sample_flow,
    doc_activities,
    load_station_daily_flow,
    sampling_triplets,
    station_cadence,
)
from river_graph.experiments.provenance import sha256_file


def equal_csv(name, frame):
    saved = pd.read_csv(ROOT/"analysis"/f"{name}.csv", dtype=DTYPES)
    # CSV calendars are compared in the canonical string representation.
    for column in frame.select_dtypes(include=["datetime64[ns]"]):
        saved[column] = pd.to_datetime(saved[column])
    pd.testing.assert_frame_equal(frame.reset_index(drop=True), saved, check_dtype=False,
                                  check_exact=False, atol=1e-10, rtol=1e-10)


def main():
    a = ROOT/"analysis"
    sources = json.loads((ROOT/"analysis_sources.json").read_text())
    for path, expected in sources["source_hashes"].items():
        assert sha256_file(Path(path)) == expected, path
    for r in sources["raw_doc_inventory"]+sources["daily_inventory"]["files"]:
        assert sha256_file(Path(r["path"])) == r["sha256"], r["path"]
    for path in CODE:
        assert sha256_file(path) == sha256_file(ROOT/"code_snapshot"/path), path
    for path, expected in sources["product_hashes"].items():
        assert sha256_file(Path(path)) == expected, path
    config = json.loads((ROOT/"config.json").read_text())
    assert config["bootstrap_draws"] == 5000 and not config["new_model_training"]
    inputs = pd.read_parquet(INPUT)
    raw, monthly, _ = read_doc(inputs)
    events = doc_activities(raw)
    pd.testing.assert_frame_equal(events, pd.read_parquet(a/"doc_activities.parquet"), check_exact=True)
    pd.testing.assert_frame_equal(monthly, pd.read_parquet(a/"monthly_reconciliation.parquet"), check_exact=True)
    triplets = sampling_triplets(inputs, events)
    changed = events.copy()
    changed["doc"] = np.random.default_rng(42).uniform(0, 10000, len(events))
    altered = sampling_triplets(inputs, changed)
    columns = [c for c in triplets if not c.startswith("sample_doc_")]
    pd.testing.assert_frame_equal(triplets[columns], altered[columns], check_exact=True)
    daily, _ = load_station_daily_flow("data/raw/nwis_dv", set(events.site_no))
    triplets = attach_sample_flow(triplets, daily)
    pd.testing.assert_frame_equal(triplets, pd.read_parquet(a/"sample_triplets.parquet"), check_exact=True)
    assert len(triplets) == 3026 and triplets.pair_id.nunique() == 59
    assert triplets.target.nunique() == 22 and triplets.component.nunique() == 11
    assert len(triplets.drop_duplicates(["target", "month_index"])) == 1092
    span = triplets[["sample_date_a", "sample_date_b", "sample_date_receiver"]]
    np.testing.assert_array_equal((span.max(axis=1)-span.min(axis=1)).dt.days, triplets.sample_span_days)
    footprints = pd.read_csv(FOOTPRINT, dtype=DTYPES)
    tables = {**compare_sampling(triplets, 5000), "hydro_date_summary": hydro_summary(triplets),
              "event_opportunity": event_opportunity(triplets, footprints),
              "calendar_examples": calendar_examples(triplets, footprints, daily)}
    cadence, gaps = station_cadence(events)
    tables.update(station_cadence=cadence, sampling_intervals=gaps)
    for name, frame in tables.items():
        equal_csv(name, frame)
    connections = tables["mixing_connections"]
    for cut, f in connections.groupby("max_span_days"):
        selected = f[f.version.eq("date_selected_activity")]
        monthly = f[f.version.eq("monthly_mean")]
        np.testing.assert_array_equal(selected.pair_id, monthly.pair_id)
        np.testing.assert_array_equal(selected.n_months, monthly.n_months)
        assert monthly.n_months.ge(24).all(), cut
    full = connections[connections.max_span_days.eq(29) & connections.version.eq("monthly_mean")].set_index("pair_id")
    previous = pd.read_csv("experiments/phase4_transfer/doc_river_signal_mechanisms_v1/analysis/mixing_connections.csv").set_index("pair_id")
    metrics = ["source_rho", "mixture_buffer_fraction", "asynchronous_buffer_fraction", "outlet_mixture_log_sd_ratio"]
    np.testing.assert_allclose(full[metrics], previous.loc[full.index, metrics], atol=1e-10, rtol=1e-10)
    for suffix in ("", "_cn"):
        receipt = json.loads((ROOT/"figures"/f"figure_sources{suffix}.json").read_text())
        for path, expected in {**receipt["inputs"], **receipt["outputs"]}.items():
            assert sha256_file(Path(path)) == expected, path
    print("Verified raw monthly reconciliation, all date-selected triplets, concentration-independent selection,")
    print("daily-flow joins, matched populations, every sampling cut and whole-system bootstrap.")


if __name__ == "__main__":
    main()
