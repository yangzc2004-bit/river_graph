"""Replay the complete branch/storage experiment and its analytical controls."""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
from analyze_doc_river_storage_transport_v1 import (
    CODE,
    DT,
    DTYPES,
    FOOTPRINT,
    ROOT,
    numerical_comparison,
    simulate,
    summaries,
)

from river_graph.experiments.provenance import sha256_file


def equal_csv(name, frame):
    saved = pd.read_csv(ROOT/"analysis"/f"{name}.csv", dtype=DTYPES)
    pd.testing.assert_frame_equal(frame.reset_index(drop=True), saved, check_dtype=False,
                                  atol=1e-10, rtol=1e-10)


def main():
    a = ROOT/"analysis"
    manifest = json.loads((ROOT/"analysis_sources.json").read_text())
    for path, expected in {**manifest["source_hashes"], **manifest["product_hashes"]}.items():
        assert sha256_file(path) == expected, path
    for path in CODE:
        assert sha256_file(path) == sha256_file(ROOT/"code_snapshot"/path), path
    f = pd.read_csv(FOOTPRINT, dtype=DTYPES)
    reps = pd.read_csv(a/"representatives.csv", dtype=DTYPES)
    scenarios, contrasts, traces = simulate(f, reps)
    assert len(f) == 59 and f.target.nunique() == 22 and f.component.nunique() == 11
    assert len(scenarios) == 1416 and len(contrasts) == 708
    equal_csv("scenario_metrics", scenarios)
    equal_csv("structural_contrasts", contrasts)
    pd.testing.assert_frame_equal(traces, pd.read_parquet(a/"representative_responses.parquet"), check_exact=True)
    receivers, cohort, classes = summaries(contrasts)
    for name, table in (("receiver_contrasts", receivers), ("cohort_summary", cohort), ("outline_context", classes)):
        equal_csv(name, table)
    np.testing.assert_allclose(scenarios.pulse_centroid, 1., atol=1e-9)
    np.testing.assert_allclose(scenarios.pulse_sd, scenarios.analytic_sd, atol=1e-9)
    np.testing.assert_allclose(scenarios.anomaly_area_fraction, 1., atol=1e-10)
    assert scenarios.steady_gain.eq(1).all() and scenarios.common_translation.ge(0).all()
    assert contrasts.storage_peak_reduction_pct.ge(-1e-8).all()
    assert contrasts.branch_peak_reduction_pct.ge(-1e-8).all()
    fine, _, _ = simulate(f, reps, dt=DT/2, keep_curves=False)
    convergence = numerical_comparison(scenarios, fine)
    equal_csv("numerical_convergence", convergence)
    limits = {"pulse_peak": 1e-4, "peak_time": DT, "pulse_sd": 1e-9,
              "pulse_centroid": 1e-9, "duration_80": 1e-4, "anomaly_area_fraction": 1e-10}
    for row in convergence.itertuples():
        assert row.max_absolute_difference <= limits[row.metric]+1e-12, row
    for suffix in ("", "_cn"):
        receipt = json.loads((ROOT/"figures"/f"figure_sources{suffix}.json").read_text())
        for path, expected in {**receipt["inputs"], **receipt["outputs"]}.items():
            assert sha256_file(path) == expected, path
    print("Verified 1,416 scenarios on 59 corridors; fixed mean, unit gain, analytical variance,")
    print("all paired structural contrasts, receiver weighting, finer-grid convergence and figures.")


if __name__ == "__main__":
    main()
