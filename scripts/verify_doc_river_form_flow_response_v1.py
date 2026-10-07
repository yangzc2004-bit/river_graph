"""Replay river-form flow responses, original pairs and flow-only thresholds."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from analyze_doc_river_form_flow_response_v1 import (
    CELLS,
    CODE,
    DATASET,
    PAIRS,
    ROOT,
    analyze,
    build_inputs,
    digest,
)

from river_graph.experiments.provenance import sha256_file


def compare_csv(frame, path):
    saved = pd.read_csv(path, dtype={c: str for c in ("station", "station_a", "station_b", "huc4", "omitted_huc4")})
    pd.testing.assert_frame_equal(frame.reset_index(drop=True), saved.reset_index(drop=True),
                                  check_dtype=False, check_exact=False, atol=1e-10, rtol=1e-10)


def main():
    sources = json.loads((ROOT/"analysis_sources.json").read_text())
    for path, expected in sources["source_hashes"].items():
        assert sha256_file(Path(path)) == expected, path
    for path in CODE:
        assert sha256_file(path) == sha256_file(ROOT/"code_snapshot"/path)
    config = json.loads((ROOT/"config.json").read_text())
    assert digest(config) == sources["config_hash"]
    data = torch.load(DATASET, map_location="cpu", weights_only=False)
    frame, pairs, hydro, thresholds = build_inputs(data)
    out = ROOT/"analysis"
    pd.testing.assert_frame_equal(hydro, pd.read_parquet(out/"flow_reference_months.parquet"), check_exact=True)
    compare_csv(thresholds, out/"flow_reference_thresholds.csv")

    changed = dict(data)
    changed["y"] = np.asarray(data["y"]).copy()
    permitted = np.zeros(changed["y"].size, bool)
    permitted[np.load(CELLS)] = True
    changed["y"].ravel()[~permitted] = 123456.
    replay, changed_pairs, changed_hydro, changed_thresholds = build_inputs(changed)
    pd.testing.assert_frame_equal(frame, replay, check_exact=True)
    pd.testing.assert_frame_equal(pairs, changed_pairs, check_exact=True)
    pd.testing.assert_frame_equal(hydro, changed_hydro, check_exact=True)
    pd.testing.assert_frame_equal(thresholds, changed_thresholds, check_exact=True)
    # Even observed DOC perturbation cannot change the flow states or reference.
    changed["y"][:] = 123456.
    altered, _, _, altered_thresholds = build_inputs(changed)
    pd.testing.assert_frame_equal(thresholds, altered_thresholds, check_exact=True)
    pd.testing.assert_frame_equal(frame[["station", "month_index", "flow_state"]],
                                  altered[["station", "month_index", "flow_state"]], check_exact=True)

    tables, products, fitted = analyze(frame, pairs, sources["bootstrap_draws"])
    for name, table in tables.items():
        compare_csv(table, out/f"{name}.csv")
    for name, product in products.items():
        pd.testing.assert_frame_equal(product, pd.read_parquet(out/f"{name}.parquet"), check_exact=True)
    assert fitted == json.loads((out/"fitted_states.json").read_text())
    original = pd.read_csv(PAIRS, dtype={"station_a": str, "station_b": str, "huc4": str})
    pd.testing.assert_frame_equal(pairs, original, check_exact=True)
    look = frame.set_index(["station", "month_index"])
    records = products["shared_pair_months"]
    for suffix in ("a", "b"):
        keys = pd.MultiIndex.from_arrays([records["station_"+suffix], records.month_index])
        np.testing.assert_array_equal(records.date, look.loc[keys, "date"])
        np.testing.assert_array_equal(records["y_true_"+suffix], look.loc[keys, "y_true"])
        np.testing.assert_array_equal(records["flow_state_"+suffix], look.loc[keys, "flow_state"])
    # Empty geographic uncertainty stays empty for sparse one-HUC4 comparisons.
    contrasts = tables["pair_response_contrasts"]
    assert contrasts[contrasts.n_huc4.eq(1)][["ci_low", "ci_high"]].isna().all().all()
    for suffix in ("", "_cn"):
        manifest = json.loads((ROOT/"figures"/f"manifest{suffix}.json").read_text())
        assert manifest["generator_sha256"] == sha256_file(Path("scripts/plot_doc_river_form_flow_response_v1.py"))
        for path, expected in {**manifest["source_hashes"], **manifest["figure_hashes"]}.items():
            assert sha256_file(Path(path)) == expected, path
    result = {"status": "passed", "n_source_stations": frame.station.nunique(), "n_observed_months": len(frame),
              "nonpermitted_doc_perturbation": "passed", "doc_independent_flow_states": "passed",
              "thresholds_use_non_doc_flow_months": bool(hydro.flow_measured.sum() > frame.discharge_valid.sum()),
              "original_pairs_preserved": len(pairs), "same_calendar_alignment": "passed",
              "replayed_tables": len(tables)+1, "replayed_fits": len(fitted["station_fits"]),
              "bootstrap_draws": sources["bootstrap_draws"], "single_region_intervals": "unavailable",
              "figure_manifests": "passed", "neural_training": False}
    (ROOT/"verification.json").write_text(json.dumps(result, indent=2)+"\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
