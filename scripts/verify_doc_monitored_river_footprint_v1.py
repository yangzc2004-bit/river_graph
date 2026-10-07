"""Replay matched river geometry, controlled routing and fixed-model inputs."""
from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd
from analyze_doc_monitored_river_footprint_v1 import (
    CODE,
    OLD,
    ROOT,
    build_geometry,
    digest,
    observed_tables,
    predict_fixed,
    score_predictions,
    summarize_errors,
)
from plot_doc_monitored_river_footprint_v1 import CACHE, mapped_corridor

from river_graph.analysis.river_monitored_footprint import (
    geometry_examples,
    matched_pulses,
)
from river_graph.experiments.provenance import sha256_file

DTYPES = {"source_a": str, "source_b": str, "target": str, "huc4": str,
          "max_leverage_receiver": str, "omitted_block": str}


def equal_table(name, computed):
    saved = pd.read_csv(ROOT/"analysis"/f"{name}.csv", dtype=DTYPES)
    pd.testing.assert_frame_equal(computed.reset_index(drop=True), saved,
        check_dtype=False, check_exact=False, atol=1e-10, rtol=1e-10)


def main():
    a = ROOT/"analysis"
    sources = json.loads((ROOT/"analysis_sources.json").read_text())
    config = json.loads((ROOT/"config.json").read_text())
    for path, expected in sources["source_hashes"].items():
        assert sha256_file(Path(path)) == expected, path
    for path in CODE:
        assert sha256_file(path) == sha256_file(ROOT/"code_snapshot"/path), path
    assert digest(config) == sources["config_hash"]
    assert config["bootstrap_draws"] == 5000 and not config["calibrators_refitted"]
    inputs, geometry, reaches = build_geometry()
    equal_table("footprints", geometry)
    equal_table("corridor_reaches", reaches)
    assert len(geometry) == 59 and geometry.target.nunique() == 22 and geometry.component.nunique() == 11
    assert len(inputs) == 3026 and len(inputs.drop_duplicates(["target", "month_index"])) == 1092
    assert geometry.station_measure_imputed.eq(False).all()
    assert not reaches.duplicated(["pair_id", "comid"]).any()
    np.testing.assert_allclose(geometry.total_path_cv, (1-geometry.common_fraction)*geometry.independent_branch_cv, atol=1e-12)
    np.testing.assert_allclose(geometry.path_a_km, geometry.branch_a_km+geometry.common_km, atol=1e-10)
    np.testing.assert_allclose(geometry.path_b_km, geometry.branch_b_km+geometry.common_km, atol=1e-10)
    reps, cuts = geometry_examples(geometry)
    equal_table("representatives", reps)
    assert cuts == config["geometry_example_cuts"]
    changed = geometry.assign(doc_gain=np.random.default_rng(1).normal(size=len(geometry))*1e6)
    pd.testing.assert_frame_equal(reps, geometry_examples(changed)[0])
    pulse_rows, traces = [], []
    for row in geometry.itertuples():
        p, trace = matched_pulses(row.path_a_km, row.path_b_km, row.common_km, row.weight_a)
        q = p.set_index("scenario")
        np.testing.assert_allclose(q.loc["actual", ["pulse_peak", "pulse_sd"]].to_numpy(float),
                                   q.loc["no_common_trunk", ["pulse_peak", "pulse_sd"]].to_numpy(float), atol=1e-13)
        assert q.loc["actual", "pulse_peak"] < q.loc["equal_branches", "pulse_peak"]
        np.testing.assert_allclose(q.loc["actual", "pulse_centroid"]-q.loc["no_common_trunk", "pulse_centroid"], row.common_fraction, atol=1e-13)
        np.testing.assert_allclose(p.anomaly_mass_fraction, 1., atol=1e-12)
        np.testing.assert_allclose(p.steady_doc, 5., atol=1e-12)
        np.testing.assert_allclose(q.loc["actual", "pulse_sd"], np.sqrt(config["pulse_sd"]**2+row.total_path_cv**2), atol=1e-12)
        p["pair_id"], p["target"], p["component"] = row.pair_id, row.target, row.component
        pulse_rows.append(p)
        if row.pair_id in set(reps.pair_id):
            trace["pair_id"] = row.pair_id
            traces.append(trace)
    equal_table("pulse_scenarios", pd.concat(pulse_rows, ignore_index=True))
    pd.testing.assert_frame_equal(pd.concat(traces, ignore_index=True), pd.read_parquet(a/"representative_pulses.parquet"), check_exact=True)
    states = json.loads((OLD/"analysis/fitted_states.json").read_text())
    pieces, predictions = predict_fixed(inputs, geometry, states, config["max_path_km"])
    pd.testing.assert_frame_equal(predictions, pd.read_parquet(a/"fixed_predictions.parquet"), check_exact=True)
    connections, receivers = score_predictions(predictions)
    equal_table("connection_errors", connections)
    equal_table("receiver_errors", receivers)
    tables = {**summarize_errors(receivers, 5000), **observed_tables(geometry, pieces, 5000)}
    for name, table in tables.items():
        equal_table(name, table)
    pd.testing.assert_frame_equal(pieces, pd.read_parquet(a/"timing_inputs.parquet"), check_exact=True)
    original = pd.read_parquet(OLD/"analysis/connection_predictions.parquet")
    original = original[original.operator.eq("mean_delay")].set_index(["pair_id", "month_index"])
    reference = predictions[predictions.operator.eq("equal_branches")].set_index(["pair_id", "month_index"])
    np.testing.assert_allclose(reference.y_pred, original.loc[reference.index, "y_pred"], atol=1e-10, rtol=0)
    np.testing.assert_array_equal(reference.q90_threshold, original.loc[reference.index, "q90_threshold"])
    # Receiver truth is never a feature: replay all fixed coefficients after a
    # large hidden-label perturbation, including the timing inputs themselves.
    altered = inputs.copy()
    altered["y_true"] = altered.y_true+9999.
    other_pieces, other_predictions = predict_fixed(altered, geometry, states, config["max_path_km"])
    np.testing.assert_array_equal(predictions.y_pred, other_predictions.y_pred)
    columns = ["same_month_proxy", "shared_delay_input", "equal_branch_delay_input", "differential_arrival_input",
               "common_only_proxy", "mean_delay_proxy", "actual_branch_proxy"]
    np.testing.assert_array_equal(pieces[columns], other_pieces[columns])
    changed_future = inputs.copy()
    last = int(inputs.month_index.max())
    changed_future.loc[changed_future.month_index.eq(last), "doc_a_now"] += 999.
    later_pieces, later_predictions = predict_fixed(changed_future, geometry, states, config["max_path_km"])
    np.testing.assert_array_equal(pieces.loc[pieces.month_index.lt(last), columns],
                                  later_pieces.loc[later_pieces.month_index.lt(last), columns])
    np.testing.assert_array_equal(predictions.loc[predictions.month_index.lt(last), "y_pred"],
                                  later_predictions.loc[later_predictions.month_index.lt(last), "y_pred"])
    sidecar = json.loads((a/"fixed_predictions.provenance.json").read_text())
    origin = json.loads((OLD/"analysis/connection_predictions.provenance.json").read_text())
    for key in ("dataset_hash", "mask_hash", "source_cells_sha256"):
        assert sidecar[key] == origin[key]
    assert sidecar["config_hash"] == digest(config)
    assert sidecar["prediction_sha256"] == sha256_file(a/"fixed_predictions.parquet")
    assert sidecar["saved_states_sha256"] == sha256_file(OLD/"analysis/fitted_states.json")
    assert sidecar["input_records_sha256"] == sha256_file(OLD/"analysis/input_records.parquet")
    assert sidecar["runtime_snapshot_hash"] == digest(sidecar["runtime_sources"])
    for path, expected in sidecar["runtime_sources"].items():
        assert sha256_file(Path(path)) == expected
    with sqlite3.connect(f"file:{CACHE}?mode=ro", uri=True) as connection:
        for suffix in ("", "_cn"):
            manifest = json.loads((ROOT/"figures"/f"manifest{suffix}.json").read_text())
            assert manifest["generator_sha256"] == sha256_file(Path("scripts/plot_doc_monitored_river_footprint_v1.py"))
            assert manifest["geometry_helper_sha256"] == sha256_file(Path("src/river_graph/analysis/river_monitored_footprint.py"))
            for path, expected in {**manifest["source_hashes"], **manifest["figure_hashes"]}.items():
                assert sha256_file(Path(path)) == expected, path
            for row in reps.itertuples():
                _, gap, gh = mapped_corridor(connection, reaches[reaches.pair_id.eq(row.pair_id)])
                assert gh == manifest["flowline_hashes"][row.pair_id]
                assert gap == manifest["max_endpoint_gaps_m"][row.pair_id] == 0.
    result = {"status": "passed", "connections": 59, "receivers": 22, "monitoring_systems": 11,
        "connection_months": 3026, "unique_receiver_months": 1092, "replayed_pulse_scenarios": 177,
        "replayed_predictions": len(predictions), "replayed_tables": len(tables)+6, "adjusted_associations": 6,
        "hidden_labels_and_future_inputs": "unchanged predictions", "saved_complete_reference": "equal within 1e-10 mg/L",
        "geometry_examples_doc_independent": True, "common_translation_mass_and_width": "passed",
        "maps": "actual cropped station-to-station geometry, zero endpoint gaps", "bootstrap_draws": 5000,
        "model_refitting": False, "external_validation": False}
    (ROOT/"verification.json").write_text(json.dumps(result, indent=2)+"\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
