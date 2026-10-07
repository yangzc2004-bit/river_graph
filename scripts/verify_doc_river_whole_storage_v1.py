"""Replay real paths, physical identities, statistics and exported figures."""
from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd
from analyze_doc_river_whole_storage_v1 import (
    CODE,
    ROOT,
    add_contrasts,
    context,
    simulate,
    summaries,
)

from river_graph.experiments.provenance import sha256_file


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--full-replay", action="store_true")
    args = parser.parse_args()
    a = ROOT/"analysis"
    manifest = json.loads((ROOT/"analysis_sources.json").read_text())
    for path, expected in {**manifest["source_hashes"], **manifest["product_hashes"]}.items():
        assert sha256_file(path) == expected, path
    assert sha256_file(ROOT/"config.json") == manifest["config_hash"]
    for path in CODE:
        assert sha256_file(path) == sha256_file(ROOT/"code_snapshot"/path), path
    panel, pairs, reps, vaa = context()
    frame = pd.read_csv(a/"scenario_metrics.csv", dtype={"station": str, "huc4": str})
    geometry = pd.read_csv(a/"network_descriptors.csv", dtype={"station": str, "huc4": str})
    assert len(panel) == 297 and panel.comid.nunique() == 295
    assert len(frame) == 297*14 and frame.station.nunique() == 297
    np.testing.assert_allclose(frame.numerical_centroid, frame.pulse_centroid, atol=1e-8, rtol=0)
    np.testing.assert_allclose(frame.numerical_sd, frame.pulse_sd, atol=1e-8, rtol=0)
    np.testing.assert_allclose(frame.anomaly_area_fraction, 1., atol=1e-10, rtol=0)
    np.testing.assert_allclose(frame.steady_gain, 1., atol=1e-12, rtol=0)
    assert frame.storage_variance.ge(0).all()
    np.testing.assert_allclose(frame.pulse_sd**2,
        frame.input_variance+frame.path_variance+frame.storage_variance, atol=1e-12)
    assert geometry.max_route_error_km.max() < 1e-7
    assert np.all(geometry.serial_storage_variance <= geometry.lumped_storage_variance+1e-10)
    selected = set(panel.station) if args.full_replay else set(reps.station)
    selected |= set(geometry.nlargest(1, "serial_storage_variance").station)
    for row in panel.itertuples():
        if row.station not in selected:
            continue
        descriptor, rows, _ = simulate(row, vaa)
        expected = frame[frame.station.eq(row.station)].reset_index(drop=True)
        replay = pd.DataFrame(rows)
        pd.testing.assert_frame_equal(replay, expected[replay.columns], check_dtype=False,
                                      atol=1e-10, rtol=1e-10)
        saved = geometry[geometry.station.eq(row.station)].iloc[0]
        for key, value in descriptor.items():
            np.testing.assert_allclose(value, saved[key], atol=1e-10, rtol=1e-10)
    np.testing.assert_allclose(add_contrasts(frame).peak_reduction_pct, frame.peak_reduction_pct, atol=1e-10)
    for name, table in summaries(frame, geometry, pairs).items():
        saved = pd.read_csv(a/f"{name}.csv", dtype={"station_a": str, "station_b": str, "huc4": str})
        pd.testing.assert_frame_equal(table, saved, check_dtype=False, atol=1e-10, rtol=1e-10)
    convergence = pd.read_csv(a/"numerical_convergence.csv")
    limits = {"pulse_peak": 1e-4, "duration_80": 1e-4, "numerical_centroid": 1e-8,
              "numerical_sd": 1e-8, "anomaly_area_fraction": 1e-10}
    for row in convergence.itertuples():
        assert row.max_absolute_difference <= limits[row.metric], row
    for suffix in ("", "_cn"):
        receipt = json.loads((ROOT/"figures"/f"figure_sources{suffix}.json").read_text())
        for path, expected in {**receipt["inputs"], **receipt["outputs"]}.items():
            assert sha256_file(path) == expected, path
    print(f"Verified 297 river instances / 295 outlets; 4,158 scenarios; {len(selected)} pathwise replays;")
    print("serial waterbody moments, fixed mean, unit gain, 5-point sensitivity, HUC4 summaries and figures.")


if __name__ == "__main__":
    main()
