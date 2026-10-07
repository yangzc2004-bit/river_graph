"""Replay actual junction geometry and paired statistics without reading DOC."""

from __future__ import annotations

import argparse
import json
import sqlite3

import numpy as np
import pandas as pd
from analyze_doc_river_confluence_mixing_v1 import (
    CACHE,
    CODE,
    COLS,
    CONFIG,
    DTYPES,
    FROZEN_PANEL,
    PAIRS,
    PREVIOUS,
    ROOT,
    VAA,
    one_junction,
    summarize,
)

from river_graph.analysis.river_confluence_geometry import (
    METRICS,
    paired_form_contrasts,
)
from river_graph.experiments.provenance import sha256_file


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--full-replay", action="store_true")
    parser.add_argument("--skip-figures", action="store_true")
    args = parser.parse_args()
    a = ROOT/"analysis"
    manifest = json.loads((ROOT/"analysis_sources.json").read_text())
    for collection in ("source_hashes", "code_hashes", "product_hashes"):
        for path, expected in manifest[collection].items():
            assert sha256_file(path) == expected, path
    for source in CODE:
        assert sha256_file(source) == sha256_file(ROOT/"code_snapshot"/source)
    geometry = pd.read_csv(a/"local_junction_geometry.csv", dtype=DTYPES)
    assert len(geometry) == 295*3 and not geometry.duplicated(["station", "scale_m"]).any()
    classes = pd.read_csv(FROZEN_PANEL, dtype=DTYPES).set_index("station")
    assert np.array_equal(geometry.cluster, classes.loc[geometry.station, "cluster"].to_numpy())
    good = geometry[geometry.status.eq("measured")]
    assert np.isfinite(good[list(METRICS)]).all().all()
    assert good.incoming_angle_deg.between(0, 180).all()
    assert good.branch_a_deflection_deg.between(0, 180).all()
    assert good.branch_b_deflection_deg.between(0, 180).all()
    assert good.downstream_sinuosity.ge(1-1e-10).all()
    assert good.junction_gap_m.le(CONFIG["gap_tolerance_m"]).all()
    np.testing.assert_allclose(good.area_weighted_deflection_deg,
                               good.weight_a*good.branch_a_deflection_deg+(1-good.weight_a)*good.branch_b_deflection_deg,
                               atol=1e-12, rtol=1e-12)
    inventory = pd.read_csv(a/"mapped_geometry_inventory.csv", dtype=DTYPES)
    assert len(inventory) == 295 and inventory.station.nunique() == 295
    # Independent coordinate calculations check the stored primary-scale values.
    lines = pd.read_parquet(a/"local_centerlines.parquet")
    primary = good[good.scale_m.eq(CONFIG["primary_scale_m"])].set_index("station")
    for station, f in lines.groupby("station"):
        chords = {}
        lengths = {}
        for segment, s in f.groupby("segment"):
            xy = s.sort_values("vertex")[["x_m", "y_m"]].to_numpy()
            lengths[segment] = np.linalg.norm(np.diff(xy, axis=0), axis=1).sum()
            chords[segment] = xy[-1]-xy[0]
            assert abs(lengths[segment]-250.) < 1e-7
        cosine = chords["branch_a"]@chords["branch_b"]/(np.linalg.norm(chords["branch_a"])*np.linalg.norm(chords["branch_b"]))
        angle = np.degrees(np.arccos(np.clip(cosine, -1, 1)))
        assert abs(angle-primary.loc[station, "incoming_angle_deg"]) < 1e-8
        sinuosity = lengths["common"]/np.linalg.norm(chords["common"])
        assert abs(sinuosity-primary.loc[station, "downstream_sinuosity"]) < 1e-8
    for name, f in zip(("form_geometry_summary", "physical_junction_reuse", "measurement_scale_sensitivity"),
                       summarize(geometry), strict=True):
        expected = pd.read_csv(a/f"{name}.csv", dtype={"receiving_stations": str, "forms": str})
        pd.testing.assert_frame_equal(f, expected, check_dtype=False, atol=1e-9, rtol=1e-9)
    pairs = pd.read_csv(PAIRS, dtype=DTYPES)
    pairs = pairs[pairs.class_a.eq(1) & pairs.class_b.eq(3)].copy()
    assert len(pairs) == 22
    contrasts, details = paired_form_contrasts(geometry, pairs, draws=CONFIG["bootstrap_draws"])
    for name, f in (("matched_form_contrasts", contrasts), ("matched_pair_geometry", details)):
        expected = pd.read_csv(a/f"{name}.csv", dtype=DTYPES)
        pd.testing.assert_frame_equal(f, expected, check_dtype=False, atol=1e-9, rtol=1e-9)
    cases = pd.read_csv(PREVIOUS/"selected_footprints.csv", dtype=DTYPES)
    cases = cases[cases.cohort.eq("whole_confluence")]
    reps = pd.read_csv(PREVIOUS/"representatives.csv", dtype=DTYPES)
    selected = cases if args.full_replay else cases[cases.station.isin(reps.station)]
    vaa = pd.read_parquet(VAA, columns=COLS).set_index("comid", drop=False)
    with sqlite3.connect(f"file:{CACHE}?mode=ro", uri=True) as connection:
        for row in selected.itertuples():
            replay, _, receipt = one_junction(row, vaa, connection)
            expected = geometry[geometry.station.eq(row.station)].reset_index(drop=True)
            pd.testing.assert_frame_equal(replay, expected, check_dtype=False, atol=1e-8, rtol=1e-9)
            saved = inventory[inventory.station.eq(row.station)].iloc[0]
            assert receipt["routing_sha256"] == saved.routing_sha256
            assert receipt["used_geometry_sha256"] == saved.used_geometry_sha256
    if not args.skip_figures:
        for suffix in ("", "_cn"):
            f = json.loads((ROOT/"figures"/f"figure_sources{suffix}.json").read_text())
            for path, expected in {**f["inputs"], **f["outputs"]}.items():
                assert sha256_file(path) == expected, path
    print(f"Verified {len(geometry)} local-scale rows, {len(selected)} direct geometry replays, unchanged labels and matched pairs.")
    print("No public mixing-event or DOC outcome values substituted for unavailable observations.")


if __name__ == "__main__":
    main()
