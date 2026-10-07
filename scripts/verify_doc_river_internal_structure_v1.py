"""Replay geometry-first profiles, fixed-mean scenarios and observed diagnostics."""
from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd
from analyze_doc_river_internal_structure_v1 import (
    CODE,
    ROOT,
    assign_profiles,
    build_geometry,
    classify_geometry,
    digest,
    map_stations,
    observed_tables,
    representatives,
    routing_scenarios,
    summarize_geometry,
)
from plot_doc_river_internal_structure_v1 import CACHE, geometry_digest

from river_graph.experiments.provenance import sha256_file

DTYPES = {"station": str, "target": str, "huc_cd": str, "huc4": str,
          "max_leverage_receiver": str, "omitted_block": str}


def equal_table(name, computed):
    saved = pd.read_csv(ROOT/"analysis"/f"{name}.csv", dtype=DTYPES)
    pd.testing.assert_frame_equal(computed.reset_index(drop=True), saved,
        check_dtype=False, check_exact=False, atol=1e-10, rtol=1e-10)


def main():
    sources = json.loads((ROOT/"analysis_sources.json").read_text())
    for path, expected in sources["source_hashes"].items():
        assert sha256_file(Path(path)) == expected, path
    for path in CODE:
        assert sha256_file(path) == sha256_file(ROOT/"code_snapshot"/path), path
    config = json.loads((ROOT/"config.json").read_text())
    assert digest(config) == sources["config_hash"]
    frame, distributions, _ = build_geometry()
    profiles, cuts = classify_geometry(frame)
    assert len(profiles) == config["geometry_networks"] == 322
    assert not profiles.comid.duplicated().any()
    assert cuts == {name: config[name] for name in ("balance_cut", "path_cut")}
    equal_table("network_profiles", profiles)
    reps = representatives(profiles)
    equal_table("representatives", reps)
    # DOC cannot change either a structural assignment or the example selection.
    altered = frame.copy()
    altered["doc_mean"] = np.random.default_rng(5).normal(size=len(frame))*10000
    second, second_cuts = classify_geometry(altered)
    assert cuts == second_cuts
    np.testing.assert_array_equal(profiles.profile, second.profile)
    pd.testing.assert_frame_equal(reps, representatives(second))
    mapping = map_stations(profiles)
    assert mapping.groupby("comid").profile.nunique().eq(1).all()
    rows, traces = [], []
    for row in profiles.itertuples():
        weights, paths = distributions[int(row.comid)]
        assert np.all(weights > 0) and np.isfinite(paths).all() and np.all(paths >= 0)
        np.testing.assert_allclose(np.dot(weights, paths), 1., rtol=0, atol=1e-12)
        table, pulse = routing_scenarios(weights, paths)
        table["comid"], table["profile"], table["cluster"] = row.comid, row.profile, int(row.cluster)
        table["station"], table["huc4"] = row.station, row.huc4
        rows.append(table)
        fractions = np.array(config["path_spread_fractions"])
        np.testing.assert_allclose(table.pulse_sd, np.sqrt(config["pulse_sd"]**2+(fractions*row.path_cv)**2), atol=1e-12)
        if row.comid in set(reps.comid):
            pulse["comid"], pulse["profile"] = row.comid, row.profile
            traces.append(pulse)
    routing = pd.concat(rows, ignore_index=True)
    equal_table("normalized_routing", routing)
    np.testing.assert_allclose(routing.anomaly_mass_fraction, 1., atol=1e-10)
    np.testing.assert_allclose(routing.steady_doc, 5., atol=1e-12)
    np.testing.assert_allclose(routing.mean_travel_delay, 1., atol=1e-12)
    np.testing.assert_allclose(routing[routing.scenario.eq("zero_spread")].pulse_peak, 1., atol=1e-10)
    assert np.isfinite(routing.select_dtypes("number")).all().all()
    pd.testing.assert_frame_equal(pd.concat(traces, ignore_index=True),
        pd.read_parquet(ROOT/"analysis/representative_pulses.parquet"), check_exact=True)
    tables = {**summarize_geometry(profiles, routing), **observed_tables(profiles, config["bootstrap_draws"])}
    for name, table in tables.items():
        equal_table(name, table)
    counts = tables["outline_profile_counts"]
    assert counts.filter(like="profile_").to_numpy().sum() == 322
    assert counts.profile_0.sum() == 2
    observed = tables["observed_receivers"]
    assert len(observed) == 22 and observed.target.nunique() == 22
    assert observed.component.nunique() == 11 and observed.n_connections.sum() == 59
    assert len(tables["mapped_flow_responses"]) == 410
    assert tables["mapped_flow_responses"].station.nunique() == 205
    associations = tables["continuous_observed_associations"]
    assert len(associations) == 4 and associations.n_receivers.eq(22).all()
    assert associations.requested_bootstrap_draws.eq(5000).all()
    assert associations.valid_bootstrap_draws.between(1, 5000).all()
    fixed = assign_profiles(frame, **config["fixed_reference_cut_points"])
    equal_table("fixed_reference_profiles", fixed)
    equal_table("fixed_reference_counts", summarize_geometry(fixed, routing)["outline_profile_counts"])
    fixed_tables = observed_tables(fixed, config["bootstrap_draws"])
    equal_table("fixed_reference_observed_summary", fixed_tables["observed_buffer_summary"])
    single = fixed_tables["observed_buffer_summary"].query("n_blocks == 1")
    assert len(single) and single[["ci_low", "ci_high"]].isna().all().all()
    with sqlite3.connect(f"file:{CACHE/'flowlines.sqlite'}?mode=ro", uri=True) as connection:
        for suffix in ("", "_cn"):
            manifest = json.loads((ROOT/"figures"/f"manifest{suffix}.json").read_text())
            assert manifest["generator_sha256"] == sha256_file(Path("scripts/plot_doc_river_internal_structure_v1.py"))
            for path, expected in {**manifest["source_hashes"], **manifest["figure_hashes"]}.items():
                assert sha256_file(Path(path)) == expected, path
            for row in reps.itertuples():
                with np.load(CACHE/"members_full"/f"comid_{int(row.comid)}.npz") as member:
                    assert len(member["comids"]) == len(np.unique(member["comids"]))
                    assert set(member["mainstem"]).issubset(set(member["comids"]))
                    assert geometry_digest(connection, member["comids"]) == manifest["representative_flowline_hashes"][str(row.comid)]
    result = {"status": "passed", "unique_geometry_networks": len(profiles), "junction_free_networks": 2,
        "geometry_and_examples_doc_independent": True, "replayed_routing_scenarios": len(routing),
        "fixed_mean_delay_and_conserved_mass": True, "observed_receivers": len(observed),
        "observed_monitoring_systems": 11, "observed_connections": 59, "bootstrap_draws": config["bootstrap_draws"],
        "replayed_tables": len(tables)+3, "replayed_pulse_products": 1,
        "adjusted_associations": 4, "single_system_sensitivity_intervals": "unavailable",
        "real_map_geometry_and_figure_manifests": "passed", "new_prediction_training": False}
    (ROOT/"verification.json").write_text(json.dumps(result, indent=2)+"\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
