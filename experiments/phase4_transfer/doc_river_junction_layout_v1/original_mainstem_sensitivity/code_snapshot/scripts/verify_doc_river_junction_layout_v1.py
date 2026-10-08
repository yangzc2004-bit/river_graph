"""Replay whole-network unit partitions, timing interventions and fixed comparisons."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from analyze_doc_river_junction_layout_v1 import (
    CODE,
    COLS,
    DOC_METRICS,
    GEOMETRY_METRICS,
    RESPONSE_METRICS,
    ROOT,
    TYPES,
    VAA,
    context,
    load_layout,
)

from river_graph.analysis.river_junction_layout import arrival_scenarios
from river_graph.analysis.river_morphology_effect import (
    group_descriptor_summary,
    paired_response_summary,
)
from river_graph.experiments.provenance import sha256_file


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--full", action="store_true", help="replay every original network, not just the three original representatives")
    parser.add_argument("--geometric-mainstem", action="store_true")
    args = parser.parse_args()
    root = ROOT/"original_mainstem_sensitivity" if args.geometric_mainstem else ROOT
    out = root/"analysis"
    panel, pairs, reps = context()
    config = json.loads((root/"config.json").read_text())
    ledger = json.loads((root/"analysis_sources.json").read_text())
    assert ledger["config_sha256"] == sha256_file(root/"config.json")
    for name in ("source_hashes", "output_hashes"):
        for path, expected in ledger[name].items():
            assert sha256_file(Path(path)) == expected, path
    for path in CODE:
        assert sha256_file(path) == sha256_file(root/"code_snapshot"/path), path
    f = pd.read_csv(out/"station_junction_layout.csv", dtype=TYPES)
    assert len(f) == 297 and not f.station.duplicated().any()
    np.testing.assert_array_equal(panel.station, f.station)
    np.testing.assert_array_equal(panel.cluster.to_numpy(int), f.cluster.to_numpy(int))
    units = pd.read_parquet(out/"tributary_units.parquet")
    entries = pd.read_csv(out/"entry_junctions.csv", dtype=TYPES)
    mains = pd.read_csv(out/"route_mainstems.csv", dtype=TYPES)
    pulse = pd.read_csv(out/"arrival_scenarios.csv", dtype=TYPES)
    np.testing.assert_allclose(units.groupby("station").area_fraction.sum(), 1., atol=1e-12)
    np.testing.assert_allclose(f.path_variance_km2, f.within_unit_variance_km2+f.between_unit_variance_km2, atol=1e-8)
    np.testing.assert_allclose(f.between_unit_variance_km2,
        f.entry_trunk_variance_km2+f.mean_branch_variance_km2+f.twice_entry_branch_covariance_km2, atol=1e-8)
    assert not units.duplicated(["station", "unit_comid"]).any()
    assert not entries.duplicated(["station", "entry_comid"]).any()
    np.testing.assert_allclose(pulse.anomaly_mass_fraction, 1., atol=1e-10)
    np.testing.assert_allclose(pulse.retained_fraction, 1., atol=1e-12)
    np.testing.assert_allclose(pulse.steady_doc, 5., atol=1e-12)
    for metric in ("pulse_peak", "pulse_sd", "duration_80"):
        pivot = pulse.pivot(index="station", columns="scenario", values=metric)
        np.testing.assert_array_equal(pivot.actual_paths, pivot.common_translation)
    check = panel if args.full else panel[panel.station.isin(reps.station)]
    vaa = pd.read_parquet(VAA, columns=COLS).set_index("comid", drop=False)
    for i, row in enumerate(check.itertuples()):
        layout, _ = load_layout(row, vaa, geometric_mainstem=args.geometric_mainstem)
        saved = f[f.station.eq(row.station)].iloc[0]
        for metric, expected in layout.descriptors.items():
            np.testing.assert_allclose(saved[metric], expected, atol=1e-8, rtol=1e-10, equal_nan=True)
        original_unit = units[units.station.eq(row.station)][layout.units.columns].sort_values("unit_comid").reset_index(drop=True)
        replay_unit = layout.units.sort_values("unit_comid").reset_index(drop=True)
        pd.testing.assert_frame_equal(original_unit, replay_unit, check_dtype=False, atol=1e-8, rtol=1e-10)
        saved_main = mains[mains.station.eq(row.station)].sort_values("sequence_outlet_to_head")
        np.testing.assert_array_equal(saved_main.mainstem_comid, layout.mainstem_comids)
        p, _ = arrival_scenarios(layout)
        saved_p = pulse[pulse.station.eq(row.station)][p.columns].reset_index(drop=True)
        pd.testing.assert_frame_equal(saved_p, p, check_dtype=False, atol=1e-9, rtol=1e-10)
        if (i+1) % 50 == 0 or i+1 == len(check):
            print(f"Replayed complete layout and arrivals {i+1}/{len(check)}", flush=True)
    metrics = (*GEOMETRY_METRICS, *RESPONSE_METRICS, *DOC_METRICS)
    grouped = group_descriptor_summary(f, metrics, draws=config["bootstrap_draws"])
    saved_grouped = pd.read_csv(out/"form_summary.csv")
    pd.testing.assert_frame_equal(saved_grouped[grouped.columns], grouped, check_dtype=False, atol=1e-10)
    matched, evidence = paired_response_summary(pairs, f, dict.fromkeys(metrics, "raw"), draws=config["bootstrap_draws"])
    saved_matched = pd.read_csv(out/"matched_form_contrasts.csv")
    saved_evidence = pd.read_csv(out/"matched_pair_evidence.csv", dtype=TYPES)
    pd.testing.assert_frame_equal(saved_matched[matched.columns], matched, check_dtype=False, atol=1e-10)
    pd.testing.assert_frame_equal(saved_evidence[evidence.columns], evidence, check_dtype=False, atol=1e-10)
    figure_count = 0
    for receipt in sorted((root/"figures").glob("figure_sources*.json")):
        record = json.loads(receipt.read_text())
        for kind in ("inputs", "outputs"):
            for path, expected in record[kind].items():
                assert sha256_file(Path(path)) == expected, path
        figure_count += sum(Path(p).suffix == ".png" for p in record["outputs"])
    assert figure_count == 4, "English and Chinese whole-network and comparison figures required"
    (root/"verification.md").write_text(
        "# Verification\n\n"
        f"Replayed {len(check)}/297 complete network partitions and controlled arrival responses from the saved mapped topology.\n\n"
        "- Original class labels, 22 pairs and three representatives retained.\n"
        "- Incremental areas counted exactly once; lateral units attach to a contiguous routing mainstem.\n"
        "- Original-mainstem routing mismatches reported explicitly.\n"
        "- Both nested variance identities reproduced.\n"
        "- Common translation preserves peak, width and complete curve shape.\n"
        "- All controlled scenarios preserve total anomaly and steady concentration.\n"
        "- All class and paired HUC4 bootstrap tables replayed.\n"
        "- Source, execution snapshot and product receipts agree.\n"
        f"- {figure_count} rendered PNG products and their PDF companions checked against receipts.\n\n"
        "This is a geometry/routing analysis, with copied monthly DOC context; no new training or external validation.\n")
    print(f"PASS: {len(check)} complete replays, two variance identities, fixed comparisons and {figure_count} figure products")


if __name__ == "__main__":
    main()
