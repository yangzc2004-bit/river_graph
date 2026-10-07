"""Replay storage partitions, summaries and matched contrasts for the synthesis."""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
from analyze_doc_river_structure_profiles_v1 import (
    COLS,
    ROOT,
    VAA,
    build_tables,
    partition_selected,
    read_context,
)

from river_graph.experiments.provenance import sha256_file


def check_hashes(mapping):
    for path, expected in mapping.items():
        if sha256_file(path) != expected:
            raise ValueError(f"Content changed: {path}")


def main():
    ledger = json.loads((ROOT/"analysis_sources.json").read_text())
    check_hashes(ledger["source_hashes"])
    check_hashes(ledger["output_hashes"])
    if sha256_file(ROOT/"config.json") != ledger["config_sha256"]:
        raise ValueError("configuration changed")
    for path, digest in ledger["source_hashes"].items():
        snapshot = ROOT/"code_snapshot"/path
        if snapshot.exists() and sha256_file(snapshot) != digest:
            raise ValueError(f"execution snapshot changed: {path}")
    panel, pairs, whole, selected, routing = read_context()
    vaa = pd.read_parquet(VAA, columns=COLS).set_index("comid", drop=False)
    partition, _ = partition_selected(selected, vaa)
    config = json.loads((ROOT/"config.json").read_text())
    tables = build_tables(panel, whole, selected, partition, routing, pairs, config["bootstrap_draws"])
    for name, expected in tables.items():
        actual = pd.read_csv(ROOT/"analysis"/f"{name}.csv", dtype={"station": str, "target": str, "huc4": str,
                            "station_a": str, "station_b": str})
        # CSVs deliberately omit pandas' in-memory row index; retain row order.
        pd.testing.assert_frame_equal(actual.reset_index(drop=True), expected.reset_index(drop=True),
                                      check_dtype=False, check_exact=False, atol=1e-11, rtol=1e-10)
    frame = tables["station_mechanism_profiles"]
    # The shared-path/independent-path representation conserves actual lengths.
    eligible = frame[frame.major_confluence_available]
    if len(frame) != 297 or len(eligible) != 295 or len(pairs) != 22:
        raise ValueError("fixed populations changed")
    np.testing.assert_allclose(frame.arrival_dispersion, frame.route_distance_cv, atol=1e-10)
    counts = tables["storage_position_counts"]
    np.testing.assert_allclose(counts.groupby("cluster").fraction.sum(), 1., atol=1e-12)
    if not eligible.corridor_storage_path_share.between(0, 1).all():
        raise ValueError("mapped storage cannot exceed the path budget")
    for name in ("figure_sources.json", "figure_sources_cn.json"):
        receipt = json.loads((ROOT/"figures"/name).read_text())
        check_hashes(receipt["input_hashes"])
        check_hashes(receipt["output_hashes"])
    result = {"status": "pass", "n_profiles": 297, "n_replayed_corridors": 295,
              "n_matched_pairs": 22, "n_matched_huc4": int(pairs.huc4.nunique()),
              "bootstrap_draws": config["bootstrap_draws"], "n_tables_replayed": len(tables),
              "n_monitored_receivers": 22, "original_classes_preserved": True,
              "source_and_figure_hashes_verified": True, "new_model_training": False}
    (ROOT/"verification.json").write_text(json.dumps(result, indent=2)+"\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
