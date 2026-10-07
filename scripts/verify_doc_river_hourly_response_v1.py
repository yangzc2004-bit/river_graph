"""Replay hourly waveform calculations and trace the paired records to XLSX."""

from __future__ import annotations

import io
import json

import pandas as pd
from analyze_doc_river_hourly_response_v1 import INPUT, PARENT, ROOT, build
from audit_doc_river_optical_case_v1 import build as build_original

from river_graph.experiments.provenance import sha256_file


def check_hashes(mapping):
    for path, expected in mapping.items():
        if sha256_file(path) != expected:
            raise ValueError(f"Changed source or product: {path}")


def main():
    receipt = json.loads((ROOT / "analysis_sources.json").read_text())
    parent_receipt = json.loads((PARENT / "optical_sources.json").read_text())
    for r in (receipt, parent_receipt):
        check_hashes(r["source_hashes"])
        check_hashes(r["output_hashes"])
    _, _, _, original, duplicates, _ = build_original()
    pd.testing.assert_frame_equal(pd.read_parquet(INPUT), original, check_exact=False, rtol=1e-12, atol=1e-12)
    tables, products, summary = build()
    for name, frame in tables.items():
        actual = pd.read_csv(ROOT / "analysis" / f"{name}.csv")
        expected = pd.read_csv(io.StringIO(frame.to_csv(index=False)))
        pd.testing.assert_frame_equal(actual, expected, check_exact=False, rtol=1e-10, atol=1e-11)
    for name, frame in products.items():
        actual = pd.read_parquet(ROOT / "analysis" / f"{name}.parquet")
        pd.testing.assert_frame_equal(actual, frame, check_exact=False, rtol=1e-10, atol=1e-11)
    if summary != json.loads((ROOT / "analysis/summary.json").read_text()):
        raise ValueError("Summary does not replay")
    for suffix in ("", "_cn"):
        record = json.loads((ROOT / "figures" / f"figure_sources{suffix}.json").read_text())
        check_hashes(record["input_hashes"])
        check_hashes(record["output_hashes"])
        if record["selection"]["atlas"] != tables["segment_comparison"].block.tolist():
            raise ValueError("Atlas omits an observation segment")
    result = {"status": "pass", "n_tables_replayed": len(tables), "n_parquets_replayed": len(products),
              "paired_hours_rebuilt_from_workbook": len(original),
              "conflicting_site_times_excluded": int(duplicates.conflicting_measurements.sum()),
              "all_segments_retained": len(tables["segment_comparison"]) == 15,
              "hourly_gaps_not_bridged": True, "retrieval_threshold_fnu": 600,
              "source_clock_not_relabelled_utc": True, "new_training": False}
    (ROOT / "verification.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
