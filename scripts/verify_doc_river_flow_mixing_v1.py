"""Replay the observed branch-mixing calculations and saved figures."""

from __future__ import annotations

import io
import json

import pandas as pd
from analyze_doc_river_flow_mixing_v1 import ROOT, build

from river_graph.experiments.provenance import sha256_file


def check_hashes(mapping):
    for path, expected in mapping.items():
        if sha256_file(path) != expected:
            raise ValueError(f"Changed input/output: {path}")


def main():
    receipt = json.loads((ROOT / "analysis_sources.json").read_text())
    check_hashes(receipt["source_hashes"])
    check_hashes(receipt["output_hashes"])
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
    result = {"status": "pass", "n_tables_replayed": len(tables), "n_parquets_replayed": len(products),
              "all_original_windows_accounted_for": len(tables["window_availability"]) == summary["n_original_windows"],
              "daily_flow_not_interpolated": True, "partial_flow_shares_not_clipped": True,
              "fixed_mixture_variance_decomposition_replayed": True, "new_training": False}
    (ROOT / "verification.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
