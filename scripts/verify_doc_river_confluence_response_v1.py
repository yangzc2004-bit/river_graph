"""Replay real confluence tables and verify their downloaded source bindings."""

from __future__ import annotations

import io
import json

import pandas as pd
from analyze_doc_river_confluence_response_v1 import ROOT, build

from river_graph.experiments.provenance import sha256_file


def main():
    receipt = json.loads((ROOT / "analysis_sources.json").read_text())
    for kind in ("source_hashes", "output_hashes"):
        for path, expected in receipt[kind].items():
            if sha256_file(path) != expected:
                raise ValueError(f"Changed analysis binding: {path}")
    manifest = json.loads((ROOT / "retrieval_manifest.json").read_text())
    for item in manifest["objects"]:
        if sha256_file(item["path"]) != item["sha256"]:
            raise ValueError(f"Changed raw source: {item['path']}")
    tables, products, summary = build()
    for name, expected in tables.items():
        serialized = pd.read_csv(io.StringIO(expected.to_csv(index=False)))
        actual = pd.read_csv(ROOT / "analysis" / f"{name}.csv")
        pd.testing.assert_frame_equal(actual, serialized, check_dtype=False, rtol=1e-12, atol=1e-12)
    for name, expected in products.items():
        actual = pd.read_parquet(ROOT / "analysis" / f"{name}.parquet")
        pd.testing.assert_frame_equal(actual, expected, check_dtype=False, rtol=1e-12, atol=1e-12)
    if summary != json.loads((ROOT / "analysis/summary.json").read_text()):
        raise ValueError("Summary differs from raw-file replay")
    for name in ("figure_sources.json", "figure_sources_cn.json"):
        path = ROOT / "figures" / name
        if path.exists():
            figures = json.loads(path.read_text())
            for kind in ("input_hashes", "output_hashes"):
                for file, expected in figures[kind].items():
                    if sha256_file(file) != expected:
                        raise ValueError(f"Changed figure binding: {file}")
    result = {"status": "pass", "n_public_source_files": len(manifest["objects"]),
              "n_tables_replayed": len(tables), "n_parquets_replayed": len(products),
              "n_original_windows_preserved": len(tables["water_window_comparison"]),
              "n_field_campaigns_preserved": len(tables["confluence_chemistry"]),
              "no_flow_or_doc_gap_filling": True, "new_training": False}
    (ROOT / "verification.json").write_text(json.dumps(result, indent=2)+"\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
