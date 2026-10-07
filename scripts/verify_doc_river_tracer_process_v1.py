"""Replay all original-point paired-tracer tables and inspect their source chains."""

from __future__ import annotations

import io
import json
from pathlib import Path

import numpy as np
import pandas as pd
from analyze_doc_river_tracer_process_v1 import ROOT, build

from river_graph.experiments.provenance import sha256_file


def main():
    receipt = json.loads((ROOT / "analysis_sources.json").read_text())
    for section in ("sources", "products"):
        for item in receipt[section]:
            if sha256_file(item["path"]) != item["sha256"]:
                raise ValueError(f"Changed paired-tracer source/product: {item['path']}")
    for item in receipt["sources"]:
        path = Path(item["path"])
        if str(path).startswith(("scripts/", "src/")) and sha256_file(path) != sha256_file(ROOT / "code_snapshot" / path):
            raise ValueError(f"Changed execution snapshot: {path}")
    tables, products, summary = build()
    for name, frame in tables.items():
        serialized = pd.read_csv(io.StringIO(frame.to_csv(index=False)))
        pd.testing.assert_frame_equal(pd.read_csv(ROOT / f"analysis/{name}.csv"), serialized,
                                      check_dtype=False, rtol=1e-12, atol=1e-12)
    for name, frame in products.items():
        pd.testing.assert_frame_equal(pd.read_parquet(ROOT / f"analysis/{name}.parquet"), frame,
                                      check_dtype=False, rtol=1e-12, atol=1e-12)
    if summary != json.loads((ROOT / "analysis/summary.json").read_text()):
        raise ValueError("Paired-tracer summary differs on full replay")
    original = products["original_sample_points"]
    unavailable = original.doc_total_raw.isna() | original.delta13c_raw.isna()
    if not original.loc[unavailable, "doc_label_raw"].isna().all():
        raise ValueError("Original unavailable laboratory values were filled")
    primary = tables["response_metrics"].query("max_gap_min == 30")
    points = products["response_points"]
    independent_errors = []
    # Independent scalar trapezoids, using actual eligible source-clock pairs.
    for row in primary.itertuples():
        sub = points[(points.code == row.code) & points.time_min.between(0, 300)].sort_values("time_min")
        total = 0.0
        for a, b in zip(sub.itertuples(), sub.iloc[1:].itertuples(), strict=False):
            width = b.time_min - a.time_min
            if width <= 30 and np.isfinite(a.reference_raw) and np.isfinite(b.reference_raw):
                total += width * (max(a.reference_raw, 0) + max(b.reference_raw, 0)) / 2
        independent_errors.append(abs(total - row.salt_area))
    if max(independent_errors) > 1e-11:
        raise ValueError("Independent segment integration differs")
    figure_count = 0
    for name in ("figure_sources.json", "figure_sources_cn.json"):
        figure = json.loads((ROOT / "figures" / name).read_text())
        for kind in ("input_hashes", "output_hashes"):
            for path, expected in figure[kind].items():
                if sha256_file(path) != expected:
                    raise ValueError(f"Changed paired-tracer figure: {path}")
        figure_count += len(figure["output_hashes"])
    result = {"status": "pass", "original_rows_replayed": len(original),
              "n_pulse_site_series": len(primary), "n_paired_additions": len(tables["paired_responses"]),
              "original_missing_doc_remains_missing": True,
              "max_independent_area_error": max(independent_errors),
              "figure_products_verified": figure_count, "new_model_training": False}
    (ROOT / "verification.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
