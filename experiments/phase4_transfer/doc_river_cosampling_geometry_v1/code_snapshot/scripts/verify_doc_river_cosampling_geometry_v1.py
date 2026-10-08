"""Verify permitted-source sampling and independently check receiving measures."""

from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd
from analyze_doc_river_cosampling_geometry_v1 import (
    ROOT,
    TYPES,
    calculate,
    load_context,
)

from river_graph.experiments.provenance import sha256_file


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--full", action="store_true")
    args = p.parse_args()
    ledger = json.loads((ROOT/"analysis_sources.json").read_text())
    for name in ("source_hashes", "output_hashes"):
        for path, expected in ledger[name].items():
            if sha256_file(path) != expected:
                raise ValueError(f"changed content: {path}")
    for r in ledger["raw_sources"]:
        if sha256_file(r["path"]) != r["sha256"]:
            raise ValueError("raw WQP archive differs")
    for suffix in ("", "_cn"):
        record = json.loads((ROOT/"figures"/f"figure_sources{suffix}.json").read_text())
        for name in ("input_hashes", "output_hashes"):
            for path, expected in record[name].items():
                if sha256_file(path) != expected:
                    raise ValueError("figure differs from recorded inputs")
    context = load_context()
    activities = pd.read_parquet(ROOT/"analysis/doc_activities.parquet")
    for station, g in activities.groupby("site_no"):
        permitted = set(context[6][np.isfinite(context[5][context[4][station]])].to_period("M"))
        if set(g.date.dt.to_period("M"))-permitted:
            raise ValueError("non-source-role DOC became accessible")
    selected = pd.read_parquet(ROOT/"analysis/same_day_activities.parquet")
    gauges = pd.read_csv(ROOT/"analysis/candidate_gauges.csv", dtype=TYPES)
    signals = pd.read_csv(ROOT/"analysis/network_signals.csv", dtype=TYPES)
    maximum = 0.
    for row in signals.query("version == 'selected_activity' and adjustment == 'within_month' and excursion_quantile == .75").itertuples():
        g = selected[selected.target.eq(row.target)]
        f = g.pivot(index="date", columns="source_order", values="doc").sort_index()
        count = pd.Series(f.index.to_period("M")).value_counts()
        f = f[np.asarray(count.reindex(f.index.to_period("M"))) >= 3]
        x = f-f.groupby(f.index.to_period("M")).transform("mean")
        source, receiving = x[list(range(row.n_sources))].to_numpy(), x[-1].to_numpy()
        w = gauges[gauges.target.eq(row.target) & gauges.selected].sort_values("source_order").area_weight.to_numpy()
        reference = w@np.sqrt(np.mean(source**2, axis=0))
        ratio = np.std(receiving)/reference
        np.testing.assert_allclose(np.log(ratio), row.outlet_sync_log_sd_ratio, atol=1e-10)
        high = source > np.quantile(source, .75, axis=0)
        rhigh = receiving > np.quantile(receiving, .75)
        coincident, solo = high.sum(axis=1) >= 2, high.sum(axis=1) == 1
        if (int(coincident.sum()), int(solo.sum()), int((coincident & rhigh).sum()), int((solo & rhigh).sum())) != (
                row.n_coincident, row.n_solo, row.n_receiver_high_coincident, row.n_receiver_high_solo):
            raise ValueError("independent excursion counts differ")
        maximum = max(maximum, abs(np.log(ratio)-row.outlet_sync_log_sd_ratio))
    if args.full:
        tables, _ = calculate(context, activities, draws=json.loads((ROOT/"config.json").read_text())["bootstrap_draws"])
        for name, actual in tables.items():
            parquet = name in ("same_day_activities", "signal_series")
            path = ROOT/"analysis"/f"{name}.{'parquet' if parquet else 'csv'}"
            saved = pd.read_parquet(path) if parquet else pd.read_csv(path, dtype=TYPES)
            if not parquet:
                from io import StringIO

                actual = pd.read_csv(StringIO(actual.to_csv(index=False)), dtype=TYPES)
            pd.testing.assert_frame_equal(actual, saved, check_dtype=False, atol=1e-10, rtol=1e-10)
    result = {"permitted_source_archive_checked": True, "independent_excursion_and_reference_checks": True,
        "maximum_reference_log_error": maximum, "full_table_replay": args.full}
    (ROOT/"verification.json").write_text(json.dumps(result, indent=2)+"\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
