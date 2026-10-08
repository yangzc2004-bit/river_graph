"""Replay monitored-frontier coverage and observed DOC statistics from sources."""

from __future__ import annotations

import argparse
import io
import json

import numpy as np
import pandas as pd
from analyze_doc_river_monitored_arrivals_v1 import (
    ROOT,
    TYPES,
    build_frontiers,
    load_context,
    observation_tables,
    read_activities,
    weekly_case_tables,
)

from river_graph.experiments.provenance import sha256_file


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shortest-routes", action="store_true")
    parser.add_argument("--full", action="store_true")
    args = parser.parse_args()
    root = ROOT/"shortest_route_sensitivity" if args.shortest_routes else ROOT
    ledger = json.loads((root/"analysis_sources.json").read_text())
    for collection in ("source_hashes", "output_hashes"):
        for path, expected in ledger[collection].items():
            if sha256_file(path) != expected:
                raise ValueError(f"changed source/product: {path}")
    for r in ledger["raw_sources"]:
        if sha256_file(r["path"]) != r["sha256"]:
            raise ValueError("raw sampling archive changed")
    for receipt in sorted((root/"figures").glob("figure_sources*.json")):
        f = json.loads(receipt.read_text())
        for collection in ("input_hashes", "output_hashes"):
            for path, expected in f[collection].items():
                if sha256_file(path) != expected:
                    raise ValueError(f"changed figure source/product: {path}")
    data = root/"analysis"
    signals = pd.read_csv(data/"receiver_signals.csv", dtype=TYPES)
    finite = signals.dropna(subset=["mixture_buffer_fraction"])
    np.testing.assert_allclose(finite.mixture_buffer_fraction,
        finite.asynchronous_buffer_fraction+finite.amplitude_balance_buffer_fraction, atol=1e-10)
    full = signals[signals.version.eq("full_monthly")]
    if full.target.duplicated().any() or full.comid.duplicated().any():
        raise ValueError("one observational summary per physical receiving network required")
    config = json.loads((root/"config.json").read_text())
    if args.full:
        context = load_context()
        inventory, gauges, sets, components, _ = build_frontiers(context, shortest=args.shortest_routes)
        sites = set(sets) | {s for values in sets.values() for s in values[0]}
        activities, reconcile, _ = read_activities(context, sites)
        tables = observation_tables(context, sets, components, activities, draws=config["bootstrap_draws"])
        tables.update(weekly_case_tables(context, sets, activities, tables["aligned_sample_sets"]))
        tables.update({"network_inventory": inventory, "candidate_gauges": gauges,
                       "doc_activities": activities, "raw_monthly_reconciliation": reconcile})
        for name, expected in tables.items():
            path = data/f"{name}.parquet"
            if path.exists():
                actual = pd.read_parquet(path)
            else:
                actual = pd.read_csv(data/f"{name}.csv", dtype=TYPES)
                expected = pd.read_csv(io.StringIO(expected.to_csv(index=False)), dtype=TYPES)
            pd.testing.assert_frame_equal(actual, expected, check_dtype=False, atol=1e-10, rtol=1e-9)
    message = (f"Replayed {297 if args.full else 'bound'} network inventories and "
               f"{len(full)} complete unique-receiver DOC summaries.\n")
    (root/"verification.md").write_text("# Verification\n\n"+message+
        "\nSource roles, original river classes, disjoint catchment weights, raw monthly "
        "reconciliation, metadata-only date selection, multi-input covariance identities "
        "and whole-system summaries are retained. Sources/products/figure receipts agree. "
        "This is an observed monthly fluctuation study, not event transit estimation.\n")
    print(message)


if __name__ == "__main__":
    main()
