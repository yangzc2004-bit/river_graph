"""Independently reconstruct co-sampling counts for every non-nested pair."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from analyze_doc_river_cosampling_geometry_v1 import ROOT, TYPES

from river_graph.experiments.provenance import sha256_file


def main():
    ledger = json.loads((ROOT/"disjoint_sampling_sources.json").read_text())
    for name in ("source_hashes", "output_hashes"):
        for path, expected in ledger[name].items():
            if sha256_file(path) != expected:
                raise ValueError(f"changed input: {path}")
    folder = ROOT/"analysis"
    events = pd.read_parquet(folder/"doc_activities.parquet")
    days = {s: set(pd.to_datetime(g.date).dt.normalize()) for s, g in events.groupby("site_no")}
    pairs = pd.read_csv(folder/"all_disjoint_pair_inventory.csv", dtype={**TYPES, "source_a": str, "source_b": str})
    candidate = pd.read_csv(folder/"candidate_gauges.csv", dtype=TYPES)
    for target, group in pairs.groupby("target"):
        g = candidate[candidate.target.eq(target)]
        parent = dict(zip(g.comid, g.downstream_candidate_comid, strict=True))
        for row in group.itertuples():
            shared = pd.DatetimeIndex(sorted(days[target] & days[row.source_a] & days[row.source_b]))
            counts = pd.Series(shared.to_period("M")).value_counts()
            dense = counts[counts >= 3]
            if (len(shared), len(counts), int(dense.sum()), len(dense)) != (
                    row.n_common_days, row.n_common_year_months, row.n_within_month_days, row.n_dense_year_months):
                raise ValueError("independent date-intersection counts differ")
            for start, forbidden in ((row.source_a_comid, row.source_b_comid), (row.source_b_comid, row.source_a_comid)):
                seen = set()
                while start >= 0:
                    if start in seen or start == forbidden:
                        raise ValueError("nested or cyclic source pair")
                    seen.add(start)
                    start = parent[start]
    receivers = pd.read_csv(folder/"network_inventory.csv", dtype=TYPES).set_index("station")
    comparisons = pd.read_csv(folder/"all_disjoint_form_pair_priorities.csv",
        dtype={"elongated_station": str, "broad_station": str})
    for row in comparisons.itertuples():
        ids = [int(receivers.loc[s, "comid"]) for s in (row.elongated_station, row.broad_station)]
        arrays = []
        for comid in ids:
            with np.load(Path("data/raw/river_planform_v1/members_full")/f"comid_{comid}.npz") as z:
                arrays.append(z["comids"])
        nested = ids[0] in arrays[1] or ids[1] in arrays[0]
        if nested != row.receivers_nested:
            raise ValueError("receiving catchment nesting flag differs")
    result = {"n_pairs_checked": len(pairs), "date_counts_independently_reconciled": True,
        "non_nested_routing_checked": True, "n_form_pair_nesting_checks": len(comparisons),
        "script_sha256": sha256_file(Path(__file__))}
    (ROOT/"disjoint_verification.json").write_text(json.dumps(result, indent=2)+"\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
