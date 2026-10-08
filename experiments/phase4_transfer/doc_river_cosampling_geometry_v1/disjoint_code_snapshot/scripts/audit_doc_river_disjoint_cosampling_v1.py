"""Audit all disjoint co-sampled gauge pairs without the nearest-frontier limit."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
from analyze_doc_river_cosampling_geometry_v1 import ROOT, TYPES

from river_graph.analysis.river_cosampling_pairs import disjoint_pair_inventory
from river_graph.experiments.provenance import sha256_file


def main():
    folder = ROOT/"analysis"
    inputs = [folder/"candidate_gauges.csv", folder/"doc_activities.parquet", folder/"network_inventory.csv"]
    g = pd.read_csv(inputs[0], dtype=TYPES)
    events = pd.read_parquet(inputs[1])
    receivers = pd.read_csv(inputs[2], dtype=TYPES).set_index("station")
    mapping_path = Path("experiments/phase4_transfer/doc_river_mechanisms_v1/analysis/station_mapping_audit.csv")
    mapping = pd.read_csv(mapping_path, dtype={"site_no": str}).set_index("site_no")
    dates = {s: pd.DatetimeIndex(f.date.unique()).sort_values() for s, f in events.groupby("site_no")}
    tables = []
    for target, f in g.groupby("target"):
        row = receivers.loc[target]
        if row.status in ("receiver_mapping_mismatch", "receiving_COMID_alias"):
            continue
        pair = disjoint_pair_inventory(f, dates, dates[target], receiver_area=mapping.loc[target, "drainage_area_km2"])
        pair["target"], pair["cluster"], pair["huc4"] = target, row.cluster, row.huc4
        tables.append(pair)
    pairs = pd.concat(tables, ignore_index=True)
    paths = []
    path = folder/"all_disjoint_pair_inventory.csv"
    pairs.to_csv(path, index=False)
    paths.append(path)
    summary = []
    for c in (1, 2, 3):
        f = pairs[pairs.cluster.eq(c)]
        summary.append({"cluster": c, "n_candidate_pairs": len(f), "n_candidate_receivers": f.target.nunique(),
            "n_same_day_receivers": f[f.same_day_eligible].target.nunique(),
            "n_within_month_receivers": f[f.within_month_eligible].target.nunique(),
            "maximum_common_days": int(f.n_common_days.max()) if len(f) else 0,
            "maximum_dense_days": int(f.n_within_month_days.max()) if len(f) else 0})
    path = folder/"all_disjoint_sampling_summary.csv"
    pd.DataFrame(summary).to_csv(path, index=False)
    paths.append(path)
    comparisons_path = folder/"same_region_form_pairs.csv"
    comparisons = pd.read_csv(comparisons_path, dtype={"huc4": str, "elongated_station": str, "broad_station": str})
    availability = pairs[pairs.official_area_consistent].groupby("target").agg(
        common_days=("n_common_days", "max"), dense_days=("n_within_month_days", "max"))
    for form in ("elongated", "broad"):
        for measure in ("common_days", "dense_days"):
            comparisons[form+"_all_disjoint_"+measure] = comparisons[form+"_station"].map(availability[measure]).fillna(0).astype(int)
    comparisons = comparisons[comparisons.area_comparable & comparisons.both_mapping_valid].copy()
    comparisons["balanced_common_days"] = comparisons[[f"{f}_all_disjoint_common_days" for f in ("elongated", "broad")]].min(axis=1)
    comparisons["balanced_dense_days"] = comparisons[[f"{f}_all_disjoint_dense_days" for f in ("elongated", "broad")]].min(axis=1)
    comparisons["total_common_days"] = comparisons.elongated_all_disjoint_common_days+comparisons.broad_all_disjoint_common_days
    comparisons = comparisons.sort_values(["balanced_common_days", "balanced_dense_days", "total_common_days", "area_ratio", "elongated_station", "broad_station"],
        ascending=[False, False, False, True, True, True]).reset_index(drop=True)
    comparisons["availability_rank"] = range(1, len(comparisons)+1)
    member_files, membership = [], {}
    for station in set(comparisons.elongated_station) | set(comparisons.broad_station):
        comid = int(receivers.loc[station, "comid"])
        member = Path("data/raw/river_planform_v1/members_full")/f"comid_{comid}.npz"
        with np.load(member) as data:
            membership[station] = set(data["comids"].tolist())
        member_files.append(member)
    comparisons["receivers_nested"] = [
        int(receivers.loc[b, "comid"]) in membership[a] or int(receivers.loc[a, "comid"]) in membership[b]
        for a, b in zip(comparisons.elongated_station, comparisons.broad_station, strict=True)]
    path = folder/"all_disjoint_form_pair_priorities.csv"
    comparisons.to_csv(path, index=False)
    paths.append(path)
    non_nested = comparisons[~comparisons.receivers_nested].copy().reset_index(drop=True)
    non_nested["observation_priority"] = range(1, len(non_nested)+1)
    path = folder/"non_nested_form_pair_priorities.csv"
    non_nested.to_csv(path, index=False)
    paths.append(path)
    code = list(map(Path, ("scripts/audit_doc_river_disjoint_cosampling_v1.py",
        "src/river_graph/analysis/river_cosampling_pairs.py", "src/river_graph/analysis/river_cosampling_geometry.py",
        "tests/test_river_cosampling_pairs.py")))
    sources = [*inputs, comparisons_path, mapping_path, ROOT/"disjoint_sampling_plan.md", *member_files, *code]
    for path in code:
        destination = ROOT/"disjoint_code_snapshot"/path
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, destination)
    (ROOT/"disjoint_sampling_sources.json").write_text(json.dumps({
        "source_hashes": {str(p): sha256_file(p) for p in sources},
        "output_hashes": {str(p): sha256_file(p) for p in paths},
        "selection": "all non-nested candidate pairs; no DOC values select pairs",
        "scope": "metadata sensitivity to nearest-frontier sampling restriction"}, indent=2)+"\n")
    print(pd.DataFrame(summary).to_string(index=False))


if __name__ == "__main__":
    main()
