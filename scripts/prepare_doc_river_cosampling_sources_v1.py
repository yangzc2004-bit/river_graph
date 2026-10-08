"""Prepare scoped evidence for the observational co-sampling research readout."""

from __future__ import annotations

import argparse
import json

import pandas as pd
from analyze_doc_river_cosampling_geometry_v1 import ROOT


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    folder = ROOT/"analysis"
    summary = json.loads((folder/"summary.json").read_text())
    disjoint = pd.read_csv(folder/"all_disjoint_sampling_summary.csv")
    priorities = pd.read_csv(folder/"all_disjoint_form_pair_priorities.csv",
        dtype={"huc4": str, "elongated_station": str, "broad_station": str})
    inventory_row = {k: summary[k] for k in ("n_station_networks", "n_physical_networks",
        "n_archive_stations", "n_accepted_activities", "n_same_day_receivers", "n_within_month_receivers")}
    priority_columns = ["availability_rank", "huc4", "elongated_station", "broad_station", "area_ratio",
        "elongated_all_disjoint_common_days", "broad_all_disjoint_common_days", "receivers_nested"]
    items = [{"id": "accepted-observation-coverage", "title": "现有档案能提供怎样的共同采样记录？", "queries": [{
        "id": "coverage-inventory", "source": {"label": "Accepted source-role DOC archive",
            "files": [{"label": "summary.json"}, {"label": "network_inventory.csv"}],
            "metricDefinitions": [{"label": "Accepted activities", "definition": "Accepted activities average result replicates within a sampling activity, after native-DOC QC and source-role month exclusion."},
                {"label": "Co-sampled receiver", "definition": "A co-sampled receiver has a metadata-selected disjoint frontier set with at least 12 shared local dates in three year-months."}],
            "filters": ["ST357 source-training cell union from splits 142/143/144", "297 previously mapped station-network instances", "Original whole-network form classes"],
            "caveats": ["The 297 station-network instances represent 295 receiving COMIDs.", "Same-day activities need not be simultaneous measurements.", "The nearest-frontier counts are separate from the all-disjoint pair sensitivity."]},
        "columns": list(inventory_row), "rows": [inventory_row]}]},
        {"id": "broad-form-sampling-gap", "title": "宽分支型缺少哪一类观测？", "queries": [{
            "id": "all-disjoint-sampling", "source": {"label": "Non-nested upstream-pair inventory",
                "files": [{"label": "all_disjoint_sampling_summary.csv"}, {"label": "all_disjoint_pair_inventory.csv"}],
                "metricDefinitions": [{"label": "Maximum common days", "definition": "Maximum common days is the largest exact source-A/source-B/receiver date intersection for any non-nested pair in each original form class."},
                    {"label": "Maximum dense days", "definition": "Maximum dense days counts dates in year-months with at least three joint local sampling dates, taking the maximum over pairs in a form."}],
                "filters": ["Mapping-valid physical receiver representatives", "Non-nested source catchments on original routes", "At least 12 common dates in three year-months for signal eligibility"],
                "caveats": ["Gauge-pair combinations share receiving networks and are not independent rivers.", "A broad network can have observations without meeting the signal-eligibility counts.", "These maxima describe the accepted source-role archive, not every observation ever collected in these catchments."],
                "evidenceFlow": [{"kind": "validation", "title": "Independent reconstruction", "text": "All 3,800 pair date counts were reconstructed from station-date intersections, and routing ancestry excludes nested source catchments."}]},
            "methods": [{"language": "calculation", "code": "Common dates = receiver dates ∩ source-A dates ∩ source-B dates. Dense dates retain only year-months containing at least three common dates."}],
            "columns": list(disjoint.columns), "rows": json.loads(disjoint.to_json(orient="records"))}]},
        {"id": "matched-form-observation-priorities", "title": "下一步优先补哪些形态对的观测？", "queries": [{
            "id": "priority-count", "source": {"label": "Same-region form-pair inventory", "files": [{"label": "all_disjoint_form_pair_priorities.csv"}],
                "metricDefinitions": [{"label": "Comparable form pair", "definition": "A comparable form pair joins an elongated and a broad receiving network in the same HUC4, with mapped catchment areas differing by at most a factor of two."}],
                "filters": ["Existing form classes 1 and 3", "Same HUC4", "Area ratio at most two", "Both receiver mappings valid"],
                "caveats": ["Form-pair priorities identify missing observations and do not establish a DOC shape effect.", "Receiving stations recur across different pairs; the 105 pair opportunities are not 105 independent experiments.", "Fourteen receiving pairs are nested along a larger river; the separate parallel-form shortlist retains 91 non-nested pairs."]},
                "columns": ["n_pair_opportunities", "n_nested_pairs", "n_non_nested_pairs"], "rows": [{"n_pair_opportunities": len(priorities),
                    "n_nested_pairs": int(priorities.receivers_nested.sum()), "n_non_nested_pairs": int((~priorities.receivers_nested).sum())}]},
            {"id": "ranked-priorities", "source": {"label": "Same-region form-pair inventory", "files": [{"label": "all_disjoint_form_pair_priorities.csv"}],
                "filters": ["First ten metadata-ranked opportunities", "No receiving DOC or effect direction used for ranking"],
                "metricDefinitions": [{"label": "Availability rank", "definition": "Availability rank orders balanced common-date count, balanced dense-date count, total common dates, area ratio and station IDs."}],
                "caveats": ["The preview shows ten of 105 recorded opportunities."]},
                "preview": {"kind": "sample", "totalRows": len(priorities),
                    "note": "First ten metadata-ranked opportunities from the complete 105-pair inventory."},
                "columns": priority_columns, "rows": json.loads(priorities.head(10)[priority_columns].to_json(orient="records"))}]}]
    from pathlib import Path

    payload = {"schemaVersion": 1, "items": items}
    english = json.loads(json.dumps(payload))
    for item, title in zip(english["items"], ("Coverage of co-sampled observations", "Sampling gaps in broad forms",
            "Observation priorities for comparable forms"), strict=True):
        item["title"] = title
    (ROOT/"inline_sources.json").write_text(json.dumps(english, indent=2)+"\n")
    destination = Path(args.output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(payload, ensure_ascii=False, indent=2)+"\n")


if __name__ == "__main__":
    main()
