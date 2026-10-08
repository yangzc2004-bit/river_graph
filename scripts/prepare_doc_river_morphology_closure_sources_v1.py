"""Prepare the scoped source receipt for the completed morphology study."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
from analyze_doc_river_morphology_closure_v1 import ROOT


def rows(frame):
    return json.loads(frame.to_json(orient="records"))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", required=True)
    args = p.parse_args()
    data = ROOT / "analysis"
    summary = json.loads((data / "summary.json").read_text())
    mechanisms = pd.read_csv(data / "mechanism_summary.csv")
    mechanism = mechanisms[mechanisms.minimum_coverage.eq(0) & mechanisms.metric.eq("pulse_peak_relative")
        & mechanisms.comparison.isin(["path_difference", "shared_spreading", "source_alignment", "junction_only"])
        & mechanisms.sigma.isin([.1, 2.])][["sigma", "comparison", "estimate", "ci_low", "ci_high", "n_receivers", "n_blocks"]]
    field = pd.read_csv(data / "observed_leave_system_out.csv")
    field = field[field.minimum_coverage.eq(0)][["omitted_system", "estimate", "ci_low", "ci_high", "n_receivers", "n_blocks", "n_positive_receivers"]]
    contrasts = pd.read_csv(data / "mechanism_form_contrasts.csv")
    contrasts = contrasts[contrasts.minimum_coverage.eq(0) & contrasts.sigma.eq(.1)
        & contrasts.scenario.isin(["actual_paths", "actual_shared_0.5"]) & contrasts.metric.eq("pulse_peak")][
            ["scenario", "estimate", "ci_low", "ci_high", "n_receivers", "n_blocks"]]
    recovery = summary["recovery"]
    extra = summary["additional_recovery"]
    record = {"form_opportunities": recovery["targeted_form_pairs"], "unique_receivers": recovery["targeted_receivers"],
        "station_requests": recovery["requested_stations"]+extra["additional_station_leads"],
        "fresh_station_downloads": recovery["downloaded_stations"]+extra["station_downloads"],
        "old_cache_fallbacks": recovery["old_cache_fallback_stations"],
        "accepted_activities": recovery["accepted_activities"]+extra["accepted_additional_activities"],
        "complete_same_day_form_pairs": recovery["complete_form_pairs_same_day"],
        "complete_within_month_form_pairs": recovery["complete_form_pairs_within_month"],
        "additional_eligible_broad_receivers_100m": extra["receivers_with_any_eligible_pair"]["100.0"],
        "additional_eligible_broad_receivers_300m": extra["receivers_with_any_eligible_pair"]["300.0"]}
    payload = {"schemaVersion": 1, "items": [
        {"id": "structural-mechanisms", "title": "路径错开与共同河段弥散如何改变峰值？", "queries": [{
            "id": "controlled-peak-changes", "source": {"label": "Real mapped paths, prescribed pulse experiments",
                "files": [{"label": "mechanism_summary.csv"}, {"label": "mechanism_scenarios.csv"}, {"label": "verification.json"}],
                "metricDefinitions": [{"label": "Peak response", "definition": "The peak response is the continuous maximum of a unit-amplitude Gaussian pulse routed through actual cropped paths at an imposed uniform speed."},
                    {"label": "Structural comparison", "definition": "Path effects compare unequal with equalized arrivals; additional spreading compares the same unequal paths with and without the declared unit-gain common kernel."}],
                "filters": ["32 receivers in seven overlapping catchment systems", "Receiver-equal paired changes", "5000 whole-system bootstrap draws"],
                "caveats": ["These are controlled mechanism responses, not measured DOC reductions.", "The spreading kernel and uniform-speed scale are imposed rather than fitted from field DOC.", "The two structural comparisons have different reference responses; their percentages cannot be added."],
                "evidenceFlow": [{"kind": "validation", "title": "Independent pulse checks", "text": "Numerical integration verified gain and mean/variance; junction-only responses match within 9e-16 in both saved route definitions."}]},
            "methods": [{"language": "calculation", "code": "Outlet pulse variance = input pulse variance + weighted arrival-clock variance + common-kernel variance. Each peak change = candidate peak / reference peak - 1."}],
            "columns": list(mechanism.columns), "rows": rows(mechanism),
            "preview": {"kind": "aggregate", "note": "Two declared input durations from the complete four-duration experiment grid."}}]},
        {"id": "measured-overlap-association", "title": "真实 DOC 观测是否支持上游高值叠加？", "queries": [{
            "id": "observed-excursion-check", "source": {"label": "Observed upstream and receiving DOC",
                "files": [{"label": "observed_leave_system_out.csv"}, {"label": "network_metrics.csv"}],
                "metricDefinitions": [{"label": "Receiving excursion difference", "definition": "The receiving excursion difference compares receiving-high probability when at least two monitored sources are high with the probability when one source is high. High values exceed the site's season/year-adjusted 75th percentile."}],
                "filters": ["Fixed source-role monthly population", "At least five coincident and five solo dates per receiver", "17 eligible receivers, six overlapping systems", "Receiver-equal mean, 5000 whole-system bootstrap draws"],
                "caveats": ["Monthly sampled excursions do not identify actual transit lag or unsampled event maxima.", "The observed association does not isolate a whole-form causal effect.", "Thirteen receivers have positive differences and four have negative differences."]},
            "methods": [{"language": "calculation", "code": "Receiver difference = receiving-high count / coincident-source dates - receiving-high count / solo-source dates. Average differences equally over receivers; repeat with each eligible system omitted."}],
            "columns": list(field.columns), "rows": rows(field)}]},
        {"id": "whole-form-comparison", "title": "宽分支型与细长型能否给出固定缓冲排名？", "queries": [{
            "id": "controlled-form-contrasts", "source": {"label": "Original whole-network classes",
                "files": [{"label": "mechanism_form_contrasts.csv"}],
                "metricDefinitions": [{"label": "Form contrast", "definition": "The form contrast is the broad-network mean unit-input peak minus the elongated-network mean peak under the same declared input and kernel rule."}],
                "filters": ["Original form classification", "Input SD / mean nominal path time = 0.1", "All represented-area coverages"],
                "caveats": ["The comparison intervals overlap zero and do not establish a form ranking.", "A form label does not specify source-clock overlap and channel process properties."]},
            "columns": list(contrasts.columns), "rows": rows(contrasts)}]},
        {"id": "bounded-observation-recovery", "title": "补齐哪些观测后，仍缺少什么？", "queries": [{
            "id": "record-recovery", "source": {"label": "Targeted WQP station records and official USGS catalogue",
                "files": [{"label": "summary.json"}, {"label": "additional_receiver_coverage.csv"}, {"label": "retrieval_manifest.json"}, {"label": "additional_retrieval.json"}],
                "links": [{"label": "Water Quality Portal", "url": "https://www.waterqualitydata.us/"},
                    {"label": "USGS time-series metadata", "url": "https://api.waterdata.usgs.gov/ogcapi/v1/collections/time-series-metadata"}],
                "metricDefinitions": [{"label": "Accepted activity", "definition": "Accepted activities contain uncensored dissolved DOC in mg/L; multiple laboratory results within one activity are averaged."},
                    {"label": "Eligible form pair", "definition": "An eligible form pair requires adequate common two-source/receiver sampling in both original forms. The saved rules require at least twelve common dates in three year-months; the dense version also requires repeated within-month dates."}],
                "filters": ["First fifteen non-nested same-HUC4, similar-area form opportunities", "Full public histories outside predictive-model role restrictions", "All 49 additional in-basin river/stream leads from the returned 1019 catalogue"],
                "caveats": ["Receiving stations recur among the fifteen opportunities; the opportunities are not independent experiments.", "Six regional station catalogues timed out, so discovery beyond existing mapped stations was incomplete in those regions.", "Additional public-station mappings are coordinate-based 100 m/300 m sensitivities, not verified NLDI hydrolocations.", "The complete USGS parameter-00681 time-series catalogue returned zero series; other providers and optical proxies are outside that statement."],
                "evidenceFlow": [{"kind": "validation", "title": "Sampling support replay", "text": "Every primary and expanded source-pair date intersection was checked directly from accepted station activity dates. The obsolete-coordinate parser was repaired using the actual saved WQP3 schema; v0 was retained."}]},
            "columns": list(record), "rows": [record]}]}]}
    destination = Path(args.output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(payload, ensure_ascii=False, indent=2)+"\n")
    english = json.loads(json.dumps(payload))
    for item, title in zip(english["items"], ("Controlled path and common-corridor mechanisms",
            "Observed joint-source excursion association", "Whole-form buffering comparison", "Bounded public observation recovery"), strict=True):
        item["title"] = title
    (ROOT / "inline_sources.json").write_text(json.dumps(english, indent=2)+"\n")


if __name__ == "__main__":
    main()
