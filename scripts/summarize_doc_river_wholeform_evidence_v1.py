"""Consolidate existing whole-form DOC evidence without refitting an analysis."""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from river_graph.experiments.provenance import sha256_file

ROOT = Path("experiments/phase4_transfer")
OUT = ROOT / "doc_river_wholeform_evidence_v1"


def select_one(frame, **filters):
    selected = frame
    for column, value in filters.items():
        selected = selected[selected[column].eq(value)]
    if len(selected) != 1:
        raise ValueError(f"Expected one source estimate: {filters}; found {len(selected)}")
    return selected.iloc[0]


def main():
    output = OUT / "analysis"
    output.mkdir(parents=True, exist_ok=True)
    form = ROOT / "doc_river_morphology_effect_v1/analysis"
    transport = ROOT / "doc_river_observed_transport_v1/analysis"
    sources = [form / "morphology_block_gains.csv", form / "paired_doc_contrasts.csv",
               form / "paired_geometry_contrasts.csv", transport / "paired_gains.csv",
               transport / "receiver_metrics.csv",
               ROOT / "doc_river_planform_typology_v1/analysis/classification_summary.json",
               ROOT / "doc_river_measured_confluences_v1/analysis/campaigns.csv",
               ROOT / "doc_river_multicatchment_pulses_v1/analysis/pulse_comparison.csv"]
    blocks, doc, geometry, gains = [pd.read_csv(p) for p in sources[:4]]
    evidence = []
    for candidate in ("footprint", "branching", "paths", "all_morphology"):
        row = select_one(blocks, space="native", candidate=candidate, reference="environment")
        evidence.append({"finding": candidate + "_predictive_information",
                         "evidence_type": "Observed station-median prediction; existing exploratory analysis",
                         "estimate": row.gain_pct, "ci_low": row.ci_low_pct,
                         "ci_high": row.ci_high_pct, "unit": "relative MAE reduction percent",
                         "sample_count": int(row.n_stations), "group_count": int(row.n_huc4),
                         "group_unit": "HUC4", "source_table": str(sources[0]),
                         "interpretation": "Extra predictive information; not a causal effect or monthly-model gain"})
    for metric in ("doc_median", "doc_cv", "harmonic_amplitude"):
        row = select_one(doc, class_a=1, class_b=3, metric=metric)
        evidence.append({"finding": "broad_minus_elongated_" + metric,
                         "evidence_type": "Observed matched, non-nested networks",
                         "estimate": row.difference_b_minus_a, "ci_low": row.ci_low,
                         "ci_high": row.ci_high, "unit": "mg/L" if metric == "doc_median" else "source metric",
                         "sample_count": int(row.n_pairs), "group_count": int(row.n_huc4),
                         "group_unit": "HUC4", "source_table": str(sources[1]),
                         "interpretation": "Interval includes zero; class ordering remains unresolved"})
    for metric in ("routing_pulse_peak", "routing_pulse_spread"):
        row = select_one(geometry, class_a=1, class_b=3, metric=metric)
        evidence.append({"finding": "broad_minus_elongated_" + metric,
                         "evidence_type": "Identical-input simulation on actual mapped paths",
                         "estimate": row.difference_b_minus_a, "ci_low": row.ci_low,
                         "ci_high": row.ci_high,
                         "unit": "output/input amplitude" if metric.endswith("peak") else "normalized delay",
                         "sample_count": int(row.n_pairs), "group_count": int(row.n_huc4),
                         "group_unit": "HUC4", "source_table": str(sources[2]),
                         "interpretation": "Routing mechanism conditional on input and travel assumptions; not field DOC response"})
    for candidate, reference in (("same_month", "background"), ("branch_arrival", "mean_delay")):
        row = select_one(gains, candidate=candidate, reference=reference,
                         group="all", metric="mae", unit="component")
        evidence.append({"finding": candidate + "_versus_" + reference,
                         "evidence_type": "Observed monthly source/receiver prediction",
                         "estimate": row.relative_reduction_pct,
                         "ci_low": row.gain_ci_low_pct, "ci_high": row.gain_ci_high_pct,
                         "unit": "relative MAE reduction percent",
                         "sample_count": int(row.n_receivers), "group_count": int(row.n_blocks),
                         "group_unit": "connected monitoring system", "source_table": str(sources[3]),
                         "interpretation": "Upstream concentration information" if candidate == "same_month"
                         else "No established extra overall information from separating branch delays"})
    pd.DataFrame(evidence).to_csv(output / "evidence_summary.csv", index=False)
    receivers = pd.read_csv(sources[4], dtype={"target": str, "huc4": str})
    receivers = receivers[receivers.operator.eq("background")].copy()
    if receivers.target.duplicated().any():
        raise ValueError("Repeated receiver in the form coverage population")
    coverage = receivers.groupby("cluster").agg(
        receivers=("target", "nunique"), connected_systems=("component", "nunique"),
        huc4_regions=("huc4", "nunique")).reset_index()
    coverage.to_csv(output / "observed_form_replication.csv", index=False)
    campaigns = pd.read_csv(sources[6])
    reversing = campaigns.groupby("location").doc_departure_pct.agg(
        minimum="min", maximum="max")
    pulses = pd.read_csv(sources[7])
    pulses = pulses[pulses.resolution.eq("hourly") & pulses.relative_prominence.eq(.2)]
    summary = {"classification": json.loads(sources[5].read_text()),
               "evidence_rows": len(evidence), "new_effect_estimates": False,
               "monthly_form_replication": coverage.to_dict("records"),
               "outlet_event_catchments": pulses.case.nunique(),
               "positive_doc_outlet_responses": int(pulses.n_positive_doc.sum()),
               "outlet_width_pairs": int(pulses.n_width_pairs.sum()),
               "local_confluence_campaigns": len(campaigns),
               "local_confluences": campaigns.location.nunique(),
               "local_confluences_with_seasonal_sign_reversal": int(
                   ((reversing.minimum < 0) & (reversing.maximum > 0)).sum()),
               "primary_conclusion": "Continuous branch organization carries observed DOC information; a general whole-form DOC ordering is not established",
               "dynamic_mechanism": "Geometry changes arrival/mixing under controlled routing; independent field attribution remains open"}
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    receipt = {"inputs": {str(p): sha256_file(p) for p in sources},
               "script": sha256_file(Path(__file__)),
               "method": "Extract existing estimates and independently count observation groups; no refit or new bootstrap"}
    (OUT / "analysis_sources.json").write_text(json.dumps(receipt, indent=2) + "\n")
    print(pd.DataFrame(evidence)[["finding", "estimate", "ci_low", "ci_high"]].to_string(index=False))
    print(coverage.to_string(index=False))


if __name__ == "__main__":
    main()
