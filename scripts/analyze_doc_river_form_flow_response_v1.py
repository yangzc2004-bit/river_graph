"""Analyze low/high flow DOC responses in fixed real river forms."""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from analyze_doc_river_form_monthly_comparison_v1 import (
    CELLS,
    DATASET,
    PAIRS,
    PANEL,
    build_monthly_records,
    digest,
)

from river_graph.analysis.river_form_flow import (
    BLOCKS,
    CONTROLS,
    HIGH_DOC,
    assign_states,
    flow_references,
    joint_evidence,
    pair_responses,
    shared_pair_records,
    station_responses,
    structure_associations,
    summarize_pairs,
    summarize_stations,
)
from river_graph.experiments.provenance import sha256_file

ROOT = Path("experiments/phase4_transfer/doc_river_form_flow_response_v1")
CODE = tuple(map(Path, ("scripts/analyze_doc_river_form_flow_response_v1.py",
    "src/river_graph/analysis/river_form_flow.py", "scripts/analyze_doc_river_form_monthly_comparison_v1.py",
    "src/river_graph/analysis/river_mechanisms.py", "src/river_graph/analysis/river_morphology_effect.py")))


def build_inputs(data=None):
    data = torch.load(DATASET, map_location="cpu", weights_only=False) if data is None else data
    monthly, pairs = build_monthly_records(data)
    index = {str(s): i for i, s in enumerate(data["site_no"])}
    qi = list(data["feature_channels"]).index("discharge")
    x, mask = np.asarray(data["x"], float), np.asarray(data["x_mask"], bool)
    rows = []
    for station, s in monthly.groupby("station", sort=True):
        months = np.arange(s.month_index.min(), s.month_index.max()+1)
        flow = x[index[station], months, qi]
        valid = mask[index[station], months, qi] & np.isfinite(flow) & (flow > 0)
        rows.append(pd.DataFrame({"station": station, "month_index": months,
                                  "log_discharge": np.log1p(np.where(valid, flow, np.nan)),
                                  "flow_measured": valid}))
    reference_months = pd.concat(rows, ignore_index=True)
    references = flow_references(reference_months)
    return assign_states(monthly, references), pairs, reference_months, references


def analyze(frame, pairs, draws):
    fits, states, ledger, coefficients = station_responses(frame)
    records = shared_pair_records(frame, pairs)
    evidence, pair_ledger, pair_states, pair_coefficients, joint = pair_responses(records, pairs)
    pair_summary, omitted = summarize_pairs(evidence, draws)
    joint_pairs, joint_ledger = joint_evidence(joint)
    joint_summary, joint_omitted = summarize_pairs(joint_pairs, draws)
    panel = frame.drop_duplicates("station")
    associations, models = structure_associations(fits, panel, draws)
    inclusion = []
    for population, fit in fits.groupby("population"):
        s = fit.merge(panel, on="station", validate="one_to_one", suffixes=("", "_panel"))
        for block, features in BLOCKS.items():
            for r in s.itertuples():
                missing = [name for name in (*CONTROLS, *features) if not np.isfinite(getattr(r, name))]
                inclusion.append({"population": population, "block": block, "station": r.station,
                                  "included": bool(r.adjusted_identified and not missing),
                                  "missing_covariates": ";".join(missing) or "none",
                                  "adjusted_identified": r.adjusted_identified})
    # Explicit season counts preserve cyclic calendar context in joint comparisons.
    calendar = joint.groupby(["pair_id", "population", "flow_state", "calendar_month"], as_index=False).size()
    tables = {"flow_response_inclusion": ledger, "station_flow_responses": fits,
              "station_state_summaries": states, "class_flow_responses": summarize_stations(fits, draws),
              "pair_inclusion": pair_ledger, "pair_state_summaries": pair_states,
              "pair_response_evidence": evidence, "pair_response_contrasts": pair_summary,
              "omitted_region_influence": omitted, "joint_state_inclusion": joint_ledger,
              "joint_state_evidence": joint_pairs, "joint_state_contrasts": joint_summary,
              "joint_omitted_region_influence": joint_omitted, "joint_calendar_counts": calendar,
              "structure_response_associations": associations, "structure_covariate_inclusion": pd.DataFrame(inclusion)}
    products = {"observed_flow_months": frame, "shared_pair_months": records, "joint_state_months": joint}
    fitted = {"station_fits": coefficients, "matched_member_fits": pair_coefficients, "structure_models": models}
    return tables, products, fitted


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bootstrap-draws", type=int, default=5000)
    args = parser.parse_args()
    out = ROOT/"analysis"
    out.mkdir(parents=True, exist_ok=True)
    frame, pairs, reference_months, references = build_inputs()
    tables, products, fitted = analyze(frame, pairs, args.bootstrap_draws)
    tables["flow_reference_thresholds"] = references
    for name, table in tables.items():
        table.to_csv(out/f"{name}.csv", index=False)
    products["flow_reference_months"] = reference_months
    for name, product in products.items():
        product.to_parquet(out/f"{name}.parquet", index=False)
    (out/"fitted_states.json").write_text(json.dumps(fitted, indent=2)+"\n")
    config = {"flow_quantiles": [1/3, 2/3], "minimum_reference_months": 24,
              "minimum_doc_flow_months": 24, "minimum_low_high_doc_months": 6, "minimum_calendar_months": 6,
              "minimum_years": 2, "minimum_joint_low_high_months": 4, "high_doc_threshold_mg_l": HIGH_DOC,
              "association_controls": CONTROLS, "association_blocks": BLOCKS, "bootstrap_draws": args.bootstrap_draws,
              "bootstrap_unit": "HUC4", "bootstrap_seed": 42, "permitted_roles": "source union142/143/144",
              "classification_and_pairs": "fixed previous geometry-only classes and covariate-only pairs",
              "main_adjusted_response": "within-station log1p DOC high-state minus low-state; season and year adjusted"}
    (ROOT/"config.json").write_text(json.dumps(config, indent=2)+"\n")
    sources = [DATASET, CELLS, PANEL, PAIRS, ROOT/"study_plan.md", *CODE]
    hashes = {str(p): sha256_file(p) for p in sources}
    for path in CODE:
        target = ROOT/"code_snapshot"/path
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, target)
    record = {"source_hashes": hashes, "config_hash": digest(config), "bootstrap_draws": args.bootstrap_draws,
              "n_observed_source_months": len(frame), "n_fixed_stations": frame.station.nunique(),
              "n_reference_months": int(reference_months.flow_measured.sum()), "previous_results_seen": True,
              "existing_models_retrained": False, "created_date": "2026-10-07"}
    (ROOT/"analysis_sources.json").write_text(json.dumps(record, indent=2)+"\n")
    print(json.dumps({k: v for k, v in record.items() if k != "source_hashes"}, indent=2))
    print(tables["class_flow_responses"].query("metric == 'adjusted_log_contrast'").to_string(index=False))
    print(tables["pair_response_contrasts"].query("class_a == 1 and class_b == 3").to_string(index=False))
    print(tables["structure_response_associations"].to_string(index=False))


if __name__ == "__main__":
    main()
