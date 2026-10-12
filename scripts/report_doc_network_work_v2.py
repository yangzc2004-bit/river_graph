"""Write an evidence-linked result memo only after independent audit passes."""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

ROOT = Path("experiments/phase4_transfer/doc_network_work_v2")
LABELS = {"fixed_fusion": "Environmental MLP + GRU + graph Transformer",
          "doc_ecological_rf": "Environmental/current-driver RF adaptation",
          "rf_history_control": "RF with 12-month history",
          "doc_hydro_gam": "Hydrological/environmental GAM adaptation",
          "doc_xgboost": "XGBoost with 12-month history",
          "ssn_covariance_adapt": "SSN covariance adaptation",
          "physt_architecture_adapt": "Official Transformer-GCN architecture adaptation"}


def table(frame, columns):
    lines = ["| " + " | ".join(columns) + " |", "| " + " | ".join("---" for _ in columns) + " |"]
    for row in frame[columns].itertuples(index=False, name=None):
        lines.append("| " + " | ".join(f"{value:.4f}" if isinstance(value, float) else str(value) for value in row) + " |")
    return "\n".join(lines)


def main():
    audit = json.loads((ROOT/"analysis/independent_audit.json").read_text())
    if audit["status"] != "passed":
        raise ValueError("independent verification is required before reporting")
    protocol = json.loads((ROOT/"protocol.json").read_text())
    summary = pd.read_csv(ROOT/"analysis/summary.csv")
    intervals = pd.DataFrame(json.loads((ROOT/"analysis/paired_comparisons.json").read_text()))
    regions = pd.read_csv(ROOT/"analysis/by_region_seed.csv", dtype={"target_huc4": str}).groupby(["model_name", "target_huc4"], as_index=False).mae.mean()
    display = summary.copy()
    display["Method"] = display.model_name.map(LABELS)
    display = display.sort_values("mae")
    point_winner = LABELS[display.iloc[0].model_name]
    rows = []
    for item in intervals[intervals.endpoint == "mae"].to_dict("records"):
        rows.append({"Comparator": LABELS[item["comparator"]], "Fusion MAE gain, %": item["gain_percent"],
            "Paired MAE difference": item["difference_fusion_minus_comparator"],
            "Exploratory 95% interval": "["+", ".join(f"{v:.4f}" for v in item["difference_interval"])+"]",
            "Regions with lower fusion MAE": item["regions_with_lower_error"]})
    fit_records = [json.loads(path.read_text()) for path in (ROOT/"regions").glob("*/fits/*/*/seed*/info.json")]
    ssn_info = [record["optimizer"] for record in fit_records if "optimizer" in record]
    geometry = protocol["preparation"]["geometry"]
    selections = {region: json.loads((ROOT/"regions"/f"huc4_{region}"/"selection.json").read_text())["selections"]
                  for region in protocol["regions"]}
    selection_rows = [{"Region": region, **{family: value["recipe_id"] for family, value in selected.items()}}
                      for region, selected in selections.items()]
    regional = regions.pivot(index="target_huc4", columns="model_name", values="mae").reset_index()
    text = f"""# Common-query DOC reconstruction comparison

Date: 2026-10-11. Status: completed and independently verified internal development comparison.

## Outcome

The lowest pooled native-MAE point estimate belongs to **{point_winner}**. The full table and paired differences below retain all comparators and adverse outcomes. Native MAE and log1p MAE are the previously fixed joint primary endpoints; better RMSE/R² cannot replace them. This comparison does not establish universal superiority or global SOTA.

All seven methods predict the same **7,262 observed station-months** at **104 stations** across five held-out HUC4 regions of the Mississippi system. DOC concentration errors are in mg C/L; R² is predictive 1−SSE/SST. Metrics are computed per seed and then averaged, not from an ensemble. Every method has 100% coverage of the declared observed queries. This is not coverage or verified accuracy over all unobserved river cells.

## Common-query results

{table(display, ["Method", "mae", "log1p_mae", "rmse", "r2", "bias", "basin_macro_mae", "coverage"])}

The contemporary RF and GAM use environment15 + current driver/mask/age/season8. Historical RF and XGBoost use the 113-column environmental/history feature table. Fusion and the official architecture adaptation receive the same underlying environment and 12-month driver information, with permitted source/held-out covariate graphs at inference. SSN additionally uses declared physical NHD mainstem distances and confluence geometry. The models have different information organization and family-specific budgets: the table compares complete common-data recipes, not isolated architecture effects.

The 15 environmental columns comprise stream order, log drainage area, slope, a station-graph headwater indicator (no observed upstream neighbour), forest/agricultural/urban/wetland fractions, climatic precipitation and temperature normals, soil organic matter, elevation, baseflow index and two coordinates. The builder uses 2019 land cover and 1991–2020 climate normals as shared static descriptors; these are not a reconstruction of historical land-cover change. The headwater/stream-order descriptors also carry structural information, so a no-message control does not remove every spatial descriptor. The eight monthly channels are temperature, log discharge, their availability masks and ages, and seasonal sin/cos. The environmental builder contains no DOC-derived summary features.

## Paired regional uncertainty

Negative fusion-minus-comparator differences favour fusion. Intervals resample five whole basin clusters after averaging seed errors. They are exploratory, unadjusted for multiple comparisons, and based on previously inspected development regions. Deterministic SSN predictions repeat across nominal seed rows; these are not independent SSN replicates.

{table(pd.DataFrame(rows), ["Comparator", "Fusion MAE gain, %", "Paired MAE difference", "Exploratory 95% interval", "Regions with lower fusion MAE"])}

## Regional MAE

{table(regional, list(regional.columns))}

## New model fidelity and source selection

SSN uses the public SSN2 exponential tail-up, tail-down and Euclidean covariance formulas, a linear environmental/current-driver mean and a static network Gaussian station effect with independent measurement noise. Exact grouped REML uses every source training observation; its algebra is tested against dense GLS. It is a Python covariance-family adaptation rather than execution of the original Wampler R/ArcMap workflow. Covariance uses NHDPlus mainstem routing; minor diversions are excluded. **{geometry['position_imputations']} station positions** use the declared midpoint imputation and **{geometry['junctions_with_equal_weight_fallback']} junctions** use equal branch shares where drainage area is missing/nonpositive. All covariance families were checked for positive semidefiniteness at representative ranges.

PhyST-HC uses the author's original `PositionalEncoding` and `ImprovedTransformerGCN` definitions unchanged, with two width recipes. The wrapper repeats environment15 alongside each eight-channel monthly driver step and adds one zero pad for two attention heads. Monthly graph panels are disjoint. Inputs, concentration target, optimization objective and compact widths replace the native SWAT-C process-state/event-scale flux workflow. It is an **official architecture adaptation, not a complete PhyST-HC reproduction**.

Public source commits: PhyST-HC `{protocol['preparation']['source_commits']['PhyST-HC']}`; SSN2 `{protocol['preparation']['source_commits']['SSN2']}`. [SSN paper](https://bg.copernicus.org/articles/21/3093/2024/), [SSN2 implementation](https://github.com/USEPA/SSN2), [PhyST-HC paper](https://www.sciencedirect.com/science/article/pii/S0022169426013454), [author code](https://github.com/VenkateshBudamala/PhyST-HC).

The 15 SSN fits and 30 neural fits were all completed. SSN optimizer success flags: **{sum(item['optimizer_success'] for item in ssn_info)}/{len(ssn_info)}**; limits and optimization messages remain in individual fit records. The neural epoch ceiling is 60 with source-validation early stopping. None of these budgeted fits establishes optimized architecture-family rankings.

All five regions' selections were frozen before any new query scoring; previous RF/GAM/XGBoost/fusion predictions were reused as hashed dependencies. Selection is by source-validation seed-mean native MAE with the predefined tie-break rule; target DOC never enters inputs or selection.

{table(pd.DataFrame(selection_rows), ["Region", "ssn_covariance_adapt", "physt_architecture_adapt"])}

## Verification and interpretation

Independent audit replayed all **{audit['candidate_checkpoint_replays']}** candidate validation checkpoints and selected target predictions. Maximum validation replay discrepancy: **{audit['maximum_validation_replay_difference']:.3g}**. Important metrics were independently recomputed against the original DOC grid, with maximum discrepancy **{audit['maximum_independent_metric_difference']:.3g}**. Masks, whole-station role separation, source normalization, source selection gates, code/dataset/model hashes and complete query coverage were checked.

This literature comparison cannot isolate the contribution of adding environmental, temporal or network information. The separate [component study](doc_fusion_component_results_20261011.md) and no-message control address that question within their own protocols. Independent rivers, target-only graphs, arbitrary virtual sites and true missing DOC cells remain outside validated scope.

The serial v1 trial was interrupted before new target scoring. V2 re-froze identical recipes and endpoints with five independent region workers and one CPU thread, selected using a source-covariate-only timing benchmark. All 45 final fits restarted under v2; partial v1 fits are not mixed into this table. Existing paper freezes and training dispatcher remain intact.

Evidence: [protocol](../experiments/phase4_transfer/doc_network_work_v2/protocol.json), [summary](../experiments/phase4_transfer/doc_network_work_v2/analysis/summary.csv), [regional/seed results](../experiments/phase4_transfer/doc_network_work_v2/analysis/by_region_seed.csv), [paired intervals](../experiments/phase4_transfer/doc_network_work_v2/analysis/paired_comparisons.json), [independent audit](../experiments/phase4_transfer/doc_network_work_v2/analysis/independent_audit.json), [design](doc_network_work_plan_20261011.md).
"""
    destination = Path("docs/doc_network_work_results_20261011.md")
    destination.write_text(text)
    print(json.dumps({"report": str(destination), "point_winner": point_winner,
                      "native_reproductions": False, "new_fits": len(fit_records)}))


if __name__ == "__main__":
    main()
