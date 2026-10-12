"""Check summary aggregation and write the completed traditional-method report."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from river_graph.experiments.provenance import sha256_file

ROOT = Path("experiments/phase4_transfer/doc_interpolation_v2")
LABELS = {"fixed_fusion": "Environmental MLP + GRU + graph Transformer",
          "seasonal_median": "Station seasonal/source seasonal median",
          "nearest_spatial": "Same-month spatial nearest neighbour",
          "euclidean_idw": "Same-month Euclidean IDW",
          "river_idw": "Same-month NHD river-distance IDW",
          "spatiotemporal_idw": "Bounded space-time IDW",
          "ordinary_kriging": "Exponential ordinary kriging",
          "spatiotemporal_kriging": "Exponential space-time ordinary kriging",
          "linear_time": "Within-station linear time interpolation",
          "pchip_time": "Within-station PCHIP"}


def markdown(frame, columns):
    lines = ["| " + " | ".join(columns) + " |", "| " + " | ".join("---" for _ in columns) + " |"]
    for row in frame[columns].itertuples(index=False, name=None):
        lines.append("| " + " | ".join(f"{v:.4f}" if isinstance(v, float) else str(v) for v in row) + " |")
    return "\n".join(lines)


def main():
    audit = json.loads((ROOT/"analysis/independent_audit.json").read_text())
    if audit["status"] != "passed":
        raise ValueError("prediction replay and independent metric audit required")
    by_seed = pd.read_csv(ROOT/"analysis/by_seed.csv")
    summary = pd.read_csv(ROOT/"analysis/summary.csv")
    largest = 0.
    for item in summary.to_dict("records"):
        group = by_seed[(by_seed.track == item["track"]) & (by_seed.gap_months == item["gap_months"]) & (by_seed.model_name == item["model_name"])]
        assert len(group) == 3
        for field in ("mae", "log1p_mae", "rmse", "r2", "bias", "query_count", "intrinsic_coverage"):
            delta = abs(float(group[field].mean())-item[field])
            largest = max(largest, delta)
            assert delta < 1e-10
    receipt = {"status": "passed", "summary_rows": len(summary), "maximum_mean_difference": largest,
               "report_generator_sha256": sha256_file(Path(__file__)),
               "summary_sha256": sha256_file(ROOT/"analysis/summary.csv")}
    (ROOT/"analysis/aggregation_audit.json").write_text(json.dumps(receipt, indent=2)+"\n")
    summary["Method"] = summary.model_name.map(LABELS)
    time = summary[summary.track == "time"].pivot(index="Method", columns="gap_months", values="mae").reset_index()
    time.columns = ["Method", "MAE:3 months", "MAE:6 months", "MAE:12 months"]
    long = summary[(summary.track == "time") & (summary.gap_months == 12)].sort_values("mae")
    space = summary[summary.track == "space"].sort_values("mae")
    short = summary[(summary.track == "time") & (summary.gap_months == 3)]
    long_fusion = long[long.model_name == "fixed_fusion"].iloc[0]
    long_median = long[long.model_name == "seasonal_median"].iloc[0]
    short_fusion = short[short.model_name == "fixed_fusion"].iloc[0]
    short_linear = short[short.model_name == "linear_time"].iloc[0]
    spatial_fusion = space[space.model_name == "fixed_fusion"].iloc[0]
    spatial_idw = space[space.model_name == "spatiotemporal_idw"].iloc[0]
    common = pd.read_csv(ROOT/"analysis/common_subset_by_seed.csv")
    common = common.groupby(["track", "gap_months", "model_name"], as_index=False).mean(numeric_only=True)
    common["Method"] = common.model_name.map(LABELS)
    counts = []
    for seed in (42, 43, 44):
        with np.load(ROOT/"masks"/f"time_g12_s{seed}.npz") as saved:
            counts.append({"seed": seed, "test_queries": len(saved["test"]), "validation_queries": len(saved["val"]), "eligible_stations": len(saved["eligible_rows"])})
    selection_rows = []
    for path in sorted((ROOT/"tasks").glob("*/selection.json")):
        for family, recipe in json.loads(path.read_text())["selections"].items():
            selection_rows.append({"Task": path.parent.name, "Method": LABELS[family], "Recipe": recipe["id"]})
    choices = pd.DataFrame(selection_rows)
    choices.to_csv(ROOT/"analysis/selected_recipes.csv", index=False)
    text = f"""# Traditional DOC interpolation comparison

Date: 2026-10-11. Completed, independently audited internal Mississippi development experiment.

## Outcome and information boundary

Lowest full-population MAE point estimates: 12-month time gaps **{long.iloc[0]['Method']}**; whole-station geographical gaps **{space.iloc[0]['Method']}**. Rankings refer to these fixed recipes and populations, not globally optimized method families. Adverse results for the fusion model are retained.

Time tasks retrain the unchanged environmental MLP + GRU + graph Transformer nine times, on the exact visible DOC labels available to interpolation. The neural model does not explicitly read DOC context as a feature. Linear/PCHIP and space-time methods may use future visible DOC; this is retrospective smoothing. Hydrological neural inputs remain the twelve past/current months of temperature/discharge, masks, ages and season. Spatial tasks reuse the previously audited fifteen source-trained fusion models under the exact same five HUC4 masks: target stations have no DOC support; validation DOC never enters interpolator neighbours.

All fourteen validation selections were completed before any new target scoring. The primary endpoints remain native MAE and log1p MAE. Native concentration errors are mg C/L; R² is predictive 1−SSE/SST. Metrics are calculated separately per seed/mask then averaged, not from a prediction ensemble. Deterministic spatial predictions repeat under nominal seed rows and are not independent fits.

## Interpretation

The fusion model's spatial MAE is {spatial_fusion.mae:.4f}, versus {spatial_idw.mae:.4f} for the lowest-MAE traditional spatial recipe, bounded space-time IDW: a {100*(spatial_idw.mae-spatial_fusion.mae)/spatial_idw.mae:.2f}% point-estimate reduction. Fusion also has the lowest spatial log1p MAE/RMSE and highest spatial R² in this comparison; the ranking is retained on the common covered subset.

Temporal advantages are smaller and endpoint-dependent. At three months, fusion native MAE is {short_fusion.mae:.4f}, versus linear interpolation {short_linear.mae:.4f}, only a {100*(short_linear.mae-short_fusion.mae)/short_linear.mae:.2f}% point-estimate reduction; linear interpolation has better log1p MAE, RMSE and R². At twelve months, fusion native MAE is {long_fusion.mae:.4f}, versus seasonal median {long_median.mae:.4f}, only a {100*(long_median.mae-long_fusion.mae)/long_median.mae:.2f}% reduction; seasonal median has lower log1p MAE, and bounded space-time IDW has better RMSE/R². These results support further evaluation of spatial reconstruction, but do not establish universal temporal or joint-endpoint superiority. No test-driven architecture or endpoint change is made here.

## Temporal population

{markdown(pd.DataFrame(counts), ['seed','test_queries','validation_queries','eligible_stations'])}

For each seed, the query/validation three-month cores are identical across 3/6/12-month gaps; larger gaps hide additional surrounding observations without changing the scored truth. Eligibility is based on dates, requires largest-gap brackets within twelve months outside the gap and at least six retained training observations, and excludes naturally unbracketed sparse stations. The counts are query occurrences across masks, not independent stations. There is no accuracy claim on naturally missing DOC cells.

## Temporal native MAE versus gap length

{markdown(time.sort_values('MAE:12 months'), list(time.columns))}

## Full temporal metrics at twelve months

{markdown(long, ['Method','mae','log1p_mae','rmse','r2','intrinsic_coverage'])}

## Full spatial population: 7,262 queries at 104 stations

{markdown(space, ['Method','mae','log1p_mae','rmse','r2','intrinsic_coverage'])}

**Coverage meaning:** every full-population row has 100% final finite coverage. Intrinsic coverage is the fraction actually answered by the selected interpolator before fallback. Any unsupported prediction uses the declared training station-season median, station median, source-season median, then global median. Thus these are complete operational recipes with a declared fallback; a kriging+fallback row must not be called pure kriging accuracy over every cell. Seasonal median and fusion have full intrinsic coverage by construction.

Spatial raw time interpolation is excluded because target stations have no observed DOC. Same-month methods can lack donors in sparse historical months. Space-time methods use source observations within predefined 6/12-month windows. River IDW uses physical cached NHD mainstem/shared-downstream distance; Euclidean methods use Earth-centered chord km. Ordinary kriging uses a positive exponential covariance, with validation-selected range/nugget and up to30 neighbours. Space-time ordinary kriging uses up to20 observations and the Euclidean norm of scaled coordinates/time. Native outputs share the inherited clamp [0,expm1(15)]; raw values and interpolation/fallback indicators are saved.

## Raw-method comparison on the exact common covered subset

All methods in each track/seed are scored on the same intersection of intrinsic coverage. The following counts and scores exclude fallback and retain fusion on the identical cells. These populations differ from the full-population tables and must not be combined in one ranking.

{markdown(common.sort_values(['track','gap_months','mae']), ['track','gap_months','Method','mae','log1p_mae','rmse','r2','query_count'])}

## Verification and scope

Independent audit replayed all **{audit['validation_replays']}** candidate validation outputs and selected test predictions, including all nine neural checkpoints. Maximum validation replay difference: **{audit['maximum_replay_difference']:.3g}**. Independent full/common-subset metric reconciliation passed, maximum main-metric discrepancy **{audit['maximum_metric_difference']:.3g}**. Aggregation independently matched all **{len(summary)}** summary rows with maximum discrepancy **{largest:.3g}**. Query truth, source-support separation, selected settings, immutable dependencies, model normalization, masks, hashes and explicit fallback coverage were checked.

Seed repetition alone does not establish population significance. These previously examined river-system data are development evidence; results do not demonstrate independent-river transfer, arbitrary virtual-site accuracy, or a fully observed continuous DOC field. The native interpolation/observation-assisted task is distinct from a future deployment without available DOC support.

V1 failed before neural training or target scoring on a shared metrics-interface mismatch. V2 corrected the frame contract, added an integration guard, re-froze unchanged tasks/models/budgets/endpoints and restarted all nine final fits. The failed v1 receipt is retained; no partial fit is mixed into these results.

The representative process-model feasibility investigation is reported separately in [INCA-C feasibility](doc_process_model_feasibility_20261011.md). A native supplied example ran successfully; no physical-model score on a different catchment is inserted here.

Evidence: [design](doc_interpolation_plan_20261011.md), [protocol](../experiments/phase4_transfer/doc_interpolation_v2/protocol.json), [summary](../experiments/phase4_transfer/doc_interpolation_v2/analysis/summary.csv), [seed/mask metrics](../experiments/phase4_transfer/doc_interpolation_v2/analysis/by_seed.csv), [common-subset metrics](../experiments/phase4_transfer/doc_interpolation_v2/analysis/common_subset_by_seed.csv), [selected settings](../experiments/phase4_transfer/doc_interpolation_v2/analysis/selected_recipes.csv), [audit](../experiments/phase4_transfer/doc_interpolation_v2/analysis/independent_audit.json).
"""
    destination = Path("docs/doc_interpolation_results_20261011.md")
    destination.write_text(text)
    print(json.dumps({"report": str(destination), "time_12month_mae_winner": long.iloc[0].model_name,
                      "space_mae_winner": space.iloc[0].model_name, "aggregation_audit": "passed"}))


if __name__ == "__main__":
    main()
