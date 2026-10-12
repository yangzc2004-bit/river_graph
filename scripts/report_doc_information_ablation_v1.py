"""Write the completed, audited information-ablation interpretation."""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

ROOT = Path("experiments/phase4_transfer/doc_information_ablation_v1")
LABELS = {"fixed_fusion": "Full environmental MLP + GRU + graph Transformer",
    "no_environment": "Remove all static environmental inputs",
    "no_time": "Remove all dynamic/time/season inputs",
    "no_river": "Remove cross-node exchange and explicit river descriptors",
    "local_no_messages": "Self-only attention; static river descriptors retained",
    "current_only": "Current-month information only; same GRU operator",
    "no_relations": "Generic dense attention; river relation categories removed"}


def table(frame, columns):
    lines = ["| " + " | ".join(columns) + " |", "| " + " | ".join("---" for _ in columns) + " |"]
    best = {}
    for column in columns:
        if column in ("mae", "log1p_mae", "rmse") or column.startswith("MAE:"):
            best[column] = frame[column].min()
        elif column == "r2":
            best[column] = frame[column].max()
    for values in frame[columns].itertuples(index=False, name=None):
        rendered = []
        for column, value in zip(columns, values, strict=True):
            if column == "mae_increase_percent":
                cell = f"{value:+.2f}%"
            else:
                cell = f"{value:.4f}" if isinstance(value, float) else str(value)
            if column in best and value == best[column]:
                cell = f"**{cell}**"
            rendered.append(cell)
        lines.append("| " + " | ".join(rendered) + " |")
    return "\n".join(lines)


def main():
    audit = json.loads((ROOT/"analysis/independent_audit.json").read_text())
    assert audit["status"] == "passed"
    protocol = json.loads((ROOT/"protocol.json").read_text())
    summary = pd.read_csv(ROOT/"analysis/summary.csv")
    summary["Method"] = summary.model_name.map(LABELS)
    spatial = summary[summary.track == "space"].set_index("model_name")
    primary = spatial.loc[["fixed_fusion", *protocol["primary_arms"]]].reset_index()
    controls = spatial.loc[["fixed_fusion", "local_no_messages", "current_only", "no_relations"]].reset_index()
    temporal = summary[summary.track == "time"].pivot(index="model_name", columns="gap_months", values="mae")
    temporal = temporal.loc[["fixed_fusion", *protocol["primary_arms"]]].reset_index()
    temporal["Method"] = temporal.model_name.map(LABELS)
    temporal = temporal.rename(columns={3: "MAE:3 months", 6: "MAE:6 months", 12: "MAE:12 months"})
    intervals = pd.DataFrame(json.loads((ROOT/"analysis/paired_comparisons.json").read_text()))
    contrasts = intervals[intervals.endpoint == "mae"].copy()
    contrasts["Method"] = contrasts.comparator.map(LABELS)
    contrasts["Removal MAE increase"] = -contrasts.difference_fusion_minus_comparator
    contrasts["Exploratory 95% interval"] = contrasts.difference_interval.map(lambda values: f"[{-values[1]:.4f}, {-values[0]:.4f}]")
    descriptions = []
    for arm in protocol["primary_arms"]:
        row = spatial.loc[arm]
        descriptions.append(f"- **{LABELS[arm]}**: spatial MAE {row.mae:.4f}; change {row.mae_increase:+.4f} mg C/L ({row.mae_increase_percent:+.2f}%).")
    history = spatial.loc["current_only"]
    relations = spatial.loc["no_relations"]
    messages = spatial.loc["local_no_messages"]
    environment = spatial.loc["no_environment"]
    dynamic = spatial.loc["no_time"]
    river = spatial.loc["no_river"]
    temporal_changes = summary[summary.track == "time"].groupby("model_name").mae_increase_percent.agg(["min", "max"])
    change_ranges = {arm: f"{temporal_changes.loc[arm, 'min']:.2f}–{temporal_changes.loc[arm, 'max']:.2f}%"
        for arm in protocol["primary_arms"]}
    long = summary[(summary.track == "time") & (summary.gap_months == 12)]
    best = long.loc[long.mae.idxmin()]
    text = """# DOC information ablation results

Date: 2026-10-11. Completed matched-input internal Mississippi development study.

## Conditional contribution estimates

Each change below compares a retrained masked-input recipe to the same full model, with matched parameter tensors, seed initialization, optimizer, label weights, stopping rule and query population. Positive error changes mean the removal worsened prediction; negative changes mean the removal improved the point estimate. These percentages are changes in prediction error, not additive fractions of DOC variation or causal source contributions.

""" + "\n".join(descriptions) + f"""

## Spatial primary removals

Five whole-station HUC4 tasks; 7,262 observed queries at 104 stations; 100% prediction coverage. MAE/RMSE are mg C/L. Log1p MAE and native MAE remain joint primary endpoints; predictive R² is 1−SSE/SST. Metrics are pooled per seed then averaged, without ensembling.

{table(primary, ['Method','mae','log1p_mae','rmse','r2','mae_increase','mae_increase_percent'])}

## Spatial explanatory controls

{table(controls, ['Method','mae','log1p_mae','rmse','r2','mae_increase_percent'])}

Historical information removal changes spatial MAE by **{history.mae_increase_percent:+.2f}%**; it uses the same twelve-step GRU with eleven zeroed past steps and zeroed current observation-age channels. This estimates the contribution of past information conditional on current values/masks/season; it does not compare GRU against alternative architectures.

Removing river relation categories while retaining dense context and every static descriptor changes spatial MAE by **{relations.mae_increase_percent:+.2f}%**. Removing cross-node messages while retaining every static descriptor changes it by **{messages.mae_increase_percent:+.2f}%**. These controls distinguish explicit relation coding from covariate exchange. Strict no-river additionally removes the four explicit static river descriptors, so its contrast combines those removals and cannot be labelled a topology-only effect.

The environmental package includes all thirteen river/ecological descriptors and two coordinates. Its removal is a static-package effect; ecological variables, location and hydraulic regime are not isolated individually. Direct river descriptors overlap this package, whereas climate/land-use descriptors and generic coordinates remain in strict no-river. The three package effects must not be added to 100%.

## Temporal primary removals

266 eligible stations; three masking seeds; the same three-month scored cores within each seed across 3/6/12-month excluded windows. Full-model checkpoints and masks are exactly those of the traditional interpolation comparison. Retrospective reconstruction uses exogenous covariates, with no DOC observation context as a model input. Largest-gap bracket eligibility restricts the evaluated population.

{table(temporal, ['Method','MAE:3 months','MAE:6 months','MAE:12 months'])}

Lowest twelve-month native MAE point estimate: **{best['Method']}**, {best.mae:.4f}. Full temporal MAE/log1p MAE/RMSE/R², bias and error changes are retained in the linked machine-readable summary. No independent population significance is inferred from three mask seeds or the repeated query cores.

The no-time model has no time-varying inputs and therefore gives a constant concentration through time at each station in evaluation mode. Even a competitive concentration MAE would not establish reproduction of temporal fluctuations. Current-only, in contrast, retains current hydrological drivers and season, so its output may vary through time without historical input.

## Paired spatial uncertainty

{table(contrasts, ['Method','Removal MAE increase','Exploratory 95% interval','regions_with_lower_error'])}

The last column counts regions with lower full-model MAE. Intervals resample five complete HUC4 clusters after averaging seed errors; they are exploratory, unadjusted and based on previously inspected regions. An interval crossing zero does not establish a consistent population advantage. Log1p endpoint intervals remain in the paired-comparison JSON and should be assessed alongside native MAE before making a joint-endpoint claim.

## Findings for the model rationale

On these matched tasks, the full model has the lowest native and log1p MAE among the four primary recipes, in both spatial reconstruction and every temporal gap length. The static package produces the largest conditional error increase: **{environment.mae_increase_percent:.2f}%** spatially and **{change_ranges['no_environment']}** temporally. Removing all dynamic and seasonal information increases spatial MAE by **{dynamic.mae_increase_percent:.2f}%** and temporal MAE by **{change_ranges['no_time']}**. For these two spatial removals, both endpoint cluster intervals exclude zero in favour of the full model; this is supportive internal evidence, with only five previously inspected clusters and no multiplicity adjustment.

The strict river removal increases spatial MAE by **{river.mae_increase_percent:.2f}%** and temporal MAE by **{change_ranges['no_river']}**, but its spatial intervals include zero for both primary endpoints. The near-zero **{relations.mae_increase_percent:.2f}%** change after removing relation categories does not support a strong claim for the current explicit upstream/downstream encoding. Dense attention retains virtually the full-model point estimate without those categories. The no-message control has a **{messages.mae_increase_percent:.2f}%** spatial MAE increase, with an interval including zero on native MAE but excluding zero on log1p MAE. This does not establish a joint-endpoint population advantage.

The current-only recipe increases native MAE by **{history.mae_increase_percent:.2f}%**, with both spatial intervals including zero. Its RMSE is **{history.rmse:.4f}** and R² **{history.r2:.4f}**, compared with **{spatial.loc['fixed_fusion'].rmse:.4f}** and **{spatial.loc['fixed_fusion'].r2:.4f}** for the full model. Thus current drivers/season and historical memory must be discussed separately: retaining history improves the average absolute-error point estimate, while the current-only recipe has better squared-error metrics. This ablation supports the three-input framework on the predefined primary metrics but leaves the unique benefit of explicit river relation coding and the optimal temporal operator unresolved.

## Verification and scope

All 141 validation selections were bound before any new test scoring: **102 new fits** and **39 reused checkpoints**. Each new removal was retrained rather than applied only after full-model fitting. Full spatial and self-only baseline checkpoints are the original matched architecture-study models; full temporal models are the original interpolation models. Four-thread spatial and one-thread temporal arithmetic, dataset and masks were retained. Every arm has **{audit['identical_parameter_count']} parameters**, with identical parameter shapes and seeded initialization; masking retains constant/bias paths rather than changing width or removing layers.

All validation/test checkpoint outputs replayed with maximum discrepancy **{audit['maximum_prediction_replay_difference']:.3g}**. Independent original-DOC query truth, source-only normalization, selected epochs, masks/hashes, coverage, paired intervals and **{audit['summary_rows']}** summary rows were checked. Maximum independent metric difference: **{audit['maximum_independent_metric_difference']:.3g}**. The full spatial model matches the prior comparator baseline. Original paper freezes/endpoints were not modified.

The supported interpretation is conditional predictive information value on these fixed tasks. Independent-river transfer, target-only inference graphs, continuous unobserved-field accuracy and arbitrary virtual sites remain separate evaluations. Adverse or uncertain removal results are retained and cannot justify changing the predefined endpoint after seeing the test scores.

Evidence: [design](doc_information_ablation_plan_20261011.md), [protocol](../experiments/phase4_transfer/doc_information_ablation_v1/protocol.json), [all metrics](../experiments/phase4_transfer/doc_information_ablation_v1/analysis/summary.csv), [seed metrics](../experiments/phase4_transfer/doc_information_ablation_v1/analysis/by_seed.csv), [regional metrics](../experiments/phase4_transfer/doc_information_ablation_v1/analysis/by_region_seed.csv), [paired intervals](../experiments/phase4_transfer/doc_information_ablation_v1/analysis/paired_comparisons.json), [independent audit](../experiments/phase4_transfer/doc_information_ablation_v1/analysis/independent_audit.json).
"""
    destination = Path("docs/doc_information_ablation_results_20261011.md")
    destination.write_text(text)
    print(json.dumps({"report": str(destination), "status": "completed_audited",
        "spatial_primary_changes_percent": {arm: float(spatial.loc[arm].mae_increase_percent) for arm in protocol["primary_arms"]}}))


if __name__ == "__main__":
    main()
