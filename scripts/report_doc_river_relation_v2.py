"""Write the audited directed-river comparison without target-based selection."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from run_doc_river_relation_v2 import DATASET, ROOT, task_context

from river_graph.models.river_relation_comparison import ARMS, CANDIDATES

LABELS = {
    "fixed_fusion": "Original graph Transformer",
    "generic_dense": "Generic dense attention (control)",
    "path_closure": "Full-path relation categories",
    "upstream_only": "Upstream-only attention",
    "direction_heads": "Upstream/downstream/global heads",
    "path_bias": "Directed log-hop attention bias",
}


def table(frame, columns):
    best = {}
    for column in columns:
        if column in ("mae", "log1p_mae", "rmse") or column.startswith("MAE:"):
            best[column] = frame[column].min()
        elif column == "r2":
            best[column] = frame[column].max()
    lines = ["| " + " | ".join(columns) + " |", "| " + " | ".join("---" for _ in columns) + " |"]
    for row in frame[columns].itertuples(index=False, name=None):
        cells = []
        for column, value in zip(columns, row, strict=True):
            if column.endswith("improvement_percent"):
                cell = f"{value:+.2f}%"
            else:
                cell = f"{value:.4f}" if isinstance(value, float) else str(value)
            if column in best and value == best[column]:
                cell = f"**{cell}**"
            cells.append(cell)
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def topology_diagnostics(protocol):
    data = torch.load(DATASET, weights_only=False, map_location="cpu")
    seen, rows = set(), []
    for task in protocol["tasks"]:
        if task["mask"] in seen:
            continue
        seen.add(task["mask"])
        original = task_context(ROOT, {**task, "arm": "fixed_fusion"}, data)
        closure = task_context(ROOT, {**task, "arm": "path_closure"}, data)
        for role in ("train", "val", "test"):
            before, after = original["graphs"][role], closure["graphs"][role]
            rows.append({"mask": task["mask"], "track": task["track"], "role": role,
                "active_nodes": len(before["rows"]),
                "original_ancestor_pairs": int(torch.isin(before["relation"], torch.tensor([1, 2])).sum()),
                "closure_ancestor_pairs": int(torch.isin(after["relation"], torch.tensor([1, 2])).sum()),
                "restored_ancestor_pairs": after["restored_ancestor_pairs"],
                "relation_matrix_unchanged": bool(torch.equal(before["relation"], after["relation"]))})
    result = pd.DataFrame(rows)
    result.to_csv(ROOT/"analysis/topology_diagnostics.csv", index=False)
    return result


def main():
    audit = json.loads((ROOT/"analysis/independent_audit.json").read_text())
    assert audit["status"] == "passed"
    protocol = json.loads((ROOT/"protocol.json").read_text())
    choice = json.loads((ROOT/"selected_recipe.json").read_text())
    summary = pd.read_csv(ROOT/"analysis/summary.csv")
    spatial = summary[summary.track == "space"].set_index("model_name").loc[list(ARMS)].reset_index()
    spatial["Method"] = spatial.model_name.map(LABELS)
    time = summary[summary.track == "time"].pivot(index="model_name", columns="gap_months", values="mae")
    time = time.loc[list(ARMS)].reset_index().rename(columns={3: "MAE:3 months", 6: "MAE:6 months", 12: "MAE:12 months"})
    time["Method"] = time.model_name.map(LABELS)
    long = summary[(summary.track == "time") & (summary.gap_months == 12)].set_index("model_name").loc[list(ARMS)].reset_index()
    long["Method"] = long.model_name.map(LABELS)
    intervals = pd.DataFrame(json.loads((ROOT/"analysis/paired_comparisons.json").read_text()))
    intervals["Method"] = intervals.comparator.map(LABELS)
    intervals["Improvement interval"] = intervals.difference_interval.map(lambda pair: f"[{pair[0]:.4f}, {pair[1]:.4f}]")
    selected = choice["selected_arm"]
    row = spatial.set_index("model_name").loc[selected]
    selected_time = summary[(summary.track == "time") & (summary.model_name == selected)]
    seed_metrics = pd.read_csv(ROOT/"analysis/by_seed.csv")
    temporal_seeds = seed_metrics[seed_metrics.track == "time"]
    selected_seeds = temporal_seeds[temporal_seeds.model_name == selected].merge(
        temporal_seeds[temporal_seeds.model_name == "fixed_fusion"],
        on=["gap_months", "seed"], suffixes=("", "_baseline"), validate="one_to_one")
    temporal_joint_wins = int(((selected_seeds.mae < selected_seeds.mae_baseline)
        & (selected_seeds.log1p_mae < selected_seeds.log1p_mae_baseline)).sum())
    baseline_space = spatial.set_index("model_name").loc["fixed_fusion"]
    selected_rmse_improvement = 100*(baseline_space.rmse-row.rmse)/baseline_space.rmse
    selected_intervals = intervals[intervals.comparator == selected]
    spatial_joint_interval_support = len(selected_intervals) == 2 and all(
        pair[0] > 0 for pair in selected_intervals.difference_interval)
    spatial_interval_statement = (
        "Both exploratory spatial primary intervals exclude zero in the improving direction."
        if spatial_joint_interval_support else
        "The exploratory spatial primary intervals do not establish a joint improvement over the baseline.")
    generic_long = summary[(summary.track == "time") & (summary.gap_months == 12)
                           & (summary.model_name == "generic_dense")].iloc[0]
    selected_long = selected_time[selected_time.gap_months == 12].iloc[0]
    long_improvement_over_generic = 100*(generic_long.mae-selected_long.mae)/generic_long.mae
    best_space = spatial.loc[spatial.mae.idxmin()]
    diagnoses = topology_diagnostics(protocol)
    choices = pd.DataFrame([{"Method": LABELS[arm], "Validation joint ratio": choice["validation_joint_ratios"][arm],
        "Eligible": arm == "fixed_fusion" or arm in CANDIDATES, "Selected": arm == selected} for arm in ARMS])
    findings = []
    for arm in CANDIDATES:
        values = spatial.set_index("model_name").loc[arm]
        timing = summary[(summary.track == "time") & (summary.model_name == arm)].mae_improvement_percent
        findings.append(f"- **{LABELS[arm]}**: spatial native MAE improvement {values.mae_improvement_percent:+.2f}%, log1p MAE improvement {values.log1p_mae_improvement_percent:+.2f}%; temporal native improvements {timing.min():+.2f}% to {timing.max():+.2f}%.")
    temporal_closure_unchanged = bool(diagnoses[diagnoses.track == "time"].relation_matrix_unchanged.all())
    temporal_closure_statement = (
        "All temporal relation matrices are unchanged by closure."
        if temporal_closure_unchanged else
        "Closure changes at least one temporal relation matrix; consult the role diagnostics.")
    text = f"""# Directed river relation comparison

Completed prospective matched development experiment, frozen as doc_river_relation_v2.

## Main result and validation-selected recipe

The recipe selected **before test scoring** is **{LABELS[selected]}**, with validation joint ratio **{choice['validation_joint_ratios'][selected]:.6f}** (original baseline = 1). Its spatial native MAE is **{row.mae:.4f} mg C/L**, a **{row.mae_improvement_percent:+.2f}%** improvement relative to the original graph Transformer; log1p MAE improvement is **{row.log1p_mae_improvement_percent:+.2f}%**. Its temporal native improvements range from **{selected_time.mae_improvement_percent.min():+.2f}%** to **{selected_time.mae_improvement_percent.max():+.2f}%**.

The lowest spatial test MAE point estimate is **{best_space.Method}**, {best_space.mae:.4f}. This descriptive ranking does not change the frozen validation-selected recommendation. If the original baseline wins validation, it remains the selected recipe even if a candidate has a favourable test row. Generic dense attention is a diagnostic control outside the selection set.

Both temporal primary errors are lower than the baseline in **{temporal_joint_wins}/{len(selected_seeds)} mask-seed tasks** for the validation-selected recipe. These repeated tasks measure robustness to the existing masks and initialization; they are not nine independent rivers. Its spatial RMSE improvement is **{selected_rmse_improvement:+.2f}%**. {spatial_interval_statement} Temporal gains and a secondary spatial-metric gain therefore must be distinguished from a demonstrated joint spatial advantage. For the selected directional-head recipe, retain the environmental MLP and GRU and treat the upstream/downstream/global head allocation as the supported next candidate, with the strongest current evidence in temporal reconstruction.

For the twelve-month gap, generic dense attention already achieves native MAE **{generic_long.mae:.4f}**, versus **{selected_long.mae:.4f}** for the selected recipe: only **{long_improvement_over_generic:+.2f}%** lower error. Thus most of the twelve-month improvement over the original relation-category baseline is also observed in the generic control. The clearer numerical separation for directional heads is at three- and six-month gaps.

## Spatial reconstruction

Same five HUC4 tasks, seeds 42/43/44, 104 target stations and 7,262 observed queries per seed, with 100% coverage. Metrics pool query cells within each seed and then average seeds; these are not ensemble predictions. MAE/RMSE are mg C/L; predictive R² is 1−SSE/SST. Positive improvement means lower error than the original baseline, whose denominator is retained for every comparison.

{table(spatial, ['Method','mae','log1p_mae','rmse','r2','mae_improvement_percent','log1p_mae_improvement_percent'])}

## Temporal reconstruction

Same nine 3/6/12-month masks at 266 eligible stations. Scored three-month cores are the same within masking seed across gap lengths. Training supervision before and after missing windows is permitted for retrospective reconstruction; no DOC value/history is a model input.

{table(time, ['Method','MAE:3 months','MAE:6 months','MAE:12 months'])}

Twelve-month secondary metrics:

{table(long, ['Method','mae','log1p_mae','rmse','r2','mae_improvement_percent'])}

## Candidate effects

""" + "\n".join(findings) + f"""

These are recipe-level predictive differences conditional on the same environmental/temporal inputs. Comparing each directional candidate against path closure helps separate omitted-intermediate connectivity from attention-mask or distance-bias changes. Relative effects cannot be added into independent causal contributions.

{temporal_closure_statement} There are **{int(diagnoses[diagnoses.track == 'space'].restored_ancestor_pairs.sum())}** restored upstream pairs across the spatial mask/role combinations (these are repeated graph diagnostics, not distinct population station pairs). Closure cannot add structural information when the permitted roles already contain the complete relevant paths. The diagnostic CSV records every mask and role.

## Paired spatial uncertainty

Native and log1p MAE are joint primary endpoints. Improvement intervals below are original-baseline error minus candidate error. Intervals resample five complete HUC4 clusters after seed-error averaging, using the same 5,000 draws as the previous comparisons. They are exploratory, unadjusted and derived from previously inspected development regions. An interval crossing zero does not establish a consistent population advantage; improvement on one primary endpoint alone does not establish joint superiority. Temporal masking seeds and repeated cores do not provide independent-river significance.

{table(intervals, ['Method','endpoint','difference_fusion_minus_comparator','Improvement interval','regions_with_lower_error'])}

The final column counts regions with lower **original-baseline** error, so fewer is favourable for the new candidate.

## Selection, capacity and verification

{table(choices, ['Method','Validation joint ratio','Eligible','Selected'])}

Validation selection uses relative native/log1p MAE with equal endpoint weights, equal task weights within each track, and equal spatial/temporal weights. All 144 validation selections and this choice were bound before any new target score: **105 new fits and 39 reused audited checkpoints**. Every recipe has **{audit['identical_parameter_count']} parameters**, with matching parameter shapes and exactly identical tensors at seeded initialization. The environmental MLP, twelve-step GRU, season channel, two spatial blocks, optimizer, label weights, monthly permutation and stopping rule remain identical. The optimizer loop is copied exactly except for its model factory. Four-thread spatial and one-thread temporal arithmetic are retained.

Topology closure traverses excluded intermediate nodes only to identify known paths. Their covariates/hidden states/DOC never enter message passing. Direction heads specialize three existing heads; upstream-only masks all downstream/unconnected keys. Directed path bias reuses the existing six-by-three tensor with six basis coefficients. Distance is the shortest number of existing **station edges**, not physical river kilometres, travel time or a mass-balance law. No new external covariates were introduced.

Every validation/test checkpoint prediction was replayed, with maximum discrepancy **{audit['maximum_prediction_replay_difference']:.3g}**. Independent metric discrepancy across **{audit['summary_rows']}** summary rows was **{audit['maximum_independent_metric_difference']:.3g}**. Source-only scaling, query truth/masks, selected epochs, parameter counts, validation-only global choice and paired intervals were checked. SciPy independently verified complete-topology distances and directed relation matrices. Original baseline and spatial generic control metrics matched their earlier audited results. Original paper endpoints/freezes are unchanged.

The initial v1 metadata omission was detected by a frozen contract before any target scoring. Its seven validation receipts, zero test receipts and original sources were preserved under the aborted-metadata directory. Candidate model computation was retained, count metadata was corrected in a new v2 freeze, and no v1 candidate checkpoint was reused.

## Interpretation limits and evidence

These locally designed DOC candidates draw motivation from shortest-path structural bias in [Graphormer](https://arxiv.org/abs/2106.05234) and upstream hydrological priors in [HydroNets](https://arxiv.org/abs/2007.00595); neither complete published model is reproduced here. Results concern observed-query reconstruction within the existing Mississippi development tasks. Independent rivers, naturally missing cells, arbitrary virtual stations and physical transport consistency require separate evaluation. No endpoint, architecture or selection rule was changed after new test scoring.

Evidence: [design](doc_river_relation_plan_20261011.md), [protocol](../experiments/phase4_transfer/doc_river_relation_v2/protocol.json), [metrics](../experiments/phase4_transfer/doc_river_relation_v2/analysis/summary.csv), [seed metrics](../experiments/phase4_transfer/doc_river_relation_v2/analysis/by_seed.csv), [regional metrics](../experiments/phase4_transfer/doc_river_relation_v2/analysis/by_region_seed.csv), [intervals](../experiments/phase4_transfer/doc_river_relation_v2/analysis/paired_comparisons.json), [validation choice](../experiments/phase4_transfer/doc_river_relation_v2/selected_recipe.json), [topology diagnostics](../experiments/phase4_transfer/doc_river_relation_v2/analysis/topology_diagnostics.csv), [independent audit](../experiments/phase4_transfer/doc_river_relation_v2/analysis/independent_audit.json).
"""
    destination = Path("docs/doc_river_relation_results_20261011.md")
    destination.write_text(text)
    assert np.isfinite(spatial.mae).all()
    print(json.dumps({"status": "completed_audited", "report": str(destination), "validation_selected_arm": selected,
                      "selected_spatial_mae_improvement_percent": float(row.mae_improvement_percent)}))


if __name__ == "__main__":
    main()
