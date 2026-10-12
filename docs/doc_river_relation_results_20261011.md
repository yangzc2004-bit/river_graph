# Directed river relation comparison

Completed prospective matched development experiment, frozen as doc_river_relation_v2.

## Main result and validation-selected recipe

The recipe selected **before test scoring** is **Upstream/downstream/global heads**, with validation joint ratio **0.990591** (original baseline = 1). Its spatial native MAE is **2.1845 mg C/L**, a **-0.03%** improvement relative to the original graph Transformer; log1p MAE improvement is **-0.87%**. Its temporal native improvements range from **+1.98%** to **+3.96%**.

The lowest spatial test MAE point estimate is **Directed log-hop attention bias**, 2.1830. This descriptive ranking does not change the frozen validation-selected recommendation. If the original baseline wins validation, it remains the selected recipe even if a candidate has a favourable test row. Generic dense attention is a diagnostic control outside the selection set.

Both temporal primary errors are lower than the baseline in **9/9 mask-seed tasks** for the validation-selected recipe. These repeated tasks measure robustness to the existing masks and initialization; they are not nine independent rivers. Its spatial RMSE improvement is **+1.18%**. The exploratory spatial primary intervals do not establish a joint improvement over the baseline. Temporal gains and a secondary spatial-metric gain therefore must be distinguished from a demonstrated joint spatial advantage. For the selected directional-head recipe, retain the environmental MLP and GRU and treat the upstream/downstream/global head allocation as the supported next candidate, with the strongest current evidence in temporal reconstruction.

For the twelve-month gap, generic dense attention already achieves native MAE **2.2876**, versus **2.2866** for the selected recipe: only **+0.04%** lower error. Thus most of the twelve-month improvement over the original relation-category baseline is also observed in the generic control. The clearer numerical separation for directional heads is at three- and six-month gaps.

## Spatial reconstruction

Same five HUC4 tasks, seeds 42/43/44, 104 target stations and 7,262 observed queries per seed, with 100% coverage. Metrics pool query cells within each seed and then average seeds; these are not ensemble predictions. MAE/RMSE are mg C/L; predictive R² is 1−SSE/SST. Positive improvement means lower error than the original baseline, whose denominator is retained for every comparison.

| Method | mae | log1p_mae | rmse | r2 | mae_improvement_percent | log1p_mae_improvement_percent |
| --- | --- | --- | --- | --- | --- | --- |
| Original graph Transformer | 2.1839 | 0.3085 | 4.6991 | 0.4972 | +0.00% | +0.00% |
| Generic dense attention (control) | 2.1847 | 0.3086 | 4.7053 | 0.4959 | -0.04% | -0.02% |
| Full-path relation categories | 2.1850 | 0.3084 | 4.7040 | 0.4962 | -0.05% | +0.03% |
| Upstream-only attention | 2.2684 | 0.3260 | 4.8446 | 0.4654 | -3.87% | -5.68% |
| Upstream/downstream/global heads | 2.1845 | 0.3112 | **4.6436** | **0.5090** | -0.03% | -0.87% |
| Directed log-hop attention bias | **2.1830** | **0.3079** | 4.7023 | 0.4965 | +0.04% | +0.20% |

## Temporal reconstruction

Same nine 3/6/12-month masks at 266 eligible stations. Scored three-month cores are the same within masking seed across gap lengths. Training supervision before and after missing windows is permitted for retrospective reconstruction; no DOC value/history is a model input.

| Method | MAE:3 months | MAE:6 months | MAE:12 months |
| --- | --- | --- | --- |
| Original graph Transformer | 2.2807 | 2.3083 | 2.3329 |
| Generic dense attention (control) | 2.2813 | 2.3101 | 2.2876 |
| Full-path relation categories | 2.2807 | 2.3083 | 2.3329 |
| Upstream-only attention | 2.3708 | 2.2876 | 2.3180 |
| Upstream/downstream/global heads | **2.1942** | **2.2168** | **2.2866** |
| Directed log-hop attention bias | 2.2792 | 2.3048 | 2.3502 |

Twelve-month secondary metrics:

| Method | mae | log1p_mae | rmse | r2 | mae_improvement_percent |
| --- | --- | --- | --- | --- | --- |
| Original graph Transformer | 2.3329 | 0.2600 | 6.1618 | 0.4229 | +0.00% |
| Generic dense attention (control) | 2.2876 | 0.2530 | 6.0975 | 0.4340 | +1.94% |
| Full-path relation categories | 2.3329 | 0.2600 | 6.1618 | 0.4229 | +0.00% |
| Upstream-only attention | 2.3180 | 0.2559 | **6.0170** | **0.4496** | +0.64% |
| Upstream/downstream/global heads | **2.2866** | **0.2497** | 6.0433 | 0.4445 | +1.98% |
| Directed log-hop attention bias | 2.3502 | 0.2624 | 6.1908 | 0.4169 | -0.74% |

## Candidate effects

- **Full-path relation categories**: spatial native MAE improvement -0.05%, log1p MAE improvement +0.03%; temporal native improvements +0.00% to +0.00%.
- **Upstream-only attention**: spatial native MAE improvement -3.87%, log1p MAE improvement -5.68%; temporal native improvements -3.95% to +0.90%.
- **Upstream/downstream/global heads**: spatial native MAE improvement -0.03%, log1p MAE improvement -0.87%; temporal native improvements +1.98% to +3.96%.
- **Directed log-hop attention bias**: spatial native MAE improvement +0.04%, log1p MAE improvement +0.20%; temporal native improvements -0.74% to +0.15%.

These are recipe-level predictive differences conditional on the same environmental/temporal inputs. Comparing each directional candidate against path closure helps separate omitted-intermediate connectivity from attention-mask or distance-bias changes. Relative effects cannot be added into independent causal contributions.

All temporal relation matrices are unchanged by closure. There are **2320** restored upstream pairs across the spatial mask/role combinations (these are repeated graph diagnostics, not distinct population station pairs). Closure cannot add structural information when the permitted roles already contain the complete relevant paths. The diagnostic CSV records every mask and role.

## Paired spatial uncertainty

Native and log1p MAE are joint primary endpoints. Improvement intervals below are original-baseline error minus candidate error. Intervals resample five complete HUC4 clusters after seed-error averaging, using the same 5,000 draws as the previous comparisons. They are exploratory, unadjusted and derived from previously inspected development regions. An interval crossing zero does not establish a consistent population advantage; improvement on one primary endpoint alone does not establish joint superiority. Temporal masking seeds and repeated cores do not provide independent-river significance.

| Method | endpoint | difference_fusion_minus_comparator | Improvement interval | regions_with_lower_error |
| --- | --- | --- | --- | --- |
| Generic dense attention (control) | mae | -0.0009 | [-0.0043, 0.0025] | 3 |
| Generic dense attention (control) | log1p_mae | -0.0001 | [-0.0007, 0.0008] | 3 |
| Full-path relation categories | mae | -0.0011 | [-0.0090, 0.0040] | 2 |
| Full-path relation categories | log1p_mae | 0.0001 | [-0.0010, 0.0013] | 3 |
| Upstream-only attention | mae | -0.0845 | [-0.1995, 0.0089] | 4 |
| Upstream-only attention | log1p_mae | -0.0175 | [-0.0452, 0.0030] | 4 |
| Upstream/downstream/global heads | mae | -0.0006 | [-0.0712, 0.0445] | 2 |
| Upstream/downstream/global heads | log1p_mae | -0.0027 | [-0.0190, 0.0079] | 3 |
| Directed log-hop attention bias | mae | 0.0009 | [-0.0052, 0.0064] | 1 |
| Directed log-hop attention bias | log1p_mae | 0.0006 | [-0.0006, 0.0021] | 1 |

The final column counts regions with lower **original-baseline** error, so fewer is favourable for the new candidate.

## Selection, capacity and verification

| Method | Validation joint ratio | Eligible | Selected |
| --- | --- | --- | --- |
| Original graph Transformer | 1.0000 | True | False |
| Generic dense attention (control) | 0.9968 | False | False |
| Full-path relation categories | 0.9995 | True | False |
| Upstream-only attention | 1.0564 | True | False |
| Upstream/downstream/global heads | 0.9906 | True | True |
| Directed log-hop attention bias | 1.0015 | True | False |

Validation selection uses relative native/log1p MAE with equal endpoint weights, equal task weights within each track, and equal spatial/temporal weights. All 144 validation selections and this choice were bound before any new target score: **105 new fits and 39 reused audited checkpoints**. Every recipe has **12565 parameters**, with matching parameter shapes and exactly identical tensors at seeded initialization. The environmental MLP, twelve-step GRU, season channel, two spatial blocks, optimizer, label weights, monthly permutation and stopping rule remain identical. The optimizer loop is copied exactly except for its model factory. Four-thread spatial and one-thread temporal arithmetic are retained.

Topology closure traverses excluded intermediate nodes only to identify known paths. Their covariates/hidden states/DOC never enter message passing. Direction heads specialize three existing heads; upstream-only masks all downstream/unconnected keys. Directed path bias reuses the existing six-by-three tensor with six basis coefficients. Distance is the shortest number of existing **station edges**, not physical river kilometres, travel time or a mass-balance law. No new external covariates were introduced.

Every validation/test checkpoint prediction was replayed, with maximum discrepancy **0**. Independent metric discrepancy across **24** summary rows was **8.88e-16**. Source-only scaling, query truth/masks, selected epochs, parameter counts, validation-only global choice and paired intervals were checked. SciPy independently verified complete-topology distances and directed relation matrices. Original baseline and spatial generic control metrics matched their earlier audited results. Original paper endpoints/freezes are unchanged.

The initial v1 metadata omission was detected by a frozen contract before any target scoring. Its seven validation receipts, zero test receipts and original sources were preserved under the aborted-metadata directory. Candidate model computation was retained, count metadata was corrected in a new v2 freeze, and no v1 candidate checkpoint was reused.

## Interpretation limits and evidence

These locally designed DOC candidates draw motivation from shortest-path structural bias in [Graphormer](https://arxiv.org/abs/2106.05234) and upstream hydrological priors in [HydroNets](https://arxiv.org/abs/2007.00595); neither complete published model is reproduced here. Results concern observed-query reconstruction within the existing Mississippi development tasks. Independent rivers, naturally missing cells, arbitrary virtual stations and physical transport consistency require separate evaluation. No endpoint, architecture or selection rule was changed after new test scoring.

Evidence: [design](doc_river_relation_plan_20261011.md), [protocol](../experiments/phase4_transfer/doc_river_relation_v2/protocol.json), [metrics](../experiments/phase4_transfer/doc_river_relation_v2/analysis/summary.csv), [seed metrics](../experiments/phase4_transfer/doc_river_relation_v2/analysis/by_seed.csv), [regional metrics](../experiments/phase4_transfer/doc_river_relation_v2/analysis/by_region_seed.csv), [intervals](../experiments/phase4_transfer/doc_river_relation_v2/analysis/paired_comparisons.json), [validation choice](../experiments/phase4_transfer/doc_river_relation_v2/selected_recipe.json), [topology diagnostics](../experiments/phase4_transfer/doc_river_relation_v2/analysis/topology_diagnostics.csv), [independent audit](../experiments/phase4_transfer/doc_river_relation_v2/analysis/independent_audit.json).
