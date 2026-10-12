# DOC information ablation results

Date: 2026-10-11. Completed matched-input internal Mississippi development study.

## Conditional contribution estimates

Each change below compares a retrained masked-input recipe to the same full model, with matched parameter tensors, seed initialization, optimizer, label weights, stopping rule and query population. Positive error changes mean the removal worsened prediction; negative changes mean the removal improved the point estimate. These percentages are changes in prediction error, not additive fractions of DOC variation or causal source contributions.

- **Remove all static environmental inputs**: spatial MAE 3.6618; change +1.4780 mg C/L (+67.68%).
- **Remove all dynamic/time/season inputs**: spatial MAE 2.2911; change +0.1072 mg C/L (+4.91%).
- **Remove cross-node exchange and explicit river descriptors**: spatial MAE 2.2137; change +0.0298 mg C/L (+1.36%).

## Spatial primary removals

Five whole-station HUC4 tasks; 7,262 observed queries at 104 stations; 100% prediction coverage. MAE/RMSE are mg C/L. Log1p MAE and native MAE remain joint primary endpoints; predictive R² is 1−SSE/SST. Metrics are pooled per seed then averaged, without ensembling.

| Method | mae | log1p_mae | rmse | r2 | mae_increase | mae_increase_percent |
| --- | --- | --- | --- | --- | --- | --- |
| Full environmental MLP + GRU + graph Transformer | **2.1839** | **0.3085** | **4.6991** | **0.4972** | 0.0000 | +0.00% |
| Remove all static environmental inputs | 3.6618 | 0.5640 | 6.6775 | -0.0189 | 1.4780 | +67.68% |
| Remove all dynamic/time/season inputs | 2.2911 | 0.3380 | 4.7802 | 0.4792 | 0.1072 | +4.91% |
| Remove cross-node exchange and explicit river descriptors | 2.2137 | 0.3176 | 4.7402 | 0.4883 | 0.0298 | +1.36% |

## Spatial explanatory controls

| Method | mae | log1p_mae | rmse | r2 | mae_increase_percent |
| --- | --- | --- | --- | --- | --- |
| Full environmental MLP + GRU + graph Transformer | **2.1839** | **0.3085** | 4.6991 | 0.4972 | +0.00% |
| Self-only attention; static river descriptors retained | 2.2318 | 0.3244 | 4.7255 | 0.4914 | +2.20% |
| Current-month information only; same GRU operator | 2.2252 | 0.3260 | **4.6209** | **0.5138** | +1.89% |
| Generic dense attention; river relation categories removed | 2.1847 | 0.3086 | 4.7053 | 0.4959 | +0.04% |

Historical information removal changes spatial MAE by **+1.89%**; it uses the same twelve-step GRU with eleven zeroed past steps and zeroed current observation-age channels. This estimates the contribution of past information conditional on current values/masks/season; it does not compare GRU against alternative architectures.

Removing river relation categories while retaining dense context and every static descriptor changes spatial MAE by **+0.04%**. Removing cross-node messages while retaining every static descriptor changes it by **+2.20%**. These controls distinguish explicit relation coding from covariate exchange. Strict no-river additionally removes the four explicit static river descriptors, so its contrast combines those removals and cannot be labelled a topology-only effect.

The environmental package includes all thirteen river/ecological descriptors and two coordinates. Its removal is a static-package effect; ecological variables, location and hydraulic regime are not isolated individually. Direct river descriptors overlap this package, whereas climate/land-use descriptors and generic coordinates remain in strict no-river. The three package effects must not be added to 100%.

## Temporal primary removals

266 eligible stations; three masking seeds; the same three-month scored cores within each seed across 3/6/12-month excluded windows. Full-model checkpoints and masks are exactly those of the traditional interpolation comparison. Retrospective reconstruction uses exogenous covariates, with no DOC observation context as a model input. Largest-gap bracket eligibility restricts the evaluated population.

| Method | MAE:3 months | MAE:6 months | MAE:12 months |
| --- | --- | --- | --- |
| Full environmental MLP + GRU + graph Transformer | **2.2807** | **2.3083** | **2.3329** |
| Remove all static environmental inputs | 2.9687 | 3.0378 | 2.8961 |
| Remove all dynamic/time/season inputs | 2.5100 | 2.5130 | 2.5201 |
| Remove cross-node exchange and explicit river descriptors | 2.4059 | 2.4286 | 2.4464 |

Lowest twelve-month native MAE point estimate: **Full environmental MLP + GRU + graph Transformer**, 2.3329. Full temporal MAE/log1p MAE/RMSE/R², bias and error changes are retained in the linked machine-readable summary. No independent population significance is inferred from three mask seeds or the repeated query cores.

The no-time model has no time-varying inputs and therefore gives a constant concentration through time at each station in evaluation mode. Even a competitive concentration MAE would not establish reproduction of temporal fluctuations. Current-only, in contrast, retains current hydrological drivers and season, so its output may vary through time without historical input.

## Paired spatial uncertainty

| Method | Removal MAE increase | Exploratory 95% interval | regions_with_lower_error |
| --- | --- | --- | --- |
| Remove all static environmental inputs | 1.4780 | [0.3765, 2.8957] | 5 |
| Remove all dynamic/time/season inputs | 0.1072 | [0.0042, 0.2784] | 4 |
| Remove cross-node exchange and explicit river descriptors | 0.0298 | [-0.0598, 0.1555] | 3 |
| Self-only attention; static river descriptors retained | 0.0480 | [-0.0349, 0.1317] | 4 |
| Current-month information only; same GRU operator | 0.0414 | [-0.0789, 0.1994] | 3 |
| Generic dense attention; river relation categories removed | 0.0009 | [-0.0025, 0.0043] | 3 |

The last column counts regions with lower full-model MAE. Intervals resample five complete HUC4 clusters after averaging seed errors; they are exploratory, unadjusted and based on previously inspected regions. An interval crossing zero does not establish a consistent population advantage. Log1p endpoint intervals remain in the paired-comparison JSON and should be assessed alongside native MAE before making a joint-endpoint claim.

## Findings for the model rationale

On these matched tasks, the full model has the lowest native and log1p MAE among the four primary recipes, in both spatial reconstruction and every temporal gap length. The static package produces the largest conditional error increase: **67.68%** spatially and **24.14–31.60%** temporally. Removing all dynamic and seasonal information increases spatial MAE by **4.91%** and temporal MAE by **8.03–10.06%**. For these two spatial removals, both endpoint cluster intervals exclude zero in favour of the full model; this is supportive internal evidence, with only five previously inspected clusters and no multiplicity adjustment.

The strict river removal increases spatial MAE by **1.36%** and temporal MAE by **4.87–5.49%**, but its spatial intervals include zero for both primary endpoints. The near-zero **0.04%** change after removing relation categories does not support a strong claim for the current explicit upstream/downstream encoding. Dense attention retains virtually the full-model point estimate without those categories. The no-message control has a **2.20%** spatial MAE increase, with an interval including zero on native MAE but excluding zero on log1p MAE. This does not establish a joint-endpoint population advantage.

The current-only recipe increases native MAE by **1.89%**, with both spatial intervals including zero. Its RMSE is **4.6209** and R² **0.5138**, compared with **4.6991** and **0.4972** for the full model. Thus current drivers/season and historical memory must be discussed separately: retaining history improves the average absolute-error point estimate, while the current-only recipe has better squared-error metrics. This ablation supports the three-input framework on the predefined primary metrics but leaves the unique benefit of explicit river relation coding and the optimal temporal operator unresolved.

## Verification and scope

All 141 validation selections were bound before any new test scoring: **102 new fits** and **39 reused checkpoints**. Each new removal was retrained rather than applied only after full-model fitting. Full spatial and self-only baseline checkpoints are the original matched architecture-study models; full temporal models are the original interpolation models. Four-thread spatial and one-thread temporal arithmetic, dataset and masks were retained. Every arm has **12565 parameters**, with identical parameter shapes and seeded initialization; masking retains constant/bias paths rather than changing width or removing layers.

All validation/test checkpoint outputs replayed with maximum discrepancy **0**. Independent original-DOC query truth, source-only normalization, selected epochs, masks/hashes, coverage, paired intervals and **19** summary rows were checked. Maximum independent metric difference: **4.44e-16**. The full spatial model matches the prior comparator baseline. Original paper freezes/endpoints were not modified.

The supported interpretation is conditional predictive information value on these fixed tasks. Independent-river transfer, target-only inference graphs, continuous unobserved-field accuracy and arbitrary virtual sites remain separate evaluations. Adverse or uncertain removal results are retained and cannot justify changing the predefined endpoint after seeing the test scores.

Evidence: [design](doc_information_ablation_plan_20261011.md), [protocol](../experiments/phase4_transfer/doc_information_ablation_v1/protocol.json), [all metrics](../experiments/phase4_transfer/doc_information_ablation_v1/analysis/summary.csv), [seed metrics](../experiments/phase4_transfer/doc_information_ablation_v1/analysis/by_seed.csv), [regional metrics](../experiments/phase4_transfer/doc_information_ablation_v1/analysis/by_region_seed.csv), [paired intervals](../experiments/phase4_transfer/doc_information_ablation_v1/analysis/paired_comparisons.json), [independent audit](../experiments/phase4_transfer/doc_information_ablation_v1/analysis/independent_audit.json).
