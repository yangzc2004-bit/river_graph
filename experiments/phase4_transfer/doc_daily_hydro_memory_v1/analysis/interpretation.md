# Interpretation: daily hydrology in the recurrent residual

## Main result

Giving the existing recurrent model twelve months of daily-flow summaries improves K0 prediction relative to giving the same projection only the current month. This is a small, reproducible matched-input result, concentrated in ordinary DOC error. It does not establish an improvement over the previous daily-head model across support budgets. K5 predictions are close, and the integrated historical-head model remains the established reference. No K-dependent route or target-selected mixture is introduced.

All nine packages passed replay. The 72 copied model/run checks and 36 off-control checks are exact; the complete off model summary/trace and full-grid predictions also reproduce the preceding daily model. All 24 models, K={0,1,3,5}, and 72 planned comparisons are reported. Intervals below are pointwise 95% paired station-bootstrap intervals (5,000 draws); seeds average within partition and partitions receive equal weight.

## Native MAE on fixed target queries

| Model, existing GRU support basis | K0 | K5 |
|---|---:|---:|
| context | 1.902823 | 1.596001 |
| off | 1.806996 | 1.589387 |
| current_only | 1.810409 | 1.587180 |
| full_history | 1.800854 | 1.586407 |
| tree_current | 1.855769 | 1.573251 |
| tree_history | 1.848927 | 1.577256 |
| off_integrated | 1.803538 | 1.564984 |
| current_only_integrated | 1.807859 | 1.567740 |
| full_history_integrated | 1.797393 | 1.567575 |
| prior_ecological_affine | 1.809217 | 1.579905 |

## What the matched memory comparison establishes

- At K0, full-history versus current-only reduces direct MAE by 0.528% (95% CI 0.071–0.948%). The native difference is -0.009555 [-0.017809, -0.001282] mg/L. The integrated difference is -0.010467 [-0.018935, -0.002204] mg/L, or 0.579%. Both improve 2/3 partitions and 7/9 training runs.
- The direct ordinary-DOC difference is −0.010472 [−0.019200, −0.001753] mg/L; its Q90 difference is −0.003711 [−0.033446, +0.028329]. The integrated ordinary difference is −0.012464 [−0.021233, −0.003619], while its Q90 difference is +0.003945 [−0.025660, +0.037929]. Thus the historical increment is not a demonstrated high-DOC improvement.
- Relative to the original off model, the full-history K0 difference is -0.006142 [-0.022309, +0.011614] direct and -0.006145 [-0.023423, +0.013350] integrated. Both intervals cross zero; the small gain against current-only does not by itself justify replacing off.
- At K5, full-history versus current-only is -0.000773 [-0.004668, +0.003263] direct and -0.000165 [-0.003415, +0.003425] integrated. This is insufficient evidence of a difference, not an equivalence test.
- Full-history versus off at K5 is -0.002980 [-0.013200, +0.008320] direct and +0.002591 [-0.002107, +0.007728] integrated. Direct Q90 MAE improves by −0.033604 [−0.066087, −0.000568], but this improvement does not persist after ecological/support integration.

## Matched tree controls

- Adding history to ExtraTrees changes K0 MAE by -0.006841 [-0.019709, +0.005357] and K5 MAE by +0.004005 [-0.012063, +0.021193]. Neither interval establishes an overall gain.
- At K0, full-history direct neural versus tree-history is -0.048073 [-0.103003, +0.004579]; the integrated neural model is -0.051535 [-0.103424, -0.003047] (2.787% gain, 95% CI 0.173–5.164%). The latter is a complete neural-plus-ecological pipeline comparison, not isolated proof that the new recurrent input caused the gain.
- The full integrated K0 model has Q90 MAE 7.1612 versus tree-history 7.5252: Δ−0.363996 [−0.574953, −0.144506]. Q90 recall is 64.00% versus 60.03%, a +3.973 percentage-point difference [1.842, 6.830]; false-Q90 rate also rises from 1.952% to 2.150%, +0.198 pp [0.039, 0.396]. Better tail detection therefore carries a small false-positive tradeoff. The prior off neural model already shows the same broad tail advantage over its matched current tree.
- At K5, tree-history scores 1.577256 versus full-history direct 1.586407 and integrated 1.567575. Neither corresponding paired interval establishes overall superiority. The current tree is numerically stronger than the history tree after support adaptation; neither is selected or discarded on that basis.
- Full-history integrated K5 remains better than the older ecological-affine reference by -0.012331 [-0.025337, -0.000878] (0.780%); however, the previously retained daily integrated model already improved this reference and has lower observed MAE in this study.

## Heterogeneity and source-validation context

| Comparison, GRU support basis | Partition 142 | Partition 143 | Partition 144 |
|---|---:|---:|---:|
| full_history_vs_current_only_gru_tuned_anchor_k0 | +0.002334 | -0.009322 | -0.021677 |
| full_history_vs_off_integrated_gru_tuned_anchor_k0 | +0.020489 | -0.024160 | -0.014765 |
| full_history_vs_off_integrated_gru_tuned_anchor_k5 | +0.013393 | -0.000629 | -0.004992 |

The full-versus-current direct K0 station contributions improve 91/172 unique stations and worsen 81; the five largest contributors supply 31.3% of positive gain. Integrated full-versus-off K5 improves 78 stations and worsens 94. Its average loss is driven by partition 142 (+0.013393), despite small gains in partitions 143 and 144. These patterns argue against a universal gain or a single-site explanation.

The source-validation K5 integrated scores were off 1.604602, current-only 1.604251, full-history 1.604384. Their near tie was known before target analysis. Direct K0 source-validation means improved from 1.804428 to 1.801147 to 1.798270, but the full-versus-current validation gain was concentrated in partition 143 and was negative in the other two partitions. All fits stopped by epoch 77, below the 120 cap. More training time is not the immediate bottleneck indicated by these traces.

## Fixed descriptive strata

The target panel contains 10,520 distinct station-months from 172 stations (12,865 partition-cell occurrences); training seeds do not increase ecological sample size. All are at stations with no visible DOC history. Observation-age strata beyond never-visible are therefore not identifiable by this design.

Current daily descriptors are fully valid at 8,141 distinct query cells and absent at 2,379; there are no partially valid cells. The causal daily-history groups contain 2,352 cells with zero valid months, 141 with 1–5, 281 with 6–11, and 7,746 with all 12. These descriptive populations are fixed before outcomes and are not new selection rules.

Full integrated versus off at K0 changes MAE from 1.205614 to 1.181576 in the no-current-daily group and from 1.957774 to 1.956208 in the fully valid group. Even the zero-history group improves descriptively (1.197176 to 1.172831). Because the models were globally refitted, improvement where no daily history is available cannot be attributed to a local historical signal. At K5, the full-history model is slightly worse in the large 12-month group (1.684858 versus 1.681264), consistent with the weak overall support-assisted increment. No availability-based route is fitted from these observations.

## Retention and next scientific question

Keep the existing daily integrated model as the main established candidate. Retain full-history as a completed mechanistic variant showing a modest K0 advantage over an equal-size current-only projection. Do not promote a new K0/K5 switching rule or universal neural-over-tree claim. The matched tree probes remain useful fixed comparators.

If continuing performance work, one focused hypothesis is to separate hydrologic persistence from the existing DOC-observation-age decay. The current GRU decay acts on the entire hidden state and receives increasing calendar age at never-observed DOC stations; the new hydrologic signal therefore has no independent freshness clock. A matched ablation can test that coupling while keeping the backbone, current head, source folds and tree controls fixed. This is a plausible next design question, not a proven cause of the present small gain.

Sources: findings.md, k_curves.csv, comparisons.csv, directions_by_partition.csv, gain_loss_concentration.csv, strata_profiles.csv, strata_populations.csv, training_choices.csv; source-only diagnostics/source_validation.md and formal verification/replay_checks.json. No source or execution changes were made during this analysis.
