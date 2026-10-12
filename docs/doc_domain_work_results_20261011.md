# River DOC model comparison results

The fixed environmental encoder, hydrological GRU and graph Transformer has the
lowest pooled native MAE and log1p MAE among the selected DOC literature model
families on these five development tasks. The strongest new full-history
competitor is XGBoost, with a 3.05% native MAE difference in favor of the fixed
fusion model. GAM has lower RMSE and higher predictive R². These results support
retaining the current model as the MAE reference while keeping GAM and XGBoost
as serious competitors; they do not establish general superiority or SOTA.

## DOC studies and reproduction status

Dedicated river DOC models exist. [Aqua-OC](https://authors.library.caltech.edu/records/b74cg-58e81)
retrieves river-reach DOC using Landsat observations, optical water types and
environmental inputs. [The Yangtze monthly DOC study](https://xblk.ecnu.edu.cn/EN/10.3969/j.issn.1000-5641.2026.03.002)
selects an environmental random forest for long-term reconstruction.
[Casas-Ruiz et al.](https://agupubs.onlinelibrary.wiley.com/doi/10.1029/2019GB006495)
model DOC spatial and hydrological patterns with GAMs.
[ELM-MOSART-DOC](https://agupubs.onlinelibrary.wiley.com/doi/10.1029/2025MS005307)
provides a current process and ML model of river transport and transformation,
including reservoir effects and predictions without DOC calibration.

The new numerical arms retain RF, GAM, XGBoost and SVR method families under the
project's available inputs. They are **literature motivated adaptations, not
complete reproductions of these papers**. Aqua-OC's Landsat matching, spectral
indices, ERA5-Land inputs and optical classification are absent. The Yangtze
dynamic vegetation, soil, precipitation and population covariates are absent.
The GAM changes mgcv splines and river-mouth flow to pyGAM ecological smooths
and local flow. INCA-C and ELM-MOSART-DOC are discussed but not numerically run:
their daily process forcing, soil or hydrological state and initialization are
absent from the current feature artifact. Numerical superiority over those
complete frameworks has not been tested.

The input sets and fixed recipes are documented in
[the experiment plan](doc_domain_work_plan_20261011.md), with primary-paper links
and concrete missing inputs. The existing full-history random forest remains
a control, so a weak contemporaneous RF is not the only tree comparator.

## Common dataset results

All methods predict the identical 7,262 hidden observed station-months at 104
stations in five HUC4 tasks. Source, validation and target stations are disjoint.
No DOC history enters any input. All regional choices were frozen using source
validation before scoring any new target prediction. There are 120 new candidate
fits; the reference fusion and historical RF predictions are reused from the
audited prior benchmark. Metrics are computed separately per seed, then averaged.

Concentration MAE, RMSE and bias are in mg C/L. Log MAE is absolute error after
the log1p transform; R² is predictive 1 minus SSE/SST, not squared correlation.

| Method | Local input columns | MAE | log1p MAE | RMSE | R² | Bias | Station macro MAE | Basin macro MAE | Source Q90 MAE |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Fixed fusion | 113 plus river relations and permitted other-node covariates | **2.1839** | **0.3085** | 4.6991 | 0.4972 | -0.7372 | **2.8404** | **2.3022** | 7.4505 |
| XGBoost | 113 | 2.2526 | 0.3245 | 4.6955 | 0.4980 | -0.4923 | 2.9800 | 2.4375 | 7.5879 |
| Hydrological ecological GAM | 23 | 2.4512 | 0.4096 | **4.5142** | **0.5361** | -0.1161 | 2.9676 | 2.5406 | **7.2137** |
| RF with history control | 113 | 2.4637 | 0.3560 | 4.8091 | 0.4735 | -0.1709 | 3.2048 | 2.6097 | 7.9034 |
| RBF SVR | 113 | 2.5375 | 0.4044 | 4.8046 | 0.4745 | -0.3193 | 2.9481 | 2.5595 | 7.3845 |
| Ecological RF | 23 | 2.9912 | 0.3671 | 7.8566 | -0.4067 | 0.5961 | 3.9006 | 3.2460 | 10.4181 |

Source Q90 denotes fold-specific high-concentration observations using training
DOC thresholds, not a verified within-station storm-peak metric. The same 1,039
tail query cells are used for every method. The pooled metrics include substantial
between-basin concentration differences; they do not establish within-river
temporal skill by themselves.

## Basin variation and uncertainty

The fixed fusion model beats XGBoost on both primary endpoints in three of five
basins, and loses in 0708 and 1019. It beats both RF controls in all five basins.
GAM and SVR also outperform fusion on MAE in some basins.

| Target HUC4 | Fixed fusion MAE | XGBoost MAE | GAM MAE | SVR MAE | RF with history MAE | Ecological RF MAE |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 0708 | 1.6877 | **1.4779** | 1.6372 | 1.5218 | 1.9016 | 1.9010 |
| 1013 | 5.9022 | 6.1489 | 5.8861 | **5.7741** | 6.5821 | 9.7732 |
| 1019 | 1.5111 | **1.3677** | 2.0600 | 2.6793 | 1.6204 | 1.6181 |
| 1030 | **1.2598** | 1.6999 | 1.3292 | 1.4287 | 1.4160 | 1.4040 |
| 1101 | **1.1502** | 1.4931 | 1.7904 | 1.3935 | 1.5284 | 1.5336 |

For XGBoost, fusion-minus-comparator MAE is -0.0687 mg C/L. The exploratory
five-basin bootstrap 95% interval is [-0.3541, 0.1153], crossing zero. Its log1p
MAE interval also crosses zero. Native MAE intervals against GAM and SVR cross
zero as well. These unadjusted intervals use five basin clusters after averaging
seed errors and cannot support a simultaneous superiority claim.

The ecological RF's large error is concentrated in basin 1013, where source
validation selected native-target leaf-3 RF. The historical RF selected log1p
leaf-3 for that basin. Consequently, their difference cannot be attributed
solely to the time window: permitted features and source-selected target
transforms both differ. The poor target result is retained without retuning.

## Model and paper implications

Native MAE and log1p MAE were fixed as joint primary endpoints before these
results. Under those endpoints the current three-part fusion remains the best
pooled point estimate in this comparison. GAM is stronger for squared error,
predictive variance explanation and the declared high-DOC tail diagnostic, so
the model is not optimal across objectives. Changing the objective after seeing
these results would require a new justified protocol, not a relabeling of the
primary endpoint.

The literature already contains long-term reach mapping, environmental DOC
prediction, river-network process models and predictions without local DOC
calibration. The contribution must therefore specify the readily obtainable
inputs, temporal memory and river relations, and demonstrate transfer benefits
under a defined information regime. Claiming the first river DOC reconstruction
model, or treating a lower MAE against these adaptations as a win over complete
Aqua-OC or ELM-MOSART-DOC, would exceed the evidence.

These five HUC4 tasks were inspected during development and belong to the same
Mississippi river system. Confirmatory external river evaluation is still
necessary. Predictions are saved for observed query cells; naturally missing
cells and arbitrary virtual station locations remain unverified.

## Verified artifacts

The independent audit recomputed native MAE, log1p MAE, RMSE, R², bias, station
macro and source-Q90 MAE, and basin macro errors from original truth and saved
predictions. Maximum discrepancy was 1.78e-15. It checked masks, source-only roles,
selection and fit receipts, frozen code, optional dependencies, identical query
coverage and deterministic repeats. The OpenMP runtime and paper source hashes
were additionally checked by the dependency preparation verifier.

- [Summary](../experiments/phase4_transfer/doc_domain_work_v1/analysis/summary.csv)
- [Results by basin and seed](../experiments/phase4_transfer/doc_domain_work_v1/analysis/by_region_seed.csv)
- [Paired basin intervals](../experiments/phase4_transfer/doc_domain_work_v1/analysis/paired_comparisons.json)
- [Independent audit](../experiments/phase4_transfer/doc_domain_work_v1/analysis/independent_audit.json)
- [Protocol](../experiments/phase4_transfer/doc_domain_work_v1/protocol.json)

Protocol canonical hash:
`fbc29a194190d2b0963ea5f44ac23e988c32899a43def8b9a0e2f3a99db0fa50`.
Full suite: 1,553 passed, three skipped, eight existing warnings. Post-freeze
DOC and historical K-shot contracts: 14 passed. Ruff passed. The repository's
original run_ladder, existing code and historical experiment artifacts were
not edited.
