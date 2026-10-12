# Unified DOC model: cross-river diagnosis and optimization

Completed 12 October 2026. Transfer errors improve, but absolute skill and single-checkpoint stability remain insufficient for a universal release.

## Outcome

Source-only held-region validation selected **Balanced + reliability-gated river messages** before either receiving case was scored. The environmental MLP, 12-month GRU and three upstream/downstream/global attention heads remain; six learned cross-message gate scalars are added (12,571 total parameters). The receiving river is evaluated alone and its DOC never fits parameters, normalization, calibration, epochs or recipe choice.

Against the matched newly trained reference, selected native/log MAE improve by 26.93%/22.89% on independent Hackensack-Passaic. Mean R2 changes from -0.5669 to 0.0353; it is barely above zero. Delaware remains a development case with R2=-0.1548. This is an improvement, not resolution of universal generalization.

## Why the previous transfer degraded

Frozen-weight sensitivities isolate model reliance; they do not prove causal carbon mechanisms. Changing graph composition with the same receiving inputs and weights changes MAE substantially:

| Inference intervention on already inspected Delaware | MAE | log-MAE | R2 |
|---|---:|---:|---:|
| reference | 1.3505 | 0.4055 | -1.0426 |
| source_context | 0.9603 | 0.3341 | -0.2690 |
| coords_neutral | 1.1093 | 0.3639 | -0.6275 |
| no_cross | 1.4928 | 0.4336 | -1.4461 |
| no_global | 1.7664 | 0.4915 | -2.1556 |
| no_directional | 1.1931 | 0.3751 | -0.7240 |
| hydro_missing | 1.2281 | 0.3734 | -0.8051 |
| eco_neutral | 2.4367 | 0.6305 | -4.5026 |

Three supported risks: (1) source validation previously included source exogenous-node context, while new receiving inference did not; restoring that context reduces MAE from 1.3505 to 0.9603, but this context-dependent mode is not deployed; (2) all receiving longitudes exceed source ranges, forest median changes from approximately 12% to 97%, and approximately 86% of receiving precipitation exceeds source ranges; (3) DOC-query flow availability falls from approximately 79% to 34%, and full-grid flow availability from 51% to 8%. Removing all cross-node information worsens the frozen model, and removing ecological context is particularly harmful. The jointly learned representation requires retraining; inference ablations are not a deployment optimization.

Context, location and hydro availability explain model sensitivity and motivate these controlled training recipes; their interactions and underlying biogeochemical causes are not fully isolated. The controlled optimization reference and prior random-station release use different source cohorts/validation policies, so the historical MAE of 1.3505 versus the new 0.9426 cannot be attributed entirely to the gate.

## Experiment design

63 fresh fits: 7 recipes x 3 source HUC2 regional partitions x 3 seeds. Three rotated validation/test HUC2 pairs are 05/10, 10/07 and 07/05; the other four HUC2 units supply source training. These source units are regional groups within the source drainage, not three independent external rivers. All 357 source sites and observed labels remain in the corpus. No station-prefix-derived HUCs are used: mixed HUC8/HUC12 metadata and dropped leading zeros were correctly parsed from huc_cd. Role masks are disjoint at whole-unit level. Source validation/test inference uses the receiving unit alone for every recipe.

Matched reference uses full source graph and cell-weighted loss. Isolated graphs change only source graph composition; balanced episodes add inverse-source-domain-label-count weights, conserving total epoch loss weight. Coordinate-neutral variants zero only normalized latitude/longitude. Sensor augmentation drops whole histories with probabilities 0.1 for temperature and 0.5 for discharge during training, correctly updating masks/ages/padding. Gated variants mix attended and self values with six learned sigmoid strengths, donor hydro availability and a 0.01 mean-gate penalty. Combined applies all changes. Every recipe uses the same core seeded tensors, widths, 12-month windows, source-only statistics, AdamW, shuffled month panels, maximum 60/minimum 15 epochs, patience 10 and one summed gradient update per panel.

All 63 source selections and the source-validation-only choice were sealed, then all 126 complete label-free receiving fields were sealed, then independent confirmation DOC was opened. The new [Hackensack-Passaic HUC02030103 case](https://pubs.usgs.gov/of/1998/0282/report.pdf) has 100 stations, 90 station edges, 533 months and 3066 observed DOC queries. It overlaps neither source nor development stations, and source COMIDs are disjoint. All 100 stations pass source-compatible stream-order eligibility; no prediction-dependent filtering occurs. Initial basin-wide WQP temperature retrieval timed out; bounded station-specific requests completed the frozen cohort, with transport failures never treated as empty observations.

## Results

All values are arithmetic means of nine fit-level metrics; no prediction ensemble. All compared query coverages are 100%. Native MAE/RMSE use mg/L, log-MAE is log1p DOC. Bold is the best numerical value for the four error/skill metrics within each population, not a test-driven architecture choice.

### source

| Recipe | MAE | log-MAE | RMSE | R2 | Bias |
|---|---:|---:|---:|---:|---:|
| Matched original model | 2.4353 | 0.3708 | 5.2828 | -0.0646 | -1.4255 |
| Isolated basin graphs | 2.3617 | 0.3572 | **5.1963** | **-0.0332** | -1.4243 |
| Isolated + basin-balanced | 2.4126 | 0.3676 | 5.2706 | -0.0724 | -1.6060 |
| Balanced + neutral coordinates | 2.4530 | 0.3746 | 5.3125 | -0.0766 | -1.6064 |
| Balanced + hydro missingness augmentation | **2.3455** | **0.3543** | 5.1983 | -0.0396 | -1.4446 |
| Balanced + reliability-gated river messages (SOURCE SELECTED) | 2.3569 | 0.3582 | 5.2177 | -0.0596 | -1.5219 |
| All proposed changes combined | 2.3813 | 0.3605 | 5.2582 | -0.0948 | -1.4906 |

### development

| Recipe | MAE | log-MAE | RMSE | R2 | Bias |
|---|---:|---:|---:|---:|---:|
| Matched original model | 1.0023 | 0.3269 | 1.4515 | -0.2679 | -0.1165 |
| Isolated basin graphs | 0.9010 | 0.3008 | **1.3823** | **-0.1408** | -0.3702 |
| Isolated + basin-balanced | 1.0299 | 0.3379 | 1.4987 | -0.3630 | -0.2174 |
| Balanced + neutral coordinates | 0.9075 | 0.3014 | 1.3937 | -0.1635 | -0.3426 |
| Balanced + hydro missingness augmentation | 0.9247 | 0.3090 | 1.4042 | -0.1793 | -0.3510 |
| Balanced + reliability-gated river messages (SOURCE SELECTED) | 0.9426 | 0.3083 | 1.3867 | -0.1548 | -0.1241 |
| All proposed changes combined | **0.8906** | **0.2898** | 1.4108 | -0.1959 | -0.3554 |

### confirmation

| Recipe | MAE | log-MAE | RMSE | R2 | Bias |
|---|---:|---:|---:|---:|---:|
| Matched original model | 1.8096 | 0.3359 | 2.2015 | -0.5669 | +1.1570 |
| Isolated basin graphs | 1.5664 | 0.2990 | 1.9774 | -0.2324 | +0.7816 |
| Isolated + basin-balanced | 1.6471 | 0.3116 | 2.0508 | -0.3301 | +0.8532 |
| Balanced + neutral coordinates | 1.7468 | 0.3293 | 2.1394 | -0.4474 | +1.0322 |
| Balanced + hydro missingness augmentation | 1.4886 | 0.2893 | 1.8993 | -0.1190 | +0.6334 |
| Balanced + reliability-gated river messages (SOURCE SELECTED) | **1.3223** | **0.2590** | **1.7642** | **0.0353** | +0.2331 |
| All proposed changes combined | 1.5837 | 0.3038 | 1.9823 | -0.2216 | +0.8023 |

## Consistency, intervals and primary checkpoint

| Population | Native wins/9 | Log wins/9 | Joint wins/9 | Positive R2/9 |
|---|---:|---:|---:|---:|
| confirmation | 8 | 7 | 7 | 7 |
| development | 7 | 7 | 7 | 2 |
| source | 6 | 5 | 5 | 1 |

Delta means selected recipe error minus matched reference. Intervals use 3000 paired station-block draws within fixed cases; shared source fits, regional/hydrological correlation and multiple comparisons limit inference.

| Population | Endpoint | Delta | Exploratory 95% interval |
|---|---|---:|---|
| source | mae | -0.07836 | [-0.12935, -0.03116] |
| source | log1p_mae | -0.01257 | [-0.02319, -0.00219] |
| development | mae | -0.05970 | [-0.08497, -0.03769] |
| development | log1p_mae | -0.01862 | [-0.02346, -0.01389] |
| confirmation | mae | -0.48733 | [-0.59589, -0.38387] |
| confirmation | log1p_mae | -0.07688 | [-0.09868, -0.05831] |

Primary checkpoint `val05_test10_balanced_gated_s42` was fixed before test results. Its values are NOT the nine-fit averages:

| Population | MAE | log-MAE | RMSE | R2 |
|---|---:|---:|---:|---:|
| source | 3.7846 | 0.4583 | 10.0542 | -0.0393 |
| development | 1.3739 | 0.4188 | 1.6954 | -0.7119 |
| confirmation | 1.7120 | 0.3179 | 2.0444 | -0.2881 |

The prechosen single checkpoint has negative R2 on both receiving rivers. A slightly positive average confirmation R2 does not make this checkpoint reliable or constitute a prediction ensemble. Retain the gated balanced-episode architecture as the current research candidate; do not advertise a frozen universal model. Development-native MAE is numerically lowest for combined, but this does not replace the source-selected recipe. Removing coordinates does not consistently improve independent confirmation, and adding every change does not beat the selected gated arm.

## Verification and remaining gap

Independent audit replayed all 63 source/receiving checkpoints and 126 complete receiving grids, checked 252 fit/field/test receipts and 126 source label roles. Prediction replay error=0; maximum independently recomputed metric difference=1.78e-15. Source normalization, disjoint roles, model capacities, data/mask/code hashes and blind selection/field gates were verified. Full regression suite: 1638 passed, 3 skipped for unavailable local artifacts; Ruff passed.

Three hard source-unit test regions also have near-zero/negative mean R2, so this protocol reveals a broader out-of-region weakness. Two external cases do not establish global transfer. Sampling on observed DOC cells, where auxiliary hydro information is richer, does not establish accuracy on naturally unobserved cells; monthly station fields do not validate arbitrary continuous locations/times. The next scientific gap is environment-domain coverage (especially humid forested small catchments) and single-model stability, with a new external case needed for further confirmatory claims.

Inspectable evidence: diagnosis/protocol.json and its prediction packages, diagnosis/environment_shift.csv, diagnosis/hydro_shift.csv, protocol.json, manifest.json, source_domains.csv, masks/, selected_model.json, all_selections.json, all_fields.json, analysis/summary.csv, analysis/by_fit.csv, analysis/paired_intervals.json, analysis/panel_consistency.csv, analysis/optimization_variability.csv, analysis/independent_audit.json and research_decision.json.
