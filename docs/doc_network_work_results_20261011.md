# Common-query DOC reconstruction comparison

Date: 2026-10-11. Status: completed and independently verified internal development comparison.

## Outcome

The lowest pooled native-MAE point estimate belongs to **Environmental MLP + GRU + graph Transformer**. The full table and paired differences below retain all comparators and adverse outcomes. Native MAE and log1p MAE are the previously fixed joint primary endpoints; better RMSE/R² cannot replace them. This comparison does not establish universal superiority or global SOTA.

All seven methods predict the same **7,262 observed station-months** at **104 stations** across five held-out HUC4 regions of the Mississippi system. DOC concentration errors are in mg C/L; R² is predictive 1−SSE/SST. Metrics are computed per seed and then averaged, not from an ensemble. Every method has 100% coverage of the declared observed queries. This is not coverage or verified accuracy over all unobserved river cells.

## Common-query results

| Method | mae | log1p_mae | rmse | r2 | bias | basin_macro_mae | coverage |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Environmental MLP + GRU + graph Transformer | 2.1839 | 0.3085 | 4.6991 | 0.4972 | -0.7372 | 2.3022 | 1.0000 |
| XGBoost with 12-month history | 2.2526 | 0.3245 | 4.6955 | 0.4980 | -0.4923 | 2.4375 | 1.0000 |
| SSN covariance adaptation | 2.3265 | 0.3578 | 4.6762 | 0.5022 | -0.3895 | 2.4714 | 1.0000 |
| Hydrological/environmental GAM adaptation | 2.4512 | 0.4096 | 4.5142 | 0.5361 | -0.1161 | 2.5406 | 1.0000 |
| RF with 12-month history | 2.4637 | 0.3560 | 4.8091 | 0.4735 | -0.1709 | 2.6097 | 1.0000 |
| Environmental/current-driver RF adaptation | 2.9912 | 0.3671 | 7.8566 | -0.4067 | 0.5961 | 3.2460 | 1.0000 |
| Official Transformer-GCN architecture adaptation | 3.1652 | 0.4478 | 6.3018 | 0.0851 | 0.0298 | 3.3613 | 1.0000 |

The contemporary RF and GAM use environment15 + current driver/mask/age/season8. Historical RF and XGBoost use the 113-column environmental/history feature table. Fusion and the official architecture adaptation receive the same underlying environment and 12-month driver information, with permitted source/held-out covariate graphs at inference. SSN additionally uses declared physical NHD mainstem distances and confluence geometry. The models have different information organization and family-specific budgets: the table compares complete common-data recipes, not isolated architecture effects.

The 15 environmental columns comprise stream order, log drainage area, slope, a station-graph headwater indicator (no observed upstream neighbour), forest/agricultural/urban/wetland fractions, climatic precipitation and temperature normals, soil organic matter, elevation, baseflow index and two coordinates. The builder uses 2019 land cover and 1991–2020 climate normals as shared static descriptors; these are not a reconstruction of historical land-cover change. The headwater/stream-order descriptors also carry structural information, so a no-message control does not remove every spatial descriptor. The eight monthly channels are temperature, log discharge, their availability masks and ages, and seasonal sin/cos. The environmental builder contains no DOC-derived summary features.

## Paired regional uncertainty

Negative fusion-minus-comparator differences favour fusion. Intervals resample five whole basin clusters after averaging seed errors. They are exploratory, unadjusted for multiple comparisons, and based on previously inspected development regions. Deterministic SSN predictions repeat across nominal seed rows; these are not independent SSN replicates.

| Comparator | Fusion MAE gain, % | Paired MAE difference | Exploratory 95% interval | Regions with lower fusion MAE |
| --- | --- | --- | --- | --- |
| Environmental/current-driver RF adaptation | 26.9895 | -0.8073 | [-2.2771, -0.1385] | 5 |
| Hydrological/environmental GAM adaptation | 10.9063 | -0.2673 | [-0.5161, 0.0204] | 3 |
| XGBoost with 12-month history | 3.0517 | -0.0687 | [-0.3541, 0.1153] | 3 |
| RF with 12-month history | 11.3574 | -0.2798 | [-0.4734, -0.1428] | 5 |
| SSN covariance adaptation | 6.1284 | -0.1426 | [-0.3131, 0.0376] | 4 |
| Official Transformer-GCN architecture adaptation | 31.0029 | -0.9813 | [-2.2232, -0.3319] | 5 |

## Regional MAE

| target_huc4 | doc_ecological_rf | doc_hydro_gam | doc_xgboost | fixed_fusion | physt_architecture_adapt | rf_history_control | ssn_covariance_adapt |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 0708 | 1.9010 | 1.6372 | 1.4779 | 1.6877 | 2.0732 | 1.9016 | 1.5255 |
| 1013 | 9.7732 | 5.8861 | 6.1489 | 5.9022 | 9.5657 | 6.5821 | 6.3040 |
| 1019 | 1.6181 | 2.0600 | 1.3677 | 1.5111 | 2.1276 | 1.6204 | 1.6422 |
| 1030 | 1.4040 | 1.3292 | 1.6999 | 1.2598 | 1.4146 | 1.4160 | 1.4650 |
| 1101 | 1.5336 | 1.7904 | 1.4931 | 1.1502 | 1.6257 | 1.5284 | 1.4203 |

## New model fidelity and source selection

SSN uses the public SSN2 exponential tail-up, tail-down and Euclidean covariance formulas, a linear environmental/current-driver mean and a static network Gaussian station effect with independent measurement noise. Exact grouped REML uses every source training observation; its algebra is tested against dense GLS. It is a Python covariance-family adaptation rather than execution of the original Wampler R/ArcMap workflow. Covariance uses NHDPlus mainstem routing; minor diversions are excluded. **2 station positions** use the declared midpoint imputation and **139 junctions** use equal branch shares where drainage area is missing/nonpositive. All covariance families were checked for positive semidefiniteness at representative ranges.

PhyST-HC uses the author's original `PositionalEncoding` and `ImprovedTransformerGCN` definitions unchanged, with two width recipes. The wrapper repeats environment15 alongside each eight-channel monthly driver step and adds one zero pad for two attention heads. Monthly graph panels are disjoint. Inputs, concentration target, optimization objective and compact widths replace the native SWAT-C process-state/event-scale flux workflow. It is an **official architecture adaptation, not a complete PhyST-HC reproduction**.

Public source commits: PhyST-HC `d9805e7d536dddaa3aad6c6740a6045322d4aa60`; SSN2 `d454f61dc877cafc570434ee819ebe4285406766`. [SSN paper](https://bg.copernicus.org/articles/21/3093/2024/), [SSN2 implementation](https://github.com/USEPA/SSN2), [PhyST-HC paper](https://www.sciencedirect.com/science/article/pii/S0022169426013454), [author code](https://github.com/VenkateshBudamala/PhyST-HC).

The 15 SSN fits and 30 neural fits were all completed. SSN optimizer success flags: **15/15**; limits and optimization messages remain in individual fit records. The neural epoch ceiling is 60 with source-validation early stopping. None of these budgeted fits establishes optimized architecture-family rankings.

All five regions' selections were frozen before any new query scoring; previous RF/GAM/XGBoost/fusion predictions were reused as hashed dependencies. Selection is by source-validation seed-mean native MAE with the predefined tie-break rule; target DOC never enters inputs or selection.

| Region | ssn_covariance_adapt | physt_architecture_adapt |
| --- | --- | --- |
| 1013 | log_network_euclidean | wider |
| 1019 | log_network_euclidean | wider |
| 0708 | log_network_euclidean | wider |
| 1030 | log_network_euclidean | compact |
| 1101 | log_network_euclidean | wider |

## Verification and interpretation

Independent audit replayed all **45** candidate validation checkpoints and selected target predictions. Maximum validation replay discrepancy: **0**. Important metrics were independently recomputed against the original DOC grid, with maximum discrepancy **8.33e-17**. Masks, whole-station role separation, source normalization, source selection gates, code/dataset/model hashes and complete query coverage were checked.

This literature comparison cannot isolate the contribution of adding environmental, temporal or network information. The separate [component study](doc_fusion_component_results_20261011.md) and no-message control address that question within their own protocols. Independent rivers, target-only graphs, arbitrary virtual sites and true missing DOC cells remain outside validated scope.

The serial v1 trial was interrupted before new target scoring. V2 re-froze identical recipes and endpoints with five independent region workers and one CPU thread, selected using a source-covariate-only timing benchmark. All 45 final fits restarted under v2; partial v1 fits are not mixed into this table. Existing paper freezes and training dispatcher remain intact.

Evidence: [protocol](../experiments/phase4_transfer/doc_network_work_v2/protocol.json), [summary](../experiments/phase4_transfer/doc_network_work_v2/analysis/summary.csv), [regional/seed results](../experiments/phase4_transfer/doc_network_work_v2/analysis/by_region_seed.csv), [paired intervals](../experiments/phase4_transfer/doc_network_work_v2/analysis/paired_comparisons.json), [independent audit](../experiments/phase4_transfer/doc_network_work_v2/analysis/independent_audit.json), [design](doc_network_work_plan_20261011.md).
