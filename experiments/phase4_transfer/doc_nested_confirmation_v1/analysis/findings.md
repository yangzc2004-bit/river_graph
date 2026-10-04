# Nested chemistry calibration: fresh station-role confirmation

Six fixed procedures are compared on station-role seeds 342–344 and training seeds 42–44.
Every fitted predictor is rebuilt using its fresh source roles. These partitions reuse the ST357 cohort;
they are not external-basin validation or new independent measurements. This script selects no model.

The main comparison is the separate chemical increment versus the previously joint-selected support basis.
Legacy calibration, the availability-only increment, the general model and chemical trees remain controls.
The neural decoder and ecological/temporal pipeline use the retained empty-edge spatial self path.
Source forests are station-blocked OOF; the complete source neural pipeline is not OOF.
Support is retrospective and may postdate a query. Current-month pH/conductance are allowed covariates.

Each run uses cell-weighted error. Seeds are averaged within a partition, then partitions receive equal weight.
The ten fixed comparisons use 5,000 joint whole-station bootstrap draws, with repeated stations sampled
jointly across partitions. Seed repeats do not increase ecological sample size. The intervals are unadjusted
95% percentile intervals; a crossing-zero interval does not establish equivalence. RMSE/R² are averages
of run metrics, not statistics calculated after pooling predictions.

## Full support curves

| Procedure | K0 MAE | K1 MAE | K3 MAE | K5 MAE | K5 log MAE |
|---|---:|---:|---:|---:|---:|
| point_integrated_legacy | 1.798880 | 1.765514 | 1.624407 | 1.591951 | 0.217298 |
| neural_chemistry_integrated_selected | 1.779435 | 1.743581 | 1.612311 | 1.584968 | 0.216153 |
| neural_chemistry_integrated_legacy | 1.779435 | 1.743581 | 1.613529 | 1.586328 | 0.216379 |
| tree_chemistry_selected | 1.800454 | 1.784413 | 1.616055 | 1.578438 | 0.215724 |
| neural_chemistry_integrated_nested | 1.779435 | 1.743581 | 1.615206 | 1.582420 | 0.215604 |
| neural_chemistry_integrated_nested_masks | 1.779435 | 1.743581 | 1.614418 | 1.587272 | 0.216407 |

## All fixed comparisons

Negative error deltas favor the nested candidate.

| Comparison | ΔMAE [95% CI] | Relative gain % [95% CI] | Q90 ΔMAE [95% CI] | Ordinary ΔMAE [95% CI] | Better partitions / runs |
|---|---:|---:|---:|---:|---:|
| nested_vs_joint_k3 | +0.002895 [-0.004210, +0.011040] | -0.180 [-0.671, +0.272] | +0.006885 [-0.012718, +0.026383] | +0.002821 [-0.004809, +0.011985] | 1/3; 5/9 |
| nested_vs_joint_k5 | -0.002548 [-0.005219, +0.000216] | +0.161 [-0.013, +0.332] | -0.022315 [-0.055586, +0.003301] | -0.001459 [-0.004696, +0.002455] | 3/3; 7/9 |
| nested_vs_legacy_k3 | +0.001677 [-0.003120, +0.007456] | -0.104 [-0.464, +0.192] | -0.017297 [-0.050324, +0.007909] | +0.003054 [-0.002540, +0.010298] | 1/3; 5/9 |
| nested_vs_legacy_k5 | -0.003908 [-0.006636, -0.001274] | +0.246 [+0.080, +0.415] | -0.033981 [-0.073935, -0.003447] | -0.001913 [-0.005024, +0.001694] | 3/3; 8/9 |
| nested_vs_general_k3 | -0.009201 [-0.018781, +0.001771] | +0.566 [-0.105, +1.168] | -0.070351 [-0.109355, -0.035730] | -0.003836 [-0.014528, +0.008462] | 2/3; 6/9 |
| nested_vs_general_k5 | -0.009531 [-0.016424, -0.003090] | +0.599 [+0.198, +1.005] | -0.077437 [-0.123847, -0.032181] | -0.004484 [-0.011491, +0.002915] | 3/3; 8/9 |
| nested_vs_masks_k3 | +0.000788 [-0.003910, +0.006617] | -0.049 [-0.402, +0.243] | -0.015661 [-0.048857, +0.009274] | +0.001901 [-0.003400, +0.008906] | 1/3; 7/9 |
| nested_vs_masks_k5 | -0.004852 [-0.008555, -0.001774] | +0.306 [+0.110, +0.527] | -0.011855 [-0.029311, +0.003919] | -0.004312 [-0.008257, -0.001277] | 3/3; 7/9 |
| nested_vs_tree_k3 | -0.000849 [-0.030100, +0.027199] | +0.053 [-1.690, +1.875] | -0.047671 [-0.137029, +0.047633] | +0.003263 [-0.029703, +0.034522] | 2/3; 5/9 |
| nested_vs_tree_k5 | +0.003982 [-0.014387, +0.021394] | -0.252 [-1.352, +0.889] | -0.053129 [-0.124246, +0.017825] | +0.008663 [-0.010054, +0.027450] | 1/3; 6/9 |

## Interpretation and coverage

Q90 means DOC at or above its source-training 90th percentile, including ties. Tail groups with fewer
than 20 unique query cells per partition are flagged unstable. Error-profile and classification CSVs
report tail/ordinary MAE, signed bias, precision, recall and false-positive rates together.
Positive signed bias means overprediction. Low K0/K1 nested increments are exactly zero by design;
their unchanged predictions are algebraic negative controls, not a statistical power finding.

The availability appendix retains both, pH-only, conductance-only and neither groups, including empty
groups. Full-grid covariate coverage at genuinely missing DOC cells is separate from accuracy on observed
held queries. No accuracy is inferred for cells without DOC truth.

Source-validation choices and conditional fold diagnostics are recorded separately. Those conditional folds
assess the incremental adapter after its parent already used all source validation; only the fresh target
queries provide the present held-station comparison. No K-specific winner is assembled from target results.

All six curves, station gain/loss concentration and every partition/seed direction remain in the output tables.
