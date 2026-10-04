# Nonlinear auxiliary chemistry decoding for sparse DOC reconstruction

The experiment replaces the preceding linear chemistry correction with a small nonlinear
eight-state chemistry decoder and a native residual readout. The underlying DOC encoder,
GRU, point predictor and support basis stay frozen. Three decoder arms share parameter
count and initialization; only the auxiliary-value/mask mode changes. Three previously fitted
ExtraTrees controls and the old linear chemistry correction are copied without retraining.
All fifteen curves and sixteen fixed contrasts are reported. The nonlinear-versus-linear
comparison changes both the feature form and parameter count; it is not proof that
nonlinearity alone causes any observed difference.

This is reconstruction conditional on same-month pH/conductivity availability, not a forecast
before those measurements arrive. Chemistry is not imputed from DOC or target evaluation
labels. The source-validation data choose head checkpoint, support parameters and ecological
mixture; target outcomes do not select a model or an availability-dependent route.

Intervals use 5,000 paired whole-station bootstrap draws, with shared station multiplicities
across overlapping partitions, seed averaging within partition and equal partition weights.
The station holdouts are reused development partitions. Repeated seeds are not new ecological samples.
Q90 includes ties at the source-training threshold. The availability appendix is descriptive
and does not introduce a new performance threshold or model-selection rule.

The loader verifies 81 copied reference-model comparisons.

## Complete K curves

| Model | K0 MAE | K1 MAE | K3 MAE | K5 MAE | K5 RMSE | K5 R² |
|---|---:|---:|---:|---:|---:|---:|
| context_gru_tuned_anchor | 1.902823 | 1.866563 | 1.634818 | 1.596001 | 3.591068 | 0.567520 |
| point_gru_tuned_anchor | 1.806996 | 1.758826 | 1.631317 | 1.589387 | 3.604470 | 0.565514 |
| tree_prior_gru_tuned_anchor | 1.855769 | 1.823650 | 1.607992 | 1.573251 | 3.566475 | 0.573434 |
| neural_no_aux_gru_tuned_anchor | 1.805284 | 1.756274 | 1.631454 | 1.582337 | 3.586397 | 0.569185 |
| neural_masks_gru_tuned_anchor | 1.803479 | 1.755835 | 1.631803 | 1.583762 | 3.585471 | 0.569409 |
| neural_chemistry_gru_tuned_anchor | 1.773213 | 1.741419 | 1.621943 | 1.579294 | 3.563277 | 0.574750 |
| tree_no_aux_gru_tuned_anchor | 1.847481 | 1.817093 | 1.607431 | 1.577596 | 3.574165 | 0.571400 |
| tree_masks_gru_tuned_anchor | 1.851629 | 1.819963 | 1.607299 | 1.579903 | 3.577304 | 0.570377 |
| tree_chemistry_gru_tuned_anchor | 1.802261 | 1.768522 | 1.565090 | 1.537905 | 3.518279 | 0.584432 |
| linear_chemistry_gru_tuned_anchor | 1.795167 | 1.749001 | 1.627932 | 1.585300 | 3.594138 | 0.567867 |
| point_integrated_gru_tuned_anchor | 1.803538 | 1.755368 | 1.620090 | 1.564984 | 3.552260 | 0.575857 |
| neural_no_aux_integrated_gru_tuned_anchor | 1.801879 | 1.752870 | 1.619850 | 1.563227 | 3.551186 | 0.576105 |
| neural_masks_integrated_gru_tuned_anchor | 1.800286 | 1.752642 | 1.620066 | 1.564150 | 3.550738 | 0.576232 |
| neural_chemistry_integrated_gru_tuned_anchor | 1.764947 | 1.738600 | 1.614240 | 1.568820 | 3.562414 | 0.574431 |
| linear_chemistry_integrated_gru_tuned_anchor | 1.792258 | 1.746092 | 1.619351 | 1.571309 | 3.573300 | 0.571816 |

## All sixteen fixed contrasts

Negative error deltas favor the candidate; classification deltas are percentage points.

| Contrast | ΔMAE [95% CI] | ΔQ90 MAE [95% CI] | Δordinary MAE [95% CI] | Δrecall [95% CI], pp | Δfalse-Q90 [95% CI], pp |
|---|---:|---:|---:|---:|---:|
| neural_chemistry_vs_no_aux_k0 | -0.032071 [-0.058517, -0.012106] | -0.140786 [-0.293780, +0.000092] | -0.020053 [-0.037797, -0.005015] | +0.612894 [-0.067563, +1.423451] | +0.061597 [-0.035562, +0.193688] |
| neural_chemistry_vs_no_aux_integrated_k0 | -0.036933 [-0.063709, -0.015899] | -0.143957 [-0.295161, -0.005332] | -0.025132 [-0.044473, -0.008615] | +0.743049 [+0.056864, +1.526670] | +0.069130 [-0.020277, +0.189941] |
| neural_chemistry_vs_neural_masks_integrated_k0 | -0.035339 [-0.062586, -0.014613] | -0.145884 [-0.295899, -0.009658] | -0.023240 [-0.043000, -0.006590] | +0.704197 [+0.067254, +1.450707] | +0.078330 [-0.009190, +0.202036] |
| neural_chemistry_vs_point_integrated_k0 | -0.038591 [-0.066865, -0.015185] | -0.159643 [-0.316980, -0.018144] | -0.025304 [-0.048207, -0.005344] | +0.831433 [+0.187046, +1.625083] | +0.089486 [+0.006466, +0.205121] |
| tree_chemistry_vs_no_aux_k0 | -0.045220 [-0.084230, -0.010683] | -0.155819 [-0.325571, -0.017395] | -0.033794 [-0.073336, +0.000031] | +0.943123 [-0.073210, +2.252733] | -0.046953 [-0.113848, +0.003284] |
| neural_integrated_vs_tree_chemistry_k0 | -0.037315 [-0.092347, +0.015480] | -0.352738 [-0.593105, -0.107987] | -0.002217 [-0.055801, +0.054054] | +4.632096 [+2.054363, +7.989531] | +0.345617 [+0.113771, +0.649052] |
| neural_chemistry_vs_no_aux_k5 | -0.003043 [-0.019578, +0.011416] | -0.053921 [-0.120090, +0.018540] | +0.002872 [-0.012357, +0.015645] | -0.137418 [-0.878712, +0.478348] | -0.048651 [-0.160050, +0.034933] |
| neural_chemistry_vs_no_aux_integrated_k5 | +0.005592 [-0.006987, +0.017343] | +0.000215 [-0.039624, +0.035537] | +0.006751 [-0.005490, +0.018145] | +0.064602 [-0.668167, +0.686636] | -0.028709 [-0.104062, +0.026517] |
| neural_chemistry_vs_neural_masks_integrated_k5 | +0.004669 [-0.007270, +0.015566] | -0.001714 [-0.042146, +0.034490] | +0.005902 [-0.005599, +0.016422] | +0.013124 [-0.720673, +0.579711] | -0.026892 [-0.102069, +0.027993] |
| neural_chemistry_vs_point_integrated_k5 | +0.003836 [-0.009955, +0.016600] | +0.000187 [-0.041405, +0.036001] | +0.004719 [-0.008982, +0.017378] | +0.092774 [-0.706839, +0.753449] | -0.008897 [-0.083110, +0.051624] |
| tree_chemistry_vs_no_aux_k5 | -0.039691 [-0.056866, -0.024888] | -0.138584 [-0.225612, -0.045865] | -0.028570 [-0.042739, -0.016310] | +0.306945 [-0.309076, +1.019414] | -0.003040 [-0.066768, +0.060155] |
| neural_integrated_vs_tree_chemistry_k5 | +0.030915 [+0.002488, +0.062056] | +0.077289 [-0.031147, +0.177020] | +0.025555 [-0.005903, +0.059518] | +0.101983 [-0.965203, +1.367728] | -0.033321 [-0.191121, +0.094992] |
| nonlinear_vs_linear_chemistry_k0 | -0.021954 [-0.040814, -0.007256] | -0.080697 [-0.193447, +0.028151] | -0.015351 [-0.028494, -0.003608] | +0.062181 [-0.598617, +0.649912] | +0.012950 [-0.075242, +0.121412] |
| nonlinear_vs_linear_chemistry_integrated_k0 | -0.027311 [-0.046387, -0.012595] | -0.085558 [-0.198619, +0.019371] | -0.020775 [-0.034706, -0.008563] | +0.192336 [-0.460300, +0.813610] | +0.012950 [-0.063557, +0.100519] |
| nonlinear_vs_linear_chemistry_k5 | -0.006006 [-0.018716, +0.004229] | -0.057721 [-0.124136, +0.010619] | -0.000312 [-0.009864, +0.008601] | -0.187448 [-0.689345, +0.231614] | -0.032903 [-0.143432, +0.048810] |
| nonlinear_vs_linear_chemistry_integrated_k5 | -0.002490 [-0.010717, +0.005152] | -0.022652 [-0.063068, +0.027040] | -0.000100 [-0.007449, +0.006622] | -0.126263 [-0.735114, +0.307698] | -0.017554 [-0.070320, +0.020467] |

## High and ordinary DOC

| Model | K | Q90 MAE | Q90 bias | Ordinary MAE | Ordinary bias | Recall | Precision | False-Q90 rate |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| context_gru_tuned_anchor | 0 | 7.558635 | -6.606236 | 1.257359 | +0.482039 | 58.898% | 78.465% | 1.909% |
| context_gru_tuned_anchor | 5 | 6.694451 | -5.165304 | 1.008430 | +0.231465 | 65.013% | 77.677% | 2.208% |
| point_gru_tuned_anchor | 0 | 7.167345 | -5.532017 | 1.194119 | +0.213345 | 64.363% | 77.424% | 2.195% |
| point_gru_tuned_anchor | 5 | 6.642277 | -5.175765 | 1.007548 | +0.166518 | 65.545% | 77.741% | 2.203% |
| tree_prior_gru_tuned_anchor | 0 | 7.535329 | -6.632345 | 1.208099 | +0.424140 | 59.350% | 78.582% | 1.917% |
| tree_prior_gru_tuned_anchor | 5 | 6.629872 | -5.092762 | 0.990932 | +0.232974 | 65.321% | 77.693% | 2.225% |
| neural_no_aux_gru_tuned_anchor | 0 | 7.153292 | -5.488847 | 1.193681 | +0.227791 | 64.392% | 77.300% | 2.208% |
| neural_no_aux_gru_tuned_anchor | 5 | 6.614268 | -5.069173 | 1.002361 | +0.167172 | 65.630% | 77.525% | 2.233% |
| neural_masks_gru_tuned_anchor | 0 | 7.154834 | -5.482413 | 1.191599 | +0.236914 | 64.371% | 77.422% | 2.193% |
| neural_masks_gru_tuned_anchor | 5 | 6.617082 | -5.071528 | 1.003681 | +0.163905 | 65.510% | 77.618% | 2.218% |
| neural_chemistry_gru_tuned_anchor | 0 | 7.012506 | -5.329012 | 1.173629 | +0.198901 | 65.005% | 77.072% | 2.269% |
| neural_chemistry_gru_tuned_anchor | 5 | 6.560347 | -5.081866 | 1.005234 | +0.157009 | 65.493% | 77.968% | 2.184% |
| tree_no_aux_gru_tuned_anchor | 0 | 7.528101 | -6.624459 | 1.199439 | +0.414847 | 59.321% | 78.272% | 1.948% |
| tree_no_aux_gru_tuned_anchor | 5 | 6.648477 | -5.147864 | 0.993375 | +0.235527 | 64.818% | 77.598% | 2.215% |
| tree_masks_gru_tuned_anchor | 0 | 7.524137 | -6.598189 | 1.204335 | +0.419986 | 59.284% | 78.194% | 1.957% |
| tree_masks_gru_tuned_anchor | 5 | 6.666255 | -5.161249 | 0.993655 | +0.233649 | 64.494% | 77.784% | 2.180% |
| tree_chemistry_gru_tuned_anchor | 0 | 7.372282 | -6.659931 | 1.165645 | +0.404965 | 60.264% | 78.909% | 1.901% |
| tree_chemistry_gru_tuned_anchor | 5 | 6.509893 | -5.120486 | 0.964805 | +0.231099 | 65.125% | 77.682% | 2.212% |
| linear_chemistry_gru_tuned_anchor | 0 | 7.093202 | -5.425076 | 1.188979 | +0.232450 | 64.942% | 77.056% | 2.256% |
| linear_chemistry_gru_tuned_anchor | 5 | 6.618068 | -5.136331 | 1.005545 | +0.158631 | 65.680% | 77.664% | 2.217% |
| point_integrated_gru_tuned_anchor | 0 | 7.179187 | -5.591957 | 1.188732 | +0.203547 | 64.064% | 77.636% | 2.157% |
| point_integrated_gru_tuned_anchor | 5 | 6.586994 | -4.965678 | 0.985640 | +0.171136 | 65.134% | 77.807% | 2.188% |
| neural_no_aux_integrated_gru_tuned_anchor | 0 | 7.163501 | -5.548879 | 1.188560 | +0.219514 | 64.153% | 77.473% | 2.177% |
| neural_no_aux_integrated_gru_tuned_anchor | 5 | 6.586966 | -4.953510 | 0.983609 | +0.168835 | 65.162% | 77.644% | 2.207% |
| neural_masks_integrated_gru_tuned_anchor | 0 | 7.165428 | -5.543014 | 1.186667 | +0.228287 | 64.192% | 77.572% | 2.168% |
| neural_masks_integrated_gru_tuned_anchor | 5 | 6.588895 | -4.954056 | 0.984457 | +0.167347 | 65.214% | 77.673% | 2.206% |
| neural_chemistry_integrated_gru_tuned_anchor | 0 | 7.019544 | -5.399239 | 1.163428 | +0.191181 | 64.896% | 77.220% | 2.247% |
| neural_chemistry_integrated_gru_tuned_anchor | 5 | 6.587181 | -5.099975 | 0.990359 | +0.166579 | 65.227% | 77.923% | 2.179% |
| linear_chemistry_integrated_gru_tuned_anchor | 0 | 7.105102 | -5.487527 | 1.184203 | +0.223680 | 64.704% | 77.165% | 2.234% |
| linear_chemistry_integrated_gru_tuned_anchor | 5 | 6.609833 | -5.060122 | 0.990460 | +0.169801 | 65.353% | 77.796% | 2.196% |

## Availability and intended reconstruction coverage

Observed-DOC evaluation and genuinely missing-DOC reconstruction have different auxiliary
data coverage. Success in the observed evaluation cannot be assumed to cover the missing grid.

| Full-grid cohort | Auxiliary availability | Cells | Denominator | Fraction |
|---|---|---:|---:|---:|
| all_station_months | both | 53815 | 233478 | 23.049% |
| all_station_months | ph_only | 815 | 233478 | 0.349% |
| all_station_months | ec_only | 9194 | 233478 | 3.938% |
| all_station_months | neither | 169654 | 233478 | 72.664% |
| doc_observed | both | 21902 | 22571 | 97.036% |
| doc_observed | ph_only | 168 | 22571 | 0.744% |
| doc_observed | ec_only | 131 | 22571 | 0.580% |
| doc_observed | neither | 370 | 22571 | 1.639% |
| doc_genuinely_missing | both | 31913 | 210907 | 15.131% |
| doc_genuinely_missing | ph_only | 647 | 210907 | 0.307% |
| doc_genuinely_missing | ec_only | 9063 | 210907 | 4.297% |
| doc_genuinely_missing | neither | 169284 | 210907 | 80.265% |

| Evaluation availability | Unique query cells | Unique stations | Split-cell occurrences |
|---|---:|---:|---:|
| both | 10216 | 170 | 12271 |
| ph_only | 55 | 7 | 55 |
| ec_only | 84 | 22 | 182 |
| neither | 165 | 9 | 357 |

## Source-validation choices

The complete checkpoint, support alpha/ridge and ecological mixing scores are saved separately.
The fixed ExtraTrees probes were trained in the preceding study and are copied with their original support choices.
Each arm keeps its own validation-fitted support parameters; no calibration setting is transferred
from the nonlinear chemistry arm to its controls. New-arm support and ecological choices are fitted
on auxiliary-active validation queries; inactive queries use an identical fixed parent fallback
for every candidate, so excluding their constant errors preserves full-product candidate ranking.
Head checkpoint selection still evaluates the complete validation query. Native source predictions
combine a forest OOF base with a source-trained frozen neural correction; the neural base is not OOF.

| Neural arm | Selected epochs | Mean initial validation MAE | Mean selected validation MAE |
|---|---:|---:|---:|
| neural_no_aux | 0–20 | 1.804428 | 1.799160 |
| neural_masks | 2–22 | 1.804428 | 1.799376 |
| neural_chemistry | 0–73 | 1.804428 | 1.769764 |

Partition/seed directions and station gain/harm concentration accompany the aggregate estimates.
Intervals crossing zero do not demonstrate equality. No new K-specific route is inferred.
