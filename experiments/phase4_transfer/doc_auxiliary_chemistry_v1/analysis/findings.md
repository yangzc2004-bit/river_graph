# Auxiliary pH and conductivity for sparse DOC reconstruction

The experiment tests whether contemporaneous auxiliary chemistry adds information to the
existing DOC model. Matched no-auxiliary, mask-only and chemistry-value arms distinguish
the availability pattern from the measured values. Neural and ExtraTrees probes use the
same auxiliary footprint. All thirteen model curves and twelve fixed contrasts are reported.

This is reconstruction conditional on same-month pH/conductivity availability, not a forecast
before those measurements arrive. Chemistry is not imputed from DOC or target evaluation
labels. The source-validation data choose head checkpoint, support parameters and ecological
mixture; target outcomes do not select a model or an availability-dependent route.

Intervals use 5,000 paired whole-station bootstrap draws, with shared station multiplicities
across overlapping partitions, seed averaging within partition and equal partition weights.
The station holdouts are reused development partitions. Repeated seeds are not new ecological samples.
Q90 includes ties at the source-training threshold. The availability appendix is descriptive
and does not introduce a new performance threshold or model-selection rule.

The loader verifies 36 copied reference-model comparisons.

## Complete K curves

| Model | K0 MAE | K1 MAE | K3 MAE | K5 MAE | K5 RMSE | K5 R² |
|---|---:|---:|---:|---:|---:|---:|
| context_gru_tuned_anchor | 1.902823 | 1.866563 | 1.634818 | 1.596001 | 3.591068 | 0.567520 |
| point_gru_tuned_anchor | 1.806996 | 1.758826 | 1.631317 | 1.589387 | 3.604470 | 0.565514 |
| tree_prior_gru_tuned_anchor | 1.855769 | 1.823650 | 1.607992 | 1.573251 | 3.566475 | 0.573434 |
| neural_no_aux_gru_tuned_anchor | 1.805225 | 1.756659 | 1.630033 | 1.581314 | 3.584033 | 0.569795 |
| neural_masks_gru_tuned_anchor | 1.802646 | 1.754012 | 1.629667 | 1.580988 | 3.583610 | 0.569872 |
| neural_chemistry_gru_tuned_anchor | 1.795167 | 1.749001 | 1.627932 | 1.585300 | 3.594138 | 0.567867 |
| tree_no_aux_gru_tuned_anchor | 1.847481 | 1.817093 | 1.607431 | 1.577596 | 3.574165 | 0.571400 |
| tree_masks_gru_tuned_anchor | 1.851629 | 1.819963 | 1.607299 | 1.579903 | 3.577304 | 0.570377 |
| tree_chemistry_gru_tuned_anchor | 1.802261 | 1.768522 | 1.565090 | 1.537905 | 3.518279 | 0.584432 |
| point_integrated_gru_tuned_anchor | 1.803538 | 1.755368 | 1.620090 | 1.564984 | 3.552260 | 0.575857 |
| neural_no_aux_integrated_gru_tuned_anchor | 1.802007 | 1.753441 | 1.618762 | 1.562256 | 3.548424 | 0.576837 |
| neural_masks_integrated_gru_tuned_anchor | 1.799529 | 1.750895 | 1.618621 | 1.564502 | 3.556501 | 0.575170 |
| neural_chemistry_integrated_gru_tuned_anchor | 1.792258 | 1.746092 | 1.619351 | 1.571309 | 3.573300 | 0.571816 |

## All twelve fixed contrasts

Negative error deltas favor the candidate; classification deltas are percentage points.

| Contrast | ΔMAE [95% CI] | ΔQ90 MAE [95% CI] | Δordinary MAE [95% CI] | Δrecall [95% CI], pp | Δfalse-Q90 [95% CI], pp |
|---|---:|---:|---:|---:|---:|
| neural_chemistry_vs_no_aux_k0 | -0.010058 [-0.023692, +0.002580] | -0.057039 [-0.099450, -0.013083] | -0.005060 [-0.018316, +0.007473] | +0.364216 [-0.055583, +0.863913] | +0.065783 [+0.020336, +0.132256] |
| neural_chemistry_vs_no_aux_integrated_k0 | -0.009749 [-0.022927, +0.002438] | -0.054751 [-0.096798, -0.011147] | -0.005001 [-0.017676, +0.006938] | +0.404040 [-0.062602, +0.937540] | +0.073316 [+0.028698, +0.138628] |
| neural_chemistry_vs_neural_masks_integrated_k0 | -0.007271 [-0.019566, +0.003668] | -0.054005 [-0.095942, -0.010459] | -0.002213 [-0.013504, +0.008278] | +0.438525 [+0.040616, +0.946351] | +0.051296 [+0.009275, +0.108954] |
| neural_chemistry_vs_point_integrated_k0 | -0.011280 [-0.027004, +0.003412] | -0.074084 [-0.127043, -0.019321] | -0.004529 [-0.021527, +0.010943] | +0.639097 [+0.224659, +1.228036] | +0.076537 [+0.026846, +0.144709] |
| tree_chemistry_vs_no_aux_k0 | -0.045220 [-0.084230, -0.010683] | -0.155819 [-0.325571, -0.017395] | -0.033794 [-0.073336, +0.000031] | +0.943123 [-0.073210, +2.252733] | -0.046953 [-0.113848, +0.003284] |
| neural_integrated_vs_tree_chemistry_k0 | -0.010003 [-0.061093, +0.040881] | -0.267180 [-0.459733, -0.082109] | +0.018558 [-0.033566, +0.074923] | +4.439761 [+2.012462, +7.685479] | +0.332667 [+0.120646, +0.602942] |
| neural_chemistry_vs_no_aux_k5 | +0.003986 [-0.004035, +0.012460] | +0.006284 [-0.016424, +0.024853] | +0.003996 [-0.004293, +0.012789] | -0.077704 [-0.418861, +0.259358] | -0.017427 [-0.059267, +0.018909] |
| neural_chemistry_vs_no_aux_integrated_k5 | +0.009053 [-0.000052, +0.019134] | +0.026778 [-0.024667, +0.085741] | +0.007417 [-0.000141, +0.015191] | +0.005340 [-0.179464, +0.215781] | -0.011849 [-0.041184, +0.013228] |
| neural_chemistry_vs_neural_masks_integrated_k5 | +0.006808 [-0.001350, +0.016020] | +0.012447 [-0.029730, +0.057779] | +0.006357 [-0.000828, +0.014045] | +0.134522 [-0.110949, +0.446596] | -0.002093 [-0.033961, +0.031761] |
| neural_chemistry_vs_point_integrated_k5 | +0.006325 [-0.003346, +0.017163] | +0.022839 [-0.031119, +0.081732] | +0.004820 [-0.003414, +0.013529] | +0.219036 [-0.060195, +0.570111] | +0.008656 [-0.024575, +0.047978] |
| tree_chemistry_vs_no_aux_k5 | -0.039691 [-0.056866, -0.024888] | -0.138584 [-0.225612, -0.045865] | -0.028570 [-0.042739, -0.016310] | +0.306945 [-0.309076, +1.019414] | -0.003040 [-0.066768, +0.060155] |
| neural_integrated_vs_tree_chemistry_k5 | +0.033405 [+0.005082, +0.064418] | +0.099941 [-0.032869, +0.220727] | +0.025655 [-0.004980, +0.058855] | +0.228246 [-0.894184, +1.581760] | -0.015768 [-0.167093, +0.104876] |

## High and ordinary DOC

| Model | K | Q90 MAE | Q90 bias | Ordinary MAE | Ordinary bias | Recall | Precision | False-Q90 rate |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| context_gru_tuned_anchor | 0 | 7.558635 | -6.606236 | 1.257359 | +0.482039 | 58.898% | 78.465% | 1.909% |
| context_gru_tuned_anchor | 5 | 6.694451 | -5.165304 | 1.008430 | +0.231465 | 65.013% | 77.677% | 2.208% |
| point_gru_tuned_anchor | 0 | 7.167345 | -5.532017 | 1.194119 | +0.213345 | 64.363% | 77.424% | 2.195% |
| point_gru_tuned_anchor | 5 | 6.642277 | -5.175765 | 1.007548 | +0.166518 | 65.545% | 77.741% | 2.203% |
| tree_prior_gru_tuned_anchor | 0 | 7.535329 | -6.632345 | 1.208099 | +0.424140 | 59.350% | 78.582% | 1.917% |
| tree_prior_gru_tuned_anchor | 5 | 6.629872 | -5.092762 | 0.990932 | +0.232974 | 65.321% | 77.693% | 2.225% |
| neural_no_aux_gru_tuned_anchor | 0 | 7.150242 | -5.472095 | 1.194039 | +0.225950 | 64.578% | 77.487% | 2.190% |
| neural_no_aux_gru_tuned_anchor | 5 | 6.611784 | -5.055866 | 1.001549 | +0.163994 | 65.758% | 77.538% | 2.234% |
| neural_masks_gru_tuned_anchor | 0 | 7.149524 | -5.472645 | 1.191135 | +0.223331 | 64.524% | 77.276% | 2.215% |
| neural_masks_gru_tuned_anchor | 5 | 6.608912 | -5.055627 | 1.001486 | +0.163800 | 65.730% | 77.566% | 2.231% |
| neural_chemistry_gru_tuned_anchor | 0 | 7.093202 | -5.425076 | 1.188979 | +0.232450 | 64.942% | 77.056% | 2.256% |
| neural_chemistry_gru_tuned_anchor | 5 | 6.618068 | -5.136331 | 1.005545 | +0.158631 | 65.680% | 77.664% | 2.217% |
| tree_no_aux_gru_tuned_anchor | 0 | 7.528101 | -6.624459 | 1.199439 | +0.414847 | 59.321% | 78.272% | 1.948% |
| tree_no_aux_gru_tuned_anchor | 5 | 6.648477 | -5.147864 | 0.993375 | +0.235527 | 64.818% | 77.598% | 2.215% |
| tree_masks_gru_tuned_anchor | 0 | 7.524137 | -6.598189 | 1.204335 | +0.419986 | 59.284% | 78.194% | 1.957% |
| tree_masks_gru_tuned_anchor | 5 | 6.666255 | -5.161249 | 0.993655 | +0.233649 | 64.494% | 77.784% | 2.180% |
| tree_chemistry_gru_tuned_anchor | 0 | 7.372282 | -6.659931 | 1.165645 | +0.404965 | 60.264% | 78.909% | 1.901% |
| tree_chemistry_gru_tuned_anchor | 5 | 6.509893 | -5.120486 | 0.964805 | +0.231099 | 65.125% | 77.682% | 2.212% |
| point_integrated_gru_tuned_anchor | 0 | 7.179187 | -5.591957 | 1.188732 | +0.203547 | 64.064% | 77.636% | 2.157% |
| point_integrated_gru_tuned_anchor | 5 | 6.586994 | -4.965678 | 0.985640 | +0.171136 | 65.134% | 77.807% | 2.188% |
| neural_no_aux_integrated_gru_tuned_anchor | 0 | 7.159853 | -5.533584 | 1.189204 | +0.218264 | 64.299% | 77.646% | 2.160% |
| neural_no_aux_integrated_gru_tuned_anchor | 5 | 6.583055 | -4.939803 | 0.983043 | +0.165250 | 65.348% | 77.692% | 2.208% |
| neural_masks_integrated_gru_tuned_anchor | 0 | 7.159107 | -5.533777 | 1.186416 | +0.215622 | 64.265% | 77.462% | 2.182% |
| neural_masks_integrated_gru_tuned_anchor | 5 | 6.597386 | -4.978542 | 0.984103 | +0.167641 | 65.218% | 77.761% | 2.198% |
| neural_chemistry_integrated_gru_tuned_anchor | 0 | 7.105102 | -5.487527 | 1.184203 | +0.223680 | 64.704% | 77.165% | 2.234% |
| neural_chemistry_integrated_gru_tuned_anchor | 5 | 6.609833 | -5.060122 | 0.990460 | +0.169801 | 65.353% | 77.796% | 2.196% |

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
Fixed-parameter ExtraTrees probes are trained on source cells, with no target-selected tree parameters.
Each arm keeps its own validation-fitted support parameters; no calibration setting is transferred
from the neural chemistry arm to its controls. New-arm support and ecological choices are fitted
on auxiliary-active validation queries; inactive queries use an identical fixed parent fallback
for every candidate, so excluding their constant errors preserves full-product candidate ranking.
Head checkpoint selection still evaluates the complete validation query. Native source predictions
combine a forest OOF base with a source-trained frozen neural correction; the neural base is not OOF.

| Neural arm | Selected epochs | Mean initial validation MAE | Mean selected validation MAE |
|---|---:|---:|---:|
| neural_no_aux | 0–62 | 1.804428 | 1.797428 |
| neural_masks | 0–21 | 1.804428 | 1.798246 |
| neural_chemistry | 2–15 | 1.804428 | 1.794668 |

Partition/seed directions and station gain/harm concentration accompany the aggregate estimates.
Intervals crossing zero do not demonstrate equality. No new K-specific route is inferred.
