# Interpretation: refreshed support bases for fixed DOC experts

## Result

Refreshing the support representation improves three-observation calibration for the off and current-only experts, mostly by reducing ordinary-DOC error in one station partition. The benefit does not extend to five-observation calibration or high-DOC reconstruction. Native predictions are unchanged. Keep the legacy-support daily integrated model as the established candidate; preserve refreshed support as a completed representation experiment, without introducing a K-dependent winner route.

The experiment changes selected hidden coordinates while holding the v4 two-dimensional readout and anchor normalization fixed. Source-validation alpha/ridge choices are refitted, and integrated arms also refit positive-K ecological gamma. Therefore the direct contrast is representation alignment plus calibration reselection; it does not isolate a basis change at fixed coefficients. Integrated comparisons add another selection layer.

## Complete legacy/refreshed curves

| Expert / stage | Basis | K0 MAE | K1 MAE | K3 MAE | K5 MAE |
|---|---|---:|---:|---:|---:|
| off | legacy | 1.806996 | 1.758826 | 1.631317 | 1.589387 |
| off | refreshed | 1.806996 | 1.758826 | 1.607645 | 1.590976 |
| current_only | legacy | 1.810409 | 1.762757 | 1.634744 | 1.587180 |
| current_only | refreshed | 1.810409 | 1.762757 | 1.611741 | 1.590835 |
| full_history | legacy | 1.800854 | 1.756293 | 1.631764 | 1.586407 |
| full_history | refreshed | 1.800854 | 1.756293 | 1.619470 | 1.593891 |
| off_integrated | legacy | 1.803538 | 1.755368 | 1.620090 | 1.564984 |
| off_integrated | refreshed | 1.803538 | 1.755368 | 1.599087 | 1.572464 |
| current_only_integrated | legacy | 1.807859 | 1.760206 | 1.622109 | 1.567740 |
| current_only_integrated | refreshed | 1.807859 | 1.760206 | 1.599909 | 1.570676 |
| full_history_integrated | legacy | 1.797393 | 1.752831 | 1.620559 | 1.567575 |
| full_history_integrated | refreshed | 1.797393 | 1.752831 | 1.610500 | 1.573159 |

Constant-only calibration curves remain in findings.md and k_curves.csv. No neural or forest training was performed.

## K0 and K1 are structural controls

K0 has no support update and reproduces the frozen base. At K1 the centered support-feature matrix and centered support residual have rank zero: a single point equals its own mean. The adapter explicitly applies shape correction only when K>1. Only the shrunk station offset remains, so changing the basis has no effect. All 54 K1 expert/pipeline/run prediction checks are bitwise identical, and their bootstrap differences are exactly [0,0]. This is algebraic invariance, not an underpowered statistical comparison.

## Three observations: benefit concentrated in ordinary DOC

| Expert / stage | Δoverall MAE [95% CI] | Δordinary MAE [95% CI] | ΔQ90 MAE [95% CI] | Better partitions / fits |
|---|---:|---:|---:|---:|
| off | -0.023671 [-0.047630, -0.001123] | -0.028789 [-0.056459, -0.004002] | +0.019447 [-0.037787, +0.073159] | 2/3; 5/9 |
| current_only | -0.023003 [-0.046626, -0.000345] | -0.028599 [-0.055705, -0.004272] | +0.023910 [-0.035416, +0.079573] | 2/3; 4/9 |
| full_history | -0.012294 [-0.032318, +0.008281] | -0.018005 [-0.039050, +0.001936] | +0.034602 [-0.023247, +0.091529] | 3/3; 5/9 |
| off_integrated | -0.021003 [-0.044403, +0.000769] | -0.026178 [-0.053517, -0.002702] | +0.022366 [-0.035710, +0.076070] | 2/3; 4/9 |
| current_only_integrated | -0.022199 [-0.047859, +0.000980] | -0.028183 [-0.057787, -0.003323] | +0.027624 [-0.031420, +0.081908] | 2/3; 5/9 |
| full_history_integrated | -0.010059 [-0.029165, +0.009297] | -0.015159 [-0.035710, +0.003197] | +0.032035 [-0.030958, +0.090472] | 3/3; 5/9 |

Off direct K3 improves 1.451% (95% CI 0.070–2.917%); current-only direct improves 1.407% (0.022–2.856%). The full-history direct estimate is 0.753% with a CI crossing zero. All three integrated point estimates improve, but each overall CI crosses zero, including the narrow off/current intervals near zero. Ordinary-DOC intervals are below zero for the off/current direct and integrated paths. All six Q90 point estimates worsen; none of these K3 tail intervals establishes a change.

| Expert / stage, K3 | Partition 142 ΔMAE | Partition 143 ΔMAE | Partition 144 ΔMAE |
|---|---:|---:|---:|
| off | -0.002175 | +0.002331 | -0.071170 |
| current_only | -0.001787 | +0.002821 | -0.070043 |
| full_history | -0.001101 | -0.004512 | -0.031270 |
| off_integrated | -0.002490 | +0.008897 | -0.069414 |
| current_only_integrated | -0.001896 | +0.008024 | -0.072725 |
| full_history_integrated | -0.001061 | -0.000785 | -0.028332 |

Partition 144 supplies most of the mean improvement. The largest five station contributions account for about 58% of positive direct K3 gain, while station improvement counts are 85/172 (off), 91/172 (current-only) and 87/172 (full-history). The improvement is not a uniform station response. Training seeds repeat the same ecological observations and do not multiply sample size.

## Five observations: the refreshed representation does not improve transfer

| Expert / stage | Δoverall MAE [95% CI] | ΔQ90 MAE [95% CI] | Better partitions / fits |
|---|---:|---:|---:|
| off | +0.001590 [-0.003770, +0.008108] | +0.021691 [-0.017139, +0.056841] | 1/3; 3/9 |
| current_only | +0.003656 [-0.008506, +0.016397] | +0.049674 [-0.007148, +0.103972] | 1/3; 3/9 |
| full_history | +0.007484 [-0.000735, +0.016935] | +0.050762 [+0.004498, +0.098961] | 0/3; 2/9 |
| off_integrated | +0.007480 [-0.000043, +0.016361] | +0.050280 [-0.002918, +0.101326] | 0/3; 2/9 |
| current_only_integrated | +0.002937 [-0.007917, +0.013788] | +0.037858 [-0.010870, +0.081961] | 1/3; 2/9 |
| full_history_integrated | +0.005584 [-0.000375, +0.012723] | +0.031837 [-0.001432, +0.064323] | 0/3; 1/9 |

Every K5 overall point estimate worsens, though the paired overall intervals include zero. The refreshed off integrated model is 1.572464 versus legacy 1.564984; current-only is 1.570676 versus 1.567740; full-history is 1.573159 versus 1.567575. Full-history direct Q90 error worsens by +0.050762 [0.004498, 0.098961] mg/L. Several refreshed K5 predictions are near the constant-only diagnostic, while the old basis retains useful shape correction. This rejects a broad replacement claim; it does not mean the refreshed basis is universally harmful.

## Detection and false positives

| Expert / stage | K | ΔQ90 recall [95% CI], pp | Δfalse-Q90 rate [95% CI], pp |
|---|---:|---:|---:|
| off | 3 | -0.802763 [-2.226174, +0.340247] | -0.015752 [-0.115026, +0.065776] |
| off | 5 | -0.118501 [-0.721454, +0.500113] | +0.035127 [-0.007536, +0.089871] |
| current_only | 3 | -0.926605 [-2.272132, +0.170140] | -0.005306 [-0.103188, +0.078554] |
| current_only | 5 | -0.036431 [-0.752392, +0.579471] | +0.062318 [-0.003545, +0.155028] |
| full_history | 3 | -0.457487 [-1.312434, +0.219441] | +0.022000 [-0.054240, +0.105933] |
| full_history | 5 | -0.234106 [-0.933154, +0.363002] | -0.020521 [-0.074408, +0.020394] |
| off_integrated | 3 | -0.431737 [-1.486304, +0.528243] | -0.000836 [-0.063762, +0.055003] |
| off_integrated | 5 | +0.016020 [-0.397920, +0.415883] | +0.008085 [-0.030320, +0.045857] |
| current_only_integrated | 3 | -0.267097 [-1.298639, +0.680295] | +0.011984 [-0.050158, +0.072953] |
| current_only_integrated | 5 | -0.066548 [-0.640749, +0.476082] | +0.028161 [-0.012623, +0.073237] |
| full_history_integrated | 3 | -0.008259 [-0.633543, +0.632394] | +0.026735 [-0.042485, +0.100153] |
| full_history_integrated | 5 | -0.562887 [-1.206284, +0.015924] | -0.005451 [-0.041720, +0.025267] |

No K3/K5 recall or false-positive comparison has a pointwise interval excluding zero. Most recall changes are negative; the high-DOC detection problem has not been solved by refreshing the basis. Precision, absolute rates, transformed-space errors and signed biases are preserved in the complete output tables.

## Source-validation context and scientific interpretation

Before target analysis, source-validation K3 improved for all three experts: direct differences −0.014389/−0.013842/−0.013362 and integrated −0.014536/−0.014399/−0.013358 for off/current/full. Target K3 mostly supports the direction, but shows stronger partition concentration. At validation K5, off integrated was slightly worse (+0.000233), current-only improved −0.001854 and full-history improved −0.002495. Those latter small validation gains did not transfer to the fixed target stations.

The results show that updating the support coordinates can change small-support calibration, but a hidden representation optimized for native residual prediction is not automatically a better station-adaptation basis. Here the unchanged v4 readout is applied to different selected hidden states; it was not retrained for those states. Calibration coefficients and ecological mixtures also change. These facts motivate task-aligned representation tests, but the current experiment cannot uniquely attribute the K5 loss to coordinate mismatch, parameter selection or the observations selected as support.

Preserve every current/legacy/refreshed product. Continue to use the already established daily integrated model as the main candidate, with this K3 result as a conditional support-adaptation finding. Do not construct a K3-refreshed/K5-legacy selector from these target outcomes. No new model, threshold, basis or route is fitted in this analysis.

All 18 predefined contrasts use 5,000 paired whole-station bootstrap draws; station identities are jointly resampled across partitions, seeds average within partitions and partitions receive equal weight. Intervals are pointwise on reused development partitions. Native full-grid products are byte-identical to the preceding memory experiment; constant/legacy parent replication and K0 invariance pass all 216 recorded checks.

Sources: findings.md, k_curves.csv, comparisons.csv, directions_by_partition.csv, gain_loss_concentration.csv, error_profiles.csv, classification.csv; source-only diagnostics/source_validation.md and formal verification/replay_checks.json.
