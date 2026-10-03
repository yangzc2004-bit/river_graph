# Ecological residual transfer on the DOC interaction model

Nine frozen-expert packages contain four competing residual memories each. Source
DOC minus station-blocked OOF context predictions provides the residual library.
The current interaction model and the support-adaptation representation remain fixed.

Each source station receives equal total weight. Global versus ecological donor pools
are crossed with an intercept-only versus concentration-dependent affine residual.
Nine ecology variables use source-station median/IQR scaling. Ecological candidates
use 20/40/80 eligible neighbors; all arms share ridge 0.1/1 and smooth absolute error.
Affine conditioning uses source-standardized log1p(context prediction), not a physical
state. No extra tail weighting is introduced in the memory objective.

The memory competes with the temporal correction:

    prediction = max(0, context + (1-gamma)*(interaction-context) + gamma*memory)

Gamma=0 exactly preserves the interaction model; gamma=1 replaces its correction.
Intermediate gamma mixes them. Memory improvement is not automatically an improvement
of the neural representation. Source-validation K0 MAE selects gamma, ridge and donors;
the usual support adapters are then fitted on the same validation protocol.

These are previously evaluated same-cohort station partitions 142–144. This development
analysis retains all four memories and does not select or promote a winner. Target
support is retrospective and query populations are fixed across models and K.

Paired intervals use 5,000 joint whole-station bootstrap draws, with repeated station
identities resampled together. Seed losses are averaged within partitions, then partitions
receive equal weight. Negative ΔMAE or positive relative reduction favors the candidate.
Q90 is source-training-derived. Bias is prediction minus observed DOC.

## Overall performance

| Saved model | K | MAE | RMSE | R² | Log MAE | Q90 MAE |
|---|---:|---:|---:|---:|---:|---:|
| ecological_affine_constant | 0 | 1.8092 | 3.953 | 0.480 | 0.2683 | 7.4318 |
| ecological_affine_constant | 5 | 1.5958 | 3.609 | 0.563 | 0.2322 | 6.7242 |
| ecological_affine_gru_tuned_anchor | 0 | 1.8092 | 3.953 | 0.480 | 0.2683 | 7.4318 |
| ecological_affine_gru_tuned_anchor | 5 | 1.5872 | 3.590 | 0.567 | 0.2302 | 6.6924 |
| ecological_bias_constant | 0 | 1.8198 | 3.973 | 0.475 | 0.2698 | 7.5246 |
| ecological_bias_constant | 5 | 1.5966 | 3.614 | 0.561 | 0.2323 | 6.7526 |
| ecological_bias_gru_tuned_anchor | 0 | 1.8198 | 3.973 | 0.475 | 0.2698 | 7.5246 |
| ecological_bias_gru_tuned_anchor | 5 | 1.5840 | 3.587 | 0.567 | 0.2293 | 6.6970 |
| global_affine_constant | 0 | 1.8352 | 3.958 | 0.479 | 0.2757 | 7.4371 |
| global_affine_constant | 5 | 1.5956 | 3.607 | 0.563 | 0.2330 | 6.7277 |
| global_affine_gru_tuned_anchor | 0 | 1.8352 | 3.958 | 0.479 | 0.2757 | 7.4371 |
| global_affine_gru_tuned_anchor | 5 | 1.5816 | 3.579 | 0.569 | 0.2296 | 6.6746 |
| global_bias_constant | 0 | 1.8346 | 3.966 | 0.477 | 0.2750 | 7.4682 |
| global_bias_constant | 5 | 1.5982 | 3.609 | 0.563 | 0.2335 | 6.7326 |
| global_bias_gru_tuned_anchor | 0 | 1.8346 | 3.966 | 0.477 | 0.2750 | 7.4682 |
| global_bias_gru_tuned_anchor | 5 | 1.5844 | 3.581 | 0.569 | 0.2302 | 6.6797 |
| interaction_constant | 0 | 1.8352 | 3.958 | 0.479 | 0.2757 | 7.4371 |
| interaction_constant | 5 | 1.5956 | 3.607 | 0.563 | 0.2330 | 6.7277 |
| interaction_gru_tuned_anchor | 0 | 1.8352 | 3.958 | 0.479 | 0.2757 | 7.4371 |
| interaction_gru_tuned_anchor | 5 | 1.5816 | 3.579 | 0.569 | 0.2296 | 6.6746 |

At K0 the two support adapters give identical predictions for a given base.
Repeated K0 rows are not independent evidence.

## Ecological similarity beyond global calibration

| Comparison | ΔMAE [95% CI] | Reduction [95% CI] | Improved partitions; fits |
|---|---:|---:|---:|
| ecological_vs_global_bias_constant_k0 | -0.0148 [-0.0269, -0.0032] | +0.81% [+0.17, +1.52] | 3/3; 8/9 |
| ecological_vs_global_affine_constant_k0 | -0.0260 [-0.0442, -0.0097] | +1.42% [+0.52, +2.45] | 3/3; 8/9 |
| ecological_vs_global_bias_constant_k5 | -0.0016 [-0.0069, +0.0037] | +0.10% [-0.21, +0.45] | 2/3; 7/9 |
| ecological_vs_global_affine_constant_k5 | +0.0002 [-0.0066, +0.0063] | -0.01% [-0.39, +0.43] | 2/3; 4/9 |
| ecological_vs_global_bias_gru_tuned_anchor_k0 | -0.0148 [-0.0269, -0.0032] | +0.81% [+0.17, +1.52] | 3/3; 8/9 |
| ecological_vs_global_affine_gru_tuned_anchor_k0 | -0.0260 [-0.0442, -0.0097] | +1.42% [+0.52, +2.45] | 3/3; 8/9 |
| ecological_vs_global_bias_gru_tuned_anchor_k5 | -0.0004 [-0.0055, +0.0043] | +0.03% [-0.26, +0.35] | 2/3; 6/9 |
| ecological_vs_global_affine_gru_tuned_anchor_k5 | +0.0056 [-0.0042, +0.0161] | -0.35% [-1.00, +0.28] | 1/3; 3/9 |

## Affine conditioning beyond a constant correction

| Comparison | ΔMAE [95% CI] | Reduction [95% CI] | Improved partitions; fits |
|---|---:|---:|---:|
| global_affine_vs_bias_constant_k0 | +0.0007 [-0.0051, +0.0076] | -0.04% [-0.41, +0.28] | 0/3; 1/9 |
| ecological_affine_vs_bias_constant_k0 | -0.0106 [-0.0224, -0.0005] | +0.58% [+0.03, +1.20] | 2/3; 7/9 |
| global_affine_vs_bias_constant_k5 | -0.0026 [-0.0054, +0.0001] | +0.16% [-0.00, +0.33] | 1/3; 3/9 |
| ecological_affine_vs_bias_constant_k5 | -0.0009 [-0.0039, +0.0018] | +0.05% [-0.12, +0.23] | 1/3; 6/9 |
| global_affine_vs_bias_gru_tuned_anchor_k0 | +0.0007 [-0.0051, +0.0076] | -0.04% [-0.41, +0.28] | 0/3; 1/9 |
| ecological_affine_vs_bias_gru_tuned_anchor_k0 | -0.0106 [-0.0224, -0.0005] | +0.58% [+0.03, +1.20] | 2/3; 7/9 |
| global_affine_vs_bias_gru_tuned_anchor_k5 | -0.0028 [-0.0054, -0.0002] | +0.18% [+0.01, +0.34] | 1/3; 3/9 |
| ecological_affine_vs_bias_gru_tuned_anchor_k5 | +0.0032 [-0.0057, +0.0132] | -0.20% [-0.82, +0.35] | 0/3; 5/9 |

## Comparison with the current interaction model

| Comparison | ΔMAE [95% CI] | Reduction [95% CI] | Improved partitions; fits |
|---|---:|---:|---:|
| global_bias_vs_interaction_constant_k0 | -0.0007 [-0.0076, +0.0051] | +0.04% [-0.28, +0.41] | 1/3; 2/9 |
| ecological_bias_vs_interaction_constant_k0 | -0.0155 [-0.0305, -0.0012] | +0.84% [+0.06, +1.71] | 3/3; 8/9 |
| global_affine_vs_interaction_constant_k0 | +0.0000 [+0.0000, +0.0000] | +0.00% [+0.00, +0.00] | 0/3; 0/9 |
| ecological_affine_vs_interaction_constant_k0 | -0.0260 [-0.0442, -0.0097] | +1.42% [+0.52, +2.45] | 3/3; 8/9 |
| global_bias_vs_interaction_constant_k5 | +0.0026 [-0.0001, +0.0054] | -0.16% [-0.34, +0.00] | 0/3; 0/9 |
| ecological_bias_vs_interaction_constant_k5 | +0.0010 [-0.0054, +0.0069] | -0.06% [-0.42, +0.35] | 1/3; 4/9 |
| global_affine_vs_interaction_constant_k5 | +0.0000 [+0.0000, +0.0000] | +0.00% [+0.00, +0.00] | 0/3; 0/9 |
| ecological_affine_vs_interaction_constant_k5 | +0.0002 [-0.0066, +0.0063] | -0.01% [-0.39, +0.43] | 2/3; 4/9 |
| global_bias_vs_interaction_gru_tuned_anchor_k0 | -0.0007 [-0.0076, +0.0051] | +0.04% [-0.28, +0.41] | 1/3; 2/9 |
| ecological_bias_vs_interaction_gru_tuned_anchor_k0 | -0.0155 [-0.0305, -0.0012] | +0.84% [+0.06, +1.71] | 3/3; 8/9 |
| global_affine_vs_interaction_gru_tuned_anchor_k0 | +0.0000 [+0.0000, +0.0000] | +0.00% [+0.00, +0.00] | 0/3; 0/9 |
| ecological_affine_vs_interaction_gru_tuned_anchor_k0 | -0.0260 [-0.0442, -0.0097] | +1.42% [+0.52, +2.45] | 3/3; 8/9 |
| global_bias_vs_interaction_gru_tuned_anchor_k5 | +0.0028 [+0.0002, +0.0054] | -0.18% [-0.34, -0.01] | 0/3; 0/9 |
| ecological_bias_vs_interaction_gru_tuned_anchor_k5 | +0.0024 [-0.0036, +0.0079] | -0.15% [-0.49, +0.24] | 1/3; 3/9 |
| global_affine_vs_interaction_gru_tuned_anchor_k5 | +0.0000 [+0.0000, +0.0000] | +0.00% [+0.00, +0.00] | 0/3; 0/9 |
| ecological_affine_vs_interaction_gru_tuned_anchor_k5 | +0.0056 [-0.0042, +0.0161] | -0.35% [-1.00, +0.28] | 1/3; 3/9 |

## Tail, ordinary concentrations and false alarms

| Saved model | K | Q90 MAE | Q90 bias | Non-tail MAE | Non-tail log MAE | False Q90 rate |
|---|---:|---:|---:|---:|---:|---:|
| interaction_gru_tuned_anchor | 0 | 7.4371 | -6.4393 | 1.1931 | 0.2505 | 1.95% |
| global_bias_gru_tuned_anchor | 0 | 7.4682 | -6.4927 | 1.1884 | 0.2493 | 1.92% |
| ecological_bias_gru_tuned_anchor | 0 | 7.5246 | -6.5785 | 1.1656 | 0.2428 | 1.88% |
| global_affine_gru_tuned_anchor | 0 | 7.4371 | -6.4393 | 1.1931 | 0.2505 | 1.95% |
| ecological_affine_gru_tuned_anchor | 0 | 7.4318 | -6.4542 | 1.1653 | 0.2423 | 1.99% |
| interaction_constant | 5 | 6.7277 | -5.2102 | 1.0030 | 0.2084 | 2.07% |
| interaction_gru_tuned_anchor | 5 | 6.6746 | -5.1066 | 0.9937 | 0.2051 | 2.14% |
| global_bias_constant | 5 | 6.7326 | -5.2109 | 1.0053 | 0.2089 | 2.06% |
| global_bias_gru_tuned_anchor | 5 | 6.6797 | -5.1074 | 0.9961 | 0.2057 | 2.13% |
| ecological_bias_constant | 5 | 6.7526 | -5.2477 | 1.0013 | 0.2073 | 2.04% |
| ecological_bias_gru_tuned_anchor | 5 | 6.6970 | -5.1419 | 0.9938 | 0.2045 | 2.14% |
| global_affine_constant | 5 | 6.7277 | -5.2102 | 1.0030 | 0.2084 | 2.07% |
| global_affine_gru_tuned_anchor | 5 | 6.6746 | -5.1066 | 0.9937 | 0.2051 | 2.14% |
| ecological_affine_constant | 5 | 6.7242 | -5.2153 | 1.0038 | 0.2076 | 2.10% |
| ecological_affine_gru_tuned_anchor | 5 | 6.6924 | -5.1519 | 0.9979 | 0.2055 | 2.15% |

False Q90 rate = P(prediction ≥ source Q90 | observation < source Q90).
All regional raw/log MAE and signed-bias intervals are retained in comparisons.csv.

## Source-validation choices

| Memory | Unchanged interaction | Mixed corrections | Memory replaces correction | Selected gamma counts | Neighbor counts | Ridge counts |
|---|---:|---:|---:|---|---|---|
| global_bias | 6/9 | 0/9 | 3/9 | 0: 6, 1: 3 | all source stations | 0.1: 3, 1: 6 |
| ecological_bias | 1/9 | 4/9 | 4/9 | 0: 1, 0.25: 2, 0.5: 2, 1: 4 | 20, 40, 80 | 0.1: 8, 1: 1 |
| global_affine | 9/9 | 0/9 | 0/9 | 0: 9 | all source stations | 1: 9 |
| ecological_affine | 1/9 | 6/9 | 2/9 | 0: 1, 0.25: 2, 0.5: 4, 1: 2 | 20, 80 | 0.1: 8, 1: 1 |

Donor profiles are still saved when gamma=0, but they do not contribute to prediction.

Global affine selects gamma=0 in all nine packages. It therefore equals the existing
interaction model at every K. Ecological-affine comparisons against global affine
and against interaction are the same numerical contrast, not independent evidence.

## Donor support on outer target stations

| Memory | Mean effective donors | Maximum donor weight | Global fallback station-fits | Mean nearest squared distance | Mean absolute K0 prediction change |
|---|---:|---:|---:|---:|---:|
| global_bias | 232.0 | 0.004 | 0/639 | not applicable | 0.0517 |
| ecological_bias | 50.8 | 0.050 | 6/639 | 0.0848 | 0.1606 |
| global_affine | 232.0 | 0.004 | 0/639 | not applicable | 0.0000 |
| ecological_affine | 28.6 | 0.050 | 6/639 | 0.0848 | 0.1850 |

Distances are mean squared differences in source-scaled ecology, undefined for global
donor pools. Donor summaries are station-fit summaries across seeds and partitions,
not independent station counts. Each donor station has uniform total weight regardless
of its number of observations. Saved diagnostics include donor observation counts,
maximum weight, ecological fallback, coefficient scales and target correction magnitude.

## Station gain and loss concentration

| Contrast with GRU support basis | Improved / worsened stations | Top-five share of positive gain | Top-five share of harm |
|---|---:|---:|---:|
| ecological_vs_global_affine_gru_tuned_anchor_k0 | 104 / 68 | 29.9% | 39.9% |
| ecological_affine_vs_bias_gru_tuned_anchor_k0 | 102 / 70 | 36.0% | 51.2% |
| ecological_affine_vs_interaction_gru_tuned_anchor_k0 | 104 / 68 | 29.9% | 39.9% |
| ecological_vs_global_affine_gru_tuned_anchor_k5 | 93 / 79 | 42.5% | 48.1% |
| ecological_affine_vs_bias_gru_tuned_anchor_k5 | 96 / 76 | 51.9% | 63.6% |
| ecological_affine_vs_interaction_gru_tuned_anchor_k5 | 93 / 79 | 42.5% | 48.1% |

Positive gains and harms have separate denominators. Contributions use equal
partition weights and combine repeated station identities.

## Interpretation of the main fixed contrasts

- **ecological_vs_global_affine_gru_tuned_anchor_k0:** overall MAE reduction +1.42% (95% CI [+0.52, +2.45]); 3/3 partitions and 8/9 fits improve. Q90 ΔMAE -0.0053 mg/L (95% CI [-0.0498, +0.0379]); non-tail ΔMAE -0.0278 mg/L (95% CI [-0.0482, -0.0107]). False Q90 rate changes by +0.039 percentage points (95% CI [-0.016, +0.110]).
- **ecological_affine_vs_interaction_gru_tuned_anchor_k0:** overall MAE reduction +1.42% (95% CI [+0.52, +2.45]); 3/3 partitions and 8/9 fits improve. Q90 ΔMAE -0.0053 mg/L (95% CI [-0.0498, +0.0379]); non-tail ΔMAE -0.0278 mg/L (95% CI [-0.0482, -0.0107]). False Q90 rate changes by +0.039 percentage points (95% CI [-0.016, +0.110]).
- **ecological_affine_vs_bias_gru_tuned_anchor_k0:** overall MAE reduction +0.58% (95% CI [+0.03, +1.20]); 2/3 partitions and 7/9 fits improve. Q90 ΔMAE -0.0929 mg/L (95% CI [-0.1458, -0.0477]); non-tail ΔMAE -0.0003 mg/L (95% CI [-0.0095, +0.0086]). False Q90 rate changes by +0.108 percentage points (95% CI [+0.027, +0.238]).
- **ecological_vs_global_affine_gru_tuned_anchor_k5:** overall MAE reduction -0.35% (95% CI [-1.00, +0.28]); 1/3 partitions and 3/9 fits improve. Q90 ΔMAE +0.0178 mg/L (95% CI [-0.0025, +0.0378]); non-tail ΔMAE +0.0043 mg/L (95% CI [-0.0065, +0.0156]). False Q90 rate changes by +0.008 percentage points (95% CI [-0.030, +0.049]).
- **ecological_affine_vs_interaction_gru_tuned_anchor_k5:** overall MAE reduction -0.35% (95% CI [-1.00, +0.28]); 1/3 partitions and 3/9 fits improve. Q90 ΔMAE +0.0178 mg/L (95% CI [-0.0025, +0.0378]); non-tail ΔMAE +0.0043 mg/L (95% CI [-0.0065, +0.0156]). False Q90 rate changes by +0.008 percentage points (95% CI [-0.030, +0.049]).
- **ecological_affine_vs_bias_gru_tuned_anchor_k5:** overall MAE reduction -0.20% (95% CI [-0.82, +0.35]); 0/3 partitions and 5/9 fits improve. Q90 ΔMAE -0.0047 mg/L (95% CI [-0.0196, +0.0113]); non-tail ΔMAE +0.0041 mg/L (95% CI [-0.0057, +0.0156]). False Q90 rate changes by +0.013 percentage points (95% CI [-0.017, +0.044]).

Ecological versus global affine tests the value of ecological donor selection within
the same residual family. Ecological affine versus ecological bias tests concentration
conditioning. Each arm selects its mixing weight on source validation, so these are
comparisons of complete fitting-and-selection procedures, not coefficients held at a
common mixing weight. Improvement over interaction alone cannot establish either
component without these controls. No physical process or causal interpretation is assigned.
