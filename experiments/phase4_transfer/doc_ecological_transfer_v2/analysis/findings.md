# Support-aware integration of ecological and temporal DOC residuals

This iteration reuses the four source-station memory profiles from v1 and the saved
temporal expert. It changes how their residual corrections are mixed after target
support becomes available. No ecological profile or neural weight is refitted.

For each K > 0, source-validation query MAE jointly chooses the mixing weight gamma
and the existing support adapter's alpha and ridge. Gamma=0 uses the temporal expert,
gamma=1 uses the ecological/global memory correction, and intermediate values mix them.
The K0 gamma is locked to v1. All K0 outputs are verified bitwise identical to v1;
their previously established gains are the same evidence, not new confirmation.

The comparison retains all ten current products and eight fixed-mixing references.
All use the same target queries and nested retrospective support sets. Target support
can follow the query in calendar time. Source validation and the three same-cohort
station partitions have been used during model development; outer outcomes do not
select a mode, K or hyperparameter in this analysis.

Intervals use 5,000 joint whole-station bootstrap draws. Repeated station identities
are resampled together across partitions; seed losses are averaged within partition,
then the three partitions receive equal weight. Positive reduction and negative
Delta MAE favor the candidate. Q90 thresholds come from source training observations.

## K curves with the existing GRU support basis

| Saved model | K0 MAE | K1 MAE | K3 MAE | K5 MAE | K5 RMSE | K5 R² | K5 log MAE |
|---|---:|---:|---:|---:|---:|---:|---:|
| interaction_gru_tuned_anchor | 1.8352 | 1.8048 | 1.6269 | 1.5816 | 3.579 | 0.569 | 0.2296 |
| global_bias_gru_tuned_anchor | 1.8346 | 1.8050 | 1.6252 | 1.5804 | 3.582 | 0.569 | 0.2280 |
| ecological_bias_gru_tuned_anchor | 1.8198 | 1.7983 | 1.6235 | 1.5798 | 3.583 | 0.569 | 0.2282 |
| global_affine_gru_tuned_anchor | 1.8352 | 1.8048 | 1.6256 | 1.5806 | 3.582 | 0.569 | 0.2282 |
| ecological_affine_gru_tuned_anchor | 1.8092 | 1.7917 | 1.6221 | 1.5799 | 3.581 | 0.569 | 0.2283 |
| fixed_global_bias_gru_tuned_anchor | 1.8346 | 1.8053 | 1.6280 | 1.5844 | 3.581 | 0.569 | 0.2302 |
| fixed_ecological_bias_gru_tuned_anchor | 1.8198 | 1.7955 | 1.6254 | 1.5840 | 3.587 | 0.567 | 0.2293 |
| fixed_global_affine_gru_tuned_anchor | 1.8352 | 1.8048 | 1.6269 | 1.5816 | 3.579 | 0.569 | 0.2296 |
| fixed_ecological_affine_gru_tuned_anchor | 1.8092 | 1.7880 | 1.6216 | 1.5872 | 3.590 | 0.567 | 0.2302 |

Full metrics, including constant-only adaptation, are in k_curves.csv.

## Joint versus fixed mixing at K5

| Comparison | Delta MAE [95% CI] | Reduction [95% CI] | Improved partitions; fits |
|---|---:|---:|---:|
| global_bias_dynamic_vs_fixed_constant_k5 | -0.0039 [-0.0083, +0.0007] | +0.25% [-0.04, +0.55] | 2/3; 5/9 |
| ecological_bias_dynamic_vs_fixed_constant_k5 | -0.0032 [-0.0077, +0.0016] | +0.20% [-0.10, +0.48] | 2/3; 5/9 |
| global_affine_dynamic_vs_fixed_constant_k5 | -0.0010 [-0.0049, +0.0031] | +0.06% [-0.19, +0.32] | 1/3; 2/9 |
| ecological_affine_dynamic_vs_fixed_constant_k5 | -0.0022 [-0.0065, +0.0024] | +0.14% [-0.15, +0.42] | 1/3; 5/9 |
| global_bias_dynamic_vs_fixed_gru_tuned_anchor_k5 | -0.0040 [-0.0082, +0.0005] | +0.25% [-0.03, +0.55] | 2/3; 5/9 |
| ecological_bias_dynamic_vs_fixed_gru_tuned_anchor_k5 | -0.0042 [-0.0082, +0.0000] | +0.27% [-0.00, +0.52] | 2/3; 5/9 |
| global_affine_dynamic_vs_fixed_gru_tuned_anchor_k5 | -0.0010 [-0.0048, +0.0030] | +0.06% [-0.18, +0.32] | 1/3; 2/9 |
| ecological_affine_dynamic_vs_fixed_gru_tuned_anchor_k5 | -0.0073 [-0.0162, +0.0006] | +0.46% [-0.03, +1.01] | 2/3; 5/9 |

## Ecological integration versus the existing interaction expert

| Comparison | Delta MAE [95% CI] | Reduction [95% CI] | Improved partitions; fits |
|---|---:|---:|---:|
| ecological_bias_vs_interaction_constant_k1 | -0.0065 [-0.0191, +0.0052] | +0.36% [-0.29, +1.07] | 2/3; 6/9 |
| ecological_affine_vs_interaction_constant_k1 | -0.0131 [-0.0269, -0.0006] | +0.73% [+0.03, +1.51] | 3/3; 7/9 |
| ecological_bias_vs_interaction_constant_k3 | +0.0067 [-0.0022, +0.0176] | -0.41% [-1.04, +0.14] | 2/3; 4/9 |
| ecological_affine_vs_interaction_constant_k3 | +0.0063 [-0.0045, +0.0188] | -0.39% [-1.12, +0.28] | 1/3; 4/9 |
| ecological_bias_vs_interaction_constant_k5 | -0.0022 [-0.0073, +0.0032] | +0.14% [-0.19, +0.48] | 2/3; 4/9 |
| ecological_affine_vs_interaction_constant_k5 | -0.0020 [-0.0072, +0.0035] | +0.13% [-0.21, +0.47] | 2/3; 4/9 |
| ecological_bias_vs_interaction_gru_tuned_anchor_k1 | -0.0065 [-0.0191, +0.0052] | +0.36% [-0.29, +1.07] | 2/3; 6/9 |
| ecological_affine_vs_interaction_gru_tuned_anchor_k1 | -0.0131 [-0.0269, -0.0006] | +0.73% [+0.03, +1.51] | 3/3; 7/9 |
| ecological_bias_vs_interaction_gru_tuned_anchor_k3 | -0.0034 [-0.0070, -0.0000] | +0.21% [+0.00, +0.46] | 3/3; 5/9 |
| ecological_affine_vs_interaction_gru_tuned_anchor_k3 | -0.0049 [-0.0089, -0.0011] | +0.30% [+0.07, +0.57] | 3/3; 7/9 |
| ecological_bias_vs_interaction_gru_tuned_anchor_k5 | -0.0018 [-0.0057, +0.0023] | +0.12% [-0.14, +0.38] | 2/3; 4/9 |
| ecological_affine_vs_interaction_gru_tuned_anchor_k5 | -0.0017 [-0.0057, +0.0025] | +0.11% [-0.15, +0.37] | 2/3; 4/9 |

## Affine versus bias memory

| Comparison | Delta MAE [95% CI] | Reduction [95% CI] | Improved partitions; fits |
|---|---:|---:|---:|
| ecological_affine_vs_bias_constant_k5 | +0.0002 [-0.0010, +0.0013] | -0.01% [-0.09, +0.06] | 1/3; 3/9 |
| ecological_affine_vs_bias_gru_tuned_anchor_k5 | +0.0001 [-0.0008, +0.0010] | -0.01% [-0.06, +0.05] | 1/3; 3/9 |

## Ecological versus global affine memory

| Comparison | Delta MAE [95% CI] | Reduction [95% CI] | Improved partitions; fits |
|---|---:|---:|---:|
| ecological_vs_global_affine_constant_k5 | -0.0010 [-0.0033, +0.0011] | +0.06% [-0.07, +0.21] | 2/3; 4/9 |
| ecological_vs_global_affine_gru_tuned_anchor_k5 | -0.0007 [-0.0025, +0.0009] | +0.05% [-0.06, +0.16] | 2/3; 5/9 |

## Tail, ordinary concentrations and false alarms

| Saved model | K | Q90 MAE | Q90 bias | Non-tail MAE | Non-tail log MAE | False Q90 rate |
|---|---:|---:|---:|---:|---:|---:|
| interaction_gru_tuned_anchor | 0 | 7.4371 | -6.4393 | 1.1931 | 0.2505 | 1.95% |
| interaction_gru_tuned_anchor | 5 | 6.6746 | -5.1066 | 0.9937 | 0.2051 | 2.14% |
| global_bias_gru_tuned_anchor | 0 | 7.4682 | -6.4927 | 1.1884 | 0.2493 | 1.92% |
| global_bias_gru_tuned_anchor | 5 | 6.6847 | -5.1430 | 0.9913 | 0.2032 | 2.14% |
| ecological_bias_gru_tuned_anchor | 0 | 7.5246 | -6.5785 | 1.1656 | 0.2428 | 1.88% |
| ecological_bias_gru_tuned_anchor | 5 | 6.6884 | -5.1442 | 0.9902 | 0.2033 | 2.13% |
| global_affine_gru_tuned_anchor | 0 | 7.4371 | -6.4393 | 1.1931 | 0.2505 | 1.95% |
| global_affine_gru_tuned_anchor | 5 | 6.6838 | -5.1394 | 0.9917 | 0.2034 | 2.14% |
| ecological_affine_gru_tuned_anchor | 0 | 7.4318 | -6.4542 | 1.1653 | 0.2423 | 1.99% |
| ecological_affine_gru_tuned_anchor | 5 | 6.6808 | -5.1324 | 0.9913 | 0.2035 | 2.14% |
| fixed_global_bias_gru_tuned_anchor | 0 | 7.4682 | -6.4927 | 1.1884 | 0.2493 | 1.92% |
| fixed_global_bias_gru_tuned_anchor | 5 | 6.6797 | -5.1074 | 0.9961 | 0.2057 | 2.13% |
| fixed_ecological_bias_gru_tuned_anchor | 0 | 7.5246 | -6.5785 | 1.1656 | 0.2428 | 1.88% |
| fixed_ecological_bias_gru_tuned_anchor | 5 | 6.6970 | -5.1419 | 0.9938 | 0.2045 | 2.14% |
| fixed_global_affine_gru_tuned_anchor | 0 | 7.4371 | -6.4393 | 1.1931 | 0.2505 | 1.95% |
| fixed_global_affine_gru_tuned_anchor | 5 | 6.6746 | -5.1066 | 0.9937 | 0.2051 | 2.14% |
| fixed_ecological_affine_gru_tuned_anchor | 0 | 7.4318 | -6.4542 | 1.1653 | 0.2423 | 1.99% |
| fixed_ecological_affine_gru_tuned_anchor | 5 | 6.6924 | -5.1519 | 0.9979 | 0.2055 | 2.15% |

Bias is predicted minus observed DOC. False Q90 rate is the fraction of non-tail
observations predicted at or above the source Q90 threshold. Regional intervals for
raw/log MAE, bias and false alarms are retained for every contrast in comparisons.csv.

## Source-validation selections

| Mode | Support adapter | K | Gamma counts | Changed from K0 | Mean validation MAE change vs fixed gamma |
|---|---|---:|---|---:|---:|
| global_bias | constant | 0 | 0: 6, 1: 3 | 0/9 | +0.00000 |
| global_bias | constant | 1 | 0: 6, 0.5: 1, 1: 2 | 1/9 | -0.00005 |
| global_bias | constant | 3 | 0: 8, 0.25: 1 | 4/9 | -0.00416 |
| global_bias | constant | 5 | 0: 6, 0.25: 3 | 6/9 | -0.00553 |
| global_bias | gru_tuned_anchor | 0 | 0: 6, 1: 3 | 0/9 | +0.00000 |
| global_bias | gru_tuned_anchor | 1 | 0: 6, 0.5: 1, 1: 2 | 1/9 | -0.00005 |
| global_bias | gru_tuned_anchor | 3 | 0: 8, 0.25: 1 | 4/9 | -0.00344 |
| global_bias | gru_tuned_anchor | 5 | 0: 6, 0.25: 3 | 6/9 | -0.00426 |
| ecological_bias | constant | 0 | 0: 1, 0.25: 2, 0.5: 2, 1: 4 | 0/9 | +0.00000 |
| ecological_bias | constant | 1 | 0: 3, 0.5: 3, 1: 3 | 3/9 | -0.00129 |
| ecological_bias | constant | 3 | 0: 4, 0.25: 3, 0.5: 2 | 5/9 | -0.00419 |
| ecological_bias | constant | 5 | 0: 4, 0.25: 4, 0.5: 1 | 8/9 | -0.00548 |
| ecological_bias | gru_tuned_anchor | 0 | 0: 1, 0.25: 2, 0.5: 2, 1: 4 | 0/9 | +0.00000 |
| ecological_bias | gru_tuned_anchor | 1 | 0: 3, 0.5: 3, 1: 3 | 3/9 | -0.00129 |
| ecological_bias | gru_tuned_anchor | 3 | 0: 4, 0.25: 4, 0.5: 1 | 5/9 | -0.00373 |
| ecological_bias | gru_tuned_anchor | 5 | 0: 4, 0.25: 5 | 7/9 | -0.00487 |
| global_affine | constant | 0 | 0: 9 | 0/9 | +0.00000 |
| global_affine | constant | 1 | 0: 9 | 0/9 | +0.00000 |
| global_affine | constant | 3 | 0: 7, 0.25: 2 | 2/9 | -0.00030 |
| global_affine | constant | 5 | 0: 6, 0.25: 3 | 3/9 | -0.00131 |
| global_affine | gru_tuned_anchor | 0 | 0: 9 | 0/9 | +0.00000 |
| global_affine | gru_tuned_anchor | 1 | 0: 9 | 0/9 | +0.00000 |
| global_affine | gru_tuned_anchor | 3 | 0: 8, 0.25: 1 | 1/9 | -0.00018 |
| global_affine | gru_tuned_anchor | 5 | 0: 6, 0.25: 3 | 3/9 | -0.00080 |
| ecological_affine | constant | 0 | 0: 1, 0.25: 2, 0.5: 4, 1: 2 | 0/9 | +0.00000 |
| ecological_affine | constant | 1 | 0: 2, 0.25: 2, 0.5: 4, 1: 1 | 3/9 | -0.00104 |
| ecological_affine | constant | 3 | 0: 1, 0.25: 5, 0.5: 3 | 4/9 | -0.00254 |
| ecological_affine | constant | 5 | 0: 3, 0.25: 5, 0.5: 1 | 8/9 | -0.00410 |
| ecological_affine | gru_tuned_anchor | 0 | 0: 1, 0.25: 2, 0.5: 4, 1: 2 | 0/9 | +0.00000 |
| ecological_affine | gru_tuned_anchor | 1 | 0: 2, 0.25: 2, 0.5: 4, 1: 1 | 3/9 | -0.00104 |
| ecological_affine | gru_tuned_anchor | 3 | 0: 2, 0.25: 5, 0.5: 2 | 4/9 | -0.00228 |
| ecological_affine | gru_tuned_anchor | 5 | 0: 3, 0.25: 6 | 7/9 | -0.00364 |

K0 selection is locked. At positive K, fixed gamma remains an available candidate.
Validation MAE therefore cannot worsen when the candidate set expands, but this does
not ensure improved target-station performance. Alpha/ridge choices and all gamma
scores are saved separately. Gamma=0 target predictions are checked against the
unchanged interaction support adapter.

## Reused ecological profiles

| Memory | Locked K0 gamma counts | Neighbor counts | Memory ridge counts |
|---|---|---|---|
| global_bias | 0: 6, 1: 3 | all source stations | 0.1: 3, 1: 6 |
| ecological_bias | 0: 1, 0.25: 2, 0.5: 2, 1: 4 | 20, 40, 80 | 0.1: 8, 1: 1 |
| global_affine | 0: 9 | all source stations | 1: 9 |
| ecological_affine | 0: 1, 0.25: 2, 0.5: 4, 1: 2 | 20, 80 | 0.1: 8, 1: 1 |

The bound v1 memory states retain the original source-only ecological scaler, donor
IDs, equal station weights, distances, missing-ecology fallback and coefficient arrays.
These profiles are reused intact, including profiles whose locked K0 gamma was zero.
Their source fitting and donor diagnostics remain documented in the v1 analysis.

## Station gain and loss concentration

| GRU-basis K5 contrast | Improved / worsened stations | Top-five positive gain | Top-five harm |
|---|---:|---:|---:|
| ecological_affine_dynamic_vs_fixed_gru_tuned_anchor_k5 | 99 / 73 | 45.6% | 56.1% |
| ecological_bias_vs_interaction_gru_tuned_anchor_k5 | 74 / 54 | 48.6% | 58.2% |
| ecological_affine_vs_interaction_gru_tuned_anchor_k5 | 102 / 70 | 42.5% | 55.5% |

Positive gain and harm have separate denominators; station contributions use equal
partition weights and combine repeated station identities.

## Main fixed K5 contrasts

- **ecological_affine_dynamic_vs_fixed_gru_tuned_anchor_k5:** MAE reduction +0.46% (95% CI [-0.03, +1.01]); 2/3 partitions and 5/9 fits improve. Q90 Delta MAE -0.0116 mg/L [-0.0282, +0.0052]; non-tail Delta MAE -0.0066 mg/L [-0.0167, +0.0022]. False Q90 rate change -0.014 percentage points [-0.046, +0.017].
- **ecological_affine_vs_interaction_gru_tuned_anchor_k5:** MAE reduction +0.11% (95% CI [-0.15, +0.37]); 2/3 partitions and 4/9 fits improve. Q90 Delta MAE +0.0062 mg/L [-0.0021, +0.0164]; non-tail Delta MAE -0.0024 mg/L [-0.0066, +0.0020]. False Q90 rate change -0.007 percentage points [-0.028, +0.013].
- **ecological_bias_vs_interaction_gru_tuned_anchor_k5:** MAE reduction +0.12% (95% CI [-0.14, +0.38]); 2/3 partitions and 4/9 fits improve. Q90 Delta MAE +0.0138 mg/L [+0.0061, +0.0247]; non-tail Delta MAE -0.0034 mg/L [-0.0074, +0.0007]. False Q90 rate change -0.012 percentage points [-0.035, +0.010].

These comparisons assess how source-validation support selects between two saved
residual explanations. Improvements do not imply newly learned neural features or a
physical mechanism. The full fixed comparison set is reported without target-based
selection or automatic model promotion.
