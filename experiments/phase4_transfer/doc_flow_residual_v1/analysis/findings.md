# Explicit discharge dynamics in the DOC residual: development results

Nine partition–seed packages contain 18 neural fits. Both new arms use tail weight 2,
a 30-epoch ceiling, patience 5, and the same existing GRU, decay and zero-initialized
native-DOC head. The ten additional head inputs contain bounded causal discharge
anomaly/change values and their observation-support indicators.

`flow_freshness` zeros the three numerical anomaly/change columns while retaining
the seven age, count and validity columns. Both arms still receive discharge through
the existing spatial/GRU inputs. The direct comparison estimates the incremental value
of explicit numerical flow changes at the readout, not the value of all flow information.
Head dimensions and nominal parameter counts match; three columns are inactive in the
freshness control, so effective capacity is not identical.

Source-validation chooses checkpoint, scale and support-adapter parameters. Partitions
142–144 were previously evaluated, making this a development comparison. All saved
models remain visible and no winner or promotion is selected from this analysis.
K-shot support is retrospective; the complete adapted model is not prospective forecasting.

Intervals use 5,000 paired whole-station bootstrap draws, with shared station identities
resampled jointly across partitions. Seed losses are averaged within each partition and
partitions receive equal weight. Positive relative reduction and negative ΔMAE favor
the candidate. Q90 uses source-training labels; signed bias is prediction minus observation.

## Overall performance

| Saved model | K | MAE | RMSE | R² | Log MAE | Q90 MAE |
|---|---:|---:|---:|---:|---:|---:|
| context_constant | 0 | 1.9028 | 4.008 | 0.467 | 0.2859 | 7.5586 |
| context_constant | 5 | 1.6094 | 3.615 | 0.562 | 0.2323 | 6.7436 |
| context_gru_tuned_anchor | 0 | 1.9028 | 4.008 | 0.467 | 0.2859 | 7.5586 |
| context_gru_tuned_anchor | 5 | 1.5960 | 3.591 | 0.568 | 0.2294 | 6.6945 |
| flow_freshness_constant | 0 | 1.8508 | 3.980 | 0.473 | 0.2813 | 7.5170 |
| flow_freshness_constant | 5 | 1.6075 | 3.623 | 0.560 | 0.2353 | 6.7624 |
| flow_freshness_gru_tuned_anchor | 0 | 1.8508 | 3.980 | 0.473 | 0.2813 | 7.5170 |
| flow_freshness_gru_tuned_anchor | 5 | 1.5888 | 3.587 | 0.568 | 0.2308 | 6.6860 |
| flow_values_constant | 0 | 1.8386 | 3.971 | 0.475 | 0.2774 | 7.4889 |
| flow_values_constant | 5 | 1.6031 | 3.624 | 0.559 | 0.2344 | 6.7633 |
| flow_values_gru_tuned_anchor | 0 | 1.8386 | 3.971 | 0.475 | 0.2774 | 7.4889 |
| flow_values_gru_tuned_anchor | 5 | 1.5839 | 3.586 | 0.568 | 0.2296 | 6.6845 |
| prior_native_tail_constant | 0 | 1.8545 | 3.971 | 0.476 | 0.2811 | 7.4446 |
| prior_native_tail_constant | 5 | 1.6091 | 3.617 | 0.561 | 0.2357 | 6.7310 |
| prior_native_tail_gru_tuned_anchor | 0 | 1.8545 | 3.971 | 0.476 | 0.2811 | 7.4446 |
| prior_native_tail_gru_tuned_anchor | 5 | 1.5959 | 3.590 | 0.567 | 0.2323 | 6.6822 |
| v4_fusion_gru_tuned_anchor | 0 | 1.9136 | 4.063 | 0.451 | 0.2853 | 7.7785 |
| v4_fusion_gru_tuned_anchor | 5 | 1.5952 | 3.588 | 0.568 | 0.2294 | 6.6886 |
| v4_fusion_tree_episodic | 0 | 1.9136 | 4.063 | 0.451 | 0.2853 | 7.7785 |
| v4_fusion_tree_episodic | 5 | 1.5952 | 3.599 | 0.566 | 0.2287 | 6.7115 |

At K0 the constant and shape adapters for each base produce exactly identical
predictions; duplicated K0 table rows are not independent evidence.

## Primary: values versus freshness

| Comparison | ΔMAE [95% CI] | Reduction [95% CI] | Improved partitions; fits |
|---|---:|---:|---:|
| values_vs_freshness_constant_k0 | -0.0122 [-0.0188, -0.0060] | +0.66% [+0.31, +1.04] | 3/3; 9/9 |
| values_vs_freshness_constant_k5 | -0.0043 [-0.0148, +0.0069] | +0.27% [-0.43, +0.92] | 2/3; 8/9 |
| values_vs_freshness_gru_tuned_anchor_k0 | -0.0122 [-0.0188, -0.0060] | +0.66% [+0.31, +1.04] | 3/3; 9/9 |
| values_vs_freshness_gru_tuned_anchor_k5 | -0.0050 [-0.0148, +0.0061] | +0.31% [-0.38, +0.93] | 2/3; 8/9 |

## Increment beyond the previous tail-weighted residual

| Comparison | ΔMAE [95% CI] | Reduction [95% CI] | Improved partitions; fits |
|---|---:|---:|---:|
| flow_freshness_vs_prior_tail_constant_k0 | -0.0037 [-0.0248, +0.0152] | +0.20% [-0.83, +1.28] | 3/3; 7/9 |
| flow_values_vs_prior_tail_constant_k0 | -0.0159 [-0.0355, +0.0016] | +0.86% [-0.09, +1.87] | 3/3; 8/9 |
| flow_freshness_vs_prior_tail_constant_k5 | -0.0016 [-0.0068, +0.0034] | +0.10% [-0.21, +0.42] | 2/3; 7/9 |
| flow_values_vs_prior_tail_constant_k5 | -0.0060 [-0.0175, +0.0067] | +0.37% [-0.41, +1.09] | 2/3; 8/9 |
| flow_freshness_vs_prior_tail_gru_tuned_anchor_k0 | -0.0037 [-0.0248, +0.0152] | +0.20% [-0.83, +1.28] | 3/3; 7/9 |
| flow_values_vs_prior_tail_gru_tuned_anchor_k0 | -0.0159 [-0.0355, +0.0016] | +0.86% [-0.09, +1.87] | 3/3; 8/9 |
| flow_freshness_vs_prior_tail_gru_tuned_anchor_k5 | -0.0071 [-0.0176, +0.0015] | +0.44% [-0.09, +1.08] | 2/3; 7/9 |
| flow_values_vs_prior_tail_gru_tuned_anchor_k5 | -0.0121 [-0.0193, -0.0044] | +0.76% [+0.27, +1.21] | 3/3; 8/9 |

## Comparison with the fixed environmental predictor

| Comparison | ΔMAE [95% CI] | Reduction [95% CI] | Improved partitions; fits |
|---|---:|---:|---:|
| flow_freshness_vs_context_constant_k0 | -0.0520 [-0.0877, -0.0187] | +2.74% [+1.02, +4.35] | 3/3; 9/9 |
| flow_values_vs_context_constant_k0 | -0.0642 [-0.0987, -0.0311] | +3.38% [+1.69, +4.93] | 3/3; 9/9 |
| flow_freshness_vs_context_constant_k5 | -0.0019 [-0.0174, +0.0147] | +0.12% [-0.94, +1.08] | 2/3; 6/9 |
| flow_values_vs_context_constant_k5 | -0.0063 [-0.0290, +0.0184] | +0.39% [-1.18, +1.79] | 2/3; 7/9 |
| flow_freshness_vs_context_gru_tuned_anchor_k0 | -0.0520 [-0.0877, -0.0187] | +2.74% [+1.02, +4.35] | 3/3; 9/9 |
| flow_values_vs_context_gru_tuned_anchor_k0 | -0.0642 [-0.0987, -0.0311] | +3.38% [+1.69, +4.93] | 3/3; 9/9 |
| flow_freshness_vs_context_gru_tuned_anchor_k5 | -0.0072 [-0.0198, +0.0051] | +0.45% [-0.33, +1.19] | 3/3; 8/9 |
| flow_values_vs_context_gru_tuned_anchor_k5 | -0.0121 [-0.0284, +0.0051] | +0.76% [-0.33, +1.74] | 3/3; 8/9 |

## Existing K5 fusion references

| Comparison | ΔMAE [95% CI] | Reduction [95% CI] | Improved partitions; fits |
|---|---:|---:|---:|
| flow_freshness_constant_vs_v4_fusion_gru_tuned_anchor_k5 | +0.0123 [-0.0100, +0.0363] | -0.77% [-2.27, +0.62] | 0/3; 3/9 |
| flow_freshness_constant_vs_v4_fusion_tree_episodic_k5 | +0.0123 [-0.0113, +0.0380] | -0.77% [-2.36, +0.69] | 0/3; 2/9 |
| flow_values_constant_vs_v4_fusion_gru_tuned_anchor_k5 | +0.0079 [-0.0225, +0.0416] | -0.50% [-2.60, +1.41] | 1/3; 4/9 |
| flow_values_constant_vs_v4_fusion_tree_episodic_k5 | +0.0079 [-0.0236, +0.0429] | -0.50% [-2.69, +1.45] | 1/3; 3/9 |
| flow_freshness_gru_tuned_anchor_vs_v4_fusion_gru_tuned_anchor_k5 | -0.0064 [-0.0189, +0.0058] | +0.40% [-0.37, +1.15] | 3/3; 8/9 |
| flow_freshness_gru_tuned_anchor_vs_v4_fusion_tree_episodic_k5 | -0.0064 [-0.0229, +0.0085] | +0.40% [-0.56, +1.37] | 2/3; 6/9 |
| flow_values_gru_tuned_anchor_vs_v4_fusion_gru_tuned_anchor_k5 | -0.0113 [-0.0274, +0.0056] | +0.71% [-0.36, +1.68] | 3/3; 8/9 |
| flow_values_gru_tuned_anchor_vs_v4_fusion_tree_episodic_k5 | -0.0113 [-0.0311, +0.0085] | +0.71% [-0.55, +1.87] | 3/3; 6/9 |

## Tail, ordinary concentrations and false alarms

| Saved model | K | Q90 MAE | Q90 bias | Non-tail MAE | Non-tail log MAE | False Q90 rate |
|---|---:|---:|---:|---:|---:|---:|
| context_gru_tuned_anchor | 0 | 7.5586 | -6.6062 | 1.2574 | 0.2614 | 1.91% |
| flow_freshness_gru_tuned_anchor | 0 | 7.5170 | -6.5350 | 1.2014 | 0.2557 | 1.94% |
| flow_values_gru_tuned_anchor | 0 | 7.4889 | -6.5020 | 1.1909 | 0.2518 | 1.93% |
| prior_native_tail_gru_tuned_anchor | 0 | 7.4446 | -6.3882 | 1.2148 | 0.2565 | 1.98% |
| v4_fusion_gru_tuned_anchor | 0 | 7.7785 | -7.1226 | 1.2406 | 0.2579 | 1.68% |
| v4_fusion_tree_episodic | 0 | 7.7785 | -7.1226 | 1.2406 | 0.2579 | 1.68% |
| context_constant | 5 | 6.7436 | -5.2589 | 1.0177 | 0.2075 | 2.15% |
| context_gru_tuned_anchor | 5 | 6.6945 | -5.1653 | 1.0084 | 0.2047 | 2.21% |
| flow_freshness_constant | 5 | 6.7624 | -5.2623 | 1.0126 | 0.2105 | 2.09% |
| flow_freshness_gru_tuned_anchor | 5 | 6.6860 | -5.1152 | 1.0010 | 0.2062 | 2.19% |
| flow_values_constant | 5 | 6.7633 | -5.2632 | 1.0073 | 0.2096 | 2.05% |
| flow_values_gru_tuned_anchor | 5 | 6.6845 | -5.1169 | 0.9953 | 0.2050 | 2.16% |
| prior_native_tail_constant | 5 | 6.7310 | -5.1925 | 1.0184 | 0.2114 | 2.14% |
| prior_native_tail_gru_tuned_anchor | 5 | 6.6822 | -5.0917 | 1.0095 | 0.2080 | 2.20% |
| v4_fusion_gru_tuned_anchor | 5 | 6.6886 | -5.1363 | 1.0080 | 0.2047 | 2.21% |
| v4_fusion_tree_episodic | 5 | 6.7115 | -5.1581 | 1.0056 | 0.2037 | 2.22% |

False Q90 rate = P(prediction ≥ source Q90 | observation < source Q90).
Raw/log MAE, signed-bias intervals and false-alarm rate differences appear in
`comparisons.csv`; regional sample counts and all K curves are saved separately.

## Station gain and loss concentration

| Contrast, GRU support basis | Stations improved / worsened | Top-five share of positive gain | Top-five share of harm |
|---|---:|---:|---:|
| values_vs_freshness_gru_tuned_anchor_k0 | 109 / 63 | 47.1% | 43.7% |
| flow_freshness_vs_prior_tail_gru_tuned_anchor_k0 | 89 / 83 | 51.3% | 47.4% |
| flow_values_vs_prior_tail_gru_tuned_anchor_k0 | 104 / 68 | 40.9% | 47.5% |
| values_vs_freshness_gru_tuned_anchor_k5 | 93 / 79 | 43.1% | 50.3% |
| flow_freshness_vs_prior_tail_gru_tuned_anchor_k5 | 94 / 78 | 66.1% | 55.2% |
| flow_values_vs_prior_tail_gru_tuned_anchor_k5 | 100 / 72 | 37.1% | 38.9% |

Station contributions retain equal partition weighting and combine repeated station
identities. Positive gain and harm have separate denominators; they do not imply
a physical process or an irreducible error floor.

## Source-validation selections

| Arm | Active fits | Context fallbacks | Selected scales (count) | Selected epochs |
|---|---:|---:|---|---|
| flow_freshness | 9/9 | 0/9 | 0.5: 3, 1: 6 | 27, 27, 30, 1, 1, 1, 1, 2, 1 |
| flow_values | 9/9 | 0/9 | 0.5: 3, 1: 6 | 27, 29, 30, 1, 1, 1, 2, 2, 1 |

Scale zero is an exact context fallback, not a learned neural gain.

## Interpretation of the fixed contrasts

- **values_vs_freshness_gru_tuned_anchor_k0:** overall MAE reduction +0.66% (95% CI [+0.31, +1.04]); 3/3 partitions, 9/9 fits improve. Q90 ΔMAE -0.0281 mg/L (95% CI [-0.0435, -0.0173]); non-tail ΔMAE -0.0105 mg/L (95% CI [-0.0176, -0.0038]). False Q90 rate changes by -0.009 percentage points (95% CI [-0.036, +0.015]).
- **flow_values_vs_prior_tail_gru_tuned_anchor_k0:** overall MAE reduction +0.86% (95% CI [-0.09, +1.87]); 3/3 partitions, 8/9 fits improve. Q90 ΔMAE +0.0443 mg/L (95% CI [+0.0090, +0.0793]); non-tail ΔMAE -0.0239 mg/L (95% CI [-0.0458, -0.0052]). False Q90 rate changes by -0.052 percentage points (95% CI [-0.105, -0.013]).
- **flow_freshness_vs_prior_tail_gru_tuned_anchor_k0:** overall MAE reduction +0.20% (95% CI [-0.83, +1.28]); 3/3 partitions, 7/9 fits improve. Q90 ΔMAE +0.0724 mg/L (95% CI [+0.0395, +0.1060]); non-tail ΔMAE -0.0134 mg/L (95% CI [-0.0380, +0.0070]). False Q90 rate changes by -0.043 percentage points (95% CI [-0.094, -0.006]).
- **values_vs_freshness_gru_tuned_anchor_k5:** overall MAE reduction +0.31% (95% CI [-0.38, +0.93]); 2/3 partitions, 8/9 fits improve. Q90 ΔMAE -0.0015 mg/L (95% CI [-0.0323, +0.0283]); non-tail ΔMAE -0.0057 mg/L (95% CI [-0.0163, +0.0066]). False Q90 rate changes by -0.030 percentage points (95% CI [-0.082, +0.008]).
- **flow_values_vs_prior_tail_gru_tuned_anchor_k5:** overall MAE reduction +0.76% (95% CI [+0.27, +1.21]); 3/3 partitions, 8/9 fits improve. Q90 ΔMAE +0.0022 mg/L (95% CI [-0.0252, +0.0264]); non-tail ΔMAE -0.0142 mg/L (95% CI [-0.0217, -0.0063]). False Q90 rate changes by -0.046 percentage points (95% CI [-0.106, -0.003]).
- **flow_freshness_vs_prior_tail_gru_tuned_anchor_k5:** overall MAE reduction +0.44% (95% CI [-0.09, +1.08]); 2/3 partitions, 7/9 fits improve. Q90 ΔMAE +0.0038 mg/L (95% CI [-0.0103, +0.0184]); non-tail ΔMAE -0.0085 mg/L (95% CI [-0.0208, +0.0009]). False Q90 rate changes by -0.016 percentage points (95% CI [-0.034, -0.003]).

All comparisons are descriptive development evidence from this fixed candidate set.
The values-versus-freshness contrast isolates the added numerical readout; comparisons
with the prior residual establish whether this change improves the existing model.
