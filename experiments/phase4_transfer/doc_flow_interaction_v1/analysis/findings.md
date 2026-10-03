# State-dependent discharge corrections for DOC: development results

Nine partition–seed packages contain 18 neural fits. Both new arms use tail weight 2,
a 30-epoch ceiling, patience 5, and the original GRU/decay initialization. The same ten
causal discharge inputs enter a zero-initialized native-DOC head. An outer product of
the 64-dimensional recurrent state and three numerical anomaly/change features adds
192 interaction weights: readout = [h, f, flatten(h outer f_numeric)].

`interaction_tuned` trains recurrence, decay and the expanded head; `interaction_frozen`
trains the identical head while retaining original recurrence and decay exactly.
The primary architecture contrast is tuned interaction versus the saved additive
full-flow predictor, which also trained recurrence under the same objective and budget.
Tuned versus frozen interaction tests recurrent updating within the expanded readout.
Frozen interaction versus additive changes both readout and trainability and is a
performance comparison. Trainable parameter counts therefore differ between new arms.

The context forest, spatial encoder and frozen-v4 support basis remain unchanged.
The recurrent state is a learned representation; its interactions describe predictive
conditioning rather than graph transport, physical coefficients or causal effects.

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
| interaction_frozen_constant | 0 | 1.8524 | 4.008 | 0.467 | 0.2733 | 7.6236 |
| interaction_frozen_constant | 5 | 1.5993 | 3.620 | 0.560 | 0.2316 | 6.7665 |
| interaction_frozen_gru_tuned_anchor | 0 | 1.8524 | 4.008 | 0.467 | 0.2733 | 7.6236 |
| interaction_frozen_gru_tuned_anchor | 5 | 1.5846 | 3.592 | 0.567 | 0.2282 | 6.7079 |
| interaction_tuned_constant | 0 | 1.8352 | 3.958 | 0.479 | 0.2757 | 7.4371 |
| interaction_tuned_constant | 5 | 1.5956 | 3.607 | 0.563 | 0.2330 | 6.7277 |
| interaction_tuned_gru_tuned_anchor | 0 | 1.8352 | 3.958 | 0.479 | 0.2757 | 7.4371 |
| interaction_tuned_gru_tuned_anchor | 5 | 1.5816 | 3.579 | 0.569 | 0.2296 | 6.6746 |
| prior_flow_values_constant | 0 | 1.8386 | 3.971 | 0.475 | 0.2774 | 7.4889 |
| prior_flow_values_constant | 5 | 1.6031 | 3.624 | 0.559 | 0.2344 | 6.7633 |
| prior_flow_values_gru_tuned_anchor | 0 | 1.8386 | 3.971 | 0.475 | 0.2774 | 7.4889 |
| prior_flow_values_gru_tuned_anchor | 5 | 1.5839 | 3.586 | 0.568 | 0.2296 | 6.6845 |
| v4_fusion_gru_tuned_anchor | 0 | 1.9136 | 4.063 | 0.451 | 0.2853 | 7.7785 |
| v4_fusion_gru_tuned_anchor | 5 | 1.5952 | 3.588 | 0.568 | 0.2294 | 6.6886 |
| v4_fusion_tree_episodic | 0 | 1.9136 | 4.063 | 0.451 | 0.2853 | 7.7785 |
| v4_fusion_tree_episodic | 5 | 1.5952 | 3.599 | 0.566 | 0.2287 | 6.7115 |

At K0 the constant and shape adapters for each base produce exactly identical
predictions; duplicated K0 table rows are not independent evidence.

## Interaction readout versus the existing additive model

| Comparison | ΔMAE [95% CI] | Reduction [95% CI] | Improved partitions; fits |
|---|---:|---:|---:|
| interaction_frozen_vs_additive_constant_k0 | +0.0138 [-0.0100, +0.0414] | -0.75% [-2.06, +0.58] | 1/3; 6/9 |
| interaction_tuned_vs_additive_constant_k0 | -0.0034 [-0.0098, +0.0037] | +0.18% [-0.20, +0.53] | 1/3; 6/9 |
| interaction_frozen_vs_additive_constant_k5 | -0.0038 [-0.0114, +0.0034] | +0.24% [-0.21, +0.71] | 2/3; 8/9 |
| interaction_tuned_vs_additive_constant_k5 | -0.0075 [-0.0123, -0.0027] | +0.47% [+0.17, +0.73] | 3/3; 9/9 |
| interaction_frozen_vs_additive_gru_tuned_anchor_k0 | +0.0138 [-0.0100, +0.0414] | -0.75% [-2.06, +0.58] | 1/3; 6/9 |
| interaction_tuned_vs_additive_gru_tuned_anchor_k0 | -0.0034 [-0.0098, +0.0037] | +0.18% [-0.20, +0.53] | 1/3; 6/9 |
| interaction_frozen_vs_additive_gru_tuned_anchor_k5 | +0.0008 [-0.0105, +0.0130] | -0.05% [-0.80, +0.67] | 1/3; 7/9 |
| interaction_tuned_vs_additive_gru_tuned_anchor_k5 | -0.0023 [-0.0114, +0.0087] | +0.14% [-0.53, +0.71] | 2/3; 7/9 |

## Recurrent updating within the interaction model

| Comparison | ΔMAE [95% CI] | Reduction [95% CI] | Improved partitions; fits |
|---|---:|---:|---:|
| tuned_vs_frozen_constant_k0 | -0.0171 [-0.0434, +0.0046] | +0.92% [-0.27, +2.08] | 3/3; 8/9 |
| tuned_vs_frozen_constant_k5 | -0.0037 [-0.0122, +0.0044] | +0.23% [-0.29, +0.73] | 3/3; 9/9 |
| tuned_vs_frozen_gru_tuned_anchor_k0 | -0.0171 [-0.0434, +0.0046] | +0.92% [-0.27, +2.08] | 3/3; 8/9 |
| tuned_vs_frozen_gru_tuned_anchor_k5 | -0.0030 [-0.0113, +0.0050] | +0.19% [-0.32, +0.68] | 3/3; 8/9 |

## Comparison with the fixed environmental predictor

| Comparison | ΔMAE [95% CI] | Reduction [95% CI] | Improved partitions; fits |
|---|---:|---:|---:|
| interaction_frozen_vs_context_constant_k0 | -0.0505 [-0.0719, -0.0266] | +2.65% [+1.33, +3.94] | 3/3; 9/9 |
| interaction_tuned_vs_context_constant_k0 | -0.0676 [-0.1010, -0.0351] | +3.55% [+1.92, +5.01] | 3/3; 9/9 |
| interaction_frozen_vs_context_constant_k5 | -0.0101 [-0.0313, +0.0128] | +0.63% [-0.82, +1.94] | 2/3; 7/9 |
| interaction_tuned_vs_context_constant_k5 | -0.0138 [-0.0383, +0.0127] | +0.86% [-0.80, +2.33] | 2/3; 7/9 |
| interaction_frozen_vs_context_gru_tuned_anchor_k0 | -0.0505 [-0.0719, -0.0266] | +2.65% [+1.33, +3.94] | 3/3; 9/9 |
| interaction_tuned_vs_context_gru_tuned_anchor_k0 | -0.0676 [-0.1010, -0.0351] | +3.55% [+1.92, +5.01] | 3/3; 9/9 |
| interaction_frozen_vs_context_gru_tuned_anchor_k5 | -0.0114 [-0.0319, +0.0117] | +0.71% [-0.74, +1.99] | 2/3; 7/9 |
| interaction_tuned_vs_context_gru_tuned_anchor_k5 | -0.0144 [-0.0379, +0.0113] | +0.90% [-0.72, +2.35] | 2/3; 7/9 |

## Existing K5 fusion references

| Comparison | ΔMAE [95% CI] | Reduction [95% CI] | Improved partitions; fits |
|---|---:|---:|---:|
| interaction_frozen_constant_vs_v4_fusion_gru_tuned_anchor_k5 | +0.0041 [-0.0250, +0.0373] | -0.26% [-2.32, +1.58] | 2/3; 5/9 |
| interaction_frozen_constant_vs_v4_fusion_tree_episodic_k5 | +0.0041 [-0.0265, +0.0385] | -0.26% [-2.42, +1.64] | 1/3; 4/9 |
| interaction_tuned_constant_vs_v4_fusion_gru_tuned_anchor_k5 | +0.0004 [-0.0309, +0.0347] | -0.02% [-2.20, +1.93] | 2/3; 5/9 |
| interaction_tuned_constant_vs_v4_fusion_tree_episodic_k5 | +0.0004 [-0.0317, +0.0360] | -0.03% [-2.29, +1.95] | 1/3; 4/9 |
| interaction_frozen_gru_tuned_anchor_vs_v4_fusion_gru_tuned_anchor_k5 | -0.0106 [-0.0310, +0.0121] | +0.66% [-0.78, +1.94] | 2/3; 7/9 |
| interaction_frozen_gru_tuned_anchor_vs_v4_fusion_tree_episodic_k5 | -0.0106 [-0.0342, +0.0154] | +0.66% [-0.97, +2.11] | 2/3; 6/9 |
| interaction_tuned_gru_tuned_anchor_vs_v4_fusion_gru_tuned_anchor_k5 | -0.0136 [-0.0371, +0.0117] | +0.85% [-0.76, +2.29] | 2/3; 7/9 |
| interaction_tuned_gru_tuned_anchor_vs_v4_fusion_tree_episodic_k5 | -0.0136 [-0.0401, +0.0148] | +0.85% [-0.96, +2.42] | 2/3; 7/9 |

## Tail, ordinary concentrations and false alarms

| Saved model | K | Q90 MAE | Q90 bias | Non-tail MAE | Non-tail log MAE | False Q90 rate |
|---|---:|---:|---:|---:|---:|---:|
| context_gru_tuned_anchor | 0 | 7.5586 | -6.6062 | 1.2574 | 0.2614 | 1.91% |
| interaction_frozen_gru_tuned_anchor | 0 | 7.6236 | -6.7483 | 1.1937 | 0.2464 | 1.81% |
| interaction_tuned_gru_tuned_anchor | 0 | 7.4371 | -6.4393 | 1.1931 | 0.2505 | 1.95% |
| prior_flow_values_gru_tuned_anchor | 0 | 7.4889 | -6.5020 | 1.1909 | 0.2518 | 1.93% |
| v4_fusion_gru_tuned_anchor | 0 | 7.7785 | -7.1226 | 1.2406 | 0.2579 | 1.68% |
| v4_fusion_tree_episodic | 0 | 7.7785 | -7.1226 | 1.2406 | 0.2579 | 1.68% |
| context_constant | 5 | 6.7436 | -5.2589 | 1.0177 | 0.2075 | 2.15% |
| context_gru_tuned_anchor | 5 | 6.6945 | -5.1653 | 1.0084 | 0.2047 | 2.21% |
| interaction_frozen_constant | 5 | 6.7665 | -5.3179 | 1.0032 | 0.2065 | 2.07% |
| interaction_frozen_gru_tuned_anchor | 5 | 6.7079 | -5.2157 | 0.9937 | 0.2032 | 2.12% |
| interaction_tuned_constant | 5 | 6.7277 | -5.2102 | 1.0030 | 0.2084 | 2.07% |
| interaction_tuned_gru_tuned_anchor | 5 | 6.6746 | -5.1066 | 0.9937 | 0.2051 | 2.14% |
| prior_flow_values_constant | 5 | 6.7633 | -5.2632 | 1.0073 | 0.2096 | 2.05% |
| prior_flow_values_gru_tuned_anchor | 5 | 6.6845 | -5.1169 | 0.9953 | 0.2050 | 2.16% |
| v4_fusion_gru_tuned_anchor | 5 | 6.6886 | -5.1363 | 1.0080 | 0.2047 | 2.21% |
| v4_fusion_tree_episodic | 5 | 6.7115 | -5.1581 | 1.0056 | 0.2037 | 2.22% |

False Q90 rate = P(prediction ≥ source Q90 | observation < source Q90).
Raw/log MAE, signed-bias intervals and false-alarm rate differences appear in
`comparisons.csv`; regional sample counts and all K curves are saved separately.

## Station gain and loss concentration

| Contrast, GRU support basis | Stations improved / worsened | Top-five share of positive gain | Top-five share of harm |
|---|---:|---:|---:|
| tuned_vs_frozen_gru_tuned_anchor_k0 | 95 / 77 | 49.6% | 55.7% |
| interaction_frozen_vs_additive_gru_tuned_anchor_k0 | 89 / 83 | 46.1% | 42.3% |
| interaction_tuned_vs_additive_gru_tuned_anchor_k0 | 85 / 87 | 27.6% | 25.5% |
| tuned_vs_frozen_gru_tuned_anchor_k5 | 88 / 84 | 59.1% | 61.9% |
| interaction_frozen_vs_additive_gru_tuned_anchor_k5 | 84 / 88 | 39.0% | 44.3% |
| interaction_tuned_vs_additive_gru_tuned_anchor_k5 | 82 / 90 | 37.2% | 43.7% |

Station contributions retain equal partition weighting and combine repeated station
identities. Positive gain and harm have separate denominators; they do not imply
a physical process or an irreducible error floor.

## Source-validation selections

| Arm | Trainable parameters | Active fits | Context fallbacks | Selected scales (count) | Selected epochs |
|---|---:|---:|---:|---|---|
| interaction_frozen | 267 | 9/9 | 0/9 | 0.25: 1, 0.5: 2, 1: 6 | 30, 20, 20, 1, 2, 2, 2, 2, 1 |
| interaction_tuned | 25739 | 9/9 | 0/9 | 0.5: 3, 1: 6 | 29, 29, 30, 6, 2, 2, 2, 2, 1 |

Scale zero is an exact context fallback, not a learned neural gain.

## Interpretation of the fixed contrasts

- **tuned_vs_frozen_gru_tuned_anchor_k0:** overall MAE reduction +0.92% (95% CI [-0.27, +2.08]); 3/3 partitions, 8/9 fits improve. Q90 ΔMAE -0.1865 mg/L (95% CI [-0.3072, -0.0635]); Q90 improves in 2/3 partitions and 6/9 fits. Non-tail ΔMAE -0.0006 mg/L (95% CI [-0.0194, +0.0196]). False Q90 rate changes by +0.141 percentage points (95% CI [+0.028, +0.323]).
- **interaction_tuned_vs_additive_gru_tuned_anchor_k0:** overall MAE reduction +0.18% (95% CI [-0.20, +0.53]); 1/3 partitions, 6/9 fits improve. Q90 ΔMAE -0.0518 mg/L (95% CI [-0.0670, -0.0352]); Q90 improves in 3/3 partitions and 9/9 fits. Non-tail ΔMAE +0.0022 mg/L (95% CI [-0.0048, +0.0100]). False Q90 rate changes by +0.024 percentage points (95% CI [-0.008, +0.064]).
- **interaction_frozen_vs_additive_gru_tuned_anchor_k0:** overall MAE reduction -0.75% (95% CI [-2.06, +0.58]); 1/3 partitions, 6/9 fits improve. Q90 ΔMAE +0.1347 mg/L (95% CI [+0.0148, +0.2535]); Q90 improves in 2/3 partitions and 6/9 fits. Non-tail ΔMAE +0.0028 mg/L (95% CI [-0.0202, +0.0250]). False Q90 rate changes by -0.117 percentage points (95% CI [-0.292, -0.004]).
- **tuned_vs_frozen_gru_tuned_anchor_k5:** overall MAE reduction +0.19% (95% CI [-0.32, +0.68]); 3/3 partitions, 8/9 fits improve. Q90 ΔMAE -0.0334 mg/L (95% CI [-0.0559, -0.0097]); Q90 improves in 3/3 partitions and 9/9 fits. Non-tail ΔMAE +0.0000 mg/L (95% CI [-0.0087, +0.0084]). False Q90 rate changes by +0.027 percentage points (95% CI [+0.006, +0.061]).
- **interaction_tuned_vs_additive_gru_tuned_anchor_k5:** overall MAE reduction +0.14% (95% CI [-0.53, +0.71]); 2/3 partitions, 7/9 fits improve. Q90 ΔMAE -0.0099 mg/L (95% CI [-0.0333, +0.0182]); Q90 improves in 2/3 partitions and 8/9 fits. Non-tail ΔMAE -0.0016 mg/L (95% CI [-0.0115, +0.0105]). False Q90 rate changes by -0.015 percentage points (95% CI [-0.074, +0.032]).
- **interaction_frozen_vs_additive_gru_tuned_anchor_k5:** overall MAE reduction -0.05% (95% CI [-0.80, +0.67]); 1/3 partitions, 7/9 fits improve. Q90 ΔMAE +0.0235 mg/L (95% CI [-0.0044, +0.0563]); Q90 improves in 1/3 partitions and 5/9 fits. Non-tail ΔMAE -0.0016 mg/L (95% CI [-0.0139, +0.0124]). False Q90 rate changes by -0.042 percentage points (95% CI [-0.106, +0.006]).

All comparisons are descriptive development evidence from this fixed candidate set.
Tuned interaction versus additive tests the added state-dependent readout; tuned
versus frozen interaction tests recurrent updating. Read tail and non-tail changes
alongside overall error before interpreting a concentration-dependent benefit.
