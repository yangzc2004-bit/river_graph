# Native-DOC temporal residual: development results

Nine completed partition–seed packages (18 neural fits) compare ordinary native-scale MAE
training with doubled weight on
source-training Q90 observations. The environmental context predictor is fixed; the
existing GRU and decay plus a zero-initialized scalar head learn the native-DOC correction.
Each base has constant and frozen-v4-GRU support adapters. All saved arms are reported.

These are previously evaluated station partitions 142–144 within the same cohort.
This development analysis does not select or automatically promote a model.
Source-validation chooses epochs, residual scale, and support adaptation; outer query labels
are used here only for evaluation. Target support is retrospective and identical across arms.

Intervals use 5,000 paired whole-station bootstrap draws with station IDs resampled
jointly across partitions. Individual-seed losses are averaged within each partition,
then partitions receive equal weight. Bias is prediction minus observation; negative bias
means underprediction. Q90 is determined only from each partition's source training labels.

## Predictions at K = 0 and K = 5

| Saved model | K | MAE | RMSE | R² | Log MAE | Q90 MAE |
|---|---:|---:|---:|---:|---:|---:|
| context_constant | 0 | 1.9028 | 4.008 | 0.467 | 0.2859 | 7.559 |
| context_constant | 5 | 1.6094 | 3.615 | 0.562 | 0.2323 | 6.744 |
| context_gru_tuned_anchor | 0 | 1.9028 | 4.008 | 0.467 | 0.2859 | 7.559 |
| context_gru_tuned_anchor | 5 | 1.5960 | 3.591 | 0.568 | 0.2294 | 6.694 |
| native_mae_constant | 0 | 1.8625 | 4.037 | 0.459 | 0.2777 | 7.737 |
| native_mae_constant | 5 | 1.6086 | 3.629 | 0.558 | 0.2337 | 6.796 |
| native_mae_gru_tuned_anchor | 0 | 1.8625 | 4.037 | 0.459 | 0.2777 | 7.737 |
| native_mae_gru_tuned_anchor | 5 | 1.5999 | 3.610 | 0.563 | 0.2318 | 6.759 |
| native_tail_constant | 0 | 1.8545 | 3.971 | 0.476 | 0.2811 | 7.445 |
| native_tail_constant | 5 | 1.6091 | 3.617 | 0.561 | 0.2357 | 6.731 |
| native_tail_gru_tuned_anchor | 0 | 1.8545 | 3.971 | 0.476 | 0.2811 | 7.445 |
| native_tail_gru_tuned_anchor | 5 | 1.5959 | 3.590 | 0.567 | 0.2323 | 6.682 |
| v4_fusion_gru_tuned_anchor | 0 | 1.9136 | 4.063 | 0.451 | 0.2853 | 7.779 |
| v4_fusion_gru_tuned_anchor | 5 | 1.5952 | 3.588 | 0.568 | 0.2294 | 6.689 |
| v4_fusion_tree_episodic | 0 | 1.9136 | 4.063 | 0.451 | 0.2853 | 7.779 |
| v4_fusion_tree_episodic | 5 | 1.5952 | 3.599 | 0.566 | 0.2287 | 6.712 |

At K = 0, the two support adapters for a given base are exactly identical.
Their repeated rows describe the same prediction, not independent evidence.

## Correction beyond the environmental predictor

| Comparison | ΔMAE [95% CI] | MAE reduction [95% CI] | Improved partitions; fits |
|---|---:|---:|---:|
| native_mae_vs_context_constant_k0 | -0.0404 [-0.0711, -0.0113] | 2.12% [0.58, 3.79] | 3/3; 9/9 |
| native_mae_vs_context_constant_k5 | -0.0008 [-0.0137, 0.0125] | 0.05% [-0.79, 0.86] | 1/3; 5/9 |
| native_mae_vs_context_gru_tuned_anchor_k0 | -0.0404 [-0.0711, -0.0113] | 2.12% [0.58, 3.79] | 3/3; 9/9 |
| native_mae_vs_context_gru_tuned_anchor_k5 | 0.0039 [-0.0150, 0.0246] | -0.25% [-1.55, 0.94] | 1/3; 4/9 |
| native_tail_vs_context_constant_k0 | -0.0483 [-0.0767, -0.0216] | 2.54% [1.17, 3.89] | 3/3; 9/9 |
| native_tail_vs_context_constant_k5 | -0.0003 [-0.0151, 0.0156] | 0.02% [-0.99, 0.94] | 2/3; 6/9 |
| native_tail_vs_context_gru_tuned_anchor_k0 | -0.0483 [-0.0767, -0.0216] | 2.54% [1.17, 3.89] | 3/3; 9/9 |
| native_tail_vs_context_gru_tuned_anchor_k5 | -0.0001 [-0.0140, 0.0147] | 0.00% [-0.95, 0.85] | 2/3; 7/9 |

## Tail-weighted versus ordinary training

| Comparison | ΔMAE [95% CI] | MAE reduction [95% CI] | Improved partitions; fits |
|---|---:|---:|---:|
| tail_vs_ordinary_constant_k0 | -0.0080 [-0.0363, 0.0174] | 0.43% [-0.98, 1.77] | 2/3; 3/9 |
| tail_vs_ordinary_constant_k5 | 0.0005 [-0.0073, 0.0081] | -0.03% [-0.52, 0.43] | 2/3; 5/9 |
| tail_vs_ordinary_gru_tuned_anchor_k0 | -0.0080 [-0.0363, 0.0174] | 0.43% [-0.98, 1.77] | 2/3; 3/9 |
| tail_vs_ordinary_gru_tuned_anchor_k5 | -0.0040 [-0.0163, 0.0075] | 0.25% [-0.48, 0.98] | 3/3; 6/9 |

## Comparison with the existing fusion references

| Comparison | ΔMAE [95% CI] | MAE reduction [95% CI] | Improved partitions; fits |
|---|---:|---:|---:|
| native_mae_constant_vs_v4_fusion_gru_tuned_anchor_k5 | 0.0134 [-0.0081, 0.0368] | -0.84% [-2.26, 0.52] | 0/3; 0/9 |
| native_mae_constant_vs_v4_fusion_tree_episodic_k5 | 0.0134 [-0.0086, 0.0376] | -0.84% [-2.33, 0.54] | 0/3; 2/9 |
| native_mae_gru_tuned_anchor_vs_v4_fusion_gru_tuned_anchor_k5 | 0.0047 [-0.0143, 0.0253] | -0.30% [-1.59, 0.89] | 1/3; 4/9 |
| native_mae_gru_tuned_anchor_vs_v4_fusion_tree_episodic_k5 | 0.0047 [-0.0175, 0.0287] | -0.30% [-1.82, 1.08] | 1/3; 4/9 |
| native_tail_constant_vs_v4_fusion_gru_tuned_anchor_k5 | 0.0139 [-0.0080, 0.0376] | -0.87% [-2.39, 0.50] | 0/3; 2/9 |
| native_tail_constant_vs_v4_fusion_tree_episodic_k5 | 0.0139 [-0.0093, 0.0390] | -0.87% [-2.50, 0.56] | 0/3; 2/9 |
| native_tail_gru_tuned_anchor_vs_v4_fusion_gru_tuned_anchor_k5 | 0.0007 [-0.0128, 0.0151] | -0.04% [-0.98, 0.80] | 2/3; 7/9 |
| native_tail_gru_tuned_anchor_vs_v4_fusion_tree_episodic_k5 | 0.0007 [-0.0175, 0.0186] | -0.05% [-1.19, 1.04] | 1/3; 5/9 |

## High-DOC improvement and ordinary-concentration cost

| Saved model | K | Q90 MAE | Q90 signed bias | Non-tail MAE | Non-tail log MAE | False Q90 rate |
|---|---:|---:|---:|---:|---:|---:|
| context_gru_tuned_anchor | 0 | 7.5586 | -6.6062 | 1.2574 | 0.2614 | 1.91% |
| native_mae_gru_tuned_anchor | 0 | 7.7375 | -6.8908 | 1.1917 | 0.2497 | 1.72% |
| native_tail_gru_tuned_anchor | 0 | 7.4446 | -6.3882 | 1.2148 | 0.2565 | 1.98% |
| v4_fusion_gru_tuned_anchor | 0 | 7.7785 | -7.1226 | 1.2406 | 0.2579 | 1.68% |
| v4_fusion_tree_episodic | 0 | 7.7785 | -7.1226 | 1.2406 | 0.2579 | 1.68% |
| context_constant | 5 | 6.7436 | -5.2589 | 1.0177 | 0.2075 | 2.15% |
| context_gru_tuned_anchor | 5 | 6.6945 | -5.1653 | 1.0084 | 0.2047 | 2.21% |
| native_mae_constant | 5 | 6.7955 | -5.3549 | 1.0106 | 0.2085 | 2.05% |
| native_mae_gru_tuned_anchor | 5 | 6.7587 | -5.2920 | 1.0050 | 0.2066 | 2.08% |
| native_tail_constant | 5 | 6.7310 | -5.1925 | 1.0184 | 0.2114 | 2.14% |
| native_tail_gru_tuned_anchor | 5 | 6.6822 | -5.0917 | 1.0095 | 0.2080 | 2.20% |
| v4_fusion_gru_tuned_anchor | 5 | 6.6886 | -5.1363 | 1.0080 | 0.2047 | 2.21% |
| v4_fusion_tree_episodic | 5 | 6.7115 | -5.1581 | 1.0056 | 0.2037 | 2.22% |

The false Q90 rate is P(prediction ≥ source Q90 | observation < source Q90).
It is not the false-discovery fraction among predicted high values. Overall, Q90 and
non-tail raw/log MAE, signed-bias intervals and absolute false-positive-rate differences are retained
in `comparisons.csv`. All K curves and signed biases are retained in `error_profiles.csv`.

## Where gains and losses occur

| Native correction vs matched context | Improved stations | Worsened stations | Top-five share of positive gain | Top-five share of harm |
|---|---:|---:|---:|---:|
| native_mae_vs_context_constant_k5 | 103 | 69 | 48.9% | 43.1% |
| native_mae_vs_context_gru_tuned_anchor_k0 | 96 | 76 | 25.1% | 41.5% |
| native_mae_vs_context_gru_tuned_anchor_k5 | 101 | 71 | 53.6% | 49.4% |
| native_tail_vs_context_constant_k5 | 99 | 73 | 47.5% | 59.3% |
| native_tail_vs_context_gru_tuned_anchor_k0 | 96 | 76 | 34.2% | 32.4% |
| native_tail_vs_context_gru_tuned_anchor_k5 | 96 | 76 | 43.8% | 61.1% |

Station contributions use the same equal-partition weighting as overall MAE.
Repeated station identities are combined. Gain and harm shares have separate positive
denominators, avoiding unstable percentages of a nearly zero net improvement.

## What source-validation selected

| Training objective | Active residual fits | Context fallbacks | Selected scale counts | Selected epochs |
|---|---:|---:|---|---|
| native_mae | 9/9 | 0/9 | 0.5: 4, 1: 5 | 30, 30, 30, 29, 22, 2, 30, 1, 30 |
| native_tail | 9/9 | 0/9 | 0.25: 1, 0.5: 4, 1: 4 | 30, 27, 28, 1, 1, 1, 15, 1, 30 |

A selected residual scale of zero is the environmental context fallback and is not
evidence of a neural correction. Training choices and every support-adapter selection
are available in `training_choices.csv` and `adapter_choices.csv`.

## Interpretation

- **native_mae, no target support (K = 0):** MAE changes from 1.9028 to 1.8625 mg/L (2.12% reduction, 95% CI [0.58, 3.79]%). 3/3 partitions and 9/9 partition–seed pairs improve. Q90 MAE changes by +0.1788 mg/L (95% CI [+0.1324, +0.2357]), equivalent to -2.37% reduction (95% CI [-3.24, -1.72]%). Non-tail MAE changes by -0.0657 mg/L (95% CI [-0.0996, -0.0329]).
- **native_tail, no target support (K = 0):** MAE changes from 1.9028 to 1.8545 mg/L (2.54% reduction, 95% CI [1.17, 3.89]%). 3/3 partitions and 9/9 partition–seed pairs improve. Q90 MAE changes by -0.1141 mg/L (95% CI [-0.2298, +0.0037]), equivalent to +1.51% reduction (95% CI [-0.05, +2.65]%). Non-tail MAE changes by -0.0426 mg/L (95% CI [-0.0685, -0.0145]).
- **native_mae, K = 5 with frozen GRU support adapter:** overall MAE is higher (-0.25% reduction; the paired interval includes zero). Q90 and non-tail MAE changes are +0.0643 and -0.0034 mg/L, respectively.
- **native_tail, K = 5 with frozen GRU support adapter:** overall MAE is lower (+0.00% reduction; the paired interval includes zero). Q90 and non-tail MAE changes are -0.0122 and +0.0010 mg/L, respectively.
- **Tail weighting versus ordinary training, K = 0:** Q90 MAE changes by -0.2929 mg/L (95% CI [-0.4165, -0.1681]); non-tail MAE changes by +0.0231 mg/L (95% CI [+0.0015, +0.0478]). The false Q90 rate changes by +0.255 percentage points (95% CI [+0.109, +0.478]).
- **Tail weighting versus ordinary training, K = 5:** Q90 MAE changes by -0.0765 mg/L (95% CI [-0.1090, -0.0474]); non-tail MAE changes by +0.0045 mg/L (95% CI [-0.0093, +0.0175]). The false Q90 rate changes by +0.128 percentage points (95% CI [+0.047, +0.251]).

The clearest gain occurs before target-station support is available. With five
support observations, the native correction adds little beyond the existing station
adapter. Tail weighting changes the allocation of error between high and ordinary DOC
rather than establishing an additional overall-MAE advantage over ordinary training.

The named comparisons evaluate both training objectives and both support adapters;
they do not define a winner from this test panel. The existing v4 GRU and tree fusion
references remain explicit performance comparators. Error concentration is descriptive
and does not identify an irreducible error floor or a physical mechanism.
