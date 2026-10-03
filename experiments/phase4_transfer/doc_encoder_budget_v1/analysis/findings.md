# Partial encoder adaptation for native DOC residual learning

Nine station-partition/seed packages contain three neural fits each. All modes retain
the concentration-interaction head: 30 direct regime/flow features and hidden-state
interactions with three numerical flow features and context concentration. The source
OOF context forest, native DOC objective, tail weight 2, 60-epoch ceiling, patience 5,
GRU initialization, support basis and ecological-memory profile remain fixed.

Raw monthly inputs now pass through the existing spatial/ecological encoder during
optimization. Its input normalization is unchanged and dropout is disabled for every
mode. Empty edges are retained: this is a test of local representation adaptation,
not renewed river-message training or evidence for graph transport.

The matched trainability ladder is:

- frozen: encoder weights remain fixed; GRU/decay and the concentration head train.
- last_self: also update the encoder's final self-path layer.
- last_self_ecology: also update its existing ecological encoder.

Selected encoder parameters use learning rate 1e-5. GRU/decay and head learning rates
remain 1e-4 and 1e-3. Every scalar head starts at zero, and source validation selects
checkpoint and residual scale using overall native query MAE. Encoder parameter
counts and selected-checkpoint movements are reported separately.

The new frozen arm is the matched control for partial updating under the same extended
60-epoch ceiling. Its preceding saved concentration reference had a 30-epoch
ceiling. Their final prediction differences therefore include the training-duration
extension; they are not a raw-versus-cached processing-equivalence test. Initial encoder
numeric parity and saved-model replay are checked separately. Historical predictions are copied exactly.
The budget extension was motivated by source-validation learning traces reaching the
earlier ceiling. The objective, tail weight, stopping patience and target endpoints are unchanged.

Each new base is also integrated with the same frozen v1 ecological-affine residual profile.
Source-validation selects the K0 residual mixing weight; positive-K selection jointly
chooses mixing and the existing support adapter. Gamma=0 recovers its direct neural base,
gamma=1 replaces the neural correction with ecological memory. Both paths use the same
frozen-v4 GRU support basis. All models and old concentration/ecological-v2 references remain
visible. This analysis does not select a winning arm or promote it from target outcomes.

Station partitions 142–144 have been examined previously. This is model development on
the same cohort. Source validation has also been reused. Target support is retrospective
and all K values use a fixed query set, with the full five reserved supports excluded.

Intervals use 5,000 paired whole-station bootstrap draws; repeated station identities
are jointly resampled across partitions. Seed losses are averaged within partition, then
the three partitions receive equal weight. Negative Delta MAE and positive relative
reduction favor the candidate. Q90 is source-training-derived. Bias is prediction minus truth.

## K curves with the existing GRU support basis

| Saved model | K0 MAE | K1 MAE | K3 MAE | K5 MAE | K5 RMSE | K5 R² | K5 log MAE |
|---|---:|---:|---:|---:|---:|---:|---:|
| context_gru_tuned_anchor | 1.9028 | 1.8666 | 1.6348 | 1.5960 | 3.591 | 0.568 | 0.2294 |
| frozen_gru_tuned_anchor | 1.8313 | 1.7975 | 1.6497 | 1.5935 | 3.592 | 0.567 | 0.2303 |
| last_self_gru_tuned_anchor | 1.8280 | 1.7950 | 1.6490 | 1.5936 | 3.594 | 0.567 | 0.2302 |
| last_self_ecology_gru_tuned_anchor | 1.8240 | 1.7862 | 1.6458 | 1.5947 | 3.613 | 0.563 | 0.2303 |
| frozen_integrated_gru_tuned_anchor | 1.8117 | 1.7852 | 1.6295 | 1.5834 | 3.575 | 0.570 | 0.2279 |
| last_self_integrated_gru_tuned_anchor | 1.8110 | 1.7838 | 1.6296 | 1.5920 | 3.598 | 0.566 | 0.2295 |
| last_self_ecology_integrated_gru_tuned_anchor | 1.8108 | 1.7769 | 1.6254 | 1.5887 | 3.594 | 0.567 | 0.2289 |
| prior_concentration_gru_tuned_anchor | 1.8436 | 1.8064 | 1.6371 | 1.5883 | 3.573 | 0.571 | 0.2288 |
| prior_ecological_affine_gru_tuned_anchor | 1.8092 | 1.7917 | 1.6221 | 1.5799 | 3.581 | 0.569 | 0.2283 |

Constant-only adaptation is retained in all CSVs. At K0 it is bitwise identical to
the GRU-support path, so its duplicate bootstrap contrasts are omitted.

## Matched partial encoder updates

| Comparison | Delta MAE [95% CI] | Reduction [95% CI] | Improved partitions; fits |
|---|---:|---:|---:|
| last_self_vs_frozen_constant_k5 | +0.0043 [-0.0011, +0.0113] | -0.27% [-0.66, +0.07] | 2/3; 5/9 |
| last_self_ecology_vs_frozen_constant_k5 | +0.0005 [-0.0083, +0.0091] | -0.03% [-0.54, +0.53] | 2/3; 7/9 |
| last_self_ecology_vs_last_self_constant_k5 | -0.0038 [-0.0095, +0.0009] | +0.24% [-0.06, +0.59] | 3/3; 8/9 |
| last_self_vs_frozen_integrated_constant_k5 | +0.0077 [+0.0012, +0.0156] | -0.48% [-0.93, -0.08] | 2/3; 6/9 |
| last_self_ecology_vs_frozen_integrated_constant_k5 | +0.0046 [-0.0009, +0.0113] | -0.29% [-0.67, +0.06] | 2/3; 7/9 |
| last_self_ecology_vs_last_self_integrated_constant_k5 | -0.0031 [-0.0079, +0.0003] | +0.19% [-0.02, +0.48] | 3/3; 8/9 |
| last_self_vs_frozen_gru_tuned_anchor_k0 | -0.0033 [-0.0098, +0.0030] | +0.18% [-0.16, +0.55] | 3/3; 7/9 |
| last_self_ecology_vs_frozen_gru_tuned_anchor_k0 | -0.0073 [-0.0235, +0.0075] | +0.40% [-0.41, +1.26] | 2/3; 7/9 |
| last_self_ecology_vs_last_self_gru_tuned_anchor_k0 | -0.0040 [-0.0188, +0.0098] | +0.22% [-0.54, +1.00] | 2/3; 6/9 |
| last_self_vs_frozen_integrated_gru_tuned_anchor_k0 | -0.0007 [-0.0070, +0.0057] | +0.04% [-0.31, +0.40] | 2/3; 5/9 |
| last_self_ecology_vs_frozen_integrated_gru_tuned_anchor_k0 | -0.0009 [-0.0167, +0.0155] | +0.05% [-0.83, +0.92] | 2/3; 5/9 |
| last_self_ecology_vs_last_self_integrated_gru_tuned_anchor_k0 | -0.0002 [-0.0147, +0.0146] | +0.01% [-0.81, +0.79] | 2/3; 5/9 |
| last_self_vs_frozen_gru_tuned_anchor_k5 | +0.0001 [-0.0027, +0.0027] | -0.01% [-0.17, +0.17] | 2/3; 6/9 |
| last_self_ecology_vs_frozen_gru_tuned_anchor_k5 | +0.0012 [-0.0083, +0.0108] | -0.08% [-0.64, +0.52] | 2/3; 7/9 |
| last_self_ecology_vs_last_self_gru_tuned_anchor_k5 | +0.0011 [-0.0078, +0.0100] | -0.07% [-0.61, +0.48] | 2/3; 6/9 |
| last_self_vs_frozen_integrated_gru_tuned_anchor_k5 | +0.0086 [+0.0018, +0.0175] | -0.54% [-1.04, -0.11] | 2/3; 6/9 |
| last_self_ecology_vs_frozen_integrated_gru_tuned_anchor_k5 | +0.0053 [-0.0007, +0.0131] | -0.33% [-0.78, +0.05] | 2/3; 7/9 |
| last_self_ecology_vs_last_self_integrated_gru_tuned_anchor_k5 | -0.0033 [-0.0081, +0.0001] | +0.21% [-0.01, +0.50] | 3/3; 7/9 |

## Comparison with the prior concentration model

| Comparison | Delta MAE [95% CI] | Reduction [95% CI] | Improved partitions; fits |
|---|---:|---:|---:|
| frozen_vs_prior_concentration_constant_k5 | +0.0054 [-0.0015, +0.0139] | -0.34% [-0.83, +0.09] | 0/3; 2/9 |
| last_self_vs_prior_concentration_constant_k5 | +0.0097 [-0.0014, +0.0244] | -0.61% [-1.44, +0.09] | 1/3; 5/9 |
| last_self_ecology_vs_prior_concentration_constant_k5 | +0.0059 [-0.0072, +0.0212] | -0.37% [-1.26, +0.45] | 2/3; 7/9 |
| frozen_vs_prior_concentration_gru_tuned_anchor_k0 | -0.0124 [-0.0318, +0.0054] | +0.67% [-0.32, +1.62] | 1/3; 3/9 |
| last_self_vs_prior_concentration_gru_tuned_anchor_k0 | -0.0156 [-0.0374, +0.0034] | +0.85% [-0.19, +1.96] | 3/3; 8/9 |
| last_self_ecology_vs_prior_concentration_gru_tuned_anchor_k0 | -0.0197 [-0.0500, +0.0054] | +1.07% [-0.32, +2.57] | 2/3; 8/9 |
| frozen_vs_prior_concentration_gru_tuned_anchor_k5 | +0.0052 [-0.0012, +0.0131] | -0.33% [-0.79, +0.08] | 0/3; 2/9 |
| last_self_vs_prior_concentration_gru_tuned_anchor_k5 | +0.0053 [-0.0013, +0.0135] | -0.33% [-0.81, +0.08] | 2/3; 6/9 |
| last_self_ecology_vs_prior_concentration_gru_tuned_anchor_k5 | +0.0064 [-0.0073, +0.0226] | -0.40% [-1.36, +0.45] | 2/3; 7/9 |

## Integrated models versus prior ecological-v2 model

| Comparison | Delta MAE [95% CI] | Reduction [95% CI] | Improved partitions; fits |
|---|---:|---:|---:|
| frozen_integrated_vs_prior_ecological_constant_k5 | -0.0013 [-0.0109, +0.0089] | +0.08% [-0.55, +0.68] | 2/3; 7/9 |
| last_self_integrated_vs_prior_ecological_constant_k5 | +0.0064 [-0.0054, +0.0197] | -0.40% [-1.20, +0.34] | 1/3; 5/9 |
| last_self_ecology_integrated_vs_prior_ecological_constant_k5 | +0.0033 [-0.0071, +0.0151] | -0.21% [-0.92, +0.46] | 1/3; 6/9 |
| frozen_integrated_vs_prior_ecological_gru_tuned_anchor_k0 | +0.0024 [-0.0247, +0.0277] | -0.13% [-1.63, +1.27] | 2/3; 4/9 |
| last_self_integrated_vs_prior_ecological_gru_tuned_anchor_k0 | +0.0018 [-0.0244, +0.0264] | -0.10% [-1.50, +1.28] | 1/3; 3/9 |
| last_self_ecology_integrated_vs_prior_ecological_gru_tuned_anchor_k0 | +0.0016 [-0.0299, +0.0316] | -0.09% [-1.83, +1.56] | 1/3; 4/9 |
| frozen_integrated_vs_prior_ecological_gru_tuned_anchor_k5 | +0.0035 [-0.0072, +0.0149] | -0.22% [-0.94, +0.46] | 2/3; 6/9 |
| last_self_integrated_vs_prior_ecological_gru_tuned_anchor_k5 | +0.0121 [-0.0017, +0.0277] | -0.77% [-1.67, +0.10] | 1/3; 6/9 |
| last_self_ecology_integrated_vs_prior_ecological_gru_tuned_anchor_k5 | +0.0088 [-0.0032, +0.0222] | -0.56% [-1.33, +0.20] | 1/3; 6/9 |

## Ecological integration versus each direct head

| Comparison | Delta MAE [95% CI] | Reduction [95% CI] | Improved partitions; fits |
|---|---:|---:|---:|
| frozen_integrated_vs_direct_constant_k5 | -0.0079 [-0.0162, -0.0012] | +0.49% [+0.07, +0.98] | 2/3; 2/9 |
| last_self_integrated_vs_direct_constant_k5 | -0.0046 [-0.0128, +0.0030] | +0.29% [-0.18, +0.77] | 2/3; 2/9 |
| last_self_ecology_integrated_vs_direct_constant_k5 | -0.0038 [-0.0128, +0.0045] | +0.24% [-0.28, +0.76] | 3/3; 3/9 |
| frozen_integrated_vs_direct_gru_tuned_anchor_k0 | -0.0196 [-0.0345, -0.0072] | +1.07% [+0.40, +1.88] | 2/3; 5/9 |
| last_self_integrated_vs_direct_gru_tuned_anchor_k0 | -0.0170 [-0.0302, -0.0060] | +0.93% [+0.33, +1.64] | 2/3; 5/9 |
| last_self_ecology_integrated_vs_direct_gru_tuned_anchor_k0 | -0.0132 [-0.0229, -0.0049] | +0.72% [+0.27, +1.25] | 2/3; 4/9 |
| frozen_integrated_vs_direct_gru_tuned_anchor_k5 | -0.0101 [-0.0193, -0.0027] | +0.63% [+0.17, +1.18] | 2/3; 3/9 |
| last_self_integrated_vs_direct_gru_tuned_anchor_k5 | -0.0016 [-0.0050, +0.0021] | +0.10% [-0.12, +0.32] | 1/3; 3/9 |
| last_self_ecology_integrated_vs_direct_gru_tuned_anchor_k5 | -0.0060 [-0.0153, +0.0021] | +0.38% [-0.13, +0.92] | 3/3; 5/9 |

## Tail, ordinary concentrations and false alarms

| Saved model | K | Q90 MAE | Q90 bias | Non-tail MAE | Non-tail log MAE | False Q90 rate |
|---|---:|---:|---:|---:|---:|---:|
| context_gru_tuned_anchor | 0 | 7.5586 | -6.6062 | 1.2574 | 0.2614 | 1.91% |
| context_gru_tuned_anchor | 5 | 6.6945 | -5.1653 | 1.0084 | 0.2047 | 2.21% |
| frozen_gru_tuned_anchor | 0 | 7.2583 | -6.0087 | 1.2104 | 0.2519 | 2.09% |
| frozen_gru_tuned_anchor | 5 | 6.6461 | -5.0372 | 1.0111 | 0.2064 | 2.22% |
| last_self_gru_tuned_anchor | 0 | 7.2532 | -5.9934 | 1.2070 | 0.2502 | 2.13% |
| last_self_gru_tuned_anchor | 5 | 6.6498 | -5.0474 | 1.0108 | 0.2062 | 2.23% |
| last_self_ecology_gru_tuned_anchor | 0 | 7.2369 | -5.8588 | 1.2044 | 0.2492 | 2.17% |
| last_self_ecology_gru_tuned_anchor | 5 | 6.6774 | -5.1456 | 1.0092 | 0.2062 | 2.23% |
| frozen_integrated_gru_tuned_anchor | 0 | 7.2795 | -6.0745 | 1.1856 | 0.2471 | 2.09% |
| frozen_integrated_gru_tuned_anchor | 5 | 6.6394 | -5.0153 | 1.0004 | 0.2035 | 2.23% |
| last_self_integrated_gru_tuned_anchor | 0 | 7.2741 | -6.0599 | 1.1853 | 0.2460 | 2.13% |
| last_self_integrated_gru_tuned_anchor | 5 | 6.6632 | -5.0996 | 1.0075 | 0.2053 | 2.21% |
| last_self_ecology_integrated_gru_tuned_anchor | 0 | 7.2513 | -5.9115 | 1.1877 | 0.2460 | 2.18% |
| last_self_ecology_integrated_gru_tuned_anchor | 5 | 6.6593 | -5.0843 | 1.0041 | 0.2046 | 2.22% |
| prior_concentration_gru_tuned_anchor | 0 | 7.2842 | -6.0655 | 1.2213 | 0.2517 | 2.08% |
| prior_concentration_gru_tuned_anchor | 5 | 6.6244 | -4.9737 | 1.0075 | 0.2047 | 2.21% |
| prior_ecological_affine_gru_tuned_anchor | 0 | 7.4318 | -6.4542 | 1.1653 | 0.2423 | 1.99% |
| prior_ecological_affine_gru_tuned_anchor | 5 | 6.6808 | -5.1324 | 0.9913 | 0.2035 | 2.14% |

False Q90 rate conditions on an observed concentration below the source Q90 threshold.
All regional raw/log MAE and bias intervals are in comparisons.csv.

## Source-validation training choices

| Mode | Total trainable parameters | Trainable encoder parameters | Selected epoch range | Scale counts | Mean validation MAE change |
|---|---:|---:|---|---|---:|
| frozen | 25823 | 0 | 5–50 | 0.5: 3, 1: 6 | -0.1049 |
| last_self | 29983 | 4160 | 5–46 | 0.5: 3, 1: 6 | -0.1077 |
| last_self_ecology | 31359 | 5536 | 5–49 | 0.5: 2, 1: 7 | -0.1215 |

| Mode | Mean spatial weight movement | Mean last-self movement | Mean ecology movement |
|---|---:|---:|---:|
| frozen | 0.000000 | 0.000000 | 0.000000 |
| last_self | 0.214920 | 0.214920 | 0.000000 |
| last_self_ecology | 0.297983 | 0.242643 | 0.172443 |

Movements are Euclidean parameter distances from initialization at the selected
checkpoint. They establish which weights changed, not whether the representation improved.

## Frozen budget extension versus preceding 30-epoch concentration model

| Support path | K | Maximum query prediction difference | Mean run absolute difference | Bitwise-equal runs |
|---|---:|---:|---:|---:|
| constant | 0 | 2.00646603 | 0.104233297 | 4/9 |
| constant | 1 | 1.90382608 | 0.0943505471 | 4/9 |
| constant | 3 | 1.98128827 | 0.0735183126 | 4/9 |
| constant | 5 | 6.18070992 | 0.0771932539 | 4/9 |
| gru_tuned_anchor | 0 | 2.00646603 | 0.104233297 | 4/9 |
| gru_tuned_anchor | 1 | 1.90382608 | 0.0943505471 | 4/9 |
| gru_tuned_anchor | 3 | 4.65977613 | 0.113021115 | 4/9 |
| gru_tuned_anchor | 5 | 6.17063939 | 0.0751496378 | 4/9 |

These products use different epoch ceilings (60 versus 30). Differences must not be
attributed solely to numerical processing. Initial raw-versus-cached encoder parity is
verified independently; the matched raw-input 60-versus-30 comparisons are reported separately.


The selected scale applies to the raw head output before adding it to context.
Saved full-grid arm_delta columns are unscaled; realized corrections also include
the selected scale and zero concentration floor. Raw head magnitudes are not directly
comparable as applied prediction corrections.

## Source-validation ecological mixing

| Integrated model | K | Gamma counts |
|---|---:|---|
| frozen_integrated_constant | 0 | 0: 4, 0.25: 2, 0.5: 2, 1: 1 |
| frozen_integrated_constant | 1 | 0: 4, 0.25: 3, 0.5: 2 |
| frozen_integrated_constant | 3 | 0: 1, 0.25: 6, 0.5: 2 |
| frozen_integrated_constant | 5 | 0: 4, 0.25: 4, 0.5: 1 |
| frozen_integrated_gru_tuned_anchor | 0 | 0: 4, 0.25: 2, 0.5: 2, 1: 1 |
| frozen_integrated_gru_tuned_anchor | 1 | 0: 4, 0.25: 3, 0.5: 2 |
| frozen_integrated_gru_tuned_anchor | 3 | 0.25: 6, 0.5: 3 |
| frozen_integrated_gru_tuned_anchor | 5 | 0: 3, 0.25: 5, 0.5: 1 |
| last_self_integrated_constant | 0 | 0: 4, 0.25: 3, 0.5: 1, 1: 1 |
| last_self_integrated_constant | 1 | 0: 4, 0.25: 3, 0.5: 2 |
| last_self_integrated_constant | 3 | 0: 1, 0.25: 7, 0.5: 1 |
| last_self_integrated_constant | 5 | 0: 5, 0.25: 3, 0.5: 1 |
| last_self_integrated_gru_tuned_anchor | 0 | 0: 4, 0.25: 3, 0.5: 1, 1: 1 |
| last_self_integrated_gru_tuned_anchor | 1 | 0: 4, 0.25: 3, 0.5: 2 |
| last_self_integrated_gru_tuned_anchor | 3 | 0.25: 6, 0.5: 3 |
| last_self_integrated_gru_tuned_anchor | 5 | 0: 4, 0.25: 5 |
| last_self_ecology_integrated_constant | 0 | 0: 5, 0.25: 3, 1: 1 |
| last_self_ecology_integrated_constant | 1 | 0: 5, 0.25: 3, 0.5: 1 |
| last_self_ecology_integrated_constant | 3 | 0: 2, 0.25: 6, 0.5: 1 |
| last_self_ecology_integrated_constant | 5 | 0: 4, 0.25: 3, 0.5: 2 |
| last_self_ecology_integrated_gru_tuned_anchor | 0 | 0: 5, 0.25: 3, 1: 1 |
| last_self_ecology_integrated_gru_tuned_anchor | 1 | 0: 5, 0.25: 3, 0.5: 1 |
| last_self_ecology_integrated_gru_tuned_anchor | 3 | 0: 1, 0.25: 5, 0.5: 3 |
| last_self_ecology_integrated_gru_tuned_anchor | 5 | 0: 3, 0.25: 5, 0.5: 1 |

Mixer and support-adapter choices are fitted independently for each direct head.
At gamma=0 integrated predictions are checked against that head's direct support adapter.

## Station gains and harms

| GRU-basis contrast | Improved / worsened stations | Top-five positive gain | Top-five harm |
|---|---:|---:|---:|
| last_self_ecology_integrated_vs_prior_ecological_gru_tuned_anchor_k0 | 94 / 78 | 34.5% | 37.6% |
| last_self_vs_frozen_gru_tuned_anchor_k0 | 95 / 77 | 45.6% | 38.5% |
| last_self_ecology_vs_frozen_gru_tuned_anchor_k0 | 94 / 78 | 41.3% | 36.7% |
| last_self_ecology_vs_last_self_gru_tuned_anchor_k0 | 90 / 82 | 42.8% | 45.1% |
| last_self_vs_frozen_integrated_gru_tuned_anchor_k0 | 87 / 85 | 42.5% | 32.4% |
| last_self_ecology_vs_frozen_integrated_gru_tuned_anchor_k0 | 91 / 81 | 44.8% | 40.8% |
| last_self_ecology_vs_last_self_integrated_gru_tuned_anchor_k0 | 90 / 82 | 43.2% | 47.9% |
| last_self_ecology_integrated_vs_prior_ecological_gru_tuned_anchor_k5 | 90 / 82 | 30.4% | 52.1% |
| last_self_vs_frozen_gru_tuned_anchor_k5 | 86 / 86 | 45.4% | 47.2% |
| last_self_ecology_vs_frozen_gru_tuned_anchor_k5 | 87 / 85 | 47.4% | 42.1% |
| last_self_ecology_vs_last_self_gru_tuned_anchor_k5 | 87 / 85 | 53.0% | 42.0% |
| last_self_vs_frozen_integrated_gru_tuned_anchor_k5 | 78 / 94 | 45.8% | 59.3% |
| last_self_ecology_vs_frozen_integrated_gru_tuned_anchor_k5 | 92 / 80 | 33.6% | 53.3% |
| last_self_ecology_vs_last_self_integrated_gru_tuned_anchor_k5 | 98 / 74 | 50.8% | 27.6% |

Positive gains and harms have separate denominators. Station contributions retain
equal partition weights and combine repeated station identities.

## Fixed primary contrasts with tail tradeoffs

- **last_self_vs_frozen_gru_tuned_anchor_k0:** MAE reduction +0.18% (95% CI [-0.16, +0.55]); 3/3 partitions and 7/9 fits improve. Q90 Delta MAE -0.0052 [-0.0222, +0.0078]; non-tail Delta MAE -0.0033 [-0.0104, +0.0034]. False Q90 change +0.038 percentage points [-0.013, +0.105].
- **last_self_ecology_vs_frozen_gru_tuned_anchor_k0:** MAE reduction +0.40% (95% CI [-0.41, +1.26]); 2/3 partitions and 7/9 fits improve. Q90 Delta MAE -0.0214 [-0.0617, +0.0164]; non-tail Delta MAE -0.0060 [-0.0243, +0.0106]. False Q90 change +0.077 percentage points [+0.015, +0.176].
- **last_self_ecology_vs_last_self_gru_tuned_anchor_k0:** MAE reduction +0.22% (95% CI [-0.54, +1.00]); 2/3 partitions and 6/9 fits improve. Q90 Delta MAE -0.0163 [-0.0554, +0.0209]; non-tail Delta MAE -0.0027 [-0.0193, +0.0122]. False Q90 change +0.039 percentage points [-0.009, +0.102].
- **last_self_ecology_vs_prior_concentration_gru_tuned_anchor_k0:** MAE reduction +1.07% (95% CI [-0.32, +2.57]); 2/3 partitions and 8/9 fits improve. Q90 Delta MAE -0.0473 [-0.1043, +0.0087]; non-tail Delta MAE -0.0169 [-0.0518, +0.0118]. False Q90 change +0.089 percentage points [+0.016, +0.196].
- **last_self_ecology_integrated_vs_prior_ecological_gru_tuned_anchor_k0:** MAE reduction -0.09% (95% CI [-1.83, +1.56]); 1/3 partitions and 4/9 fits improve. Q90 Delta MAE -0.1805 [-0.2824, -0.0677]; non-tail Delta MAE +0.0224 [-0.0103, +0.0562]. False Q90 change +0.193 percentage points [+0.096, +0.326].
- **last_self_vs_frozen_gru_tuned_anchor_k5:** MAE reduction -0.01% (95% CI [-0.17, +0.17]); 2/3 partitions and 6/9 fits improve. Q90 Delta MAE +0.0037 [-0.0049, +0.0108]; non-tail Delta MAE -0.0003 [-0.0032, +0.0024]. False Q90 change +0.002 percentage points [-0.013, +0.016].
- **last_self_ecology_vs_frozen_gru_tuned_anchor_k5:** MAE reduction -0.08% (95% CI [-0.64, +0.52]); 2/3 partitions and 7/9 fits improve. Q90 Delta MAE +0.0313 [-0.0114, +0.0781]; non-tail Delta MAE -0.0019 [-0.0121, +0.0058]. False Q90 change +0.004 percentage points [-0.036, +0.045].
- **last_self_ecology_vs_last_self_gru_tuned_anchor_k5:** MAE reduction -0.07% (95% CI [-0.61, +0.48]); 2/3 partitions and 6/9 fits improve. Q90 Delta MAE +0.0276 [-0.0095, +0.0683]; non-tail Delta MAE -0.0016 [-0.0112, +0.0063]. False Q90 change +0.002 percentage points [-0.040, +0.048].
- **last_self_ecology_vs_prior_concentration_gru_tuned_anchor_k5:** MAE reduction -0.40% (95% CI [-1.36, +0.45]); 2/3 partitions and 7/9 fits improve. Q90 Delta MAE +0.0530 [-0.0313, +0.1437]; non-tail Delta MAE +0.0017 [-0.0110, +0.0126]. False Q90 change +0.020 percentage points [-0.034, +0.084].
- **last_self_ecology_integrated_vs_prior_ecological_gru_tuned_anchor_k5:** MAE reduction -0.56% (95% CI [-1.33, +0.20]); 1/3 partitions and 6/9 fits improve. Q90 Delta MAE -0.0215 [-0.0771, +0.0324]; non-tail Delta MAE +0.0129 [+0.0010, +0.0269]. False Q90 change +0.089 percentage points [+0.032, +0.168].

The matched controls test which existing representations benefit from the same residual
objective. Any improvement of the ecology-integrated
product can also involve a changed source-validation mixing weight. It does not alone
establish a better neural representation or a physical ecological/transport mechanism.
