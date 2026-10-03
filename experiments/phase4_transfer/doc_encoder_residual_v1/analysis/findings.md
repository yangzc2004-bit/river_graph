# Partial encoder adaptation for native DOC residual learning

Nine station-partition/seed packages contain three neural fits each. All modes retain
the concentration-interaction head: 30 direct regime/flow features and hidden-state
interactions with three numerical flow features and context concentration. The source
OOF context forest, native DOC objective, tail weight 2, 30-epoch ceiling, patience 5,
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

The new frozen arm is the matched control for partial updating. The previous saved
concentration model used cached encodings. Batched raw encoding can introduce small
floating-point changes that optimization amplifies, so final prediction equality is
measured rather than assumed. Historical reference products themselves are copied exactly.

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
| frozen_gru_tuned_anchor | 1.8436 | 1.8064 | 1.6371 | 1.5883 | 3.573 | 0.571 | 0.2288 |
| last_self_gru_tuned_anchor | 1.8357 | 1.8015 | 1.6345 | 1.5883 | 3.573 | 0.571 | 0.2290 |
| last_self_ecology_gru_tuned_anchor | 1.8299 | 1.7945 | 1.6482 | 1.5865 | 3.572 | 0.571 | 0.2287 |
| frozen_integrated_gru_tuned_anchor | 1.8246 | 1.7947 | 1.6193 | 1.5859 | 3.578 | 0.570 | 0.2281 |
| last_self_integrated_gru_tuned_anchor | 1.8187 | 1.7903 | 1.6184 | 1.5858 | 3.577 | 0.570 | 0.2283 |
| last_self_ecology_integrated_gru_tuned_anchor | 1.8136 | 1.7845 | 1.6168 | 1.5844 | 3.577 | 0.570 | 0.2278 |
| prior_concentration_gru_tuned_anchor | 1.8436 | 1.8064 | 1.6371 | 1.5883 | 3.573 | 0.571 | 0.2288 |
| prior_ecological_affine_gru_tuned_anchor | 1.8092 | 1.7917 | 1.6221 | 1.5799 | 3.581 | 0.569 | 0.2283 |

Constant-only adaptation is retained in all CSVs. At K0 it is bitwise identical to
the GRU-support path, so its duplicate bootstrap contrasts are omitted.

## Matched partial encoder updates

| Comparison | Delta MAE [95% CI] | Reduction [95% CI] | Improved partitions; fits |
|---|---:|---:|---:|
| last_self_vs_frozen_constant_k5 | +0.0003 [-0.0030, +0.0026] | -0.02% [-0.17, +0.19] | 1/3; 7/9 |
| last_self_ecology_vs_frozen_constant_k5 | -0.0012 [-0.0066, +0.0026] | +0.08% [-0.16, +0.41] | 2/3; 8/9 |
| last_self_ecology_vs_last_self_constant_k5 | -0.0015 [-0.0037, +0.0003] | +0.09% [-0.02, +0.23] | 3/3; 7/9 |
| last_self_vs_frozen_integrated_constant_k5 | +0.0009 [-0.0017, +0.0028] | -0.06% [-0.18, +0.10] | 2/3; 6/9 |
| last_self_ecology_vs_frozen_integrated_constant_k5 | -0.0003 [-0.0038, +0.0024] | +0.02% [-0.15, +0.23] | 1/3; 7/9 |
| last_self_ecology_vs_last_self_integrated_constant_k5 | -0.0012 [-0.0026, +0.0001] | +0.08% [-0.00, +0.17] | 2/3; 6/9 |
| last_self_vs_frozen_gru_tuned_anchor_k0 | -0.0079 [-0.0185, +0.0001] | +0.43% [-0.01, +0.96] | 3/3; 8/9 |
| last_self_ecology_vs_frozen_gru_tuned_anchor_k0 | -0.0137 [-0.0306, -0.0013] | +0.74% [+0.07, +1.60] | 3/3; 9/9 |
| last_self_ecology_vs_last_self_gru_tuned_anchor_k0 | -0.0058 [-0.0132, -0.0000] | +0.32% [+0.00, +0.68] | 3/3; 8/9 |
| last_self_vs_frozen_integrated_gru_tuned_anchor_k0 | -0.0059 [-0.0160, +0.0022] | +0.32% [-0.13, +0.86] | 2/3; 6/9 |
| last_self_ecology_vs_frozen_integrated_gru_tuned_anchor_k0 | -0.0110 [-0.0270, +0.0008] | +0.60% [-0.05, +1.44] | 2/3; 7/9 |
| last_self_ecology_vs_last_self_integrated_gru_tuned_anchor_k0 | -0.0051 [-0.0118, -0.0000] | +0.28% [+0.00, +0.62] | 3/3; 7/9 |
| last_self_vs_frozen_gru_tuned_anchor_k5 | -0.0000 [-0.0033, +0.0023] | +0.00% [-0.15, +0.20] | 2/3; 7/9 |
| last_self_ecology_vs_frozen_gru_tuned_anchor_k5 | -0.0018 [-0.0070, +0.0019] | +0.11% [-0.12, +0.43] | 3/3; 8/9 |
| last_self_ecology_vs_last_self_gru_tuned_anchor_k5 | -0.0018 [-0.0039, -0.0001] | +0.11% [+0.01, +0.24] | 3/3; 6/9 |
| last_self_vs_frozen_integrated_gru_tuned_anchor_k5 | -0.0001 [-0.0028, +0.0018] | +0.01% [-0.11, +0.17] | 2/3; 7/9 |
| last_self_ecology_vs_frozen_integrated_gru_tuned_anchor_k5 | -0.0015 [-0.0052, +0.0012] | +0.09% [-0.08, +0.32] | 3/3; 7/9 |
| last_self_ecology_vs_last_self_integrated_gru_tuned_anchor_k5 | -0.0014 [-0.0028, -0.0001] | +0.09% [+0.01, +0.18] | 2/3; 5/9 |

## Comparison with the prior concentration model

| Comparison | Delta MAE [95% CI] | Reduction [95% CI] | Improved partitions; fits |
|---|---:|---:|---:|
| frozen_vs_prior_concentration_constant_k5 | +0.0000 [+0.0000, +0.0000] | +0.00% [+0.00, +0.00] | 0/3; 0/9 |
| last_self_vs_prior_concentration_constant_k5 | +0.0003 [-0.0030, +0.0026] | -0.02% [-0.17, +0.19] | 1/3; 7/9 |
| last_self_ecology_vs_prior_concentration_constant_k5 | -0.0012 [-0.0066, +0.0026] | +0.08% [-0.16, +0.41] | 2/3; 8/9 |
| frozen_vs_prior_concentration_gru_tuned_anchor_k0 | +0.0000 [+0.0000, +0.0000] | +0.00% [+0.00, +0.00] | 0/3; 0/9 |
| last_self_vs_prior_concentration_gru_tuned_anchor_k0 | -0.0079 [-0.0185, +0.0001] | +0.43% [-0.01, +0.96] | 3/3; 8/9 |
| last_self_ecology_vs_prior_concentration_gru_tuned_anchor_k0 | -0.0137 [-0.0306, -0.0013] | +0.74% [+0.07, +1.60] | 3/3; 9/9 |
| frozen_vs_prior_concentration_gru_tuned_anchor_k5 | +0.0000 [+0.0000, +0.0000] | +0.00% [+0.00, +0.00] | 0/3; 0/9 |
| last_self_vs_prior_concentration_gru_tuned_anchor_k5 | -0.0000 [-0.0033, +0.0023] | +0.00% [-0.15, +0.20] | 2/3; 7/9 |
| last_self_ecology_vs_prior_concentration_gru_tuned_anchor_k5 | -0.0018 [-0.0070, +0.0019] | +0.11% [-0.12, +0.43] | 3/3; 8/9 |

## Integrated models versus prior ecological-v2 model

| Comparison | Delta MAE [95% CI] | Reduction [95% CI] | Improved partitions; fits |
|---|---:|---:|---:|
| frozen_integrated_vs_prior_ecological_constant_k5 | -0.0008 [-0.0104, +0.0095] | +0.05% [-0.58, +0.66] | 2/3; 7/9 |
| last_self_integrated_vs_prior_ecological_constant_k5 | +0.0001 [-0.0095, +0.0107] | -0.01% [-0.65, +0.60] | 2/3; 5/9 |
| last_self_ecology_integrated_vs_prior_ecological_constant_k5 | -0.0011 [-0.0106, +0.0093] | +0.07% [-0.56, +0.67] | 2/3; 6/9 |
| frozen_integrated_vs_prior_ecological_gru_tuned_anchor_k0 | +0.0154 [-0.0114, +0.0419] | -0.85% [-2.37, +0.60] | 1/3; 3/9 |
| last_self_integrated_vs_prior_ecological_gru_tuned_anchor_k0 | +0.0095 [-0.0142, +0.0334] | -0.53% [-1.90, +0.74] | 0/3; 3/9 |
| last_self_ecology_integrated_vs_prior_ecological_gru_tuned_anchor_k0 | +0.0044 [-0.0212, +0.0293] | -0.24% [-1.69, +1.12] | 1/3; 4/9 |
| frozen_integrated_vs_prior_ecological_gru_tuned_anchor_k5 | +0.0060 [-0.0062, +0.0195] | -0.38% [-1.21, +0.40] | 1/3; 5/9 |
| last_self_integrated_vs_prior_ecological_gru_tuned_anchor_k5 | +0.0059 [-0.0055, +0.0182] | -0.37% [-1.14, +0.34] | 2/3; 5/9 |
| last_self_ecology_integrated_vs_prior_ecological_gru_tuned_anchor_k5 | +0.0045 [-0.0066, +0.0164] | -0.28% [-1.02, +0.42] | 2/3; 5/9 |

## Ecological integration versus each direct head

| Comparison | Delta MAE [95% CI] | Reduction [95% CI] | Improved partitions; fits |
|---|---:|---:|---:|
| frozen_integrated_vs_direct_constant_k5 | -0.0020 [-0.0054, +0.0012] | +0.12% [-0.08, +0.34] | 1/3; 3/9 |
| last_self_integrated_vs_direct_constant_k5 | -0.0013 [-0.0046, +0.0020] | +0.08% [-0.12, +0.30] | 1/3; 2/9 |
| last_self_ecology_integrated_vs_direct_constant_k5 | -0.0011 [-0.0055, +0.0035] | +0.07% [-0.22, +0.36] | 1/3; 2/9 |
| frozen_integrated_vs_direct_gru_tuned_anchor_k0 | -0.0190 [-0.0337, -0.0068] | +1.03% [+0.37, +1.82] | 2/3; 5/9 |
| last_self_integrated_vs_direct_gru_tuned_anchor_k0 | -0.0170 [-0.0302, -0.0061] | +0.93% [+0.33, +1.63] | 2/3; 5/9 |
| last_self_ecology_integrated_vs_direct_gru_tuned_anchor_k0 | -0.0163 [-0.0296, -0.0053] | +0.89% [+0.30, +1.61] | 2/3; 5/9 |
| frozen_integrated_vs_direct_gru_tuned_anchor_k5 | -0.0024 [-0.0052, +0.0002] | +0.15% [-0.01, +0.33] | 2/3; 4/9 |
| last_self_integrated_vs_direct_gru_tuned_anchor_k5 | -0.0025 [-0.0054, +0.0003] | +0.16% [-0.02, +0.35] | 2/3; 4/9 |
| last_self_ecology_integrated_vs_direct_gru_tuned_anchor_k5 | -0.0021 [-0.0058, +0.0016] | +0.13% [-0.09, +0.37] | 2/3; 3/9 |

## Tail, ordinary concentrations and false alarms

| Saved model | K | Q90 MAE | Q90 bias | Non-tail MAE | Non-tail log MAE | False Q90 rate |
|---|---:|---:|---:|---:|---:|---:|
| context_gru_tuned_anchor | 0 | 7.5586 | -6.6062 | 1.2574 | 0.2614 | 1.91% |
| context_gru_tuned_anchor | 5 | 6.6945 | -5.1653 | 1.0084 | 0.2047 | 2.21% |
| frozen_gru_tuned_anchor | 0 | 7.2842 | -6.0655 | 1.2213 | 0.2517 | 2.08% |
| frozen_gru_tuned_anchor | 5 | 6.6244 | -4.9737 | 1.0075 | 0.2047 | 2.21% |
| last_self_gru_tuned_anchor | 0 | 7.2720 | -6.0265 | 1.2137 | 0.2500 | 2.10% |
| last_self_gru_tuned_anchor | 5 | 6.6226 | -4.9630 | 1.0077 | 0.2050 | 2.22% |
| last_self_ecology_gru_tuned_anchor | 0 | 7.2546 | -5.9594 | 1.2090 | 0.2485 | 2.11% |
| last_self_ecology_gru_tuned_anchor | 5 | 6.6204 | -4.9538 | 1.0059 | 0.2047 | 2.22% |
| frozen_integrated_gru_tuned_anchor | 0 | 7.3052 | -6.1253 | 1.1971 | 0.2470 | 2.08% |
| frozen_integrated_gru_tuned_anchor | 5 | 6.6390 | -5.0230 | 1.0032 | 0.2038 | 2.21% |
| last_self_integrated_gru_tuned_anchor | 0 | 7.2934 | -6.0924 | 1.1919 | 0.2458 | 2.09% |
| last_self_integrated_gru_tuned_anchor | 5 | 6.6371 | -5.0142 | 1.0032 | 0.2040 | 2.21% |
| last_self_ecology_integrated_gru_tuned_anchor | 0 | 7.2748 | -6.0367 | 1.1880 | 0.2446 | 2.10% |
| last_self_ecology_integrated_gru_tuned_anchor | 5 | 6.6394 | -5.0194 | 1.0014 | 0.2035 | 2.21% |
| prior_concentration_gru_tuned_anchor | 0 | 7.2842 | -6.0655 | 1.2213 | 0.2517 | 2.08% |
| prior_concentration_gru_tuned_anchor | 5 | 6.6244 | -4.9737 | 1.0075 | 0.2047 | 2.21% |
| prior_ecological_affine_gru_tuned_anchor | 0 | 7.4318 | -6.4542 | 1.1653 | 0.2423 | 1.99% |
| prior_ecological_affine_gru_tuned_anchor | 5 | 6.6808 | -5.1324 | 0.9913 | 0.2035 | 2.14% |

False Q90 rate conditions on an observed concentration below the source Q90 threshold.
All regional raw/log MAE and bias intervals are in comparisons.csv.

## Source-validation training choices

| Mode | Total trainable parameters | Trainable encoder parameters | Selected epoch range | Scale counts | Mean validation MAE change |
|---|---:|---:|---|---|---:|
| frozen | 25823 | 0 | 5–30 | 0.5: 3, 1: 6 | -0.0878 |
| last_self | 29983 | 4160 | 5–30 | 0.5: 3, 1: 6 | -0.0938 |
| last_self_ecology | 31359 | 5536 | 5–30 | 0.5: 3, 1: 6 | -0.0980 |

| Mode | Mean spatial weight movement | Mean last-self movement | Mean ecology movement |
|---|---:|---:|---:|
| frozen | 0.000000 | 0.000000 | 0.000000 |
| last_self | 0.194490 | 0.194490 | 0.000000 |
| last_self_ecology | 0.263512 | 0.217212 | 0.148907 |

Movements are Euclidean parameter distances from initialization at the selected
checkpoint. They establish which weights changed, not whether the representation improved.

## Frozen processing versus the cached predecessor

| Support path | K | Maximum query prediction difference | Mean run absolute difference | Bitwise-equal runs |
|---|---:|---:|---:|---:|
| constant | 0 | 0 | 0 | 9/9 |
| constant | 1 | 0 | 0 | 9/9 |
| constant | 3 | 0 | 0 | 9/9 |
| constant | 5 | 0 | 0 | 9/9 |
| gru_tuned_anchor | 0 | 0 | 0 | 9/9 |
| gru_tuned_anchor | 1 | 0 | 0 | 9/9 |
| gru_tuned_anchor | 3 | 0 | 0 | 9/9 |
| gru_tuned_anchor | 5 | 0 | 0 | 9/9 |

These compare final trained products and are empirical processing diagnostics.
No equivalence threshold is fitted to the observed differences. Raw-versus-cached
initial encoder features and saved-model replay are checked independently by the verifier.


The selected scale applies to the raw head output before adding it to context.
Saved full-grid arm_delta columns are unscaled; realized corrections also include
the selected scale and zero concentration floor. Raw head magnitudes are not directly
comparable as applied prediction corrections.

## Source-validation ecological mixing

| Integrated model | K | Gamma counts |
|---|---:|---|
| frozen_integrated_constant | 0 | 0: 4, 0.25: 2, 0.5: 2, 1: 1 |
| frozen_integrated_constant | 1 | 0: 4, 0.25: 3, 0.5: 2 |
| frozen_integrated_constant | 3 | 0: 3, 0.25: 4, 0.5: 2 |
| frozen_integrated_constant | 5 | 0: 4, 0.25: 4, 0.5: 1 |
| frozen_integrated_gru_tuned_anchor | 0 | 0: 4, 0.25: 2, 0.5: 2, 1: 1 |
| frozen_integrated_gru_tuned_anchor | 1 | 0: 4, 0.25: 3, 0.5: 2 |
| frozen_integrated_gru_tuned_anchor | 3 | 0: 2, 0.25: 4, 0.5: 3 |
| frozen_integrated_gru_tuned_anchor | 5 | 0: 4, 0.25: 5 |
| last_self_integrated_constant | 0 | 0: 4, 0.25: 3, 0.5: 1, 1: 1 |
| last_self_integrated_constant | 1 | 0: 4, 0.25: 3, 0.5: 2 |
| last_self_integrated_constant | 3 | 0: 3, 0.25: 5, 0.5: 1 |
| last_self_integrated_constant | 5 | 0: 5, 0.25: 3, 0.5: 1 |
| last_self_integrated_gru_tuned_anchor | 0 | 0: 4, 0.25: 3, 0.5: 1, 1: 1 |
| last_self_integrated_gru_tuned_anchor | 1 | 0: 4, 0.25: 3, 0.5: 2 |
| last_self_integrated_gru_tuned_anchor | 3 | 0: 2, 0.25: 4, 0.5: 3 |
| last_self_integrated_gru_tuned_anchor | 5 | 0: 4, 0.25: 5 |
| last_self_ecology_integrated_constant | 0 | 0: 4, 0.25: 3, 0.5: 1, 1: 1 |
| last_self_ecology_integrated_constant | 1 | 0: 4, 0.25: 3, 0.5: 2 |
| last_self_ecology_integrated_constant | 3 | 0: 3, 0.25: 5, 0.5: 1 |
| last_self_ecology_integrated_constant | 5 | 0: 4, 0.25: 4, 0.5: 1 |
| last_self_ecology_integrated_gru_tuned_anchor | 0 | 0: 4, 0.25: 3, 0.5: 1, 1: 1 |
| last_self_ecology_integrated_gru_tuned_anchor | 1 | 0: 4, 0.25: 3, 0.5: 2 |
| last_self_ecology_integrated_gru_tuned_anchor | 3 | 0: 2, 0.25: 4, 0.5: 3 |
| last_self_ecology_integrated_gru_tuned_anchor | 5 | 0: 3, 0.25: 6 |

Mixer and support-adapter choices are fitted independently for each direct head.
At gamma=0 integrated predictions are checked against that head's direct support adapter.

## Station gains and harms

| GRU-basis contrast | Improved / worsened stations | Top-five positive gain | Top-five harm |
|---|---:|---:|---:|
| last_self_ecology_integrated_vs_prior_ecological_gru_tuned_anchor_k0 | 88 / 84 | 33.7% | 44.5% |
| last_self_vs_frozen_gru_tuned_anchor_k0 | 107 / 65 | 63.6% | 36.3% |
| last_self_ecology_vs_frozen_gru_tuned_anchor_k0 | 100 / 72 | 50.5% | 29.7% |
| last_self_ecology_vs_last_self_gru_tuned_anchor_k0 | 103 / 69 | 44.4% | 33.2% |
| last_self_vs_frozen_integrated_gru_tuned_anchor_k0 | 85 / 87 | 63.6% | 27.4% |
| last_self_ecology_vs_frozen_integrated_gru_tuned_anchor_k0 | 94 / 78 | 57.4% | 31.0% |
| last_self_ecology_vs_last_self_integrated_gru_tuned_anchor_k0 | 99 / 73 | 49.2% | 30.0% |
| last_self_ecology_integrated_vs_prior_ecological_gru_tuned_anchor_k5 | 87 / 85 | 36.9% | 56.1% |
| last_self_vs_frozen_gru_tuned_anchor_k5 | 93 / 79 | 57.8% | 42.7% |
| last_self_ecology_vs_frozen_gru_tuned_anchor_k5 | 95 / 77 | 46.9% | 39.6% |
| last_self_ecology_vs_last_self_gru_tuned_anchor_k5 | 95 / 77 | 35.3% | 34.7% |
| last_self_vs_frozen_integrated_gru_tuned_anchor_k5 | 93 / 79 | 57.4% | 44.9% |
| last_self_ecology_vs_frozen_integrated_gru_tuned_anchor_k5 | 97 / 75 | 43.0% | 42.5% |
| last_self_ecology_vs_last_self_integrated_gru_tuned_anchor_k5 | 98 / 74 | 29.3% | 31.7% |

Positive gains and harms have separate denominators. Station contributions retain
equal partition weights and combine repeated station identities.

## Fixed primary contrasts with tail tradeoffs

- **last_self_vs_frozen_gru_tuned_anchor_k0:** MAE reduction +0.43% (95% CI [-0.01, +0.96]); 3/3 partitions and 8/9 fits improve. Q90 Delta MAE -0.0122 [-0.0239, +0.0013]; non-tail Delta MAE -0.0075 [-0.0200, +0.0017]. False Q90 change +0.015 percentage points [+0.000, +0.037].
- **last_self_ecology_vs_frozen_gru_tuned_anchor_k0:** MAE reduction +0.74% (95% CI [+0.07, +1.60]); 3/3 partitions and 9/9 fits improve. Q90 Delta MAE -0.0295 [-0.0573, -0.0005]; non-tail Delta MAE -0.0123 [-0.0322, +0.0018]. False Q90 change +0.030 percentage points [+0.003, +0.066].
- **last_self_ecology_vs_last_self_gru_tuned_anchor_k0:** MAE reduction +0.32% (95% CI [+0.00, +0.68]); 3/3 partitions and 8/9 fits improve. Q90 Delta MAE -0.0173 [-0.0353, +0.0010]; non-tail Delta MAE -0.0048 [-0.0130, +0.0015]. False Q90 change +0.014 percentage points [-0.001, +0.034].
- **last_self_ecology_vs_prior_concentration_gru_tuned_anchor_k0:** MAE reduction +0.74% (95% CI [+0.07, +1.60]); 3/3 partitions and 9/9 fits improve. Q90 Delta MAE -0.0295 [-0.0573, -0.0005]; non-tail Delta MAE -0.0123 [-0.0322, +0.0018]. False Q90 change +0.030 percentage points [+0.003, +0.066].
- **last_self_ecology_integrated_vs_prior_ecological_gru_tuned_anchor_k0:** MAE reduction -0.24% (95% CI [-1.69, +1.12]); 1/3 partitions and 4/9 fits improve. Q90 Delta MAE -0.1570 [-0.2398, -0.0621]; non-tail Delta MAE +0.0228 [-0.0018, +0.0498]. False Q90 change +0.116 percentage points [+0.049, +0.212].
- **last_self_vs_frozen_gru_tuned_anchor_k5:** MAE reduction +0.00% (95% CI [-0.15, +0.20]); 2/3 partitions and 7/9 fits improve. Q90 Delta MAE -0.0018 [-0.0068, +0.0014]; non-tail Delta MAE +0.0002 [-0.0038, +0.0027]. False Q90 change +0.009 percentage points [-0.004, +0.029].
- **last_self_ecology_vs_frozen_gru_tuned_anchor_k5:** MAE reduction +0.11% (95% CI [-0.12, +0.43]); 3/3 partitions and 8/9 fits improve. Q90 Delta MAE -0.0040 [-0.0128, +0.0023]; non-tail Delta MAE -0.0016 [-0.0079, +0.0025]. False Q90 change +0.008 percentage points [-0.017, +0.029].
- **last_self_ecology_vs_last_self_gru_tuned_anchor_k5:** MAE reduction +0.11% (95% CI [+0.01, +0.24]); 3/3 partitions and 6/9 fits improve. Q90 Delta MAE -0.0022 [-0.0068, +0.0017]; non-tail Delta MAE -0.0018 [-0.0043, +0.0000]. False Q90 change -0.001 percentage points [-0.027, +0.019].
- **last_self_ecology_vs_prior_concentration_gru_tuned_anchor_k5:** MAE reduction +0.11% (95% CI [-0.12, +0.43]); 3/3 partitions and 8/9 fits improve. Q90 Delta MAE -0.0040 [-0.0128, +0.0023]; non-tail Delta MAE -0.0016 [-0.0079, +0.0025]. False Q90 change +0.008 percentage points [-0.017, +0.029].
- **last_self_ecology_integrated_vs_prior_ecological_gru_tuned_anchor_k5:** MAE reduction -0.28% (95% CI [-1.02, +0.42]); 2/3 partitions and 5/9 fits improve. Q90 Delta MAE -0.0414 [-0.0637, -0.0214]; non-tail Delta MAE +0.0102 [-0.0016, +0.0240]. False Q90 change +0.073 percentage points [+0.009, +0.164].

The matched controls test which existing representations benefit from the same residual
objective. Any improvement of the ecology-integrated
product can also involve a changed source-validation mixing weight. It does not alone
establish a better neural representation or a physical ecological/transport mechanism.
