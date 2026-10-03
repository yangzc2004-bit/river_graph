# Regime-conditioned DOC residual heads

Nine station-partition/seed packages contain three native-DOC neural fits each. The
same cached spatial encoding, original observation-aware GRU initialization, source
OOF context base, 30-epoch budget, patience 5 and tail weight 2 are used by all arms.

Every head sees the same 30 direct features: ten causal-flow features, nine watershed
ecology values plus nine availability flags, and source-normalized context concentration
plus its flag. Ecology uses source-station median/IQR; concentration uses station-balanced
source OOF log1p prediction mean/SD. Normalized values use z/(1+abs(z)) compression.

The matched head ladder changes only which extra values interact with the hidden state:

- additive: the existing three numerical flow interactions; new regime inputs enter additively.
- concentration: also hidden × predicted concentration (64 additional weights).
- ecological: also hidden × nine ecological values (576 further weights).

All three train GRU/decay and a zero-initialized signed native-unit head. Different interaction
capacities are reported explicitly. No additional tail weighting or concentration threshold
is used as a routing rule. The concentration input is an OOF prediction during training,
not the true concentration. The head-level ecology is already available in the model's
spatial encoder; this experiment tests whether direct conditioning improves its use.

Each new base is also integrated with the same frozen v1 ecological-affine residual profile.
Source-validation selects the K0 residual mixing weight; positive-K selection jointly
chooses mixing and the existing support adapter. Gamma=0 recovers its direct neural base,
gamma=1 replaces the neural correction with ecological memory. Both paths use the same
frozen-v4 GRU support basis. All models and old interaction/ecological-v2 references remain
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
| additive_gru_tuned_anchor | 1.8345 | 1.8066 | 1.6267 | 1.5874 | 3.580 | 0.569 | 0.2311 |
| concentration_gru_tuned_anchor | 1.8436 | 1.8064 | 1.6371 | 1.5883 | 3.573 | 0.571 | 0.2288 |
| ecological_gru_tuned_anchor | 1.8797 | 1.8308 | 1.6518 | 1.5860 | 3.573 | 0.571 | 0.2290 |
| additive_ecology_gru_tuned_anchor | 1.8176 | 1.7952 | 1.6191 | 1.5841 | 3.583 | 0.569 | 0.2295 |
| concentration_ecology_gru_tuned_anchor | 1.8246 | 1.7947 | 1.6193 | 1.5859 | 3.578 | 0.570 | 0.2281 |
| ecological_ecology_gru_tuned_anchor | 1.8555 | 1.8121 | 1.6359 | 1.5856 | 3.583 | 0.569 | 0.2282 |
| prior_interaction_gru_tuned_anchor | 1.8352 | 1.8048 | 1.6269 | 1.5816 | 3.579 | 0.569 | 0.2296 |
| prior_ecological_affine_gru_tuned_anchor | 1.8092 | 1.7917 | 1.6221 | 1.5799 | 3.581 | 0.569 | 0.2283 |

Constant-only adaptation is retained in all CSVs. At K0 it is bitwise identical to
the GRU-support path, so its duplicate bootstrap contrasts are omitted.

## Matched interactions: concentration and ecology

| Comparison | Delta MAE [95% CI] | Reduction [95% CI] | Improved partitions; fits |
|---|---:|---:|---:|
| concentration_vs_additive_constant_k5 | +0.0003 [-0.0084, +0.0101] | -0.02% [-0.61, +0.55] | 2/3; 6/9 |
| ecological_vs_concentration_constant_k5 | +0.0043 [-0.0052, +0.0137] | -0.27% [-0.88, +0.32] | 1/3; 3/9 |
| concentration_vs_additive_ecology_constant_k5 | +0.0010 [-0.0061, +0.0088] | -0.06% [-0.53, +0.40] | 1/3; 5/9 |
| ecological_vs_concentration_ecology_constant_k5 | +0.0021 [-0.0042, +0.0081] | -0.13% [-0.52, +0.26] | 1/3; 2/9 |
| concentration_vs_additive_gru_tuned_anchor_k0 | +0.0091 [-0.0174, +0.0361] | -0.50% [-1.91, +1.02] | 1/3; 3/9 |
| ecological_vs_concentration_gru_tuned_anchor_k0 | +0.0360 [-0.0027, +0.0770] | -1.95% [-4.36, +0.14] | 1/3; 3/9 |
| concentration_vs_additive_ecology_gru_tuned_anchor_k0 | +0.0070 [-0.0188, +0.0329] | -0.38% [-1.76, +1.10] | 2/3; 2/9 |
| ecological_vs_concentration_ecology_gru_tuned_anchor_k0 | +0.0310 [-0.0035, +0.0675] | -1.70% [-3.89, +0.19] | 0/3; 1/9 |
| concentration_vs_additive_gru_tuned_anchor_k5 | +0.0009 [-0.0076, +0.0107] | -0.05% [-0.64, +0.51] | 2/3; 5/9 |
| ecological_vs_concentration_gru_tuned_anchor_k5 | -0.0023 [-0.0137, +0.0081] | +0.15% [-0.55, +0.83] | 2/3; 4/9 |
| concentration_vs_additive_ecology_gru_tuned_anchor_k5 | +0.0018 [-0.0049, +0.0097] | -0.11% [-0.59, +0.32] | 1/3; 3/9 |
| ecological_vs_concentration_ecology_gru_tuned_anchor_k5 | -0.0003 [-0.0066, +0.0054] | +0.02% [-0.35, +0.40] | 2/3; 4/9 |

## New direct heads versus prior temporal interaction

| Comparison | Delta MAE [95% CI] | Reduction [95% CI] | Improved partitions; fits |
|---|---:|---:|---:|
| additive_vs_prior_interaction_constant_k5 | -0.0012 [-0.0105, +0.0087] | +0.07% [-0.56, +0.65] | 2/3; 4/9 |
| concentration_vs_prior_interaction_constant_k5 | -0.0008 [-0.0139, +0.0129] | +0.05% [-0.79, +0.88] | 2/3; 7/9 |
| ecological_vs_prior_interaction_constant_k5 | +0.0034 [-0.0110, +0.0173] | -0.22% [-1.07, +0.67] | 2/3; 5/9 |
| additive_vs_prior_interaction_gru_tuned_anchor_k0 | -0.0007 [-0.0301, +0.0305] | +0.04% [-1.76, +1.57] | 2/3; 6/9 |
| concentration_vs_prior_interaction_gru_tuned_anchor_k0 | +0.0084 [-0.0239, +0.0413] | -0.46% [-2.27, +1.29] | 1/3; 5/9 |
| ecological_vs_prior_interaction_gru_tuned_anchor_k0 | +0.0444 [-0.0147, +0.1069] | -2.42% [-6.05, +0.77] | 1/3; 4/9 |
| additive_vs_prior_interaction_gru_tuned_anchor_k5 | +0.0058 [-0.0052, +0.0177] | -0.37% [-1.13, +0.32] | 2/3; 4/9 |
| concentration_vs_prior_interaction_gru_tuned_anchor_k5 | +0.0067 [-0.0077, +0.0227] | -0.42% [-1.40, +0.50] | 2/3; 5/9 |
| ecological_vs_prior_interaction_gru_tuned_anchor_k5 | +0.0044 [-0.0080, +0.0166] | -0.28% [-1.02, +0.50] | 2/3; 5/9 |

## Integrated heads versus prior ecological-v2 model

| Comparison | Delta MAE [95% CI] | Reduction [95% CI] | Improved partitions; fits |
|---|---:|---:|---:|
| additive_ecology_vs_prior_ecological_constant_k5 | -0.0017 [-0.0100, +0.0071] | +0.11% [-0.45, +0.61] | 3/3; 4/9 |
| concentration_ecology_vs_prior_ecological_constant_k5 | -0.0008 [-0.0104, +0.0095] | +0.05% [-0.58, +0.66] | 2/3; 7/9 |
| ecological_ecology_vs_prior_ecological_constant_k5 | +0.0013 [-0.0091, +0.0114] | -0.08% [-0.71, +0.57] | 2/3; 6/9 |
| additive_ecology_vs_prior_ecological_gru_tuned_anchor_k0 | +0.0084 [-0.0177, +0.0363] | -0.47% [-2.13, +0.93] | 1/3; 4/9 |
| concentration_ecology_vs_prior_ecological_gru_tuned_anchor_k0 | +0.0154 [-0.0114, +0.0419] | -0.85% [-2.37, +0.60] | 1/3; 3/9 |
| ecological_ecology_vs_prior_ecological_gru_tuned_anchor_k0 | +0.0463 [-0.0057, +0.1023] | -2.56% [-5.91, +0.29] | 0/3; 2/9 |
| additive_ecology_vs_prior_ecological_gru_tuned_anchor_k5 | +0.0042 [-0.0053, +0.0145] | -0.26% [-0.93, +0.34] | 1/3; 3/9 |
| concentration_ecology_vs_prior_ecological_gru_tuned_anchor_k5 | +0.0060 [-0.0062, +0.0195] | -0.38% [-1.21, +0.40] | 1/3; 5/9 |
| ecological_ecology_vs_prior_ecological_gru_tuned_anchor_k5 | +0.0057 [-0.0069, +0.0192] | -0.36% [-1.21, +0.44] | 2/3; 6/9 |

## Ecological integration versus each direct head

| Comparison | Delta MAE [95% CI] | Reduction [95% CI] | Improved partitions; fits |
|---|---:|---:|---:|
| additive_integrated_vs_direct_constant_k5 | -0.0026 [-0.0065, +0.0014] | +0.16% [-0.08, +0.43] | 2/3; 3/9 |
| concentration_integrated_vs_direct_constant_k5 | -0.0020 [-0.0054, +0.0012] | +0.12% [-0.08, +0.34] | 1/3; 3/9 |
| ecological_integrated_vs_direct_constant_k5 | -0.0041 [-0.0104, +0.0020] | +0.26% [-0.12, +0.69] | 1/3; 5/9 |
| additive_integrated_vs_direct_gru_tuned_anchor_k0 | -0.0169 [-0.0297, -0.0059] | +0.92% [+0.32, +1.61] | 2/3; 5/9 |
| concentration_integrated_vs_direct_gru_tuned_anchor_k0 | -0.0190 [-0.0337, -0.0068] | +1.03% [+0.37, +1.82] | 2/3; 5/9 |
| ecological_integrated_vs_direct_gru_tuned_anchor_k0 | -0.0241 [-0.0376, -0.0119] | +1.28% [+0.62, +1.99] | 2/3; 6/9 |
| additive_integrated_vs_direct_gru_tuned_anchor_k5 | -0.0034 [-0.0074, +0.0006] | +0.21% [-0.04, +0.49] | 2/3; 5/9 |
| concentration_integrated_vs_direct_gru_tuned_anchor_k5 | -0.0024 [-0.0052, +0.0002] | +0.15% [-0.01, +0.33] | 2/3; 4/9 |
| ecological_integrated_vs_direct_gru_tuned_anchor_k5 | -0.0004 [-0.0106, +0.0109] | +0.02% [-0.66, +0.69] | 1/3; 4/9 |

## Tail, ordinary concentrations and false alarms

| Saved model | K | Q90 MAE | Q90 bias | Non-tail MAE | Non-tail log MAE | False Q90 rate |
|---|---:|---:|---:|---:|---:|---:|
| context_gru_tuned_anchor | 0 | 7.5586 | -6.6062 | 1.2574 | 0.2614 | 1.91% |
| context_gru_tuned_anchor | 5 | 6.6945 | -5.1653 | 1.0084 | 0.2047 | 2.21% |
| additive_gru_tuned_anchor | 0 | 7.3258 | -6.2097 | 1.2062 | 0.2549 | 2.01% |
| additive_gru_tuned_anchor | 5 | 6.6461 | -5.0395 | 1.0039 | 0.2071 | 2.18% |
| concentration_gru_tuned_anchor | 0 | 7.2842 | -6.0655 | 1.2213 | 0.2517 | 2.08% |
| concentration_gru_tuned_anchor | 5 | 6.6244 | -4.9737 | 1.0075 | 0.2047 | 2.21% |
| ecological_gru_tuned_anchor | 0 | 7.2986 | -6.0285 | 1.2591 | 0.2679 | 2.03% |
| ecological_gru_tuned_anchor | 5 | 6.6296 | -4.9604 | 1.0044 | 0.2049 | 2.23% |
| additive_ecology_gru_tuned_anchor | 0 | 7.3423 | -6.2579 | 1.1851 | 0.2506 | 2.01% |
| additive_ecology_gru_tuned_anchor | 5 | 6.6603 | -5.0834 | 0.9985 | 0.2053 | 2.19% |
| concentration_ecology_gru_tuned_anchor | 0 | 7.3052 | -6.1253 | 1.1971 | 0.2470 | 2.08% |
| concentration_ecology_gru_tuned_anchor | 5 | 6.6390 | -5.0230 | 1.0032 | 0.2038 | 2.21% |
| ecological_ecology_gru_tuned_anchor | 0 | 7.3131 | -6.1077 | 1.2302 | 0.2623 | 2.02% |
| ecological_ecology_gru_tuned_anchor | 5 | 6.6676 | -5.0685 | 0.9995 | 0.2036 | 2.19% |
| prior_interaction_gru_tuned_anchor | 0 | 7.4371 | -6.4393 | 1.1931 | 0.2505 | 1.95% |
| prior_interaction_gru_tuned_anchor | 5 | 6.6746 | -5.1066 | 0.9937 | 0.2051 | 2.14% |
| prior_ecological_affine_gru_tuned_anchor | 0 | 7.4318 | -6.4542 | 1.1653 | 0.2423 | 1.99% |
| prior_ecological_affine_gru_tuned_anchor | 5 | 6.6808 | -5.1324 | 0.9913 | 0.2035 | 2.14% |

False Q90 rate conditions on an observed concentration below the source Q90 threshold.
All regional raw/log MAE and bias intervals are in comparisons.csv.

## Source-validation training choices

| Head | Trainable parameters | Head parameters | Selected epoch range | Scale counts | Mean validation MAE change |
|---|---:|---:|---|---|---:|
| additive | 25759 | 287 | 2–30 | 0.5: 3, 1: 6 | -0.0967 |
| concentration | 25823 | 351 | 5–30 | 0.5: 3, 1: 6 | -0.0878 |
| ecological | 26399 | 927 | 4–30 | 0.5: 3, 1: 6 | -0.0907 |

The selected scale applies to the raw head output before adding it to context.
Saved full-grid arm_delta columns are unscaled; realized corrections also include
the selected scale and zero concentration floor. Raw head magnitudes are not directly
comparable as applied prediction corrections.

## Source-validation ecological mixing

| Integrated model | K | Gamma counts |
|---|---:|---|
| additive_ecology_constant | 0 | 0: 4, 0.25: 3, 0.5: 1, 1: 1 |
| additive_ecology_constant | 1 | 0: 4, 0.25: 3, 0.5: 2 |
| additive_ecology_constant | 3 | 0: 2, 0.25: 5, 0.5: 2 |
| additive_ecology_constant | 5 | 0: 5, 0.25: 3, 0.5: 1 |
| additive_ecology_gru_tuned_anchor | 0 | 0: 4, 0.25: 3, 0.5: 1, 1: 1 |
| additive_ecology_gru_tuned_anchor | 1 | 0: 4, 0.25: 3, 0.5: 2 |
| additive_ecology_gru_tuned_anchor | 3 | 0: 1, 0.25: 5, 0.5: 3 |
| additive_ecology_gru_tuned_anchor | 5 | 0: 3, 0.25: 5, 0.5: 1 |
| concentration_ecology_constant | 0 | 0: 4, 0.25: 2, 0.5: 2, 1: 1 |
| concentration_ecology_constant | 1 | 0: 4, 0.25: 3, 0.5: 2 |
| concentration_ecology_constant | 3 | 0: 3, 0.25: 4, 0.5: 2 |
| concentration_ecology_constant | 5 | 0: 4, 0.25: 4, 0.5: 1 |
| concentration_ecology_gru_tuned_anchor | 0 | 0: 4, 0.25: 2, 0.5: 2, 1: 1 |
| concentration_ecology_gru_tuned_anchor | 1 | 0: 4, 0.25: 3, 0.5: 2 |
| concentration_ecology_gru_tuned_anchor | 3 | 0: 2, 0.25: 4, 0.5: 3 |
| concentration_ecology_gru_tuned_anchor | 5 | 0: 4, 0.25: 5 |
| ecological_ecology_constant | 0 | 0: 3, 0.25: 4, 0.5: 2 |
| ecological_ecology_constant | 1 | 0: 3, 0.25: 4, 0.5: 2 |
| ecological_ecology_constant | 3 | 0.25: 7, 0.5: 1, 1: 1 |
| ecological_ecology_constant | 5 | 0: 2, 0.25: 5, 0.5: 2 |
| ecological_ecology_gru_tuned_anchor | 0 | 0: 3, 0.25: 4, 0.5: 2 |
| ecological_ecology_gru_tuned_anchor | 1 | 0: 3, 0.25: 4, 0.5: 2 |
| ecological_ecology_gru_tuned_anchor | 3 | 0: 1, 0.25: 2, 0.5: 5, 1: 1 |
| ecological_ecology_gru_tuned_anchor | 5 | 0: 2, 0.25: 5, 0.5: 2 |

Mixer and support-adapter choices are fitted independently for each direct head.
At gamma=0 integrated predictions are checked against that head's direct support adapter.

## Station gains and harms

| GRU-basis contrast | Improved / worsened stations | Top-five positive gain | Top-five harm |
|---|---:|---:|---:|
| ecological_vs_prior_interaction_gru_tuned_anchor_k0 | 91 / 81 | 29.2% | 56.0% |
| ecological_ecology_vs_prior_ecological_gru_tuned_anchor_k0 | 84 / 88 | 33.7% | 56.9% |
| ecological_vs_concentration_gru_tuned_anchor_k0 | 84 / 88 | 31.1% | 60.7% |
| ecological_vs_concentration_ecology_gru_tuned_anchor_k0 | 87 / 85 | 28.9% | 62.3% |
| ecological_vs_prior_interaction_gru_tuned_anchor_k5 | 90 / 82 | 34.6% | 45.5% |
| ecological_ecology_vs_prior_ecological_gru_tuned_anchor_k5 | 89 / 83 | 39.4% | 60.7% |
| ecological_vs_concentration_gru_tuned_anchor_k5 | 95 / 77 | 49.9% | 51.6% |
| ecological_vs_concentration_ecology_gru_tuned_anchor_k5 | 85 / 87 | 48.1% | 40.4% |

Positive gains and harms have separate denominators. Station contributions retain
equal partition weights and combine repeated station identities.

## Fixed primary contrasts with tail tradeoffs

- **concentration_vs_additive_gru_tuned_anchor_k0:** MAE reduction -0.50% (95% CI [-1.91, +1.02]); 1/3 partitions and 3/9 fits improve. Q90 Delta MAE -0.0416 [-0.0770, +0.0018]; non-tail Delta MAE +0.0151 [-0.0156, +0.0464]. False Q90 change +0.073 percentage points [+0.027, +0.138].
- **ecological_vs_concentration_gru_tuned_anchor_k0:** MAE reduction -1.95% (95% CI [-4.36, +0.14]); 1/3 partitions and 3/9 fits improve. Q90 Delta MAE +0.0145 [-0.0352, +0.0787]; non-tail Delta MAE +0.0379 [-0.0043, +0.0833]. False Q90 change -0.048 percentage points [-0.124, +0.013].
- **ecological_vs_prior_interaction_gru_tuned_anchor_k0:** MAE reduction -2.42% (95% CI [-6.05, +0.77]); 1/3 partitions and 4/9 fits improve. Q90 Delta MAE -0.1385 [-0.2441, -0.0129]; non-tail Delta MAE +0.0660 [+0.0056, +0.1347]. False Q90 change +0.084 percentage points [-0.007, +0.207].
- **ecological_ecology_vs_prior_ecological_gru_tuned_anchor_k0:** MAE reduction -2.56% (95% CI [-5.91, +0.29]); 0/3 partitions and 2/9 fits improve. Q90 Delta MAE -0.1187 [-0.2173, +0.0062]; non-tail Delta MAE +0.0649 [+0.0111, +0.1271]. False Q90 change +0.032 percentage points [-0.048, +0.110].
- **concentration_vs_additive_gru_tuned_anchor_k5:** MAE reduction -0.05% (95% CI [-0.64, +0.51]); 2/3 partitions and 5/9 fits improve. Q90 Delta MAE -0.0216 [-0.0372, -0.0002]; non-tail Delta MAE +0.0036 [-0.0061, +0.0149]. False Q90 change +0.031 percentage points [-0.001, +0.078].
- **ecological_vs_concentration_gru_tuned_anchor_k5:** MAE reduction +0.15% (95% CI [-0.55, +0.83]); 2/3 partitions and 4/9 fits improve. Q90 Delta MAE +0.0052 [-0.0175, +0.0259]; non-tail Delta MAE -0.0031 [-0.0157, +0.0086]. False Q90 change +0.017 percentage points [-0.007, +0.045].
- **ecological_vs_prior_interaction_gru_tuned_anchor_k5:** MAE reduction -0.28% (95% CI [-1.02, +0.50]); 2/3 partitions and 5/9 fits improve. Q90 Delta MAE -0.0449 [-0.0749, -0.0167]; non-tail Delta MAE +0.0108 [-0.0018, +0.0240]. False Q90 change +0.083 percentage points [+0.021, +0.177].
- **ecological_ecology_vs_prior_ecological_gru_tuned_anchor_k5:** MAE reduction -0.36% (95% CI [-1.21, +0.44]); 2/3 partitions and 6/9 fits improve. Q90 Delta MAE -0.0132 [-0.0314, +0.0112]; non-tail Delta MAE +0.0082 [-0.0055, +0.0239]. False Q90 change +0.058 percentage points [+0.010, +0.129].

The matched controls distinguish access to additional inputs from conditioning the
recurrent-state correction on those inputs. Any improvement of the ecology-integrated
product can also involve a changed source-validation mixing weight. It does not alone
establish a better neural representation or a physical ecological/transport mechanism.
