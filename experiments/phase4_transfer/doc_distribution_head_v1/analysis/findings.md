# Conditional distribution heads for sparse DOC reconstruction

Three heads are compared: the retained point model, a single conditional distribution,
and a conditional mixture. Density models provide median point predictions. Forests,
ecological residual profiles, support representations, and query cells remain unchanged.
The density models are fitted on source training data; source-validation native MAE
selects their checkpoint and blend, including the exact zero-blend fallback. A better
likelihood fit alone does not establish a better reconstruction model.

All fourteen model curves and twelve specified contrasts are reported. Direct comparisons
include the source-validation support calibration; integrated comparisons additionally
include source-validation-selected ecological mixing. Constant-only support is retained
as a diagnostic. The principal contrasts use the unchanged GRU support basis at K0/K5.

All intervals use 5,000 paired whole-station bootstrap draws. Repeated station identities
are resampled jointly across partitions. Seed means average within each partition, then
partitions receive equal weight. Q90 includes ties at the source-training threshold.
These are reused development partitions; target outcomes do not choose a head, blend,
checkpoint, support calibration, ecological mixture or K-specific route.

The loader verifies 54 copied-context/point model comparisons.

## Complete K curves

| Model | K0 MAE | K1 MAE | K3 MAE | K5 MAE | K5 RMSE | K5 R² |
|---|---:|---:|---:|---:|---:|---:|
| context_constant | 1.902823 | 1.866563 | 1.643669 | 1.609381 | 3.614595 | 0.562102 |
| context_gru_tuned_anchor | 1.902823 | 1.866563 | 1.634818 | 1.596001 | 3.591068 | 0.567520 |
| point_constant | 1.806996 | 1.758826 | 1.641713 | 1.591869 | 3.611671 | 0.563919 |
| point_gru_tuned_anchor | 1.806996 | 1.758826 | 1.631317 | 1.589387 | 3.604470 | 0.565514 |
| single_constant | 1.807848 | 1.761706 | 1.639395 | 1.591330 | 3.608595 | 0.564536 |
| single_gru_tuned_anchor | 1.807848 | 1.761706 | 1.630858 | 1.587454 | 3.587161 | 0.568946 |
| mixture_constant | 1.801902 | 1.755767 | 1.633394 | 1.587703 | 3.604830 | 0.565192 |
| mixture_gru_tuned_anchor | 1.801902 | 1.755767 | 1.634876 | 1.582850 | 3.597131 | 0.566916 |
| point_integrated_constant | 1.803538 | 1.755368 | 1.632934 | 1.574914 | 3.572022 | 0.571864 |
| point_integrated_gru_tuned_anchor | 1.803538 | 1.755368 | 1.620090 | 1.564984 | 3.552260 | 0.575857 |
| single_integrated_constant | 1.805217 | 1.759075 | 1.631058 | 1.572862 | 3.568811 | 0.572704 |
| single_integrated_gru_tuned_anchor | 1.805217 | 1.759075 | 1.620028 | 1.563876 | 3.550941 | 0.576178 |
| mixture_integrated_constant | 1.799271 | 1.753136 | 1.627379 | 1.579142 | 3.594343 | 0.567479 |
| mixture_integrated_gru_tuned_anchor | 1.799271 | 1.753136 | 1.623813 | 1.562012 | 3.545380 | 0.577380 |

## Density medians versus point head: direct

| Comparison | ΔMAE [95% CI] | ΔQ90 MAE [95% CI] | Δordinary MAE [95% CI] | Δrecall [95% CI], pp | Δfalse-Q90 [95% CI], pp |
|---|---:|---:|---:|---:|---:|
| single_vs_point_gru_tuned_anchor_k0 | +0.000852 [-0.019416, +0.020469] | +0.025456 [-0.067076, +0.136359] | -0.002986 [-0.023875, +0.015596] | -2.054234 [-4.280614, -0.519413] | -0.087527 [-0.205803, +0.007044] |
| single_vs_point_gru_tuned_anchor_k5 | -0.001933 [-0.013086, +0.007971] | -0.007327 [-0.056391, +0.040638] | -0.001968 [-0.012401, +0.007281] | -0.061683 [-0.621517, +0.575148] | -0.013501 [-0.136866, +0.098484] |
| mixture_vs_point_gru_tuned_anchor_k0 | -0.005094 [-0.027677, +0.015423] | +0.024144 [-0.065274, +0.126205] | -0.009294 [-0.033732, +0.012514] | -1.075676 [-2.342068, -0.014254] | -0.046374 [-0.176169, +0.058660] |
| mixture_vs_point_gru_tuned_anchor_k5 | -0.006536 [-0.018790, +0.006107] | +0.001037 [-0.060187, +0.058814] | -0.007704 [-0.019805, +0.005322] | +0.051478 [-0.392269, +0.544996] | +0.016741 [-0.062383, +0.100119] |

## Density medians versus point head: integrated

| Comparison | ΔMAE [95% CI] | ΔQ90 MAE [95% CI] | Δordinary MAE [95% CI] | Δrecall [95% CI], pp | Δfalse-Q90 [95% CI], pp |
|---|---:|---:|---:|---:|---:|
| single_vs_point_integrated_gru_tuned_anchor_k0 | +0.001679 [-0.020358, +0.022722] | +0.020178 [-0.071631, +0.130330] | -0.001389 [-0.024652, +0.019184] | -1.914847 [-4.136773, -0.373835] | -0.072461 [-0.188402, +0.020188] |
| single_vs_point_integrated_gru_tuned_anchor_k5 | -0.001108 [-0.005610, +0.003585] | +0.006642 [-0.008207, +0.020197] | -0.002047 [-0.007063, +0.002800] | +0.251077 [-0.085222, +0.639766] | -0.006954 [-0.051596, +0.036079] |
| mixture_vs_point_integrated_gru_tuned_anchor_k0 | -0.004267 [-0.028346, +0.017739] | +0.018866 [-0.068414, +0.120354] | -0.007698 [-0.034826, +0.015683] | -0.936289 [-2.207938, +0.123763] | -0.031308 [-0.159675, +0.076505] |
| mixture_vs_point_integrated_gru_tuned_anchor_k5 | -0.002972 [-0.011097, +0.004623] | -0.008754 [-0.040835, +0.023025] | -0.002274 [-0.011285, +0.006016] | +0.161720 [-0.161527, +0.492852] | +0.003496 [-0.050508, +0.058632] |

## Mixture versus single distribution: direct

| Comparison | ΔMAE [95% CI] | ΔQ90 MAE [95% CI] | Δordinary MAE [95% CI] | Δrecall [95% CI], pp | Δfalse-Q90 [95% CI], pp |
|---|---:|---:|---:|---:|---:|
| mixture_vs_single_gru_tuned_anchor_k0 | -0.005946 [-0.020508, +0.006932] | -0.001312 [-0.063368, +0.044142] | -0.006309 [-0.020916, +0.007848] | +0.978558 [-0.193667, +2.824555] | +0.041153 [-0.022389, +0.129721] |
| mixture_vs_single_gru_tuned_anchor_k5 | -0.004604 [-0.014506, +0.005864] | +0.008364 [-0.032444, +0.049410] | -0.005735 [-0.015868, +0.004333] | +0.113161 [-0.433388, +0.606895] | +0.030243 [-0.057820, +0.145444] |

## Mixture versus single distribution: integrated

| Comparison | ΔMAE [95% CI] | ΔQ90 MAE [95% CI] | Δordinary MAE [95% CI] | Δrecall [95% CI], pp | Δfalse-Q90 [95% CI], pp |
|---|---:|---:|---:|---:|---:|
| mixture_vs_single_integrated_gru_tuned_anchor_k0 | -0.005946 [-0.020508, +0.006932] | -0.001312 [-0.063368, +0.044142] | -0.006309 [-0.020916, +0.007848] | +0.978558 [-0.193667, +2.824555] | +0.041153 [-0.022389, +0.129721] |
| mixture_vs_single_integrated_gru_tuned_anchor_k5 | -0.001864 [-0.006868, +0.002679] | -0.015396 [-0.040520, +0.009127] | -0.000226 [-0.005643, +0.004809] | -0.089357 [-0.498906, +0.283897] | +0.010450 [-0.010695, +0.035929] |

## Tail and ordinary DOC with detection tradeoffs

| Model | K | Q90 MAE | Q90 bias | Ordinary MAE | Ordinary bias | Recall | Precision | False-Q90 rate |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| context_gru_tuned_anchor | 0 | 7.558635 | -6.606236 | 1.257359 | +0.482039 | 58.898% | 78.465% | 1.909% |
| context_gru_tuned_anchor | 5 | 6.694451 | -5.165304 | 1.008430 | +0.231465 | 65.013% | 77.677% | 2.208% |
| point_gru_tuned_anchor | 0 | 7.167345 | -5.532017 | 1.194119 | +0.213345 | 64.363% | 77.424% | 2.195% |
| point_gru_tuned_anchor | 5 | 6.642277 | -5.175765 | 1.007548 | +0.166518 | 65.545% | 77.741% | 2.203% |
| single_gru_tuned_anchor | 0 | 7.192801 | -5.632162 | 1.191133 | +0.242859 | 62.309% | 77.512% | 2.107% |
| single_gru_tuned_anchor | 5 | 6.634950 | -5.004055 | 1.005580 | +0.174582 | 65.484% | 77.826% | 2.190% |
| mixture_gru_tuned_anchor | 0 | 7.191489 | -5.684372 | 1.184825 | +0.175937 | 63.287% | 77.410% | 2.148% |
| mixture_gru_tuned_anchor | 5 | 6.643314 | -5.112246 | 0.999844 | +0.158348 | 65.597% | 77.661% | 2.220% |
| point_integrated_gru_tuned_anchor | 0 | 7.179187 | -5.591957 | 1.188732 | +0.203547 | 64.064% | 77.636% | 2.157% |
| point_integrated_gru_tuned_anchor | 5 | 6.586994 | -4.965678 | 0.985640 | +0.171136 | 65.134% | 77.807% | 2.188% |
| single_integrated_gru_tuned_anchor | 0 | 7.199364 | -5.659347 | 1.187343 | +0.238935 | 62.150% | 77.646% | 2.085% |
| single_integrated_gru_tuned_anchor | 5 | 6.593637 | -4.936693 | 0.983593 | +0.173099 | 65.385% | 77.940% | 2.181% |
| mixture_integrated_gru_tuned_anchor | 0 | 7.198052 | -5.711557 | 1.181034 | +0.172012 | 63.128% | 77.545% | 2.126% |
| mixture_integrated_gru_tuned_anchor | 5 | 6.578240 | -4.921038 | 0.983366 | +0.168989 | 65.296% | 77.804% | 2.191% |

## Source-validation fitting and model selection

Epoch zero uses a source-initialized constant density; it is not necessarily the exact
point fallback. Only correction scale zero guarantees unchanged native point predictions.
All feature normalization and density initialization statistics use source training rows.
The source base includes a source-trained neural correction over forest OOF predictions;
the entire neural source base is not claimed to be OOF.

Saved source-training and source-validation likelihood traces are separate from native-MAE
checkpoint/blend choices. MAE ties prefer the lower correction scale, then the earlier epoch.
Epoch>0 training-NLL traces average online minibatch losses; selected_source_nll evaluates
the complete source set at the selected checkpoint. Component collapse and scale diagnostics describe the conditional
density fit. They are not additional target endpoints or evidence of calibrated uncertainty.
Any density intervals are exploratory distribution diagnostics; this study promotes models
only through point-prediction performance. All selected blending and support/mixing parameters
are preserved with their source-validation candidate scores.

| Arm | Selected epochs | Exact point fallbacks | Mean baseline MAE | Mean selected MAE | Mean initial NLL | Mean selected NLL |
|---|---:|---:|---:|---:|---:|---:|
| single | 0–25 | 1/9 | 1.804428 | 1.786599 | 0.476837 | 0.449816 |
| mixture | 0–50 | 1/9 | 1.804428 | 1.774160 | 0.472704 | 0.404354 |

Partition/seed directions and station gain/harm concentration accompany aggregate effects.
Intervals crossing zero do not establish equivalence. No target-based route is produced.
