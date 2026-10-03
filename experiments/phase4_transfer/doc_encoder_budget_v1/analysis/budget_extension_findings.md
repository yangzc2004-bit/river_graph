# Matched 60-versus-30 epoch budget comparison

The source-validation traces at the earlier ceiling motivated a longer maximum budget.
The objective, tail weight, learning rates, initialization, early-stopping patience, data,
fixed target queries, source thresholds and candidate modes remain unchanged. Each budget
uses its own source-validation-selected checkpoint, scale and support/ecological mixing.
Thus integrated-product differences include changes in those validation choices as well as
training duration. No model or target threshold is selected by this comparison.

The reference is `experiments/phase4_transfer/doc_encoder_residual_v1`. All nine packages are bound to their saved files.
The same 5,000 joint whole-station bootstrap draws and equal-partition estimator are used.
K0 constant and GRU-support predictions coincide; their repeated rows describe the same evidence.
Negative Delta MAE or false-positive-rate change favors the extended budget.

## Overall and concentration-specific paired effects

| Mode / support path | K | Overall Delta MAE [95% CI] | Reduction [95% CI] | Q90 Delta MAE [95% CI] | Non-tail Delta MAE [95% CI] | False Q90 change, percentage points [95% CI] | Improved partitions; fits |
|---|---:|---:|---:|---:|---:|---:|---:|
| frozen_constant | 0 | -0.0124 [-0.0318, +0.0054] | +0.67% [-0.32, +1.62] | -0.0258 [-0.0592, +0.0049] | -0.0109 [-0.0331, +0.0102] | +0.0123 [-0.0247, +0.0575] | 1/3; 3/9 |
| frozen_constant | 5 | +0.0054 [-0.0015, +0.0139] | -0.34% [-0.83, +0.09] | +0.0218 [-0.0291, +0.0725] | +0.0038 [-0.0015, +0.0091] | +0.0014 [-0.0303, +0.0306] | 0/3; 2/9 |
| frozen_gru_tuned_anchor | 0 | -0.0124 [-0.0318, +0.0054] | +0.67% [-0.32, +1.62] | -0.0258 [-0.0592, +0.0049] | -0.0109 [-0.0331, +0.0102] | +0.0123 [-0.0247, +0.0575] | 1/3; 3/9 |
| frozen_gru_tuned_anchor | 5 | +0.0052 [-0.0012, +0.0131] | -0.33% [-0.79, +0.08] | +0.0216 [-0.0259, +0.0688] | +0.0036 [-0.0015, +0.0087] | +0.0162 [-0.0164, +0.0492] | 0/3; 2/9 |
| frozen_integrated_constant | 0 | -0.0129 [-0.0322, +0.0050] | +0.71% [-0.29, +1.68] | -0.0258 [-0.0579, +0.0040] | -0.0116 [-0.0338, +0.0096] | +0.0148 [-0.0196, +0.0579] | 1/3; 3/9 |
| frozen_integrated_constant | 5 | -0.0005 [-0.0046, +0.0030] | +0.03% [-0.19, +0.28] | -0.0025 [-0.0188, +0.0093] | -0.0002 [-0.0045, +0.0035] | +0.0050 [-0.0130, +0.0279] | 1/3; 2/9 |
| frozen_integrated_gru_tuned_anchor | 0 | -0.0129 [-0.0322, +0.0050] | +0.71% [-0.29, +1.68] | -0.0258 [-0.0579, +0.0040] | -0.0116 [-0.0338, +0.0096] | +0.0148 [-0.0196, +0.0579] | 1/3; 3/9 |
| frozen_integrated_gru_tuned_anchor | 5 | -0.0025 [-0.0065, +0.0007] | +0.16% [-0.05, +0.39] | +0.0004 [-0.0107, +0.0120] | -0.0027 [-0.0072, +0.0007] | +0.0162 [-0.0027, +0.0415] | 1/3; 3/9 |
| last_self_constant | 0 | -0.0077 [-0.0228, +0.0056] | +0.42% [-0.31, +1.21] | -0.0188 [-0.0518, +0.0048] | -0.0067 [-0.0233, +0.0084] | +0.0348 [-0.0263, +0.1178] | 1/3; 3/9 |
| last_self_constant | 5 | +0.0095 [-0.0013, +0.0237] | -0.59% [-1.40, +0.08] | +0.0529 [-0.0275, +0.1411] | +0.0052 [-0.0028, +0.0125] | +0.0067 [-0.0479, +0.0611] | 1/3; 1/9 |
| last_self_gru_tuned_anchor | 0 | -0.0077 [-0.0228, +0.0056] | +0.42% [-0.31, +1.21] | -0.0188 [-0.0518, +0.0048] | -0.0067 [-0.0233, +0.0084] | +0.0348 [-0.0263, +0.1178] | 1/3; 3/9 |
| last_self_gru_tuned_anchor | 5 | +0.0053 [-0.0009, +0.0131] | -0.34% [-0.78, +0.06] | +0.0272 [-0.0222, +0.0780] | +0.0032 [-0.0014, +0.0075] | +0.0086 [-0.0259, +0.0370] | 1/3; 2/9 |
| last_self_integrated_constant | 0 | -0.0077 [-0.0228, +0.0056] | +0.43% [-0.32, +1.21] | -0.0193 [-0.0519, +0.0042] | -0.0066 [-0.0232, +0.0086] | +0.0399 [-0.0200, +0.1225] | 1/3; 3/9 |
| last_self_integrated_constant | 5 | +0.0062 [-0.0004, +0.0143] | -0.39% [-0.86, +0.02] | +0.0265 [-0.0219, +0.0781] | +0.0042 [-0.0008, +0.0096] | +0.0000 [-0.0270, +0.0240] | 1/3; 2/9 |
| last_self_integrated_gru_tuned_anchor | 0 | -0.0077 [-0.0228, +0.0056] | +0.43% [-0.32, +1.21] | -0.0193 [-0.0519, +0.0042] | -0.0066 [-0.0232, +0.0086] | +0.0399 [-0.0200, +0.1225] | 1/3; 3/9 |
| last_self_integrated_gru_tuned_anchor | 5 | +0.0062 [-0.0002, +0.0139] | -0.39% [-0.83, +0.01] | +0.0261 [-0.0206, +0.0752] | +0.0042 [-0.0005, +0.0091] | +0.0000 [-0.0363, +0.0288] | 1/3; 2/9 |
| last_self_ecology_constant | 0 | -0.0059 [-0.0227, +0.0108] | +0.32% [-0.62, +1.18] | -0.0178 [-0.0543, +0.0139] | -0.0046 [-0.0234, +0.0146] | +0.0599 [+0.0057, +0.1348] | 2/3; 3/9 |
| last_self_ecology_constant | 5 | +0.0071 [-0.0038, +0.0211] | -0.45% [-1.25, +0.24] | +0.0514 [-0.0193, +0.1286] | +0.0027 [-0.0066, +0.0116] | +0.0109 [-0.0420, +0.0710] | 1/3; 1/9 |
| last_self_ecology_gru_tuned_anchor | 0 | -0.0059 [-0.0227, +0.0108] | +0.32% [-0.62, +1.18] | -0.0178 [-0.0543, +0.0139] | -0.0046 [-0.0234, +0.0146] | +0.0599 [+0.0057, +0.1348] | 2/3; 3/9 |
| last_self_ecology_gru_tuned_anchor | 5 | +0.0082 [-0.0034, +0.0233] | -0.51% [-1.39, +0.21] | +0.0570 [-0.0214, +0.1452] | +0.0033 [-0.0060, +0.0125] | +0.0124 [-0.0383, +0.0676] | 1/3; 1/9 |
| last_self_ecology_integrated_constant | 0 | -0.0028 [-0.0197, +0.0147] | +0.16% [-0.85, +1.05] | -0.0235 [-0.0629, +0.0100] | -0.0004 [-0.0193, +0.0197] | +0.0771 [+0.0215, +0.1548] | 2/3; 3/9 |
| last_self_ecology_integrated_constant | 5 | +0.0044 [-0.0014, +0.0117] | -0.28% [-0.70, +0.09] | +0.0202 [-0.0229, +0.0641] | +0.0028 [-0.0021, +0.0075] | +0.0075 [-0.0188, +0.0370] | 2/3; 3/9 |
| last_self_ecology_integrated_gru_tuned_anchor | 0 | -0.0028 [-0.0197, +0.0147] | +0.16% [-0.85, +1.05] | -0.0235 [-0.0629, +0.0100] | -0.0004 [-0.0193, +0.0197] | +0.0771 [+0.0215, +0.1548] | 2/3; 3/9 |
| last_self_ecology_integrated_gru_tuned_anchor | 5 | +0.0043 [-0.0014, +0.0113] | -0.27% [-0.68, +0.09] | +0.0199 [-0.0209, +0.0610] | +0.0027 [-0.0021, +0.0074] | +0.0163 [-0.0188, +0.0537] | 2/3; 3/9 |

The CSV retains raw/log MAE and signed bias for every region, plus partition/seed
directions and station-level gain and harm contributions.

## Validation-selected duration

| Mode | Selected epoch: 30-budget range | Selected epoch: extended range | Fits selecting epoch >30 | Mean source-validation MAE change, extended minus 30 |
|---|---:|---:|---:|---:|
| frozen | 5–30 | 5–50 | 5/9 | -0.01711 |
| last_self | 5–30 | 5–46 | 4/9 | -0.01382 |
| last_self_ecology | 5–30 | 5–49 | 5/9 | -0.02352 |

A fit may stop before its maximum budget. Longer ceilings alone do not guarantee longer
selected training. Same-cohort target results remain development evidence.
