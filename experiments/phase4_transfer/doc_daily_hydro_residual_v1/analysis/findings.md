# Daily hydrology in the existing DOC residual encoder

Three matched arms share the same partially trainable ecological encoder, original
GRU initialization, 38-channel readout, source OOF context base, 120-epoch cap and patience 5.
All retain cell-weighted native MAE with weight two at or above source Q90.

- monthly: all eight appended daily-flow channels are zero.
- availability: daily numerical values are zero, with availability flags and fractions retained.
- daily: all eight daily-flow channels are supplied.

The daily-minus-availability comparison separates daily numerical information from
hydrological availability metadata. Daily-minus-monthly measures their joint contribution.
The three numerical daily channels interact with recurrent state; the feature count
and trainable head dimensions are identical across arms. The context forests, source OOF
predictions, source folds, ecological memory and GRU support basis remain fixed.
Current-month daily summaries serve retrospective monthly reconstruction, not month-start
forecasting. The monthly discharge footprint and coverage rules are fixed in the bound
daily feature metadata; no new DOC target label is used to create these inputs.

Checkpoint and residual scale use unweighted pooled source-validation MAE. Support
and ecological mixing are fitted separately using identical validation episodes and grids.
The no-message self path remains in use; this experiment does not establish river-message gain.

All 20 saved models and all K values are retained. The expanded monthly head is newly
trained and need not reproduce the earlier 30-channel head trajectory. The carried
context, prior encoder and ecological-affine reference predictions must reproduce exactly.
The reference checks contain 72 model/run comparisons, all verified exactly.

Intervals use 5,000 paired whole-station bootstrap draws, jointly resampling repeated
station identities across partitions. Seed means are averaged within partition; partitions
have equal weight. Negative error or false-positive deltas favor the candidate; positive
recall deltas indicate improved detection. Q90 uses source-training labels and includes threshold ties.
These are previously examined development partitions, and source validation has been reused.
No model, threshold, support count or mixture is selected using these target comparisons.

## Complete K curves

| Model (GRU support basis) | K0 MAE | K1 MAE | K3 MAE | K5 MAE | K5 RMSE | K5 R² |
|---|---:|---:|---:|---:|---:|---:|
| context_gru_tuned_anchor | 1.9028 | 1.8666 | 1.6348 | 1.5960 | 3.5911 | 0.5675 |
| monthly_gru_tuned_anchor | 1.8228 | 1.7864 | 1.6462 | 1.5909 | 3.5917 | 0.5671 |
| availability_gru_tuned_anchor | 1.8219 | 1.7850 | 1.6456 | 1.5947 | 3.6125 | 0.5630 |
| daily_gru_tuned_anchor | 1.8070 | 1.7588 | 1.6313 | 1.5894 | 3.6045 | 0.5655 |
| monthly_integrated_gru_tuned_anchor | 1.8096 | 1.7771 | 1.6266 | 1.5890 | 3.5946 | 0.5665 |
| availability_integrated_gru_tuned_anchor | 1.8087 | 1.7757 | 1.6261 | 1.5887 | 3.5943 | 0.5666 |
| daily_integrated_gru_tuned_anchor | 1.8035 | 1.7554 | 1.6201 | 1.5650 | 3.5523 | 0.5759 |
| prior_encoder_gru_tuned_anchor | 1.8240 | 1.7862 | 1.6458 | 1.5947 | 3.6125 | 0.5630 |
| prior_encoder_integrated_gru_tuned_anchor | 1.8108 | 1.7769 | 1.6254 | 1.5887 | 3.5943 | 0.5666 |
| prior_ecological_affine_gru_tuned_anchor | 1.8092 | 1.7917 | 1.6221 | 1.5799 | 3.5814 | 0.5689 |

Constant-only support adaptation is retained in all CSVs. Its K0 predictions duplicate
the GRU-basis K0 predictions, so duplicate K0 bootstrap comparisons are omitted.

## Tail accuracy, ordinary error and classification

| Model | K | Q90 MAE | Q90 bias | Ordinary MAE | Ordinary bias | Q90 recall | Q90 precision | False-Q90 rate |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| context_gru_tuned_anchor | 0 | 7.5586 | -6.6062 | 1.2574 | +0.4820 | 58.90% | 78.47% | 1.91% |
| context_gru_tuned_anchor | 5 | 6.6945 | -5.1653 | 1.0084 | +0.2315 | 65.01% | 77.68% | 2.21% |
| monthly_gru_tuned_anchor | 0 | 7.2406 | -5.8647 | 1.2026 | +0.2891 | 61.56% | 76.96% | 2.14% |
| monthly_gru_tuned_anchor | 5 | 6.6484 | -5.0447 | 1.0079 | +0.1934 | 64.87% | 77.44% | 2.22% |
| availability_gru_tuned_anchor | 0 | 7.2420 | -5.8797 | 1.2016 | +0.2853 | 61.68% | 77.05% | 2.14% |
| availability_gru_tuned_anchor | 5 | 6.6775 | -5.1516 | 1.0093 | +0.1928 | 64.87% | 77.45% | 2.22% |
| daily_gru_tuned_anchor | 0 | 7.1673 | -5.5320 | 1.1941 | +0.2133 | 64.36% | 77.42% | 2.19% |
| daily_gru_tuned_anchor | 5 | 6.6423 | -5.1758 | 1.0075 | +0.1665 | 65.55% | 77.74% | 2.20% |
| monthly_integrated_gru_tuned_anchor | 0 | 7.2550 | -5.9174 | 1.1859 | +0.2461 | 61.67% | 76.92% | 2.15% |
| monthly_integrated_gru_tuned_anchor | 5 | 6.6607 | -5.0908 | 1.0043 | +0.1932 | 64.85% | 77.44% | 2.22% |
| availability_integrated_gru_tuned_anchor | 0 | 7.2561 | -5.9316 | 1.1848 | +0.2426 | 61.83% | 77.00% | 2.15% |
| availability_integrated_gru_tuned_anchor | 5 | 6.6595 | -5.0883 | 1.0042 | +0.1927 | 64.88% | 77.42% | 2.22% |
| daily_integrated_gru_tuned_anchor | 0 | 7.1792 | -5.5920 | 1.1887 | +0.2035 | 64.06% | 77.64% | 2.16% |
| daily_integrated_gru_tuned_anchor | 5 | 6.5870 | -4.9657 | 0.9856 | +0.1711 | 65.13% | 77.81% | 2.19% |
| prior_encoder_gru_tuned_anchor | 0 | 7.2369 | -5.8588 | 1.2044 | +0.2745 | 61.88% | 76.82% | 2.17% |
| prior_encoder_gru_tuned_anchor | 5 | 6.6774 | -5.1456 | 1.0092 | +0.1918 | 64.84% | 77.35% | 2.23% |
| prior_encoder_integrated_gru_tuned_anchor | 0 | 7.2513 | -5.9115 | 1.1877 | +0.2315 | 61.99% | 76.77% | 2.18% |
| prior_encoder_integrated_gru_tuned_anchor | 5 | 6.6593 | -5.0843 | 1.0041 | +0.1921 | 64.88% | 77.42% | 2.22% |
| prior_ecological_affine_gru_tuned_anchor | 0 | 7.4318 | -6.4542 | 1.1653 | +0.1960 | 60.08% | 78.00% | 1.99% |
| prior_ecological_affine_gru_tuned_anchor | 5 | 6.6808 | -5.1324 | 0.9913 | +0.1782 | 64.26% | 78.04% | 2.14% |

Recall conditions on true high DOC; false-Q90 rate conditions on ordinary observations.
A lower false-positive rate alone is not better selectivity if recall also collapses. Precision is descriptive; paired intervals are reported for recall and false-positive rate.

## Daily numerical information and availability controls

| Comparison | Overall Delta MAE [95% CI] | Q90 Delta MAE | Ordinary Delta MAE | Recall change [95% CI], pp | False-Q90 change [95% CI], pp |
|---|---:|---:|---:|---:|---:|
| daily_vs_monthly_constant_k5 | -0.0096 [-0.0306, +0.0110] | -0.0403 [-0.0933, +0.0012] | -0.0060 [-0.0291, +0.0159] | +0.7561 [-0.2548, +2.0450] | -0.0039 [-0.1006, +0.0773] |
| availability_vs_monthly_constant_k5 | -0.0008 [-0.0024, +0.0005] | -0.0017 [-0.0042, +0.0012] | -0.0007 [-0.0026, +0.0008] | +0.0316 [+0.0000, +0.1253] | +0.0092 [-0.0082, +0.0309] |
| daily_vs_availability_constant_k5 | -0.0089 [-0.0295, +0.0113] | -0.0386 [-0.0930, +0.0034] | -0.0053 [-0.0279, +0.0159] | +0.7246 [-0.2873, +2.0118] | -0.0131 [-0.1028, +0.0586] |
| daily_vs_monthly_integrated_constant_k5 | -0.0223 [-0.0374, -0.0089] | -0.0610 [-0.1187, -0.0057] | -0.0180 [-0.0337, -0.0044] | +0.3069 [-0.4824, +1.2207] | -0.0392 [-0.1340, +0.0239] |
| availability_vs_monthly_integrated_constant_k5 | +0.0024 [-0.0039, +0.0092] | +0.0191 [-0.0125, +0.0534] | +0.0008 [-0.0056, +0.0062] | -0.0947 [-0.2977, +0.0000] | +0.0060 [-0.0248, +0.0387] |
| daily_vs_availability_integrated_constant_k5 | -0.0247 [-0.0429, -0.0093] | -0.0800 [-0.1704, +0.0027] | -0.0188 [-0.0351, -0.0045] | +0.4016 [-0.4310, +1.4072] | -0.0452 [-0.1509, +0.0329] |
| daily_vs_monthly_gru_tuned_anchor_k0 | -0.0158 [-0.0603, +0.0224] | -0.0733 [-0.2070, +0.0407] | -0.0085 [-0.0551, +0.0333] | +2.8075 [+0.5684, +5.6333] | +0.0513 [-0.1100, +0.2246] |
| availability_vs_monthly_gru_tuned_anchor_k0 | -0.0009 [-0.0029, +0.0010] | +0.0014 [-0.0037, +0.0065] | -0.0011 [-0.0033, +0.0009] | +0.1263 [-0.1173, +0.4632] | -0.0061 [-0.0306, +0.0142] |
| daily_vs_availability_gru_tuned_anchor_k0 | -0.0149 [-0.0586, +0.0222] | -0.0747 [-0.2089, +0.0387] | -0.0074 [-0.0530, +0.0335] | +2.6812 [+0.3635, +5.5297] | +0.0575 [-0.1000, +0.2300] |
| daily_vs_monthly_integrated_gru_tuned_anchor_k0 | -0.0060 [-0.0461, +0.0303] | -0.0758 [-0.2021, +0.0309] | +0.0028 [-0.0389, +0.0415] | +2.3957 [+0.4019, +4.8900] | +0.0034 [-0.1621, +0.1725] |
| availability_vs_monthly_integrated_gru_tuned_anchor_k0 | -0.0009 [-0.0030, +0.0009] | +0.0011 [-0.0040, +0.0062] | -0.0011 [-0.0033, +0.0009] | +0.1661 [-0.0766, +0.5005] | -0.0036 [-0.0281, +0.0178] |
| daily_vs_availability_integrated_gru_tuned_anchor_k0 | -0.0051 [-0.0445, +0.0302] | -0.0769 [-0.2016, +0.0294] | +0.0039 [-0.0372, +0.0415] | +2.2296 [+0.1763, +4.7442] | +0.0070 [-0.1527, +0.1765] |
| daily_vs_monthly_gru_tuned_anchor_k5 | -0.0015 [-0.0241, +0.0219] | -0.0061 [-0.0850, +0.0656] | -0.0003 [-0.0244, +0.0225] | +0.6745 [-0.3266, +1.8834] | -0.0161 [-0.1035, +0.0477] |
| availability_vs_monthly_gru_tuned_anchor_k5 | +0.0038 [-0.0022, +0.0116] | +0.0291 [-0.0100, +0.0758] | +0.0014 [-0.0038, +0.0058] | +0.0000 [-0.2168, +0.2299] | +0.0001 [-0.0359, +0.0338] |
| daily_vs_availability_gru_tuned_anchor_k5 | -0.0054 [-0.0259, +0.0146] | -0.0352 [-0.0838, +0.0022] | -0.0017 [-0.0238, +0.0190] | +0.6745 [-0.2828, +1.8114] | -0.0162 [-0.1062, +0.0530] |
| daily_vs_monthly_integrated_gru_tuned_anchor_k5 | -0.0240 [-0.0397, -0.0102] | -0.0737 [-0.1389, -0.0132] | -0.0187 [-0.0345, -0.0051] | +0.2885 [-0.4601, +1.1732] | -0.0335 [-0.1305, +0.0320] |
| availability_vs_monthly_integrated_gru_tuned_anchor_k5 | -0.0003 [-0.0013, +0.0005] | -0.0012 [-0.0031, +0.0007] | -0.0002 [-0.0014, +0.0008] | +0.0316 [+0.0000, +0.1182] | +0.0031 [+0.0000, +0.0112] |
| daily_vs_availability_integrated_gru_tuned_anchor_k5 | -0.0237 [-0.0392, -0.0101] | -0.0725 [-0.1367, -0.0122] | -0.0185 [-0.0339, -0.0053] | +0.2569 [-0.4742, +1.0932] | -0.0365 [-0.1341, +0.0291] |

## Integrated models versus the fixed ecological reference

| Comparison | Overall Delta MAE [95% CI] | Q90 Delta MAE | Ordinary Delta MAE | Recall change [95% CI], pp | False-Q90 change [95% CI], pp |
|---|---:|---:|---:|---:|---:|
| monthly_integrated_vs_prior_ecological_constant_k5 | +0.0036 [-0.0070, +0.0157] | -0.0511 [-0.1021, -0.0046] | +0.0107 [+0.0003, +0.0232] | +1.2564 [+0.5133, +2.2499] | +0.1102 [+0.0444, +0.2066] |
| availability_integrated_vs_prior_ecological_constant_k5 | +0.0060 [-0.0074, +0.0217] | -0.0321 [-0.1116, +0.0469] | +0.0115 [-0.0008, +0.0257] | +1.1617 [+0.4067, +2.1521] | +0.1162 [+0.0469, +0.2154] |
| daily_integrated_vs_prior_ecological_constant_k5 | -0.0187 [-0.0343, -0.0049] | -0.1121 [-0.1540, -0.0736] | -0.0072 [-0.0236, +0.0072] | +1.5633 [+0.4351, +2.9958] | +0.0710 [+0.0049, +0.1595] |
| monthly_integrated_vs_prior_ecological_gru_tuned_anchor_k0 | +0.0004 [-0.0298, +0.0294] | -0.1768 [-0.2767, -0.0671] | +0.0207 [-0.0096, +0.0529] | +1.5924 [+0.5747, +3.1049] | +0.1649 [+0.0730, +0.2873] |
| availability_integrated_vs_prior_ecological_gru_tuned_anchor_k0 | -0.0006 [-0.0312, +0.0286] | -0.1756 [-0.2734, -0.0687] | +0.0196 [-0.0119, +0.0522] | +1.7585 [+0.6641, +3.4281] | +0.1613 [+0.0722, +0.2810] |
| daily_integrated_vs_prior_ecological_gru_tuned_anchor_k0 | -0.0057 [-0.0616, +0.0470] | -0.2526 [-0.4278, -0.0899] | +0.0234 [-0.0332, +0.0800] | +3.9881 [+1.6681, +7.0746] | +0.1683 [-0.0175, +0.3818] |
| monthly_integrated_vs_prior_ecological_gru_tuned_anchor_k5 | +0.0091 [-0.0032, +0.0228] | -0.0200 [-0.0745, +0.0342] | +0.0131 [+0.0008, +0.0275] | +0.5833 [+0.1301, +1.2386] | +0.0858 [+0.0295, +0.1639] |
| availability_integrated_vs_prior_ecological_gru_tuned_anchor_k5 | +0.0088 [-0.0031, +0.0222] | -0.0213 [-0.0760, +0.0324] | +0.0129 [+0.0009, +0.0270] | +0.6148 [+0.1554, +1.2963] | +0.0888 [+0.0320, +0.1682] |
| daily_integrated_vs_prior_ecological_gru_tuned_anchor_k5 | -0.0149 [-0.0278, -0.0031] | -0.0938 [-0.1303, -0.0586] | -0.0056 [-0.0188, +0.0077] | +0.8717 [-0.1186, +2.1161] | +0.0523 [-0.0133, +0.1290] |

## Ecological integration versus direct residuals

| Comparison | Overall Delta MAE [95% CI] | Q90 Delta MAE | Ordinary Delta MAE | Recall change [95% CI], pp | False-Q90 change [95% CI], pp |
|---|---:|---:|---:|---:|---:|
| monthly_integrated_vs_direct_constant_k5 | -0.0043 [-0.0131, +0.0035] | -0.0139 [-0.0541, +0.0245] | -0.0035 [-0.0111, +0.0041] | +0.0758 [-0.0949, +0.2793] | +0.0032 [-0.0339, +0.0406] |
| availability_integrated_vs_direct_constant_k5 | -0.0011 [-0.0051, +0.0029] | +0.0068 [-0.0024, +0.0168] | -0.0021 [-0.0063, +0.0021] | -0.0505 [-0.2290, +0.1080] | +0.0000 [-0.0189, +0.0210] |
| daily_integrated_vs_direct_constant_k5 | -0.0170 [-0.0408, +0.0032] | -0.0346 [-0.1412, +0.0681] | -0.0155 [-0.0366, +0.0039] | -0.3734 [-0.9660, +0.0806] | -0.0321 [-0.1101, +0.0453] |
| monthly_integrated_vs_direct_gru_tuned_anchor_k0 | -0.0132 [-0.0228, -0.0049] | +0.0144 [-0.0037, +0.0351] | -0.0167 [-0.0280, -0.0078] | +0.1131 [-0.1711, +0.4680] | +0.0103 [-0.0221, +0.0486] |
| availability_integrated_vs_direct_gru_tuned_anchor_k0 | -0.0133 [-0.0228, -0.0050] | +0.0141 [-0.0040, +0.0348] | -0.0167 [-0.0279, -0.0079] | +0.1530 [-0.1455, +0.5415] | +0.0128 [-0.0173, +0.0503] |
| daily_integrated_vs_direct_gru_tuned_anchor_k0 | -0.0035 [-0.0098, +0.0028] | +0.0118 [-0.0030, +0.0274] | -0.0054 [-0.0123, +0.0015] | -0.2987 [-0.6057, -0.0817] | -0.0377 [-0.0728, -0.0136] |
| monthly_integrated_vs_direct_gru_tuned_anchor_k5 | -0.0019 [-0.0062, +0.0023] | +0.0123 [+0.0025, +0.0225] | -0.0036 [-0.0083, +0.0009] | -0.0253 [-0.1728, +0.1289] | +0.0017 [-0.0255, +0.0352] |
| availability_integrated_vs_direct_gru_tuned_anchor_k5 | -0.0060 [-0.0152, +0.0020] | -0.0180 [-0.0676, +0.0261] | -0.0051 [-0.0121, +0.0023] | +0.0063 [-0.1952, +0.2366] | +0.0046 [-0.0328, +0.0461] |
| daily_integrated_vs_direct_gru_tuned_anchor_k5 | -0.0244 [-0.0488, -0.0049] | -0.0553 [-0.1812, +0.0554] | -0.0219 [-0.0404, -0.0048] | -0.4113 [-1.1807, +0.3553] | -0.0158 [-0.0679, +0.0369] |

## Direct residuals versus context

| Comparison | Overall Delta MAE [95% CI] | Q90 Delta MAE | Ordinary Delta MAE | Recall change [95% CI], pp | False-Q90 change [95% CI], pp |
|---|---:|---:|---:|---:|---:|
| monthly_vs_context_constant_k5 | -0.0079 [-0.0342, +0.0214] | -0.0434 [-0.1499, +0.0576] | -0.0035 [-0.0310, +0.0269] | +0.3045 [-0.3503, +1.1883] | +0.0413 [-0.0581, +0.1505] |
| availability_vs_context_constant_k5 | -0.0087 [-0.0351, +0.0206] | -0.0451 [-0.1511, +0.0541] | -0.0042 [-0.0316, +0.0265] | +0.3361 [-0.3324, +1.2483] | +0.0505 [-0.0493, +0.1616] |
| daily_vs_context_constant_k5 | -0.0175 [-0.0546, +0.0211] | -0.0838 [-0.2200, +0.0341] | -0.0095 [-0.0480, +0.0312] | +1.0606 [-0.2015, +2.6309] | +0.0373 [-0.1002, +0.1709] |
| monthly_vs_context_gru_tuned_anchor_k0 | -0.0801 [-0.1262, -0.0359] | -0.3180 [-0.5069, -0.1243] | -0.0547 [-0.1027, -0.0068] | +2.6580 [+1.2557, +4.6360] | +0.2340 [+0.0952, +0.4339] |
| availability_vs_context_gru_tuned_anchor_k0 | -0.0809 [-0.1278, -0.0362] | -0.3166 [-0.5028, -0.1254] | -0.0558 [-0.1048, -0.0072] | +2.7842 [+1.3215, +4.9243] | +0.2278 [+0.0951, +0.4195] |
| daily_vs_context_gru_tuned_anchor_k0 | -0.0958 [-0.1705, -0.0281] | -0.3913 [-0.6280, -0.1546] | -0.0632 [-0.1413, +0.0094] | +5.4655 [+2.3881, +9.3711] | +0.2853 [+0.0822, +0.5405] |
| monthly_vs_context_gru_tuned_anchor_k5 | -0.0051 [-0.0342, +0.0270] | -0.0460 [-0.1320, +0.0334] | -0.0005 [-0.0316, +0.0365] | -0.1423 [-1.0817, +0.8547] | +0.0110 [-0.1107, +0.1179] |
| availability_vs_context_gru_tuned_anchor_k5 | -0.0013 [-0.0321, +0.0336] | -0.0170 [-0.1367, +0.1013] | +0.0008 [-0.0315, +0.0392] | -0.1423 [-1.0886, +0.8848] | +0.0111 [-0.1094, +0.1232] |
| daily_vs_context_gru_tuned_anchor_k5 | -0.0066 [-0.0465, +0.0349] | -0.0522 [-0.1961, +0.0813] | -0.0009 [-0.0417, +0.0430] | +0.5322 [-0.8474, +2.1034] | -0.0051 [-0.1588, +0.1173] |

## Source-validation choices

| Hydrology arm | Selected epoch range | Total epochs executed | Scale counts | Mean selected validation MAE |
|---|---:|---:|---|---:|
| monthly | 5–47 | 295 | 0.5: 2, 1: 7 | 1.870033 |
| availability | 5–47 | 304 | 0.5: 2, 1: 7 | 1.869337 |
| daily | 27–72 | 473 | 1: 9 | 1.804428 |

All three arms use the same source-training objective. Encoder/GRU parameter movements and support choices are
saved in the training and adapter CSVs.

| Integrated model | K | Selected gamma counts |
|---|---:|---|
| monthly_integrated_constant | 0 | 0: 5, 0.25: 3, 1: 1 |
| monthly_integrated_constant | 1 | 0: 5, 0.25: 3, 0.5: 1 |
| monthly_integrated_constant | 3 | 0: 2, 0.25: 6, 0.5: 1 |
| monthly_integrated_constant | 5 | 0: 4, 0.25: 3, 0.5: 2 |
| monthly_integrated_gru_tuned_anchor | 0 | 0: 5, 0.25: 3, 1: 1 |
| monthly_integrated_gru_tuned_anchor | 1 | 0: 5, 0.25: 3, 0.5: 1 |
| monthly_integrated_gru_tuned_anchor | 3 | 0: 2, 0.25: 4, 0.5: 3 |
| monthly_integrated_gru_tuned_anchor | 5 | 0: 3, 0.25: 5, 0.5: 1 |
| availability_integrated_constant | 0 | 0: 5, 0.25: 3, 1: 1 |
| availability_integrated_constant | 1 | 0: 5, 0.25: 3, 0.5: 1 |
| availability_integrated_constant | 3 | 0: 2, 0.25: 6, 0.5: 1 |
| availability_integrated_constant | 5 | 0: 5, 0.25: 2, 0.5: 2 |
| availability_integrated_gru_tuned_anchor | 0 | 0: 5, 0.25: 3, 1: 1 |
| availability_integrated_gru_tuned_anchor | 1 | 0: 5, 0.25: 3, 0.5: 1 |
| availability_integrated_gru_tuned_anchor | 3 | 0: 2, 0.25: 4, 0.5: 3 |
| availability_integrated_gru_tuned_anchor | 5 | 0: 3, 0.25: 5, 0.5: 1 |
| daily_integrated_constant | 0 | 0: 7, 0.25: 2 |
| daily_integrated_constant | 1 | 0: 7, 0.25: 2 |
| daily_integrated_constant | 3 | 0: 2, 0.25: 4, 0.5: 3 |
| daily_integrated_constant | 5 | 0: 2, 0.25: 1, 0.5: 6 |
| daily_integrated_gru_tuned_anchor | 0 | 0: 7, 0.25: 2 |
| daily_integrated_gru_tuned_anchor | 1 | 0: 7, 0.25: 2 |
| daily_integrated_gru_tuned_anchor | 3 | 0: 1, 0.25: 5, 0.5: 3 |
| daily_integrated_gru_tuned_anchor | 5 | 0: 1, 0.25: 4, 0.5: 4 |

Interpret direct feature contrasts before integrated ones: the latter also include changes
in source-validation-selected ecological mixing and positive-K support calibration.
A narrow or sign-changing contrast does not establish equivalence. Station gain/harm
concentration and partition/seed directions are retained alongside the overall means.
