# Selectivity and station balance in the existing DOC residual

Four source-training objectives share the same partially trainable encoder, original
initialization, concentration head, source OOF context base, 60-epoch cap and patience 5.
All use the same uniform cell shuffle; only source loss weighting/asymmetry changes.

- tail2: current cell-weighted MAE, with weight two at or above source Q90.
- mae: ordinary cell-weighted MAE, with no extra tail weight.
- selective: tail2 plus 0.5 times the positive prediction error on ordinary source cells.
- station: tail weights normalized to unit total within each source station, then equal station averaging.

The fixed full-source mean weight normalizes each minibatch. Checkpoint and residual scale
still use unweighted pooled source-validation MAE. Support and ecological mixing are fitted
separately for each arm using the same validation episodes, candidate grids and frozen support basis.
No combined selective/station objective is fitted. The no-message self path remains in use.

All 24 saved models and all K values are retained. Tail2 directly and after ecological
integration reproduces the preceding encoder wherever the prior stopping decision
was already complete. Shared training prefixes must always match exactly; a changed
prediction is allowed only when extra epochs can follow a previously binding ceiling.
Prior encoder and ecological-affine reference products themselves are copied exactly.
The control checks contain 144 support-path/K/run comparisons, with
144 bitwise-identical predictions and
144 required identical comparisons.

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
| tail2_gru_tuned_anchor | 1.8240 | 1.7862 | 1.6458 | 1.5947 | 3.6125 | 0.5630 |
| mae_gru_tuned_anchor | 1.8215 | 1.7918 | 1.6247 | 1.5812 | 3.5856 | 0.5683 |
| selective_gru_tuned_anchor | 1.8228 | 1.7792 | 1.6182 | 1.5818 | 3.6039 | 0.5643 |
| station_gru_tuned_anchor | 1.8454 | 1.8095 | 1.6373 | 1.6052 | 3.6182 | 0.5615 |
| tail2_integrated_gru_tuned_anchor | 1.8108 | 1.7769 | 1.6254 | 1.5887 | 3.5943 | 0.5666 |
| mae_integrated_gru_tuned_anchor | 1.8086 | 1.7836 | 1.6295 | 1.5765 | 3.5812 | 0.5692 |
| selective_integrated_gru_tuned_anchor | 1.8136 | 1.7758 | 1.6124 | 1.5740 | 3.5833 | 0.5686 |
| station_integrated_gru_tuned_anchor | 1.8250 | 1.7943 | 1.6205 | 1.6046 | 3.6208 | 0.5609 |
| prior_encoder_gru_tuned_anchor | 1.8240 | 1.7862 | 1.6458 | 1.5947 | 3.6125 | 0.5630 |
| prior_encoder_integrated_gru_tuned_anchor | 1.8108 | 1.7769 | 1.6254 | 1.5887 | 3.5943 | 0.5666 |
| prior_ecological_affine_gru_tuned_anchor | 1.8092 | 1.7917 | 1.6221 | 1.5799 | 3.5814 | 0.5689 |

Constant-only support adaptation is retained in all CSVs. Its K0 predictions duplicate
the GRU-basis K0 predictions, so duplicate K0 bootstrap comparisons are omitted.

## Tail accuracy, ordinary error and classification

| Model | K | Q90 MAE | Q90 bias | Ordinary MAE | Ordinary bias | Q90 recall | False-Q90 rate |
|---|---:|---:|---:|---:|---:|---:|---:|
| context_gru_tuned_anchor | 0 | 7.5586 | -6.6062 | 1.2574 | +0.4820 | 58.90% | 1.91% |
| context_gru_tuned_anchor | 5 | 6.6945 | -5.1653 | 1.0084 | +0.2315 | 65.01% | 2.21% |
| tail2_gru_tuned_anchor | 0 | 7.2369 | -5.8588 | 1.2044 | +0.2745 | 61.88% | 2.17% |
| tail2_gru_tuned_anchor | 5 | 6.6774 | -5.1456 | 1.0092 | +0.1918 | 64.84% | 2.23% |
| mae_gru_tuned_anchor | 0 | 7.5309 | -6.4883 | 1.1676 | +0.1200 | 57.87% | 1.80% |
| mae_gru_tuned_anchor | 5 | 6.6872 | -5.1102 | 0.9922 | +0.1542 | 64.20% | 2.15% |
| selective_gru_tuned_anchor | 0 | 7.4977 | -6.0890 | 1.1732 | -0.0084 | 59.92% | 1.95% |
| selective_gru_tuned_anchor | 5 | 6.7317 | -5.1723 | 0.9883 | +0.1169 | 64.56% | 2.18% |
| station_gru_tuned_anchor | 0 | 7.2331 | -5.9226 | 1.2281 | +0.3347 | 61.87% | 2.29% |
| station_gru_tuned_anchor | 5 | 6.6845 | -5.1236 | 1.0198 | +0.1985 | 64.84% | 2.26% |
| tail2_integrated_gru_tuned_anchor | 0 | 7.2513 | -5.9115 | 1.1877 | +0.2315 | 61.99% | 2.18% |
| tail2_integrated_gru_tuned_anchor | 5 | 6.6593 | -5.0843 | 1.0041 | +0.1921 | 64.88% | 2.22% |
| mae_integrated_gru_tuned_anchor | 0 | 7.5085 | -6.4647 | 1.1561 | +0.1199 | 58.64% | 1.85% |
| mae_integrated_gru_tuned_anchor | 5 | 6.6841 | -5.1158 | 0.9873 | +0.1595 | 64.28% | 2.14% |
| selective_integrated_gru_tuned_anchor | 0 | 7.4786 | -6.1356 | 1.1654 | +0.0035 | 60.28% | 1.96% |
| selective_integrated_gru_tuned_anchor | 5 | 6.6912 | -5.0994 | 0.9838 | +0.1404 | 64.48% | 2.19% |
| station_integrated_gru_tuned_anchor | 0 | 7.2568 | -6.0011 | 1.2022 | +0.2492 | 62.01% | 2.27% |
| station_integrated_gru_tuned_anchor | 5 | 6.6938 | -5.1586 | 1.0181 | +0.1954 | 64.90% | 2.26% |
| prior_encoder_gru_tuned_anchor | 0 | 7.2369 | -5.8588 | 1.2044 | +0.2745 | 61.88% | 2.17% |
| prior_encoder_gru_tuned_anchor | 5 | 6.6774 | -5.1456 | 1.0092 | +0.1918 | 64.84% | 2.23% |
| prior_encoder_integrated_gru_tuned_anchor | 0 | 7.2513 | -5.9115 | 1.1877 | +0.2315 | 61.99% | 2.18% |
| prior_encoder_integrated_gru_tuned_anchor | 5 | 6.6593 | -5.0843 | 1.0041 | +0.1921 | 64.88% | 2.22% |
| prior_ecological_affine_gru_tuned_anchor | 0 | 7.4318 | -6.4542 | 1.1653 | +0.1960 | 60.08% | 1.99% |
| prior_ecological_affine_gru_tuned_anchor | 5 | 6.6808 | -5.1324 | 0.9913 | +0.1782 | 64.26% | 2.14% |

Recall conditions on true high DOC; false-Q90 rate conditions on ordinary observations.
A lower false-positive rate alone is not better selectivity if recall also collapses.

## Matched loss contrasts

| Comparison | Overall Delta MAE [95% CI] | Q90 Delta MAE | Ordinary Delta MAE | Recall change [95% CI], pp | False-Q90 change [95% CI], pp |
|---|---:|---:|---:|---:|---:|
| mae_vs_tail2_constant_k5 | -0.0012 [-0.0156, +0.0119] | +0.0851 [+0.0552, +0.1263] | -0.0115 [-0.0290, +0.0029] | -1.3302 [-2.2639, -0.5945] | -0.1358 [-0.2610, -0.0481] |
| selective_vs_tail2_constant_k5 | -0.0066 [-0.0219, +0.0075] | +0.0835 [+0.0411, +0.1386] | -0.0173 [-0.0355, -0.0024] | -0.9048 [-1.7454, -0.2050] | -0.1134 [-0.2178, -0.0405] |
| station_vs_tail2_constant_k5 | +0.0151 [+0.0029, +0.0298] | +0.0382 [-0.0068, +0.0794] | +0.0126 [-0.0002, +0.0289] | -0.0690 [-0.6286, +0.4313] | +0.0333 [-0.0276, +0.1114] |
| mae_vs_tail2_integrated_constant_k5 | -0.0072 [-0.0202, +0.0042] | +0.0600 [+0.0305, +0.0989] | -0.0155 [-0.0313, -0.0034] | -1.3958 [-2.3736, -0.6406] | -0.1340 [-0.2482, -0.0559] |
| selective_vs_tail2_integrated_constant_k5 | -0.0110 [-0.0262, +0.0020] | +0.0566 [+0.0110, +0.1087] | -0.0194 [-0.0369, -0.0060] | -1.0588 [-1.8714, -0.4149] | -0.0894 [-0.1810, -0.0247] |
| station_vs_tail2_integrated_constant_k5 | +0.0142 [+0.0032, +0.0279] | +0.0374 [-0.0074, +0.0819] | +0.0118 [+0.0008, +0.0266] | -0.1277 [-0.5439, +0.2456] | +0.0092 [-0.0540, +0.0753] |
| mae_vs_tail2_gru_tuned_anchor_k0 | -0.0025 [-0.0441, +0.0355] | +0.2940 [+0.1771, +0.4224] | -0.0368 [-0.0847, +0.0015] | -4.0037 [-6.4834, -2.0851] | -0.3695 [-0.6404, -0.1824] |
| selective_vs_tail2_gru_tuned_anchor_k0 | -0.0012 [-0.0445, +0.0381] | +0.2608 [+0.1762, +0.3582] | -0.0311 [-0.0775, +0.0104] | -1.9601 [-3.7435, -0.2733] | -0.2235 [-0.3916, -0.1037] |
| station_vs_tail2_gru_tuned_anchor_k0 | +0.0214 [-0.0049, +0.0565] | -0.0038 [-0.0477, +0.0382] | +0.0238 [-0.0060, +0.0653] | -0.0107 [-0.8558, +0.8380] | +0.1171 [+0.0030, +0.2959] |
| mae_vs_tail2_integrated_gru_tuned_anchor_k0 | -0.0022 [-0.0348, +0.0280] | +0.2572 [+0.1482, +0.3732] | -0.0316 [-0.0677, -0.0023] | -3.3539 [-5.4217, -1.7395] | -0.3347 [-0.5865, -0.1661] |
| selective_vs_tail2_integrated_gru_tuned_anchor_k0 | +0.0028 [-0.0320, +0.0352] | +0.2273 [+0.1464, +0.3128] | -0.0223 [-0.0586, +0.0107] | -1.7076 [-3.3490, -0.2091] | -0.2268 [-0.3962, -0.1044] |
| station_vs_tail2_integrated_gru_tuned_anchor_k0 | +0.0142 [-0.0116, +0.0481] | +0.0055 [-0.0365, +0.0477] | +0.0145 [-0.0143, +0.0541] | +0.0238 [-0.8237, +0.8783] | +0.0908 [-0.0200, +0.2698] |
| mae_vs_tail2_gru_tuned_anchor_k5 | -0.0135 [-0.0353, +0.0048] | +0.0098 [-0.0603, +0.0847] | -0.0170 [-0.0404, +0.0010] | -0.6421 [-1.4410, -0.0127] | -0.0801 [-0.1862, -0.0042] |
| selective_vs_tail2_gru_tuned_anchor_k5 | -0.0129 [-0.0332, +0.0046] | +0.0543 [+0.0029, +0.1157] | -0.0209 [-0.0438, -0.0016] | -0.2841 [-0.9903, +0.3947] | -0.0473 [-0.1216, +0.0090] |
| station_vs_tail2_gru_tuned_anchor_k5 | +0.0105 [-0.0009, +0.0260] | +0.0071 [-0.0164, +0.0315] | +0.0106 [-0.0025, +0.0285] | -0.0015 [-0.3847, +0.3581] | +0.0270 [-0.0172, +0.0809] |
| mae_vs_tail2_integrated_gru_tuned_anchor_k5 | -0.0122 [-0.0291, +0.0020] | +0.0248 [-0.0158, +0.0710] | -0.0168 [-0.0367, -0.0017] | -0.5974 [-1.2719, -0.1016] | -0.0835 [-0.1618, -0.0276] |
| selective_vs_tail2_integrated_gru_tuned_anchor_k5 | -0.0146 [-0.0324, +0.0004] | +0.0319 [-0.0124, +0.0816] | -0.0203 [-0.0411, -0.0044] | -0.3992 [-1.0209, +0.1035] | -0.0347 [-0.0875, +0.0081] |
| station_vs_tail2_integrated_gru_tuned_anchor_k5 | +0.0159 [+0.0052, +0.0294] | +0.0345 [-0.0083, +0.0755] | +0.0140 [+0.0029, +0.0290] | +0.0257 [-0.3088, +0.3371] | +0.0319 [-0.0103, +0.0927] |

## Station balance versus ordinary overprediction pressure

| Comparison | Overall Delta MAE [95% CI] | Q90 Delta MAE | Ordinary Delta MAE | Recall change [95% CI], pp | False-Q90 change [95% CI], pp |
|---|---:|---:|---:|---:|---:|
| station_vs_selective_constant_k5 | +0.0217 [-0.0011, +0.0504] | -0.0453 [-0.1374, +0.0323] | +0.0300 [+0.0056, +0.0619] | +0.8358 [-0.1003, +1.8797] | +0.1467 [+0.0366, +0.3009] |
| station_vs_selective_integrated_constant_k5 | +0.0252 [+0.0043, +0.0526] | -0.0193 [-0.1097, +0.0672] | +0.0312 [+0.0101, +0.0619] | +0.9310 [+0.2291, +1.8007] | +0.0986 [-0.0006, +0.2249] |
| station_vs_selective_gru_tuned_anchor_k0 | +0.0226 [-0.0336, +0.0926] | -0.2646 [-0.3827, -0.1587] | +0.0549 [-0.0076, +0.1340] | +1.9494 [-0.2479, +4.0582] | +0.3406 [+0.1461, +0.6042] |
| station_vs_selective_integrated_gru_tuned_anchor_k0 | +0.0114 [-0.0360, +0.0709] | -0.2217 [-0.3269, -0.1210] | +0.0368 [-0.0150, +0.1043] | +1.7314 [-0.3346, +3.7237] | +0.3176 [+0.1238, +0.5778] |
| station_vs_selective_gru_tuned_anchor_k5 | +0.0234 [-0.0026, +0.0565] | -0.0471 [-0.1119, +0.0021] | +0.0315 [+0.0028, +0.0702] | +0.2827 [-0.6717, +1.2043] | +0.0743 [-0.0075, +0.1782] |
| station_vs_selective_integrated_gru_tuned_anchor_k5 | +0.0306 [+0.0072, +0.0594] | +0.0026 [-0.0835, +0.0856] | +0.0343 [+0.0101, +0.0686] | +0.4249 [-0.2598, +1.1934] | +0.0667 [-0.0023, +0.1642] |

## Integrated models versus the fixed ecological reference

| Comparison | Overall Delta MAE [95% CI] | Q90 Delta MAE | Ordinary Delta MAE | Recall change [95% CI], pp | False-Q90 change [95% CI], pp |
|---|---:|---:|---:|---:|---:|
| tail2_integrated_vs_prior_ecological_constant_k5 | +0.0033 [-0.0071, +0.0151] | -0.0525 [-0.1044, -0.0061] | +0.0105 [+0.0004, +0.0229] | +1.2564 [+0.5133, +2.2499] | +0.1164 [+0.0500, +0.2148] |
| mae_integrated_vs_prior_ecological_constant_k5 | -0.0039 [-0.0151, +0.0056] | +0.0075 [-0.0203, +0.0343] | -0.0050 [-0.0179, +0.0052] | -0.1394 [-0.7017, +0.5004] | -0.0176 [-0.0613, +0.0230] |
| selective_integrated_vs_prior_ecological_constant_k5 | -0.0077 [-0.0204, +0.0029] | +0.0041 [-0.0148, +0.0274] | -0.0088 [-0.0243, +0.0034] | +0.1977 [-0.2343, +0.8567] | +0.0270 [-0.0217, +0.0860] |
| station_integrated_vs_prior_ecological_constant_k5 | +0.0175 [+0.0004, +0.0381] | -0.0152 [-0.1022, +0.0706] | +0.0224 [+0.0066, +0.0423] | +1.1287 [+0.4482, +2.0554] | +0.1257 [+0.0493, +0.2373] |
| tail2_integrated_vs_prior_ecological_gru_tuned_anchor_k0 | +0.0016 [-0.0299, +0.0316] | -0.1805 [-0.2824, -0.0677] | +0.0224 [-0.0103, +0.0562] | +1.9144 [+0.7730, +3.6686] | +0.1931 [+0.0958, +0.3264] |
| mae_integrated_vs_prior_ecological_gru_tuned_anchor_k0 | -0.0006 [-0.0421, +0.0350] | +0.0767 [+0.0150, +0.1561] | -0.0092 [-0.0563, +0.0306] | -1.4395 [-3.0222, -0.0256] | -0.1416 [-0.3104, -0.0222] |
| selective_integrated_vs_prior_ecological_gru_tuned_anchor_k0 | +0.0044 [-0.0419, +0.0457] | +0.0468 [-0.0487, +0.1598] | +0.0001 [-0.0514, +0.0457] | +0.2068 [-1.0707, +2.0589] | -0.0337 [-0.1574, +0.0650] |
| station_integrated_vs_prior_ecological_gru_tuned_anchor_k0 | +0.0158 [-0.0118, +0.0423] | -0.1749 [-0.2656, -0.0693] | +0.0369 [+0.0108, +0.0671] | +1.9382 [+0.9388, +3.4943] | +0.2839 [+0.1301, +0.5079] |
| tail2_integrated_vs_prior_ecological_gru_tuned_anchor_k5 | +0.0088 [-0.0032, +0.0222] | -0.0215 [-0.0771, +0.0324] | +0.0129 [+0.0010, +0.0269] | +0.6148 [+0.1554, +1.2963] | +0.0888 [+0.0320, +0.1682] |
| mae_integrated_vs_prior_ecological_gru_tuned_anchor_k5 | -0.0034 [-0.0124, +0.0041] | +0.0033 [-0.0172, +0.0226] | -0.0040 [-0.0144, +0.0046] | +0.0175 [-0.4407, +0.6171] | +0.0053 [-0.0232, +0.0359] |
| selective_integrated_vs_prior_ecological_gru_tuned_anchor_k5 | -0.0059 [-0.0166, +0.0031] | +0.0104 [-0.0103, +0.0318] | -0.0074 [-0.0207, +0.0030] | +0.2156 [-0.2986, +0.9168] | +0.0541 [+0.0147, +0.1155] |
| station_integrated_vs_prior_ecological_gru_tuned_anchor_k5 | +0.0247 [+0.0051, +0.0479] | +0.0131 [-0.0758, +0.1030] | +0.0269 [+0.0078, +0.0512] | +0.6406 [+0.1214, +1.3903] | +0.1208 [+0.0484, +0.2319] |

## Ecological integration versus direct residuals

| Comparison | Overall Delta MAE [95% CI] | Q90 Delta MAE | Ordinary Delta MAE | Recall change [95% CI], pp | False-Q90 change [95% CI], pp |
|---|---:|---:|---:|---:|---:|
| tail2_integrated_vs_direct_constant_k5 | -0.0038 [-0.0128, +0.0045] | -0.0134 [-0.0528, +0.0246] | -0.0030 [-0.0109, +0.0053] | +0.0758 [-0.0949, +0.2793] | -0.0028 [-0.0411, +0.0344] |
| mae_integrated_vs_direct_constant_k5 | -0.0098 [-0.0210, +0.0001] | -0.0385 [-0.0940, +0.0089] | -0.0070 [-0.0161, +0.0019] | +0.0102 [-0.3038, +0.3362] | -0.0010 [-0.0368, +0.0367] |
| selective_integrated_vs_direct_constant_k5 | -0.0082 [-0.0197, +0.0017] | -0.0403 [-0.0886, +0.0026] | -0.0051 [-0.0147, +0.0045] | -0.0782 [-0.4555, +0.2533] | +0.0212 [-0.0170, +0.0755] |
| station_integrated_vs_direct_constant_k5 | -0.0047 [-0.0119, +0.0023] | -0.0142 [-0.0481, +0.0195] | -0.0038 [-0.0101, +0.0031] | +0.0170 [-0.1470, +0.2127] | -0.0269 [-0.0797, +0.0064] |
| tail2_integrated_vs_direct_gru_tuned_anchor_k0 | -0.0132 [-0.0229, -0.0049] | +0.0144 [-0.0036, +0.0351] | -0.0167 [-0.0280, -0.0078] | +0.1131 [-0.1711, +0.4680] | +0.0103 [-0.0221, +0.0486] |
| mae_integrated_vs_direct_gru_tuned_anchor_k0 | -0.0129 [-0.0237, -0.0016] | -0.0224 [-0.0521, +0.0025] | -0.0115 [-0.0230, +0.0007] | +0.7629 [+0.2263, +1.4560] | +0.0450 [+0.0033, +0.1076] |
| selective_integrated_vs_direct_gru_tuned_anchor_k0 | -0.0092 [-0.0172, -0.0014] | -0.0191 [-0.0428, +0.0015] | -0.0079 [-0.0163, +0.0004] | +0.3657 [+0.0475, +0.7422] | +0.0070 [-0.0202, +0.0434] |
| station_integrated_vs_direct_gru_tuned_anchor_k0 | -0.0204 [-0.0368, -0.0067] | +0.0237 [+0.0018, +0.0525] | -0.0260 [-0.0457, -0.0108] | +0.1476 [-0.2568, +0.6197] | -0.0160 [-0.0740, +0.0259] |
| tail2_integrated_vs_direct_gru_tuned_anchor_k5 | -0.0060 [-0.0153, +0.0021] | -0.0181 [-0.0677, +0.0263] | -0.0051 [-0.0122, +0.0024] | +0.0379 [-0.1427, +0.2655] | -0.0047 [-0.0451, +0.0326] |
| mae_integrated_vs_direct_gru_tuned_anchor_k5 | -0.0048 [-0.0118, +0.0026] | -0.0031 [-0.0275, +0.0207] | -0.0049 [-0.0122, +0.0029] | +0.0826 [-0.2368, +0.4104] | -0.0082 [-0.0385, +0.0203] |
| selective_integrated_vs_direct_gru_tuned_anchor_k5 | -0.0077 [-0.0192, +0.0026] | -0.0405 [-0.0880, +0.0017] | -0.0044 [-0.0142, +0.0056] | -0.0772 [-0.4277, +0.2392] | +0.0078 [-0.0322, +0.0587] |
| station_integrated_vs_direct_gru_tuned_anchor_k5 | -0.0005 [-0.0030, +0.0020] | +0.0093 [+0.0036, +0.0156] | -0.0017 [-0.0045, +0.0010] | +0.0651 [-0.0649, +0.2222] | +0.0001 [-0.0114, +0.0113] |

## Direct residuals versus context

| Comparison | Overall Delta MAE [95% CI] | Q90 Delta MAE | Ordinary Delta MAE | Recall change [95% CI], pp | False-Q90 change [95% CI], pp |
|---|---:|---:|---:|---:|---:|
| tail2_vs_context_constant_k5 | -0.0087 [-0.0354, +0.0208] | -0.0454 [-0.1523, +0.0546] | -0.0042 [-0.0318, +0.0268] | +0.3045 [-0.3503, +1.1883] | +0.0535 [-0.0465, +0.1646] |
| mae_vs_context_constant_k5 | -0.0099 [-0.0388, +0.0200] | +0.0397 [-0.0596, +0.1400] | -0.0158 [-0.0462, +0.0151] | -1.0257 [-2.0805, -0.0790] | -0.0823 [-0.2295, +0.0315] |
| selective_vs_context_constant_k5 | -0.0153 [-0.0455, +0.0156] | +0.0381 [-0.0443, +0.1195] | -0.0216 [-0.0543, +0.0124] | -0.6003 [-1.6412, +0.4436] | -0.0599 [-0.1909, +0.0430] |
| station_vs_context_constant_k5 | +0.0064 [-0.0246, +0.0409] | -0.0072 [-0.1423, +0.1235] | +0.0084 [-0.0225, +0.0429] | +0.2355 [-0.4126, +1.0918] | +0.0869 [-0.0068, +0.2211] |
| tail2_vs_context_gru_tuned_anchor_k0 | -0.0788 [-0.1276, -0.0326] | -0.3217 [-0.5134, -0.1245] | -0.0530 [-0.1037, -0.0024] | +2.9799 [+1.4411, +5.2270] | +0.2621 [+0.1190, +0.4746] |
| mae_vs_context_gru_tuned_anchor_k0 | -0.0813 [-0.1509, -0.0207] | -0.0277 [-0.1531, +0.1283] | -0.0898 [-0.1673, -0.0208] | -1.0237 [-2.6206, +0.4276] | -0.1073 [-0.2315, -0.0185] |
| selective_vs_context_gru_tuned_anchor_k0 | -0.0801 [-0.1536, -0.0136] | -0.0609 [-0.2334, +0.1457] | -0.0841 [-0.1640, -0.0081] | +1.0198 [-0.3651, +3.0215] | +0.0386 [-0.0733, +0.1478] |
| station_vs_context_gru_tuned_anchor_k0 | -0.0574 [-0.0989, -0.0203] | -0.3255 [-0.5069, -0.1313] | -0.0292 [-0.0651, +0.0090] | +2.9693 [+1.5627, +5.0323] | +0.3792 [+0.1743, +0.6740] |
| tail2_vs_context_gru_tuned_anchor_k5 | -0.0013 [-0.0322, +0.0337] | -0.0171 [-0.1377, +0.1019] | +0.0008 [-0.0315, +0.0391] | -0.1739 [-1.1121, +0.8437] | +0.0205 [-0.0978, +0.1357] |
| mae_vs_context_gru_tuned_anchor_k5 | -0.0148 [-0.0409, +0.0119] | -0.0072 [-0.0747, +0.0597] | -0.0162 [-0.0450, +0.0127] | -0.8159 [-2.0853, +0.3447] | -0.0596 [-0.1803, +0.0347] |
| selective_vs_context_gru_tuned_anchor_k5 | -0.0142 [-0.0442, +0.0160] | +0.0372 [-0.0452, +0.1179] | -0.0202 [-0.0519, +0.0128] | -0.4580 [-1.6431, +0.7136] | -0.0268 [-0.1394, +0.0692] |
| station_vs_context_gru_tuned_anchor_k5 | +0.0092 [-0.0246, +0.0460] | -0.0099 [-0.1227, +0.0975] | +0.0114 [-0.0244, +0.0531] | -0.1753 [-1.0793, +0.8284] | +0.0475 [-0.0624, +0.1707] |

## Source-validation choices

| Loss arm | Selected epoch range | Total epochs executed | Scale counts | Mean selected validation MAE |
|---|---:|---:|---|---:|
| tail2 | 5–49 | 307 | 0.5: 2, 1: 7 | 1.869505 |
| mae | 2–60 | 342 | 1: 9 | 1.882749 |
| selective | 19–57 | 407 | 0.5: 2, 1: 7 | 1.854275 |
| station | 0–60 | 304 | 0: 1, 0.25: 2, 1: 6 | 1.885577 |

Training loss levels differ across objectives and should not be treated as directly
comparable performance scores. Encoder/GRU parameter movements and support choices are
saved in the training and adapter CSVs.

| Integrated model | K | Selected gamma counts |
|---|---:|---|
| tail2_integrated_constant | 0 | 0: 5, 0.25: 3, 1: 1 |
| tail2_integrated_constant | 1 | 0: 5, 0.25: 3, 0.5: 1 |
| tail2_integrated_constant | 3 | 0: 2, 0.25: 6, 0.5: 1 |
| tail2_integrated_constant | 5 | 0: 4, 0.25: 3, 0.5: 2 |
| tail2_integrated_gru_tuned_anchor | 0 | 0: 5, 0.25: 3, 1: 1 |
| tail2_integrated_gru_tuned_anchor | 1 | 0: 5, 0.25: 3, 0.5: 1 |
| tail2_integrated_gru_tuned_anchor | 3 | 0: 1, 0.25: 5, 0.5: 3 |
| tail2_integrated_gru_tuned_anchor | 5 | 0: 3, 0.25: 5, 0.5: 1 |
| mae_integrated_constant | 0 | 0: 3, 0.25: 4, 0.5: 2 |
| mae_integrated_constant | 1 | 0: 3, 0.25: 5, 0.5: 1 |
| mae_integrated_constant | 3 | 0: 3, 0.25: 2, 0.5: 4 |
| mae_integrated_constant | 5 | 0: 3, 0.25: 3, 0.5: 3 |
| mae_integrated_gru_tuned_anchor | 0 | 0: 3, 0.25: 4, 0.5: 2 |
| mae_integrated_gru_tuned_anchor | 1 | 0: 3, 0.25: 5, 0.5: 1 |
| mae_integrated_gru_tuned_anchor | 3 | 0: 2, 0.25: 2, 0.5: 5 |
| mae_integrated_gru_tuned_anchor | 5 | 0: 2, 0.25: 4, 0.5: 3 |
| selective_integrated_constant | 0 | 0: 4, 0.25: 4, 0.5: 1 |
| selective_integrated_constant | 1 | 0: 5, 0.25: 3, 0.5: 1 |
| selective_integrated_constant | 3 | 0: 3, 0.25: 2, 0.5: 4 |
| selective_integrated_constant | 5 | 0: 3, 0.25: 4, 0.5: 2 |
| selective_integrated_gru_tuned_anchor | 0 | 0: 4, 0.25: 4, 0.5: 1 |
| selective_integrated_gru_tuned_anchor | 1 | 0: 5, 0.25: 3, 0.5: 1 |
| selective_integrated_gru_tuned_anchor | 3 | 0: 3, 0.25: 1, 0.5: 5 |
| selective_integrated_gru_tuned_anchor | 5 | 0: 3, 0.25: 3, 0.5: 3 |
| station_integrated_constant | 0 | 0: 4, 0.25: 2, 0.5: 2, 1: 1 |
| station_integrated_constant | 1 | 0: 4, 0.25: 2, 0.5: 1, 1: 2 |
| station_integrated_constant | 3 | 0: 3, 0.25: 3, 0.5: 1, 1: 2 |
| station_integrated_constant | 5 | 0: 5, 0.25: 3, 1: 1 |
| station_integrated_gru_tuned_anchor | 0 | 0: 4, 0.25: 2, 0.5: 2, 1: 1 |
| station_integrated_gru_tuned_anchor | 1 | 0: 4, 0.25: 2, 0.5: 1, 1: 2 |
| station_integrated_gru_tuned_anchor | 3 | 0: 3, 0.25: 1, 0.5: 3, 1: 2 |
| station_integrated_gru_tuned_anchor | 5 | 0: 5, 0.25: 3, 1: 1 |

Interpret direct loss contrasts before integrated ones: the latter also include changes
in source-validation-selected ecological mixing and positive-K support calibration.
A narrow or sign-changing contrast does not establish equivalence. Station gain/harm
concentration and partition/seed directions are retained alongside the overall means.
