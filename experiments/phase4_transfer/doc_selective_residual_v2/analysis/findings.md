# Selectivity and station balance in the existing DOC residual

Four source-training objectives share the same partially trainable encoder, original
initialization, concentration head, source OOF context base, 120-epoch cap and patience 5.
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
| mae_gru_tuned_anchor | 1.8215 | 1.7918 | 1.6253 | 1.5815 | 3.5861 | 0.5682 |
| selective_gru_tuned_anchor | 1.8231 | 1.7794 | 1.6181 | 1.5817 | 3.6042 | 0.5643 |
| station_gru_tuned_anchor | 1.8449 | 1.8083 | 1.6371 | 1.6094 | 3.6399 | 0.5571 |
| tail2_integrated_gru_tuned_anchor | 1.8108 | 1.7769 | 1.6254 | 1.5887 | 3.5943 | 0.5666 |
| mae_integrated_gru_tuned_anchor | 1.8086 | 1.7836 | 1.6301 | 1.5768 | 3.5816 | 0.5691 |
| selective_integrated_gru_tuned_anchor | 1.8139 | 1.7760 | 1.6124 | 1.5727 | 3.5788 | 0.5695 |
| station_integrated_gru_tuned_anchor | 1.8245 | 1.7931 | 1.6203 | 1.6043 | 3.6210 | 0.5608 |
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
| mae_gru_tuned_anchor | 0 | 7.5357 | -6.4887 | 1.1671 | +0.1001 | 57.84% | 1.80% |
| mae_gru_tuned_anchor | 5 | 6.6889 | -5.1091 | 0.9924 | +0.1496 | 64.20% | 2.15% |
| selective_gru_tuned_anchor | 0 | 7.4980 | -6.0910 | 1.1736 | -0.0052 | 59.98% | 1.95% |
| selective_gru_tuned_anchor | 5 | 6.7321 | -5.1736 | 0.9881 | +0.1186 | 64.56% | 2.18% |
| station_gru_tuned_anchor | 0 | 7.2334 | -5.9261 | 1.2275 | +0.3319 | 61.87% | 2.29% |
| station_gru_tuned_anchor | 5 | 6.7151 | -5.2297 | 1.0215 | +0.1967 | 64.90% | 2.25% |
| tail2_integrated_gru_tuned_anchor | 0 | 7.2513 | -5.9115 | 1.1877 | +0.2315 | 61.99% | 2.18% |
| tail2_integrated_gru_tuned_anchor | 5 | 6.6593 | -5.0843 | 1.0041 | +0.1921 | 64.88% | 2.22% |
| mae_integrated_gru_tuned_anchor | 0 | 7.5133 | -6.4651 | 1.1556 | +0.1001 | 58.61% | 1.84% |
| mae_integrated_gru_tuned_anchor | 5 | 6.6858 | -5.1146 | 0.9874 | +0.1550 | 64.28% | 2.14% |
| selective_integrated_gru_tuned_anchor | 0 | 7.4788 | -6.1375 | 1.1657 | +0.0067 | 60.35% | 1.96% |
| selective_integrated_gru_tuned_anchor | 5 | 6.6872 | -5.0856 | 0.9827 | +0.1400 | 64.45% | 2.18% |
| station_integrated_gru_tuned_anchor | 0 | 7.2571 | -6.0046 | 1.2016 | +0.2464 | 62.01% | 2.27% |
| station_integrated_gru_tuned_anchor | 5 | 6.6938 | -5.1601 | 1.0178 | +0.1947 | 64.90% | 2.26% |
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
| mae_vs_tail2_constant_k5 | -0.0008 [-0.0154, +0.0124] | +0.0865 [+0.0563, +0.1285] | -0.0113 [-0.0290, +0.0033] | -1.3618 [-2.3122, -0.6161] | -0.1328 [-0.2575, -0.0466] |
| selective_vs_tail2_constant_k5 | -0.0067 [-0.0221, +0.0074] | +0.0839 [+0.0424, +0.1380] | -0.0175 [-0.0356, -0.0024] | -0.9048 [-1.7454, -0.2050] | -0.1134 [-0.2178, -0.0405] |
| station_vs_tail2_constant_k5 | +0.0145 [+0.0025, +0.0289] | +0.0383 [-0.0075, +0.0811] | +0.0120 [-0.0007, +0.0276] | +0.0257 [-0.5486, +0.5967] | +0.0333 [-0.0289, +0.1132] |
| mae_vs_tail2_integrated_constant_k5 | -0.0068 [-0.0199, +0.0047] | +0.0614 [+0.0314, +0.1013] | -0.0153 [-0.0315, -0.0030] | -1.4274 [-2.4039, -0.6759] | -0.1310 [-0.2441, -0.0546] |
| selective_vs_tail2_integrated_constant_k5 | -0.0061 [-0.0232, +0.0080] | +0.0924 [+0.0673, +0.1235] | -0.0175 [-0.0379, -0.0025] | -1.0588 [-1.8476, -0.4378] | -0.0986 [-0.1896, -0.0355] |
| station_vs_tail2_integrated_constant_k5 | +0.0173 [+0.0041, +0.0333] | +0.0587 [-0.0159, +0.1394] | +0.0131 [+0.0013, +0.0263] | -0.0962 [-0.6315, +0.3887] | +0.0184 [-0.0611, +0.1066] |
| mae_vs_tail2_gru_tuned_anchor_k0 | -0.0025 [-0.0438, +0.0359] | +0.2988 [+0.1836, +0.4265] | -0.0373 [-0.0845, +0.0020] | -4.0352 [-6.5476, -2.0908] | -0.3725 [-0.6450, -0.1843] |
| selective_vs_tail2_gru_tuned_anchor_k0 | -0.0009 [-0.0438, +0.0380] | +0.2611 [+0.1753, +0.3584] | -0.0308 [-0.0764, +0.0105] | -1.8970 [-3.6116, -0.2489] | -0.2235 [-0.3916, -0.1037] |
| station_vs_tail2_gru_tuned_anchor_k0 | +0.0209 [-0.0063, +0.0575] | -0.0035 [-0.0476, +0.0377] | +0.0231 [-0.0072, +0.0663] | -0.0107 [-0.8558, +0.8380] | +0.1171 [+0.0030, +0.2959] |
| mae_vs_tail2_integrated_gru_tuned_anchor_k0 | -0.0022 [-0.0349, +0.0289] | +0.2620 [+0.1541, +0.3772] | -0.0321 [-0.0684, -0.0022] | -3.3854 [-5.4713, -1.7589] | -0.3378 [-0.5896, -0.1682] |
| selective_vs_tail2_integrated_gru_tuned_anchor_k0 | +0.0031 [-0.0311, +0.0348] | +0.2275 [+0.1458, +0.3136] | -0.0220 [-0.0579, +0.0108] | -1.6444 [-3.2238, -0.1706] | -0.2268 [-0.3962, -0.1044] |
| station_vs_tail2_integrated_gru_tuned_anchor_k0 | +0.0137 [-0.0129, +0.0495] | +0.0058 [-0.0358, +0.0475] | +0.0139 [-0.0156, +0.0554] | +0.0238 [-0.8237, +0.8783] | +0.0908 [-0.0200, +0.2698] |
| mae_vs_tail2_gru_tuned_anchor_k5 | -0.0132 [-0.0350, +0.0053] | +0.0115 [-0.0589, +0.0867] | -0.0168 [-0.0405, +0.0012] | -0.6421 [-1.4410, -0.0127] | -0.0801 [-0.1862, -0.0042] |
| selective_vs_tail2_gru_tuned_anchor_k5 | -0.0130 [-0.0332, +0.0043] | +0.0547 [+0.0046, +0.1149] | -0.0211 [-0.0440, -0.0018] | -0.2841 [-0.9903, +0.3947] | -0.0473 [-0.1216, +0.0090] |
| station_vs_tail2_gru_tuned_anchor_k5 | +0.0147 [+0.0028, +0.0291] | +0.0377 [-0.0074, +0.0803] | +0.0123 [-0.0002, +0.0278] | +0.0617 [-0.3837, +0.5037] | +0.0240 [-0.0308, +0.0905] |
| mae_vs_tail2_integrated_gru_tuned_anchor_k5 | -0.0119 [-0.0290, +0.0025] | +0.0265 [-0.0145, +0.0735] | -0.0167 [-0.0370, -0.0012] | -0.5974 [-1.2719, -0.1016] | -0.0835 [-0.1618, -0.0276] |
| selective_vs_tail2_integrated_gru_tuned_anchor_k5 | -0.0160 [-0.0345, -0.0004] | +0.0279 [-0.0279, +0.0856] | -0.0214 [-0.0426, -0.0051] | -0.4308 [-1.0586, +0.0536] | -0.0409 [-0.0952, +0.0015] |
| station_vs_tail2_integrated_gru_tuned_anchor_k5 | +0.0156 [+0.0050, +0.0287] | +0.0345 [-0.0087, +0.0760] | +0.0136 [+0.0028, +0.0280] | +0.0257 [-0.3088, +0.3371] | +0.0319 [-0.0103, +0.0927] |

## Station balance versus ordinary overprediction pressure

| Comparison | Overall Delta MAE [95% CI] | Q90 Delta MAE | Ordinary Delta MAE | Recall change [95% CI], pp | False-Q90 change [95% CI], pp |
|---|---:|---:|---:|---:|---:|
| station_vs_selective_constant_k5 | +0.0212 [-0.0013, +0.0494] | -0.0456 [-0.1377, +0.0323] | +0.0294 [+0.0054, +0.0607] | +0.9305 [-0.0306, +2.0077] | +0.1467 [+0.0361, +0.3021] |
| station_vs_selective_integrated_constant_k5 | +0.0234 [+0.0033, +0.0492] | -0.0337 [-0.1204, +0.0443] | +0.0306 [+0.0100, +0.0599] | +0.9626 [+0.2392, +1.8251] | +0.1170 [+0.0164, +0.2526] |
| station_vs_selective_gru_tuned_anchor_k0 | +0.0217 [-0.0342, +0.0919] | -0.2646 [-0.3834, -0.1581] | +0.0539 [-0.0087, +0.1332] | +1.8863 [-0.3097, +3.9400] | +0.3406 [+0.1461, +0.6042] |
| station_vs_selective_integrated_gru_tuned_anchor_k0 | +0.0105 [-0.0371, +0.0710] | -0.2218 [-0.3274, -0.1207] | +0.0358 [-0.0158, +0.1036] | +1.6682 [-0.3836, +3.6369] | +0.3176 [+0.1238, +0.5778] |
| station_vs_selective_gru_tuned_anchor_k5 | +0.0277 [+0.0011, +0.0602] | -0.0170 [-0.1162, +0.0722] | +0.0334 [+0.0052, +0.0702] | +0.3458 [-0.6577, +1.3292] | +0.0713 [-0.0187, +0.1902] |
| station_vs_selective_integrated_gru_tuned_anchor_k5 | +0.0316 [+0.0078, +0.0611] | +0.0066 [-0.0877, +0.1016] | +0.0351 [+0.0108, +0.0692] | +0.4565 [-0.2014, +1.2009] | +0.0728 [+0.0040, +0.1717] |

## Integrated models versus the fixed ecological reference

| Comparison | Overall Delta MAE [95% CI] | Q90 Delta MAE | Ordinary Delta MAE | Recall change [95% CI], pp | False-Q90 change [95% CI], pp |
|---|---:|---:|---:|---:|---:|
| tail2_integrated_vs_prior_ecological_constant_k5 | +0.0033 [-0.0071, +0.0151] | -0.0525 [-0.1044, -0.0061] | +0.0105 [+0.0004, +0.0229] | +1.2564 [+0.5133, +2.2499] | +0.1164 [+0.0500, +0.2148] |
| mae_integrated_vs_prior_ecological_constant_k5 | -0.0035 [-0.0149, +0.0063] | +0.0089 [-0.0194, +0.0362] | -0.0047 [-0.0181, +0.0059] | -0.1710 [-0.7308, +0.4500] | -0.0145 [-0.0562, +0.0245] |
| selective_integrated_vs_prior_ecological_constant_k5 | -0.0028 [-0.0201, +0.0129] | +0.0398 [-0.0110, +0.0932] | -0.0069 [-0.0264, +0.0081] | +0.1977 [-0.1745, +0.7909] | +0.0178 [-0.0259, +0.0697] |
| station_integrated_vs_prior_ecological_constant_k5 | +0.0206 [+0.0015, +0.0440] | +0.0062 [-0.1137, +0.1280] | +0.0237 [+0.0074, +0.0435] | +1.1602 [+0.4271, +2.1449] | +0.1349 [+0.0473, +0.2653] |
| tail2_integrated_vs_prior_ecological_gru_tuned_anchor_k0 | +0.0016 [-0.0299, +0.0316] | -0.1805 [-0.2824, -0.0677] | +0.0224 [-0.0103, +0.0562] | +1.9144 [+0.7730, +3.6686] | +0.1931 [+0.0958, +0.3264] |
| mae_integrated_vs_prior_ecological_gru_tuned_anchor_k0 | -0.0006 [-0.0417, +0.0356] | +0.0815 [+0.0190, +0.1670] | -0.0097 [-0.0562, +0.0303] | -1.4711 [-3.0423, -0.0602] | -0.1447 [-0.3141, -0.0251] |
| selective_integrated_vs_prior_ecological_gru_tuned_anchor_k0 | +0.0047 [-0.0406, +0.0456] | +0.0471 [-0.0482, +0.1590] | +0.0004 [-0.0506, +0.0453] | +0.2700 [-1.0226, +2.1143] | -0.0337 [-0.1574, +0.0650] |
| station_integrated_vs_prior_ecological_gru_tuned_anchor_k0 | +0.0153 [-0.0131, +0.0429] | -0.1747 [-0.2636, -0.0711] | +0.0363 [+0.0093, +0.0674] | +1.9382 [+0.9388, +3.4943] | +0.2839 [+0.1301, +0.5079] |
| tail2_integrated_vs_prior_ecological_gru_tuned_anchor_k5 | +0.0088 [-0.0032, +0.0222] | -0.0215 [-0.0771, +0.0324] | +0.0129 [+0.0010, +0.0269] | +0.6148 [+0.1554, +1.2963] | +0.0888 [+0.0320, +0.1682] |
| mae_integrated_vs_prior_ecological_gru_tuned_anchor_k5 | -0.0031 [-0.0122, +0.0047] | +0.0050 [-0.0160, +0.0253] | -0.0038 [-0.0145, +0.0049] | +0.0175 [-0.4407, +0.6171] | +0.0053 [-0.0232, +0.0359] |
| selective_integrated_vs_prior_ecological_gru_tuned_anchor_k5 | -0.0072 [-0.0182, +0.0018] | +0.0064 [-0.0102, +0.0302] | -0.0086 [-0.0222, +0.0021] | +0.1841 [-0.3080, +0.8618] | +0.0480 [+0.0123, +0.1024] |
| station_integrated_vs_prior_ecological_gru_tuned_anchor_k5 | +0.0244 [+0.0049, +0.0474] | +0.0131 [-0.0766, +0.1036] | +0.0265 [+0.0077, +0.0503] | +0.6406 [+0.1214, +1.3903] | +0.1208 [+0.0484, +0.2319] |

## Ecological integration versus direct residuals

| Comparison | Overall Delta MAE [95% CI] | Q90 Delta MAE | Ordinary Delta MAE | Recall change [95% CI], pp | False-Q90 change [95% CI], pp |
|---|---:|---:|---:|---:|---:|
| tail2_integrated_vs_direct_constant_k5 | -0.0038 [-0.0128, +0.0045] | -0.0134 [-0.0528, +0.0246] | -0.0030 [-0.0109, +0.0053] | +0.0758 [-0.0949, +0.2793] | -0.0028 [-0.0411, +0.0344] |
| mae_integrated_vs_direct_constant_k5 | -0.0098 [-0.0210, +0.0001] | -0.0385 [-0.0940, +0.0089] | -0.0070 [-0.0161, +0.0019] | +0.0102 [-0.3038, +0.3362] | -0.0010 [-0.0368, +0.0367] |
| selective_integrated_vs_direct_constant_k5 | -0.0032 [-0.0090, +0.0026] | -0.0049 [-0.0173, +0.0063] | -0.0030 [-0.0093, +0.0029] | -0.0782 [-0.4415, +0.2389] | +0.0120 [-0.0193, +0.0566] |
| station_integrated_vs_direct_constant_k5 | -0.0010 [-0.0028, +0.0008] | +0.0070 [+0.0013, +0.0125] | -0.0019 [-0.0039, +0.0000] | -0.0461 [-0.1744, +0.0569] | -0.0177 [-0.0506, +0.0071] |
| tail2_integrated_vs_direct_gru_tuned_anchor_k0 | -0.0132 [-0.0229, -0.0049] | +0.0144 [-0.0036, +0.0351] | -0.0167 [-0.0280, -0.0078] | +0.1131 [-0.1711, +0.4680] | +0.0103 [-0.0221, +0.0486] |
| mae_integrated_vs_direct_gru_tuned_anchor_k0 | -0.0129 [-0.0237, -0.0016] | -0.0224 [-0.0521, +0.0025] | -0.0115 [-0.0230, +0.0007] | +0.7629 [+0.2263, +1.4560] | +0.0450 [+0.0033, +0.1076] |
| selective_integrated_vs_direct_gru_tuned_anchor_k0 | -0.0092 [-0.0172, -0.0014] | -0.0191 [-0.0428, +0.0015] | -0.0079 [-0.0163, +0.0004] | +0.3657 [+0.0475, +0.7422] | +0.0070 [-0.0202, +0.0434] |
| station_integrated_vs_direct_gru_tuned_anchor_k0 | -0.0204 [-0.0368, -0.0067] | +0.0237 [+0.0018, +0.0525] | -0.0260 [-0.0457, -0.0108] | +0.1476 [-0.2568, +0.6197] | -0.0160 [-0.0740, +0.0259] |
| tail2_integrated_vs_direct_gru_tuned_anchor_k5 | -0.0060 [-0.0153, +0.0021] | -0.0181 [-0.0677, +0.0263] | -0.0051 [-0.0122, +0.0024] | +0.0379 [-0.1427, +0.2655] | -0.0047 [-0.0451, +0.0326] |
| mae_integrated_vs_direct_gru_tuned_anchor_k5 | -0.0048 [-0.0118, +0.0026] | -0.0031 [-0.0275, +0.0207] | -0.0049 [-0.0122, +0.0029] | +0.0826 [-0.2368, +0.4104] | -0.0082 [-0.0385, +0.0203] |
| selective_integrated_vs_direct_gru_tuned_anchor_k5 | -0.0090 [-0.0213, +0.0015] | -0.0449 [-0.1056, +0.0061] | -0.0054 [-0.0147, +0.0040] | -0.1088 [-0.4494, +0.1596] | +0.0017 [-0.0370, +0.0475] |
| station_integrated_vs_direct_gru_tuned_anchor_k5 | -0.0051 [-0.0135, +0.0017] | -0.0213 [-0.0706, +0.0218] | -0.0037 [-0.0094, +0.0024] | +0.0019 [-0.2578, +0.2621] | +0.0032 [-0.0305, +0.0402] |

## Direct residuals versus context

| Comparison | Overall Delta MAE [95% CI] | Q90 Delta MAE | Ordinary Delta MAE | Recall change [95% CI], pp | False-Q90 change [95% CI], pp |
|---|---:|---:|---:|---:|---:|
| tail2_vs_context_constant_k5 | -0.0087 [-0.0354, +0.0208] | -0.0454 [-0.1523, +0.0546] | -0.0042 [-0.0318, +0.0268] | +0.3045 [-0.3503, +1.1883] | +0.0535 [-0.0465, +0.1646] |
| mae_vs_context_constant_k5 | -0.0095 [-0.0385, +0.0207] | +0.0412 [-0.0587, +0.1417] | -0.0155 [-0.0463, +0.0160] | -1.0573 [-2.1275, -0.1029] | -0.0793 [-0.2261, +0.0346] |
| selective_vs_context_constant_k5 | -0.0154 [-0.0456, +0.0155] | +0.0385 [-0.0452, +0.1208] | -0.0217 [-0.0544, +0.0120] | -0.6003 [-1.6412, +0.4436] | -0.0599 [-0.1909, +0.0430] |
| station_vs_context_constant_k5 | +0.0058 [-0.0254, +0.0406] | -0.0071 [-0.1439, +0.1255] | +0.0077 [-0.0231, +0.0421] | +0.3302 [-0.3471, +1.2418] | +0.0869 [-0.0092, +0.2231] |
| tail2_vs_context_gru_tuned_anchor_k0 | -0.0788 [-0.1276, -0.0326] | -0.3217 [-0.5134, -0.1245] | -0.0530 [-0.1037, -0.0024] | +2.9799 [+1.4411, +5.2270] | +0.2621 [+0.1190, +0.4746] |
| mae_vs_context_gru_tuned_anchor_k0 | -0.0813 [-0.1504, -0.0205] | -0.0229 [-0.1515, +0.1392] | -0.0903 [-0.1672, -0.0199] | -1.0553 [-2.6435, +0.3747] | -0.1104 [-0.2352, -0.0228] |
| selective_vs_context_gru_tuned_anchor_k0 | -0.0797 [-0.1526, -0.0140] | -0.0607 [-0.2316, +0.1451] | -0.0838 [-0.1632, -0.0086] | +1.0829 [-0.3148, +3.0916] | +0.0386 [-0.0733, +0.1478] |
| station_vs_context_gru_tuned_anchor_k0 | -0.0580 [-0.0998, -0.0203] | -0.3253 [-0.5053, -0.1334] | -0.0298 [-0.0664, +0.0094] | +2.9693 [+1.5627, +5.0323] | +0.3792 [+0.1743, +0.6740] |
| tail2_vs_context_gru_tuned_anchor_k5 | -0.0013 [-0.0322, +0.0337] | -0.0171 [-0.1377, +0.1019] | +0.0008 [-0.0315, +0.0391] | -0.1739 [-1.1121, +0.8437] | +0.0205 [-0.0978, +0.1357] |
| mae_vs_context_gru_tuned_anchor_k5 | -0.0145 [-0.0410, +0.0127] | -0.0055 [-0.0739, +0.0622] | -0.0160 [-0.0453, +0.0132] | -0.8159 [-2.0853, +0.3447] | -0.0596 [-0.1803, +0.0347] |
| selective_vs_context_gru_tuned_anchor_k5 | -0.0143 [-0.0442, +0.0158] | +0.0377 [-0.0460, +0.1193] | -0.0203 [-0.0520, +0.0124] | -0.4580 [-1.6431, +0.7136] | -0.0268 [-0.1394, +0.0692] |
| station_vs_context_gru_tuned_anchor_k5 | +0.0134 [-0.0222, +0.0538] | +0.0207 [-0.1310, +0.1730] | +0.0131 [-0.0240, +0.0552] | -0.1122 [-1.0837, +0.9478] | +0.0444 [-0.0781, +0.1868] |

## Source-validation choices

| Loss arm | Selected epoch range | Total epochs executed | Scale counts | Mean selected validation MAE |
|---|---:|---:|---|---:|
| tail2 | 5–49 | 307 | 0.5: 2, 1: 7 | 1.869505 |
| mae | 2–67 | 359 | 1: 9 | 1.881015 |
| selective | 19–61 | 413 | 0.5: 2, 1: 7 | 1.854002 |
| station | 0–69 | 323 | 0: 1, 0.25: 2, 1: 6 | 1.884681 |

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
| selective_integrated_constant | 5 | 0: 4, 0.25: 3, 0.5: 2 |
| selective_integrated_gru_tuned_anchor | 0 | 0: 4, 0.25: 4, 0.5: 1 |
| selective_integrated_gru_tuned_anchor | 1 | 0: 5, 0.25: 3, 0.5: 1 |
| selective_integrated_gru_tuned_anchor | 3 | 0: 3, 0.25: 1, 0.5: 5 |
| selective_integrated_gru_tuned_anchor | 5 | 0: 3, 0.25: 3, 0.5: 3 |
| station_integrated_constant | 0 | 0: 4, 0.25: 2, 0.5: 2, 1: 1 |
| station_integrated_constant | 1 | 0: 4, 0.25: 2, 0.5: 1, 1: 2 |
| station_integrated_constant | 3 | 0: 3, 0.25: 3, 0.5: 1, 1: 2 |
| station_integrated_constant | 5 | 0: 6, 0.25: 2, 1: 1 |
| station_integrated_gru_tuned_anchor | 0 | 0: 4, 0.25: 2, 0.5: 2, 1: 1 |
| station_integrated_gru_tuned_anchor | 1 | 0: 4, 0.25: 2, 0.5: 1, 1: 2 |
| station_integrated_gru_tuned_anchor | 3 | 0: 3, 0.25: 1, 0.5: 3, 1: 2 |
| station_integrated_gru_tuned_anchor | 5 | 0: 5, 0.25: 3, 1: 1 |

Interpret direct loss contrasts before integrated ones: the latter also include changes
in source-validation-selected ecological mixing and positive-K support calibration.
A narrow or sign-changing contrast does not establish equivalence. Station gain/harm
concentration and partition/seed directions are retained alongside the overall means.
