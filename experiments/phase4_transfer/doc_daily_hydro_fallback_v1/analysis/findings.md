# Availability-aware daily/monthly fallback

The monthly and daily neural experts remain frozen. A hybrid combines their native
predictions using the fixed label-free availability rule: any valid daily descriptor selects daily; otherwise monthly.
Support adapters and ecological mixing use source validation only, on the same fixed
episodes and candidate grids. The route never reads DOC labels; its fixed availability rule
was motivated by the preceding experiment, before inspecting this experiment’s target results.

Direct and ecological-integrated pipelines are reported separately. A change in
the integrated product can include a changed ecological mixture or support adapter;
it is not solely the effect of switching the underlying expert.

All 20 saved models and all K values are retained, including historical references.
The copied monthly/daily controls and original references must reproduce exactly.
The reference checks contain 144 model/run comparisons, all verified exactly.

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
| daily_gru_tuned_anchor | 1.8070 | 1.7588 | 1.6313 | 1.5894 | 3.6045 | 0.5655 |
| hybrid_gru_tuned_anchor | 1.7976 | 1.7522 | 1.6306 | 1.5882 | 3.6037 | 0.5657 |
| monthly_integrated_gru_tuned_anchor | 1.8096 | 1.7771 | 1.6266 | 1.5890 | 3.5946 | 0.5665 |
| daily_integrated_gru_tuned_anchor | 1.8035 | 1.7554 | 1.6201 | 1.5650 | 3.5523 | 0.5759 |
| hybrid_integrated_gru_tuned_anchor | 1.7944 | 1.7489 | 1.6194 | 1.5714 | 3.5743 | 0.5714 |
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
| daily_gru_tuned_anchor | 0 | 7.1673 | -5.5320 | 1.1941 | +0.2133 | 64.36% | 77.42% | 2.19% |
| daily_gru_tuned_anchor | 5 | 6.6423 | -5.1758 | 1.0075 | +0.1665 | 65.55% | 77.74% | 2.20% |
| hybrid_gru_tuned_anchor | 0 | 7.1696 | -5.5311 | 1.1832 | +0.2277 | 64.01% | 77.39% | 2.18% |
| hybrid_gru_tuned_anchor | 5 | 6.6367 | -5.1602 | 1.0069 | +0.1700 | 65.64% | 77.73% | 2.21% |
| monthly_integrated_gru_tuned_anchor | 0 | 7.2550 | -5.9174 | 1.1859 | +0.2461 | 61.67% | 76.92% | 2.15% |
| monthly_integrated_gru_tuned_anchor | 5 | 6.6607 | -5.0908 | 1.0043 | +0.1932 | 64.85% | 77.44% | 2.22% |
| daily_integrated_gru_tuned_anchor | 0 | 7.1792 | -5.5920 | 1.1887 | +0.2035 | 64.06% | 77.64% | 2.16% |
| daily_integrated_gru_tuned_anchor | 5 | 6.5870 | -4.9657 | 0.9856 | +0.1711 | 65.13% | 77.81% | 2.19% |
| hybrid_integrated_gru_tuned_anchor | 0 | 7.1819 | -5.5919 | 1.1780 | +0.2179 | 63.69% | 77.60% | 2.15% |
| hybrid_integrated_gru_tuned_anchor | 5 | 6.6151 | -5.0577 | 0.9900 | +0.1747 | 65.21% | 77.76% | 2.20% |
| prior_encoder_gru_tuned_anchor | 0 | 7.2369 | -5.8588 | 1.2044 | +0.2745 | 61.88% | 76.82% | 2.17% |
| prior_encoder_gru_tuned_anchor | 5 | 6.6774 | -5.1456 | 1.0092 | +0.1918 | 64.84% | 77.35% | 2.23% |
| prior_encoder_integrated_gru_tuned_anchor | 0 | 7.2513 | -5.9115 | 1.1877 | +0.2315 | 61.99% | 76.77% | 2.18% |
| prior_encoder_integrated_gru_tuned_anchor | 5 | 6.6593 | -5.0843 | 1.0041 | +0.1921 | 64.88% | 77.42% | 2.22% |
| prior_ecological_affine_gru_tuned_anchor | 0 | 7.4318 | -6.4542 | 1.1653 | +0.1960 | 60.08% | 78.00% | 1.99% |
| prior_ecological_affine_gru_tuned_anchor | 5 | 6.6808 | -5.1324 | 0.9913 | +0.1782 | 64.26% | 78.04% | 2.14% |

Recall conditions on true high DOC; false-Q90 rate conditions on ordinary observations.
A lower false-positive rate alone is not better selectivity if recall also collapses. Precision is descriptive; paired intervals are reported for recall and false-positive rate.

## Hybrid versus frozen monthly and daily pipelines

| Comparison | Overall Delta MAE [95% CI] | Q90 Delta MAE | Ordinary Delta MAE | Recall change [95% CI], pp | False-Q90 change [95% CI], pp |
|---|---:|---:|---:|---:|---:|
| hybrid_vs_daily_constant_k5 | -0.0012 [-0.0037, +0.0013] | -0.0041 [-0.0127, +0.0033] | -0.0009 [-0.0037, +0.0019] | +0.0879 [-0.0929, +0.3728] | -0.0017 [-0.0234, +0.0203] |
| hybrid_vs_monthly_constant_k5 | -0.0108 [-0.0316, +0.0098] | -0.0444 [-0.0985, -0.0019] | -0.0068 [-0.0293, +0.0147] | +0.8440 [-0.0966, +2.0757] | -0.0056 [-0.0958, +0.0702] |
| hybrid_vs_daily_integrated_constant_k5 | +0.0042 [-0.0019, +0.0121] | +0.0184 [-0.0242, +0.0672] | +0.0028 [-0.0018, +0.0079] | +0.1627 [-0.0628, +0.5013] | +0.0075 [-0.0198, +0.0342] |
| hybrid_vs_monthly_integrated_constant_k5 | -0.0181 [-0.0325, -0.0054] | -0.0426 [-0.0758, -0.0138] | -0.0152 [-0.0306, -0.0016] | +0.4696 [-0.3169, +1.4323] | -0.0316 [-0.1129, +0.0230] |
| hybrid_vs_daily_gru_tuned_anchor_k0 | -0.0094 [-0.0207, +0.0008] | +0.0023 [-0.0253, +0.0415] | -0.0110 [-0.0230, -0.0010] | -0.3560 [-1.7068, +0.3841] | -0.0099 [-0.0530, +0.0170] |
| hybrid_vs_monthly_gru_tuned_anchor_k0 | -0.0252 [-0.0683, +0.0128] | -0.0710 [-0.2007, +0.0400] | -0.0194 [-0.0640, +0.0213] | +2.4516 [+0.3960, +5.0345] | +0.0415 [-0.1148, +0.2071] |
| hybrid_vs_daily_integrated_gru_tuned_anchor_k0 | -0.0092 [-0.0201, +0.0006] | +0.0028 [-0.0242, +0.0420] | -0.0108 [-0.0224, -0.0012] | -0.3759 [-1.7241, +0.3497] | -0.0099 [-0.0530, +0.0170] |
| hybrid_vs_monthly_integrated_gru_tuned_anchor_k0 | -0.0152 [-0.0538, +0.0203] | -0.0730 [-0.1969, +0.0298] | -0.0080 [-0.0481, +0.0303] | +2.0198 [+0.1978, +4.2999] | -0.0065 [-0.1683, +0.1584] |
| hybrid_vs_daily_gru_tuned_anchor_k5 | -0.0012 [-0.0040, +0.0019] | -0.0056 [-0.0137, +0.0002] | -0.0007 [-0.0038, +0.0028] | +0.0996 [+0.0000, +0.3334] | +0.0078 [-0.0172, +0.0345] |
| hybrid_vs_monthly_gru_tuned_anchor_k5 | -0.0027 [-0.0250, +0.0201] | -0.0118 [-0.0919, +0.0607] | -0.0010 [-0.0246, +0.0216] | +0.7741 [-0.2022, +1.9587] | -0.0083 [-0.0913, +0.0501] |
| hybrid_vs_daily_integrated_gru_tuned_anchor_k5 | +0.0064 [+0.0003, +0.0149] | +0.0281 [-0.0168, +0.0834] | +0.0044 [+0.0002, +0.0092] | +0.0796 [-0.1398, +0.3205] | +0.0109 [-0.0176, +0.0409] |
| hybrid_vs_monthly_integrated_gru_tuned_anchor_k5 | -0.0176 [-0.0313, -0.0055] | -0.0456 [-0.0776, -0.0171] | -0.0143 [-0.0289, -0.0016] | +0.3681 [-0.3146, +1.1554] | -0.0226 [-0.1075, +0.0338] |

## Direct and integrated hybrid versus the fixed ecological reference

| Comparison | Overall Delta MAE [95% CI] | Q90 Delta MAE | Ordinary Delta MAE | Recall change [95% CI], pp | False-Q90 change [95% CI], pp |
|---|---:|---:|---:|---:|---:|
| hybrid_vs_prior_ecological_constant_k5 | -0.0029 [-0.0310, +0.0258] | -0.0816 [-0.2038, +0.0268] | +0.0074 [-0.0203, +0.0354] | +2.0247 [+0.6351, +3.7724] | +0.1014 [+0.0107, +0.2198] |
| hybrid_integrated_vs_prior_ecological_constant_k5 | -0.0145 [-0.0317, +0.0012] | -0.0937 [-0.1604, -0.0377] | -0.0044 [-0.0211, +0.0109] | +1.7260 [+0.5773, +3.2212] | +0.0785 [+0.0190, +0.1649] |
| hybrid_vs_prior_ecological_gru_tuned_anchor_k0 | -0.0116 [-0.0701, +0.0433] | -0.2621 [-0.4409, -0.0910] | +0.0179 [-0.0410, +0.0763] | +3.9308 [+1.8081, +6.6664] | +0.1961 [+0.0034, +0.4152] |
| hybrid_integrated_vs_prior_ecological_gru_tuned_anchor_k0 | -0.0148 [-0.0690, +0.0367] | -0.2498 [-0.4193, -0.0894] | +0.0127 [-0.0427, +0.0677] | +3.6122 [+1.5760, +6.2429] | +0.1584 [-0.0279, +0.3636] |
| hybrid_vs_prior_ecological_gru_tuned_anchor_k5 | +0.0083 [-0.0177, +0.0366] | -0.0441 [-0.1710, +0.0796] | +0.0156 [-0.0087, +0.0417] | +1.3826 [+0.1778, +2.8315] | +0.0758 [-0.0022, +0.1586] |
| hybrid_integrated_vs_prior_ecological_gru_tuned_anchor_k5 | -0.0085 [-0.0229, +0.0056] | -0.0656 [-0.1314, -0.0083] | -0.0012 [-0.0148, +0.0125] | +0.9514 [-0.0014, +2.1618] | +0.0632 [+0.0062, +0.1304] |

## Ecological integration versus direct residuals

| Comparison | Overall Delta MAE [95% CI] | Q90 Delta MAE | Ordinary Delta MAE | Recall change [95% CI], pp | False-Q90 change [95% CI], pp |
|---|---:|---:|---:|---:|---:|
| monthly_integrated_vs_direct_constant_k5 | -0.0043 [-0.0131, +0.0035] | -0.0139 [-0.0541, +0.0245] | -0.0035 [-0.0111, +0.0041] | +0.0758 [-0.0949, +0.2793] | +0.0032 [-0.0339, +0.0406] |
| daily_integrated_vs_direct_constant_k5 | -0.0170 [-0.0408, +0.0032] | -0.0346 [-0.1412, +0.0681] | -0.0155 [-0.0366, +0.0039] | -0.3734 [-0.9660, +0.0806] | -0.0321 [-0.1101, +0.0453] |
| hybrid_integrated_vs_direct_constant_k5 | -0.0116 [-0.0290, +0.0040] | -0.0121 [-0.0730, +0.0523] | -0.0119 [-0.0289, +0.0041] | -0.2987 [-0.7817, +0.1090] | -0.0229 [-0.0894, +0.0476] |
| monthly_integrated_vs_direct_gru_tuned_anchor_k0 | -0.0132 [-0.0228, -0.0049] | +0.0144 [-0.0037, +0.0351] | -0.0167 [-0.0280, -0.0078] | +0.1131 [-0.1711, +0.4680] | +0.0103 [-0.0221, +0.0486] |
| daily_integrated_vs_direct_gru_tuned_anchor_k0 | -0.0035 [-0.0098, +0.0028] | +0.0118 [-0.0030, +0.0274] | -0.0054 [-0.0123, +0.0015] | -0.2987 [-0.6057, -0.0817] | -0.0377 [-0.0728, -0.0136] |
| hybrid_integrated_vs_direct_gru_tuned_anchor_k0 | -0.0032 [-0.0089, +0.0027] | +0.0123 [-0.0027, +0.0279] | -0.0052 [-0.0114, +0.0013] | -0.3186 [-0.6285, -0.0996] | -0.0377 [-0.0728, -0.0136] |
| monthly_integrated_vs_direct_gru_tuned_anchor_k5 | -0.0019 [-0.0062, +0.0023] | +0.0123 [+0.0025, +0.0225] | -0.0036 [-0.0083, +0.0009] | -0.0253 [-0.1728, +0.1289] | +0.0017 [-0.0255, +0.0352] |
| daily_integrated_vs_direct_gru_tuned_anchor_k5 | -0.0244 [-0.0488, -0.0049] | -0.0553 [-0.1812, +0.0554] | -0.0219 [-0.0404, -0.0048] | -0.4113 [-1.1807, +0.3553] | -0.0158 [-0.0679, +0.0369] |
| hybrid_integrated_vs_direct_gru_tuned_anchor_k5 | -0.0168 [-0.0341, -0.0013] | -0.0216 [-0.0928, +0.0465] | -0.0169 [-0.0330, -0.0014] | -0.4312 [-1.1137, +0.1739] | -0.0127 [-0.0584, +0.0325] |

## Direct residuals versus context

| Comparison | Overall Delta MAE [95% CI] | Q90 Delta MAE | Ordinary Delta MAE | Recall change [95% CI], pp | False-Q90 change [95% CI], pp |
|---|---:|---:|---:|---:|---:|
| monthly_vs_context_constant_k5 | -0.0079 [-0.0342, +0.0214] | -0.0434 [-0.1499, +0.0576] | -0.0035 [-0.0310, +0.0269] | +0.3045 [-0.3503, +1.1883] | +0.0413 [-0.0581, +0.1505] |
| daily_vs_context_constant_k5 | -0.0175 [-0.0546, +0.0211] | -0.0838 [-0.2200, +0.0341] | -0.0095 [-0.0480, +0.0312] | +1.0606 [-0.2015, +2.6309] | +0.0373 [-0.1002, +0.1709] |
| hybrid_vs_context_constant_k5 | -0.0187 [-0.0559, +0.0191] | -0.0878 [-0.2233, +0.0310] | -0.0104 [-0.0485, +0.0298] | +1.1485 [-0.0860, +2.6573] | +0.0357 [-0.0975, +0.1655] |
| monthly_vs_context_gru_tuned_anchor_k0 | -0.0801 [-0.1262, -0.0359] | -0.3180 [-0.5069, -0.1243] | -0.0547 [-0.1027, -0.0068] | +2.6580 [+1.2557, +4.6360] | +0.2340 [+0.0952, +0.4339] |
| daily_vs_context_gru_tuned_anchor_k0 | -0.0958 [-0.1705, -0.0281] | -0.3913 [-0.6280, -0.1546] | -0.0632 [-0.1413, +0.0094] | +5.4655 [+2.3881, +9.3711] | +0.2853 [+0.0822, +0.5405] |
| hybrid_vs_context_gru_tuned_anchor_k0 | -0.1052 [-0.1772, -0.0389] | -0.3890 [-0.6194, -0.1564] | -0.0742 [-0.1504, -0.0051] | +5.1095 [+2.3067, +8.6528] | +0.2754 [+0.0710, +0.5292] |
| monthly_vs_context_gru_tuned_anchor_k5 | -0.0051 [-0.0342, +0.0270] | -0.0460 [-0.1320, +0.0334] | -0.0005 [-0.0316, +0.0365] | -0.1423 [-1.0817, +0.8547] | +0.0110 [-0.1107, +0.1179] |
| daily_vs_context_gru_tuned_anchor_k5 | -0.0066 [-0.0465, +0.0349] | -0.0522 [-0.1961, +0.0813] | -0.0009 [-0.0417, +0.0430] | +0.5322 [-0.8474, +2.1034] | -0.0051 [-0.1588, +0.1173] |
| hybrid_vs_context_gru_tuned_anchor_k5 | -0.0078 [-0.0471, +0.0333] | -0.0578 [-0.2023, +0.0761] | -0.0015 [-0.0414, +0.0416] | +0.6318 [-0.7021, +2.1523] | +0.0027 [-0.1479, +0.1264] |

## Source-validation choices

The expert-training checkpoints are frozen; this experiment does not train another backbone.
The fixed router is serialized in router.json; fitted support and ecological choices are saved in CSVs.

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
| daily_integrated_constant | 0 | 0: 7, 0.25: 2 |
| daily_integrated_constant | 1 | 0: 7, 0.25: 2 |
| daily_integrated_constant | 3 | 0: 2, 0.25: 4, 0.5: 3 |
| daily_integrated_constant | 5 | 0: 2, 0.25: 1, 0.5: 6 |
| daily_integrated_gru_tuned_anchor | 0 | 0: 7, 0.25: 2 |
| daily_integrated_gru_tuned_anchor | 1 | 0: 7, 0.25: 2 |
| daily_integrated_gru_tuned_anchor | 3 | 0: 1, 0.25: 5, 0.5: 3 |
| daily_integrated_gru_tuned_anchor | 5 | 0: 1, 0.25: 4, 0.5: 4 |
| hybrid_integrated_constant | 0 | 0: 7, 0.25: 2 |
| hybrid_integrated_constant | 1 | 0: 7, 0.25: 2 |
| hybrid_integrated_constant | 3 | 0: 2, 0.25: 4, 0.5: 3 |
| hybrid_integrated_constant | 5 | 0: 2, 0.25: 3, 0.5: 4 |
| hybrid_integrated_gru_tuned_anchor | 0 | 0: 7, 0.25: 2 |
| hybrid_integrated_gru_tuned_anchor | 1 | 0: 7, 0.25: 2 |
| hybrid_integrated_gru_tuned_anchor | 3 | 0: 1, 0.25: 5, 0.5: 3 |
| hybrid_integrated_gru_tuned_anchor | 5 | 0: 1, 0.25: 5, 0.5: 3 |

Interpret direct feature contrasts before integrated ones: the latter also include changes
in source-validation-selected ecological mixing and positive-K support calibration.
A narrow or sign-changing contrast does not establish equivalence. Station gain/harm
concentration and partition/seed directions are retained alongside the overall means.
