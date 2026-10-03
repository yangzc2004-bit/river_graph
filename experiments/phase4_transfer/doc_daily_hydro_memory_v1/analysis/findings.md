# Daily hydrological state in the existing DOC temporal residual

The three neural arms retain the same spatial/ecological encoder, original GRU
initialization, 38-channel scalar head, source OOF context base, 120-epoch cap and patience 5.
All use cell-weighted native MAE with weight two at or above source-training Q90.

- off: daily descriptors enter only the current scalar residual head.
- current_only: a zero-initialized daily projection also enters the target GRU step.
- full_history: the same projection enters every valid step of the causal 12-month GRU window.

Current_only and full_history have the same additional 512 projection weights; their
contrast isolates historical exposure with matched parameter count. Off has no projection
and reproduces the previous daily-head-only model. Full_history versus off changes both
the recurrent input and parameter count, so it does not isolate history by itself.

Two ExtraTrees probes clone the previously selected context-forest parameters without
new tuning. Both have 147 columns: unchanged context information plus twelve daily
slots. The current probe zeros historical numeric/availability slots, retaining identical
valid-date flags. The history probe retains all twelve causal slots. They are independent
comparators and do not replace the neural forest base or its OOF residual targets.

Current-month daily summaries support retrospective monthly reconstruction, not a forecast
issued before that month. The original discharge footprint, source data and support basis remain fixed.
The no-message spatial self path remains in use; this experiment does not establish river-message gain.

Epoch and scale selection uses unweighted source-validation K0 MAE. Support and ecological
mixing choices use the same validation episodes and grids, independently for each arm.
All 24 models and all K curves are retained, including matched tree information controls.
Exactly 72 copied model/run checks and 36 historical off-control checks are recorded.
Historical equivalence is required for the formal matched budget; an explicit smoke-only
unmatched-budget option records its different cap and does not assert retraining equivalence.

Intervals use 5,000 paired whole-station bootstrap draws, jointly resampling the same
station identities across partitions. Seed means are averaged within each partition and
partitions receive equal weight. Negative error/FPR deltas favor the candidate; positive
recall deltas favor detection. Q90 includes ties at the source-training threshold.
These are reused development partitions; no new target outcome chooses a model, K,
checkpoint, scale, support adapter, ecological mixture, or descriptive threshold.

## Complete K curves

| Model (GRU support basis) | K0 MAE | K1 MAE | K3 MAE | K5 MAE | K5 RMSE | K5 R² |
|---|---:|---:|---:|---:|---:|---:|
| context_gru_tuned_anchor | 1.9028 | 1.8666 | 1.6348 | 1.5960 | 3.5911 | 0.5675 |
| off_gru_tuned_anchor | 1.8070 | 1.7588 | 1.6313 | 1.5894 | 3.6045 | 0.5655 |
| current_only_gru_tuned_anchor | 1.8104 | 1.7628 | 1.6347 | 1.5872 | 3.5853 | 0.5695 |
| full_history_gru_tuned_anchor | 1.8009 | 1.7563 | 1.6318 | 1.5864 | 3.5838 | 0.5700 |
| tree_current_gru_tuned_anchor | 1.8558 | 1.8236 | 1.6080 | 1.5733 | 3.5665 | 0.5734 |
| tree_history_gru_tuned_anchor | 1.8489 | 1.8151 | 1.6047 | 1.5773 | 3.5691 | 0.5725 |
| off_integrated_gru_tuned_anchor | 1.8035 | 1.7554 | 1.6201 | 1.5650 | 3.5523 | 0.5759 |
| current_only_integrated_gru_tuned_anchor | 1.8079 | 1.7602 | 1.6221 | 1.5677 | 3.5554 | 0.5754 |
| full_history_integrated_gru_tuned_anchor | 1.7974 | 1.7528 | 1.6206 | 1.5676 | 3.5576 | 0.5751 |
| prior_daily_gru_tuned_anchor | 1.8070 | 1.7588 | 1.6313 | 1.5894 | 3.6045 | 0.5655 |
| prior_daily_integrated_gru_tuned_anchor | 1.8035 | 1.7554 | 1.6201 | 1.5650 | 3.5523 | 0.5759 |
| prior_ecological_affine_gru_tuned_anchor | 1.8092 | 1.7917 | 1.6221 | 1.5799 | 3.5814 | 0.5689 |

Constant-only support adaptation is retained in all CSVs. Its K0 predictions duplicate
the GRU-basis K0 predictions, so duplicate K0 bootstrap comparisons are omitted.

## Tail accuracy, ordinary error and classification

| Model | K | Q90 MAE | Q90 bias | Ordinary MAE | Ordinary bias | Q90 recall | Q90 precision | False-Q90 rate |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| context_gru_tuned_anchor | 0 | 7.5586 | -6.6062 | 1.2574 | +0.4820 | 58.90% | 78.47% | 1.91% |
| context_gru_tuned_anchor | 5 | 6.6945 | -5.1653 | 1.0084 | +0.2315 | 65.01% | 77.68% | 2.21% |
| off_gru_tuned_anchor | 0 | 7.1673 | -5.5320 | 1.1941 | +0.2133 | 64.36% | 77.42% | 2.19% |
| off_gru_tuned_anchor | 5 | 6.6423 | -5.1758 | 1.0075 | +0.1665 | 65.55% | 77.74% | 2.20% |
| current_only_gru_tuned_anchor | 0 | 7.1512 | -5.5241 | 1.1996 | +0.2269 | 64.16% | 77.60% | 2.17% |
| current_only_gru_tuned_anchor | 5 | 6.5968 | -5.0628 | 1.0099 | +0.1776 | 65.48% | 77.74% | 2.20% |
| full_history_gru_tuned_anchor | 0 | 7.1475 | -5.5211 | 1.1891 | +0.2226 | 64.16% | 77.55% | 2.18% |
| full_history_gru_tuned_anchor | 5 | 6.6087 | -5.0510 | 1.0079 | +0.1737 | 65.15% | 77.47% | 2.22% |
| tree_current_gru_tuned_anchor | 0 | 7.5353 | -6.6323 | 1.2081 | +0.4241 | 59.35% | 78.58% | 1.92% |
| tree_current_gru_tuned_anchor | 5 | 6.6299 | -5.0928 | 0.9909 | +0.2330 | 65.32% | 77.69% | 2.23% |
| tree_history_gru_tuned_anchor | 0 | 7.5252 | -6.6393 | 1.2015 | +0.3890 | 60.03% | 78.31% | 1.95% |
| tree_history_gru_tuned_anchor | 5 | 6.6389 | -5.1577 | 0.9940 | +0.2325 | 64.52% | 77.80% | 2.17% |
| off_integrated_gru_tuned_anchor | 0 | 7.1792 | -5.5920 | 1.1887 | +0.2035 | 64.06% | 77.64% | 2.16% |
| off_integrated_gru_tuned_anchor | 5 | 6.5870 | -4.9657 | 0.9856 | +0.1711 | 65.13% | 77.81% | 2.19% |
| current_only_integrated_gru_tuned_anchor | 0 | 7.1573 | -5.5500 | 1.1960 | +0.2238 | 64.08% | 77.70% | 2.15% |
| current_only_integrated_gru_tuned_anchor | 5 | 6.5842 | -4.9659 | 0.9894 | +0.1776 | 65.11% | 77.92% | 2.18% |
| full_history_integrated_gru_tuned_anchor | 0 | 7.1612 | -5.6152 | 1.1835 | +0.2135 | 64.00% | 77.77% | 2.15% |
| full_history_integrated_gru_tuned_anchor | 5 | 6.5949 | -4.9655 | 0.9880 | +0.1748 | 65.19% | 77.83% | 2.19% |
| prior_daily_gru_tuned_anchor | 0 | 7.1673 | -5.5320 | 1.1941 | +0.2133 | 64.36% | 77.42% | 2.19% |
| prior_daily_gru_tuned_anchor | 5 | 6.6423 | -5.1758 | 1.0075 | +0.1665 | 65.55% | 77.74% | 2.20% |
| prior_daily_integrated_gru_tuned_anchor | 0 | 7.1792 | -5.5920 | 1.1887 | +0.2035 | 64.06% | 77.64% | 2.16% |
| prior_daily_integrated_gru_tuned_anchor | 5 | 6.5870 | -4.9657 | 0.9856 | +0.1711 | 65.13% | 77.81% | 2.19% |
| prior_ecological_affine_gru_tuned_anchor | 0 | 7.4318 | -6.4542 | 1.1653 | +0.1960 | 60.08% | 78.00% | 1.99% |
| prior_ecological_affine_gru_tuned_anchor | 5 | 6.6808 | -5.1324 | 0.9913 | +0.1782 | 64.26% | 78.04% | 2.14% |

Recall conditions on true high DOC; false-Q90 rate conditions on ordinary observations.
A lower false-positive rate alone is not better selectivity if recall also collapses. Precision is descriptive; paired intervals are reported for recall and false-positive rate.

## Current input and historical memory

| Comparison | Overall Delta MAE [95% CI] | Q90 Delta MAE | Ordinary Delta MAE | Recall change [95% CI], pp | False-Q90 change [95% CI], pp |
|---|---:|---:|---:|---:|---:|
| full_history_vs_current_only_constant_k5 | -0.0015 [-0.0056, +0.0025] | +0.0047 [-0.0100, +0.0194] | -0.0020 [-0.0061, +0.0020] | -0.3225 [-0.6903, -0.0553] | +0.0065 [-0.0314, +0.0485] |
| full_history_vs_off_constant_k5 | +0.0009 [-0.0057, +0.0090] | -0.0125 [-0.0331, +0.0075] | +0.0026 [-0.0045, +0.0121] | -0.2652 [-0.6682, +0.0604] | -0.0096 [-0.0493, +0.0246] |
| current_only_vs_off_constant_k5 | +0.0024 [-0.0028, +0.0099] | -0.0172 [-0.0287, -0.0065] | +0.0047 [-0.0014, +0.0137] | +0.0573 [-0.1074, +0.2240] | -0.0162 [-0.0410, +0.0025] |
| full_history_vs_current_only_integrated_constant_k5 | -0.0024 [-0.0067, +0.0015] | -0.0001 [-0.0155, +0.0114] | -0.0029 [-0.0076, +0.0013] | -0.1724 [-0.5817, +0.0473] | +0.0046 [-0.0254, +0.0390] |
| full_history_vs_off_integrated_constant_k5 | -0.0015 [-0.0053, +0.0025] | -0.0111 [-0.0286, +0.0025] | -0.0005 [-0.0043, +0.0035] | +0.0248 [-0.3173, +0.3253] | +0.0008 [-0.0314, +0.0327] |
| current_only_vs_off_integrated_constant_k5 | +0.0010 [-0.0016, +0.0040] | -0.0110 [-0.0177, -0.0051] | +0.0024 [-0.0006, +0.0062] | +0.1972 [+0.0255, +0.4317] | -0.0038 [-0.0244, +0.0133] |
| full_history_vs_current_only_gru_tuned_anchor_k0 | -0.0096 [-0.0178, -0.0013] | -0.0037 [-0.0334, +0.0283] | -0.0105 [-0.0192, -0.0018] | -0.0024 [-0.4782, +0.4497] | +0.0193 [-0.0518, +0.1110] |
| full_history_vs_off_gru_tuned_anchor_k0 | -0.0061 [-0.0223, +0.0116] | -0.0199 [-0.0581, +0.0212] | -0.0050 [-0.0231, +0.0160] | -0.2040 [-0.7870, +0.3786] | -0.0099 [-0.0899, +0.0796] |
| current_only_vs_off_gru_tuned_anchor_k0 | +0.0034 [-0.0111, +0.0222] | -0.0161 [-0.0407, +0.0075] | +0.0055 [-0.0111, +0.0270] | -0.2015 [-0.7893, +0.2695] | -0.0293 [-0.0745, +0.0108] |
| full_history_vs_current_only_integrated_gru_tuned_anchor_k0 | -0.0105 [-0.0189, -0.0022] | +0.0039 [-0.0257, +0.0379] | -0.0125 [-0.0212, -0.0036] | -0.0821 [-0.5948, +0.4174] | +0.0018 [-0.0660, +0.0809] |
| full_history_vs_off_integrated_gru_tuned_anchor_k0 | -0.0061 [-0.0234, +0.0133] | -0.0180 [-0.0540, +0.0203] | -0.0052 [-0.0243, +0.0174] | -0.0646 [-0.5914, +0.5065] | -0.0049 [-0.0795, +0.0732] |
| current_only_vs_off_integrated_gru_tuned_anchor_k0 | +0.0043 [-0.0092, +0.0219] | -0.0219 [-0.0500, +0.0046] | +0.0072 [-0.0082, +0.0272] | +0.0175 [-0.5934, +0.5339] | -0.0067 [-0.0476, +0.0308] |
| full_history_vs_current_only_gru_tuned_anchor_k5 | -0.0008 [-0.0047, +0.0033] | +0.0119 [-0.0044, +0.0260] | -0.0020 [-0.0058, +0.0018] | -0.3249 [-0.7953, +0.0087] | +0.0231 [-0.0191, +0.0743] |
| full_history_vs_off_gru_tuned_anchor_k5 | -0.0030 [-0.0132, +0.0083] | -0.0336 [-0.0661, -0.0006] | +0.0003 [-0.0104, +0.0137] | -0.3924 [-0.9392, +0.0374] | +0.0212 [-0.0230, +0.0769] |
| current_only_vs_off_gru_tuned_anchor_k5 | -0.0022 [-0.0124, +0.0091] | -0.0455 [-0.0772, -0.0148] | +0.0023 [-0.0075, +0.0156] | -0.0675 [-0.5797, +0.5455] | -0.0019 [-0.0348, +0.0364] |
| full_history_vs_current_only_integrated_gru_tuned_anchor_k5 | -0.0002 [-0.0034, +0.0034] | +0.0107 [-0.0117, +0.0301] | -0.0014 [-0.0046, +0.0016] | +0.0811 [-0.1317, +0.3049] | +0.0117 [-0.0090, +0.0388] |
| full_history_vs_off_integrated_gru_tuned_anchor_k5 | +0.0026 [-0.0021, +0.0077] | +0.0079 [-0.0255, +0.0388] | +0.0023 [-0.0016, +0.0061] | +0.0607 [-0.2540, +0.3493] | -0.0007 [-0.0268, +0.0290] |
| current_only_vs_off_integrated_gru_tuned_anchor_k5 | +0.0028 [-0.0006, +0.0064] | -0.0027 [-0.0181, +0.0107] | +0.0037 [+0.0001, +0.0077] | -0.0204 [-0.2719, +0.2169] | -0.0124 [-0.0296, +0.0024] |

## Neural models versus matched tree information

| Comparison | Overall Delta MAE [95% CI] | Q90 Delta MAE | Ordinary Delta MAE | Recall change [95% CI], pp | False-Q90 change [95% CI], pp |
|---|---:|---:|---:|---:|---:|
| off_vs_tree_current_constant_k5 | -0.0021 [-0.0342, +0.0309] | -0.0582 [-0.1709, +0.0417] | +0.0046 [-0.0310, +0.0399] | +1.4176 [+0.4787, +2.7554] | +0.0866 [-0.0034, +0.1981] |
| current_only_vs_tree_current_constant_k5 | +0.0003 [-0.0298, +0.0312] | -0.0754 [-0.1865, +0.0234] | +0.0092 [-0.0228, +0.0419] | +1.4749 [+0.5487, +2.7839] | +0.0704 [-0.0237, +0.1816] |
| full_history_vs_tree_history_constant_k5 | +0.0036 [-0.0281, +0.0367] | -0.0429 [-0.1723, +0.0768] | +0.0095 [-0.0234, +0.0423] | +1.4317 [+0.1920, +3.0496] | +0.0632 [-0.0433, +0.1771] |
| off_integrated_vs_tree_current_constant_k5 | -0.0191 [-0.0381, -0.0013] | -0.0928 [-0.1378, -0.0415] | -0.0110 [-0.0328, +0.0105] | +1.0442 [+0.3728, +2.0281] | +0.0545 [+0.0016, +0.1306] |
| current_only_integrated_vs_tree_current_constant_k5 | -0.0181 [-0.0364, -0.0011] | -0.1037 [-0.1511, -0.0503] | -0.0085 [-0.0289, +0.0114] | +1.2413 [+0.5005, +2.3626] | +0.0508 [-0.0036, +0.1252] |
| full_history_integrated_vs_tree_history_constant_k5 | -0.0157 [-0.0358, +0.0029] | -0.0760 [-0.1291, -0.0250] | -0.0091 [-0.0323, +0.0131] | +1.3482 [+0.2613, +2.7929] | +0.0415 [-0.0648, +0.1707] |
| off_vs_tree_current_gru_tuned_anchor_k0 | -0.0488 [-0.1121, +0.0125] | -0.3680 [-0.6154, -0.1175] | -0.0140 [-0.0791, +0.0514] | +5.0133 [+2.1186, +8.6079] | +0.2779 [+0.0849, +0.5141] |
| current_only_vs_tree_current_gru_tuned_anchor_k0 | -0.0454 [-0.1029, +0.0092] | -0.3841 [-0.6316, -0.1414] | -0.0085 [-0.0618, +0.0468] | +4.8118 [+2.2484, +8.1199] | +0.2487 [+0.0557, +0.4746] |
| full_history_vs_tree_history_gru_tuned_anchor_k0 | -0.0481 [-0.1030, +0.0046] | -0.3777 [-0.6046, -0.1441] | -0.0124 [-0.0647, +0.0416] | +4.1319 [+1.9070, +7.0576] | +0.2306 [+0.0619, +0.4489] |
| off_integrated_vs_tree_current_gru_tuned_anchor_k0 | -0.0522 [-0.1143, +0.0066] | -0.3561 [-0.5982, -0.1163] | -0.0194 [-0.0821, +0.0434] | +4.7146 [+1.9647, +8.2293] | +0.2403 [+0.0538, +0.4640] |
| current_only_integrated_vs_tree_current_gru_tuned_anchor_k0 | -0.0479 [-0.1044, +0.0057] | -0.3780 [-0.6229, -0.1385] | -0.0121 [-0.0642, +0.0426] | +4.7322 [+2.1909, +7.9892] | +0.2336 [+0.0454, +0.4487] |
| full_history_integrated_vs_tree_history_gru_tuned_anchor_k0 | -0.0515 [-0.1034, -0.0030] | -0.3640 [-0.5750, -0.1445] | -0.0180 [-0.0668, +0.0324] | +3.9726 [+1.8423, +6.8303] | +0.1980 [+0.0390, +0.3963] |
| off_vs_tree_current_gru_tuned_anchor_k5 | +0.0161 [-0.0203, +0.0551] | +0.0124 [-0.1168, +0.1369] | +0.0166 [-0.0233, +0.0610] | +0.2238 [-0.9658, +1.6234] | -0.0217 [-0.1698, +0.0941] |
| current_only_vs_tree_current_gru_tuned_anchor_k5 | +0.0139 [-0.0190, +0.0484] | -0.0331 [-0.1438, +0.0677] | +0.0190 [-0.0180, +0.0594] | +0.1563 [-1.0713, +1.5850] | -0.0236 [-0.1618, +0.0898] |
| full_history_vs_tree_history_gru_tuned_anchor_k5 | +0.0092 [-0.0200, +0.0389] | -0.0302 [-0.1554, +0.0873] | +0.0139 [-0.0164, +0.0448] | +0.6357 [-0.5234, +2.0486] | +0.0592 [-0.0467, +0.1720] |
| off_integrated_vs_tree_current_gru_tuned_anchor_k5 | -0.0083 [-0.0360, +0.0214] | -0.0429 [-0.1006, +0.0207] | -0.0053 [-0.0371, +0.0297] | -0.1875 [-1.4239, +1.1908] | -0.0374 [-0.1680, +0.0581] |
| current_only_integrated_vs_tree_current_gru_tuned_anchor_k5 | -0.0055 [-0.0317, +0.0226] | -0.0456 [-0.1028, +0.0120] | -0.0016 [-0.0324, +0.0322] | -0.2079 [-1.3418, +1.0461] | -0.0498 [-0.1857, +0.0466] |
| full_history_integrated_vs_tree_history_gru_tuned_anchor_k5 | -0.0097 [-0.0290, +0.0094] | -0.0439 [-0.1068, +0.0147] | -0.0060 [-0.0285, +0.0170] | +0.6775 [-0.3147, +1.9622] | +0.0216 [-0.0865, +0.1401] |

## Historical daily information in tree models

| Comparison | Overall Delta MAE [95% CI] | Q90 Delta MAE | Ordinary Delta MAE | Recall change [95% CI], pp | False-Q90 change [95% CI], pp |
|---|---:|---:|---:|---:|---:|
| tree_history_vs_tree_current_constant_k5 | -0.0048 [-0.0119, +0.0018] | -0.0278 [-0.0629, +0.0083] | -0.0023 [-0.0084, +0.0036] | -0.2792 [-1.0320, +0.2925] | +0.0138 [-0.0951, +0.1187] |
| tree_history_vs_tree_current_gru_tuned_anchor_k0 | -0.0068 [-0.0197, +0.0054] | -0.0101 [-0.0750, +0.0578] | -0.0066 [-0.0186, +0.0060] | +0.6775 [-0.3818, +1.6803] | +0.0374 [-0.0530, +0.1377] |
| tree_history_vs_tree_current_gru_tuned_anchor_k5 | +0.0040 [-0.0121, +0.0212] | +0.0090 [-0.0317, +0.0537] | +0.0031 [-0.0142, +0.0226] | -0.8043 [-2.0263, +0.2001] | -0.0597 [-0.1807, +0.0368] |

## Integrated models versus the fixed ecological reference

| Comparison | Overall Delta MAE [95% CI] | Q90 Delta MAE | Ordinary Delta MAE | Recall change [95% CI], pp | False-Q90 change [95% CI], pp |
|---|---:|---:|---:|---:|---:|
| off_integrated_vs_prior_ecological_constant_k5 | -0.0187 [-0.0343, -0.0049] | -0.1121 [-0.1540, -0.0736] | -0.0072 [-0.0236, +0.0072] | +1.5633 [+0.4351, +2.9958] | +0.0710 [+0.0049, +0.1595] |
| current_only_integrated_vs_prior_ecological_constant_k5 | -0.0177 [-0.0334, -0.0041] | -0.1230 [-0.1700, -0.0810] | -0.0048 [-0.0210, +0.0094] | +1.7605 [+0.5293, +3.3584] | +0.0672 [+0.0018, +0.1571] |
| full_history_integrated_vs_prior_ecological_constant_k5 | -0.0201 [-0.0361, -0.0065] | -0.1232 [-0.1754, -0.0787] | -0.0077 [-0.0236, +0.0066] | +1.5881 [+0.3308, +3.1709] | +0.0718 [+0.0088, +0.1602] |
| off_integrated_vs_prior_ecological_gru_tuned_anchor_k0 | -0.0057 [-0.0616, +0.0470] | -0.2526 [-0.4278, -0.0899] | +0.0234 [-0.0332, +0.0800] | +3.9881 [+1.6681, +7.0746] | +0.1683 [-0.0175, +0.3818] |
| current_only_integrated_vs_prior_ecological_gru_tuned_anchor_k0 | -0.0014 [-0.0535, +0.0478] | -0.2745 [-0.4526, -0.1056] | +0.0307 [-0.0192, +0.0817] | +4.0056 [+1.8375, +6.8752] | +0.1617 [-0.0335, +0.3775] |
| full_history_integrated_vs_prior_ecological_gru_tuned_anchor_k0 | -0.0118 [-0.0616, +0.0346] | -0.2705 [-0.4482, -0.1006] | +0.0182 [-0.0296, +0.0665] | +3.9235 [+1.7780, +6.7268] | +0.1634 [-0.0107, +0.3803] |
| off_integrated_vs_prior_ecological_gru_tuned_anchor_k5 | -0.0149 [-0.0278, -0.0031] | -0.0938 [-0.1303, -0.0586] | -0.0056 [-0.0188, +0.0077] | +0.8717 [-0.1186, +2.1161] | +0.0523 [-0.0133, +0.1290] |
| current_only_integrated_vs_prior_ecological_gru_tuned_anchor_k5 | -0.0122 [-0.0252, -0.0002] | -0.0965 [-0.1385, -0.0610] | -0.0019 [-0.0151, +0.0111] | +0.8513 [-0.0909, +2.0062] | +0.0399 [-0.0211, +0.1104] |
| full_history_integrated_vs_prior_ecological_gru_tuned_anchor_k5 | -0.0123 [-0.0253, -0.0009] | -0.0858 [-0.1419, -0.0426] | -0.0033 [-0.0155, +0.0088] | +0.9324 [-0.0052, +2.1085] | +0.0516 [-0.0044, +0.1217] |

## Ecological integration versus direct residuals

| Comparison | Overall Delta MAE [95% CI] | Q90 Delta MAE | Ordinary Delta MAE | Recall change [95% CI], pp | False-Q90 change [95% CI], pp |
|---|---:|---:|---:|---:|---:|
| off_integrated_vs_direct_constant_k5 | -0.0170 [-0.0408, +0.0032] | -0.0346 [-0.1412, +0.0681] | -0.0155 [-0.0366, +0.0039] | -0.3734 [-0.9660, +0.0806] | -0.0321 [-0.1101, +0.0453] |
| current_only_integrated_vs_direct_constant_k5 | -0.0184 [-0.0411, +0.0005] | -0.0283 [-0.1337, +0.0746] | -0.0178 [-0.0376, -0.0001] | -0.2336 [-0.6668, +0.0936] | -0.0197 [-0.0982, +0.0654] |
| full_history_integrated_vs_direct_constant_k5 | -0.0194 [-0.0403, -0.0016] | -0.0331 [-0.1316, +0.0627] | -0.0187 [-0.0360, -0.0021] | -0.0835 [-0.7370, +0.4701] | -0.0216 [-0.0944, +0.0571] |
| off_integrated_vs_direct_gru_tuned_anchor_k0 | -0.0035 [-0.0098, +0.0028] | +0.0118 [-0.0030, +0.0274] | -0.0054 [-0.0123, +0.0015] | -0.2987 [-0.6057, -0.0817] | -0.0377 [-0.0728, -0.0136] |
| current_only_integrated_vs_direct_gru_tuned_anchor_k0 | -0.0026 [-0.0057, +0.0007] | +0.0061 [-0.0005, +0.0128] | -0.0036 [-0.0071, -0.0000] | -0.0796 [-0.1658, -0.0180] | -0.0151 [-0.0351, -0.0022] |
| full_history_integrated_vs_direct_gru_tuned_anchor_k0 | -0.0035 [-0.0121, +0.0061] | +0.0137 [-0.0119, +0.0402] | -0.0056 [-0.0150, +0.0047] | -0.1593 [-0.4894, +0.1271] | -0.0326 [-0.0779, +0.0000] |
| off_integrated_vs_direct_gru_tuned_anchor_k5 | -0.0244 [-0.0488, -0.0049] | -0.0553 [-0.1812, +0.0554] | -0.0219 [-0.0404, -0.0048] | -0.4113 [-1.1807, +0.3553] | -0.0158 [-0.0679, +0.0369] |
| current_only_integrated_vs_direct_gru_tuned_anchor_k5 | -0.0194 [-0.0367, -0.0046] | -0.0126 [-0.0990, +0.0709] | -0.0205 [-0.0353, -0.0067] | -0.3642 [-0.8111, -0.0161] | -0.0262 [-0.0893, +0.0388] |
| full_history_integrated_vs_direct_gru_tuned_anchor_k5 | -0.0188 [-0.0348, -0.0045] | -0.0137 [-0.0884, +0.0600] | -0.0199 [-0.0337, -0.0064] | +0.0418 [-0.4744, +0.5996] | -0.0376 [-0.0965, +0.0237] |

## Direct residuals versus context

| Comparison | Overall Delta MAE [95% CI] | Q90 Delta MAE | Ordinary Delta MAE | Recall change [95% CI], pp | False-Q90 change [95% CI], pp |
|---|---:|---:|---:|---:|---:|
| off_vs_context_constant_k5 | -0.0175 [-0.0546, +0.0211] | -0.0838 [-0.2200, +0.0341] | -0.0095 [-0.0480, +0.0312] | +1.0606 [-0.2015, +2.6309] | +0.0373 [-0.1002, +0.1709] |
| current_only_vs_context_constant_k5 | -0.0151 [-0.0510, +0.0217] | -0.1010 [-0.2378, +0.0147] | -0.0048 [-0.0414, +0.0336] | +1.1179 [-0.1119, +2.6502] | +0.0212 [-0.1237, +0.1579] |
| full_history_vs_context_constant_k5 | -0.0166 [-0.0519, +0.0194] | -0.0963 [-0.2325, +0.0187] | -0.0069 [-0.0421, +0.0300] | +0.7955 [-0.4238, +2.2692] | +0.0277 [-0.1056, +0.1675] |
| tree_current_vs_context_constant_k5 | -0.0154 [-0.0380, +0.0051] | -0.0256 [-0.0826, +0.0259] | -0.0141 [-0.0371, +0.0065] | -0.3570 [-1.2811, +0.4957] | -0.0492 [-0.1564, +0.0324] |
| tree_history_vs_context_constant_k5 | -0.0202 [-0.0413, -0.0010] | -0.0534 [-0.1042, -0.0039] | -0.0164 [-0.0386, +0.0037] | -0.6362 [-1.7652, +0.4486] | -0.0354 [-0.1578, +0.0791] |
| off_vs_context_gru_tuned_anchor_k0 | -0.0958 [-0.1705, -0.0281] | -0.3913 [-0.6280, -0.1546] | -0.0632 [-0.1413, +0.0094] | +5.4655 [+2.3881, +9.3711] | +0.2853 [+0.0822, +0.5405] |
| current_only_vs_context_gru_tuned_anchor_k0 | -0.0924 [-0.1598, -0.0306] | -0.4074 [-0.6429, -0.1712] | -0.0578 [-0.1248, +0.0053] | +5.2640 [+2.4287, +8.9652] | +0.2561 [+0.0537, +0.4970] |
| full_history_vs_context_gru_tuned_anchor_k0 | -0.1020 [-0.1726, -0.0379] | -0.4111 [-0.6651, -0.1613] | -0.0682 [-0.1389, -0.0038] | +5.2615 [+2.3660, +8.9872] | +0.2754 [+0.0914, +0.5159] |
| tree_current_vs_context_gru_tuned_anchor_k0 | -0.0471 [-0.0785, -0.0208] | -0.0233 [-0.0811, +0.0244] | -0.0493 [-0.0835, -0.0212] | +0.4521 [-0.3751, +1.4571] | +0.0074 [-0.0714, +0.0933] |
| tree_history_vs_context_gru_tuned_anchor_k0 | -0.0539 [-0.0892, -0.0245] | -0.0334 [-0.0955, +0.0378] | -0.0559 [-0.0952, -0.0236] | +1.1296 [+0.2528, +2.1362] | +0.0448 [-0.0213, +0.1343] |
| off_vs_context_gru_tuned_anchor_k5 | -0.0066 [-0.0465, +0.0349] | -0.0522 [-0.1961, +0.0813] | -0.0009 [-0.0417, +0.0430] | +0.5322 [-0.8474, +2.1034] | -0.0051 [-0.1588, +0.1173] |
| current_only_vs_context_gru_tuned_anchor_k5 | -0.0088 [-0.0452, +0.0278] | -0.0976 [-0.2246, +0.0122] | +0.0015 [-0.0357, +0.0415] | +0.4647 [-0.8940, +1.9640] | -0.0070 [-0.1612, +0.1226] |
| full_history_vs_context_gru_tuned_anchor_k5 | -0.0096 [-0.0455, +0.0264] | -0.0858 [-0.2162, +0.0280] | -0.0006 [-0.0367, +0.0380] | +0.1398 [-1.3038, +1.6566] | +0.0161 [-0.1234, +0.1377] |
| tree_current_vs_context_gru_tuned_anchor_k5 | -0.0228 [-0.0422, -0.0068] | -0.0646 [-0.1144, -0.0261] | -0.0175 [-0.0356, -0.0016] | +0.3084 [-0.3620, +1.2308] | +0.0166 [-0.0701, +0.0991] |
| tree_history_vs_context_gru_tuned_anchor_k5 | -0.0187 [-0.0403, +0.0007] | -0.0556 [-0.1077, -0.0037] | -0.0144 [-0.0366, +0.0061] | -0.4959 [-1.7347, +0.6675] | -0.0431 [-0.1593, +0.0541] |

## Source-validation choices

| Hydrology arm | Selected epoch range | Total epochs executed | Scale counts | Mean selected validation MAE |
|---|---:|---:|---|---:|
| off | 27–72 | 473 | 1: 9 | 1.804428 |
| current_only | 31–57 | 438 | 1: 9 | 1.801147 |
| full_history | 31–67 | 470 | 1: 9 | 1.798270 |

| Tree probe | Source-validation MAE | Feature count | Number of fits |
|---|---:|---:|---:|
| tree_current | 1.944702 | 147 | 9 |
| tree_history | 1.940824 | 147 | 9 |

Tree validation scores are descriptive and do not select probe parameters.

All three arms use the same source-training objective. Encoder/GRU parameter movements and support choices are
saved in the training and adapter CSVs.

| Integrated model | K | Selected gamma counts |
|---|---:|---|
| off_integrated_constant | 0 | 0: 7, 0.25: 2 |
| off_integrated_constant | 1 | 0: 7, 0.25: 2 |
| off_integrated_constant | 3 | 0: 2, 0.25: 4, 0.5: 3 |
| off_integrated_constant | 5 | 0: 2, 0.25: 1, 0.5: 6 |
| off_integrated_gru_tuned_anchor | 0 | 0: 7, 0.25: 2 |
| off_integrated_gru_tuned_anchor | 1 | 0: 7, 0.25: 2 |
| off_integrated_gru_tuned_anchor | 3 | 0: 1, 0.25: 5, 0.5: 3 |
| off_integrated_gru_tuned_anchor | 5 | 0: 1, 0.25: 4, 0.5: 4 |
| current_only_integrated_constant | 0 | 0: 8, 0.25: 1 |
| current_only_integrated_constant | 1 | 0: 8, 0.25: 1 |
| current_only_integrated_constant | 3 | 0: 2, 0.25: 4, 0.5: 3 |
| current_only_integrated_constant | 5 | 0: 2, 0.25: 1, 0.5: 6 |
| current_only_integrated_gru_tuned_anchor | 0 | 0: 8, 0.25: 1 |
| current_only_integrated_gru_tuned_anchor | 1 | 0: 8, 0.25: 1 |
| current_only_integrated_gru_tuned_anchor | 3 | 0.25: 6, 0.5: 3 |
| current_only_integrated_gru_tuned_anchor | 5 | 0: 2, 0.25: 2, 0.5: 5 |
| full_history_integrated_constant | 0 | 0: 6, 0.25: 3 |
| full_history_integrated_constant | 1 | 0: 6, 0.25: 3 |
| full_history_integrated_constant | 3 | 0: 1, 0.25: 5, 0.5: 3 |
| full_history_integrated_constant | 5 | 0: 1, 0.25: 4, 0.5: 4 |
| full_history_integrated_gru_tuned_anchor | 0 | 0: 6, 0.25: 3 |
| full_history_integrated_gru_tuned_anchor | 1 | 0: 6, 0.25: 3 |
| full_history_integrated_gru_tuned_anchor | 3 | 0.25: 6, 0.5: 3 |
| full_history_integrated_gru_tuned_anchor | 5 | 0: 1, 0.25: 4, 0.5: 4 |

Interpret direct feature contrasts before integrated ones: the latter also include changes
in source-validation-selected ecological mixing and positive-K support calibration.
A narrow or sign-changing contrast does not establish equivalence. Station gain/harm
concentration and partition/seed directions are retained alongside the overall means.
