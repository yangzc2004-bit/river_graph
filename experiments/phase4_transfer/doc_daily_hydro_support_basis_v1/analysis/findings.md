# Aligning few-shot support representations with selected DOC experts

This experiment keeps the off, current-only and full-history neural experts, their native
base predictions, the context forest and ecological memory fixed. No new neural or forest
fit is performed. It compares three support representations for each direct and integrated expert:

- constant: the existing station-offset calibration diagnostic;
- legacy: the previously used v4 two-dimensional GRU support basis;
- refreshed: the selected current expert's hidden states projected through the unchanged
  v4 two-dimensional readout, with the same 32-calendar-anchor centering and station scalar RMS normalization.

The refreshed basis has no newly fitted PCA or readout. Support/mixing parameters are selected
on the same source-validation episodes and grids. The direct contrasts measure representation
alignment together with alpha/ridge reselection; integrated contrasts additionally include
ecological gamma reselection. They are interpreted in that order. The full-timeline anchor
normalization is label-free but retrospective, as in the legacy protocol.

All 18 models and K={0,1,3,5} are retained. The three support representations have exact
K0 invariance. Constant and legacy predictions reproduce their corresponding parent-memory
products exactly (216 copied/control model-run checks). There is no target-based
choice of expert, basis, K, route or calibration parameter.

Intervals use 5,000 paired whole-station bootstrap draws. Repeated station identities
are resampled jointly across partitions. Cell-pooled seed scores average within each
partition, then partitions receive equal weight. Q90 uses source-training thresholds with ties
included. Recall and false-Q90 rate are reported together; precision is descriptive.

These are previously examined development partitions. The eighteen fixed basis contrasts
are reported together; their pointwise intervals do not constitute an independent external confirmation.

## Complete K curves

| Model | K0 MAE | K1 MAE | K3 MAE | K5 MAE | K5 RMSE | K5 R² |
|---|---:|---:|---:|---:|---:|---:|
| off_constant | 1.806996 | 1.758826 | 1.641713 | 1.591869 | 3.611671 | 0.563919 |
| off_legacy | 1.806996 | 1.758826 | 1.631317 | 1.589387 | 3.604470 | 0.565514 |
| off_refreshed | 1.806996 | 1.758826 | 1.607645 | 1.590976 | 3.606256 | 0.565288 |
| current_only_constant | 1.810409 | 1.762757 | 1.658361 | 1.594289 | 3.609479 | 0.564477 |
| current_only_legacy | 1.810409 | 1.762757 | 1.634744 | 1.587180 | 3.585251 | 0.569494 |
| current_only_refreshed | 1.810409 | 1.762757 | 1.611741 | 1.590835 | 3.603520 | 0.565992 |
| full_history_constant | 1.800854 | 1.756293 | 1.639374 | 1.592814 | 3.606771 | 0.565212 |
| full_history_legacy | 1.800854 | 1.756293 | 1.631764 | 1.586407 | 3.583787 | 0.569975 |
| full_history_refreshed | 1.800854 | 1.756293 | 1.619470 | 1.593891 | 3.606962 | 0.565149 |
| off_integrated_constant | 1.803538 | 1.755368 | 1.632934 | 1.574914 | 3.572022 | 0.571864 |
| off_integrated_legacy | 1.803538 | 1.755368 | 1.620090 | 1.564984 | 3.552260 | 0.575857 |
| off_integrated_refreshed | 1.803538 | 1.755368 | 1.599087 | 1.572464 | 3.568902 | 0.572656 |
| current_only_integrated_constant | 1.807859 | 1.760206 | 1.636268 | 1.575899 | 3.569804 | 0.572402 |
| current_only_integrated_legacy | 1.807859 | 1.760206 | 1.622109 | 1.567740 | 3.555403 | 0.575439 |
| current_only_integrated_refreshed | 1.807859 | 1.760206 | 1.599909 | 1.570676 | 3.565024 | 0.573629 |
| full_history_integrated_constant | 1.797393 | 1.752831 | 1.632506 | 1.573453 | 3.565038 | 0.573423 |
| full_history_integrated_legacy | 1.797393 | 1.752831 | 1.620559 | 1.567575 | 3.557570 | 0.575091 |
| full_history_integrated_refreshed | 1.797393 | 1.752831 | 1.610500 | 1.573159 | 3.564928 | 0.573430 |

## Direct support-basis contrasts

| Comparison | ΔMAE [95% CI] | ΔQ90 MAE [95% CI] | Δordinary MAE [95% CI] | Δrecall [95% CI], pp | Δfalse-Q90 [95% CI], pp |
|---|---:|---:|---:|---:|---:|
| off_refreshed_vs_legacy_k1 | +0.000000 [+0.000000, +0.000000] | +0.000000 [+0.000000, +0.000000] | +0.000000 [+0.000000, +0.000000] | +0.000000 [+0.000000, +0.000000] | +0.000000 [+0.000000, +0.000000] |
| off_refreshed_vs_legacy_k3 | -0.023671 [-0.047630, -0.001123] | +0.019447 [-0.037787, +0.073159] | -0.028789 [-0.056459, -0.004002] | -0.802763 [-2.226174, +0.340247] | -0.015752 [-0.115026, +0.065776] |
| off_refreshed_vs_legacy_k5 | +0.001590 [-0.003770, +0.008108] | +0.021691 [-0.017139, +0.056841] | -0.000789 [-0.004995, +0.003701] | -0.118501 [-0.721454, +0.500113] | +0.035127 [-0.007536, +0.089871] |
| current_only_refreshed_vs_legacy_k1 | +0.000000 [+0.000000, +0.000000] | +0.000000 [+0.000000, +0.000000] | +0.000000 [+0.000000, +0.000000] | +0.000000 [+0.000000, +0.000000] | +0.000000 [+0.000000, +0.000000] |
| current_only_refreshed_vs_legacy_k3 | -0.023003 [-0.046626, -0.000345] | +0.023910 [-0.035416, +0.079573] | -0.028599 [-0.055705, -0.004272] | -0.926605 [-2.272132, +0.170140] | -0.005306 [-0.103188, +0.078554] |
| current_only_refreshed_vs_legacy_k5 | +0.003656 [-0.008506, +0.016397] | +0.049674 [-0.007148, +0.103972] | -0.001318 [-0.014013, +0.009625] | -0.036431 [-0.752392, +0.579471] | +0.062318 [-0.003545, +0.155028] |
| full_history_refreshed_vs_legacy_k1 | +0.000000 [+0.000000, +0.000000] | +0.000000 [+0.000000, +0.000000] | +0.000000 [+0.000000, +0.000000] | +0.000000 [+0.000000, +0.000000] | +0.000000 [+0.000000, +0.000000] |
| full_history_refreshed_vs_legacy_k3 | -0.012294 [-0.032318, +0.008281] | +0.034602 [-0.023247, +0.091529] | -0.018005 [-0.039050, +0.001936] | -0.457487 [-1.312434, +0.219441] | +0.022000 [-0.054240, +0.105933] |
| full_history_refreshed_vs_legacy_k5 | +0.007484 [-0.000735, +0.016935] | +0.050762 [+0.004498, +0.098961] | +0.002687 [-0.004644, +0.010611] | -0.234106 [-0.933154, +0.363002] | -0.020521 [-0.074408, +0.020394] |

## Integrated support-basis contrasts

| Comparison | ΔMAE [95% CI] | ΔQ90 MAE [95% CI] | Δordinary MAE [95% CI] | Δrecall [95% CI], pp | Δfalse-Q90 [95% CI], pp |
|---|---:|---:|---:|---:|---:|
| off_integrated_refreshed_vs_legacy_k1 | +0.000000 [+0.000000, +0.000000] | +0.000000 [+0.000000, +0.000000] | +0.000000 [+0.000000, +0.000000] | +0.000000 [+0.000000, +0.000000] | +0.000000 [+0.000000, +0.000000] |
| off_integrated_refreshed_vs_legacy_k3 | -0.021003 [-0.044403, +0.000769] | +0.022366 [-0.035710, +0.076070] | -0.026178 [-0.053517, -0.002702] | -0.431737 [-1.486304, +0.528243] | -0.000836 [-0.063762, +0.055003] |
| off_integrated_refreshed_vs_legacy_k5 | +0.007480 [-0.000043, +0.016361] | +0.050280 [-0.002918, +0.101326] | +0.002720 [-0.002544, +0.008834] | +0.016020 [-0.397920, +0.415883] | +0.008085 [-0.030320, +0.045857] |
| current_only_integrated_refreshed_vs_legacy_k1 | +0.000000 [+0.000000, +0.000000] | +0.000000 [+0.000000, +0.000000] | +0.000000 [+0.000000, +0.000000] | +0.000000 [+0.000000, +0.000000] | +0.000000 [+0.000000, +0.000000] |
| current_only_integrated_refreshed_vs_legacy_k3 | -0.022199 [-0.047859, +0.000980] | +0.027624 [-0.031420, +0.081908] | -0.028183 [-0.057787, -0.003323] | -0.267097 [-1.298639, +0.680295] | +0.011984 [-0.050158, +0.072953] |
| current_only_integrated_refreshed_vs_legacy_k5 | +0.002937 [-0.007917, +0.013788] | +0.037858 [-0.010870, +0.081961] | -0.001188 [-0.012584, +0.008248] | -0.066548 [-0.640749, +0.476082] | +0.028161 [-0.012623, +0.073237] |
| full_history_integrated_refreshed_vs_legacy_k1 | +0.000000 [+0.000000, +0.000000] | +0.000000 [+0.000000, +0.000000] | +0.000000 [+0.000000, +0.000000] | +0.000000 [+0.000000, +0.000000] | +0.000000 [+0.000000, +0.000000] |
| full_history_integrated_refreshed_vs_legacy_k3 | -0.010059 [-0.029165, +0.009297] | +0.032035 [-0.030958, +0.090472] | -0.015159 [-0.035710, +0.003197] | -0.008259 [-0.633543, +0.632394] | +0.026735 [-0.042485, +0.100153] |
| full_history_integrated_refreshed_vs_legacy_k5 | +0.005584 [-0.000375, +0.012723] | +0.031837 [-0.001432, +0.064323] | +0.002300 [-0.002978, +0.008135] | -0.562887 [-1.206284, +0.015924] | -0.005451 [-0.041720, +0.025267] |

## Tail error and classification at K5

| Model | Q90 MAE | Q90 bias | Ordinary MAE | Ordinary bias | Recall | Precision | False-Q90 rate |
|---|---:|---:|---:|---:|---:|---:|---:|
| off_constant | 6.659850 | -5.189831 | 1.008181 | +0.163216 | 65.301% | 77.775% | 2.186% |
| off_legacy | 6.642277 | -5.175765 | 1.007548 | +0.166518 | 65.545% | 77.741% | 2.203% |
| off_refreshed | 6.663967 | -5.195390 | 1.006759 | +0.170366 | 65.427% | 77.470% | 2.239% |
| current_only_constant | 6.642632 | -5.170667 | 1.012841 | +0.171224 | 65.359% | 77.939% | 2.170% |
| current_only_legacy | 6.596805 | -5.062814 | 1.009882 | +0.177557 | 65.478% | 77.743% | 2.201% |
| current_only_refreshed | 6.646478 | -5.177305 | 1.008564 | +0.171747 | 65.441% | 77.300% | 2.264% |
| full_history_constant | 6.647304 | -5.158449 | 1.010828 | +0.167587 | 65.036% | 77.814% | 2.176% |
| full_history_legacy | 6.608673 | -5.051019 | 1.007873 | +0.173658 | 65.153% | 77.465% | 2.225% |
| full_history_refreshed | 6.659435 | -5.166476 | 1.010560 | +0.171887 | 64.919% | 77.610% | 2.204% |
| off_integrated_constant | 6.625268 | -5.040134 | 0.992639 | +0.176215 | 64.928% | 78.023% | 2.154% |
| off_integrated_legacy | 6.586994 | -4.965678 | 0.985640 | +0.171136 | 65.134% | 77.807% | 2.188% |
| off_integrated_refreshed | 6.637274 | -5.046084 | 0.988360 | +0.173087 | 65.150% | 77.788% | 2.196% |
| current_only_integrated_constant | 6.614317 | -5.021112 | 0.995055 | +0.179762 | 65.125% | 78.112% | 2.150% |
| current_only_integrated_legacy | 6.584248 | -4.965898 | 0.989373 | +0.177573 | 65.114% | 77.915% | 2.175% |
| current_only_integrated_refreshed | 6.622106 | -5.023395 | 0.988185 | +0.173984 | 65.047% | 77.716% | 2.203% |
| full_history_integrated_constant | 6.614192 | -4.988958 | 0.992174 | +0.174815 | 64.953% | 77.990% | 2.154% |
| full_history_integrated_legacy | 6.594934 | -4.965475 | 0.987956 | +0.174788 | 65.195% | 77.826% | 2.187% |
| full_history_integrated_refreshed | 6.626771 | -4.982619 | 0.990256 | +0.172287 | 64.632% | 77.719% | 2.181% |

## Source-validation calibration choices

The exact per-run alpha/ridge/gamma choices and every gamma candidate score are preserved
in the CSVs. Prediction changes at positive K can include parameter reselection as well as
the changed basis. K0 equality verifies that this experiment does not change native reconstruction.

| Model | K | Alpha counts | Ridge counts |
|---|---:|---|---|
| off_constant | 0 | 0.0: 9 | infinity: 9 |
| off_constant | 1 | 0.0: 3, 0.25: 6 | infinity: 9 |
| off_constant | 3 | 0.25: 3, 0.5: 3, 0.75: 3 | infinity: 9 |
| off_constant | 5 | 0.5: 6, 0.75: 1, 1.0: 2 | infinity: 9 |
| off_legacy | 0 | 0.0: 9 | infinity: 9 |
| off_legacy | 1 | 0.0: 3, 0.25: 6 | infinity: 9 |
| off_legacy | 3 | 0.25: 2, 0.5: 4, 0.75: 3 | 10.0: 4, 1.0: 4, infinity: 1 |
| off_legacy | 5 | 0.5: 6, 1.0: 3 | 1.0: 7, 10.0: 2 |
| off_refreshed | 0 | 0.0: 9 | infinity: 9 |
| off_refreshed | 1 | 0.0: 3, 0.25: 6 | infinity: 9 |
| off_refreshed | 3 | 0.5: 6, 0.75: 3 | 1.0: 5, 10.0: 4 |
| off_refreshed | 5 | 0.5: 6, 1.0: 3 | 1.0: 5, 10.0: 3, infinity: 1 |
| current_only_constant | 0 | 0.0: 9 | infinity: 9 |
| current_only_constant | 1 | 0.0: 3, 0.25: 6 | infinity: 9 |
| current_only_constant | 3 | 0.25: 4, 0.5: 2, 0.75: 3 | infinity: 9 |
| current_only_constant | 5 | 0.5: 6, 0.75: 1, 1.0: 2 | infinity: 9 |
| current_only_legacy | 0 | 0.0: 9 | infinity: 9 |
| current_only_legacy | 1 | 0.0: 3, 0.25: 6 | infinity: 8, 10.0: 1 |
| current_only_legacy | 3 | 0.25: 2, 0.5: 4, 0.75: 3 | 10.0: 4, 1.0: 4, infinity: 1 |
| current_only_legacy | 5 | 0.5: 5, 0.75: 1, 1.0: 3 | 1.0: 7, 10.0: 2 |
| current_only_refreshed | 0 | 0.0: 9 | infinity: 9 |
| current_only_refreshed | 1 | 0.0: 3, 0.25: 6 | infinity: 8, 10.0: 1 |
| current_only_refreshed | 3 | 0.5: 6, 0.75: 3 | 1.0: 5, 10.0: 4 |
| current_only_refreshed | 5 | 0.5: 5, 0.75: 1, 1.0: 3 | 1.0: 6, 10.0: 2, infinity: 1 |
| full_history_constant | 0 | 0.0: 9 | infinity: 9 |
| full_history_constant | 1 | 0.0: 3, 0.25: 6 | infinity: 9 |
| full_history_constant | 3 | 0.25: 3, 0.5: 4, 0.75: 2 | infinity: 9 |
| full_history_constant | 5 | 0.5: 6, 0.75: 1, 1.0: 2 | infinity: 9 |
| full_history_legacy | 0 | 0.0: 9 | infinity: 9 |
| full_history_legacy | 1 | 0.0: 3, 0.25: 6 | infinity: 8, 10.0: 1 |
| full_history_legacy | 3 | 0.25: 2, 0.5: 4, 0.75: 3 | 10.0: 5, 1.0: 4 |
| full_history_legacy | 5 | 0.5: 5, 0.75: 1, 1.0: 3 | 1.0: 7, 10.0: 2 |
| full_history_refreshed | 0 | 0.0: 9 | infinity: 9 |
| full_history_refreshed | 1 | 0.0: 3, 0.25: 6 | infinity: 8, 10.0: 1 |
| full_history_refreshed | 3 | 0.25: 1, 0.5: 5, 0.75: 3 | 1.0: 6, 10.0: 2, infinity: 1 |
| full_history_refreshed | 5 | 0.5: 6, 1.0: 3 | 1.0: 5, infinity: 2, 10.0: 2 |

| Integrated model | K | Gamma counts |
|---|---:|---|
| off_integrated_constant | 0 | 0.0: 7, 0.25: 2 |
| off_integrated_constant | 1 | 0.0: 7, 0.25: 2 |
| off_integrated_constant | 3 | 0.0: 2, 0.25: 4, 0.5: 3 |
| off_integrated_constant | 5 | 0.0: 2, 0.25: 1, 0.5: 6 |
| off_integrated_legacy | 0 | 0.0: 7, 0.25: 2 |
| off_integrated_legacy | 1 | 0.0: 7, 0.25: 2 |
| off_integrated_legacy | 3 | 0.0: 1, 0.25: 5, 0.5: 3 |
| off_integrated_legacy | 5 | 0.0: 1, 0.25: 4, 0.5: 4 |
| off_integrated_refreshed | 0 | 0.0: 7, 0.25: 2 |
| off_integrated_refreshed | 1 | 0.0: 7, 0.25: 2 |
| off_integrated_refreshed | 3 | 0.0: 1, 0.25: 5, 0.5: 3 |
| off_integrated_refreshed | 5 | 0.0: 1, 0.25: 2, 0.5: 6 |
| current_only_integrated_constant | 0 | 0.0: 8, 0.25: 1 |
| current_only_integrated_constant | 1 | 0.0: 8, 0.25: 1 |
| current_only_integrated_constant | 3 | 0.0: 2, 0.25: 4, 0.5: 3 |
| current_only_integrated_constant | 5 | 0.0: 2, 0.25: 1, 0.5: 6 |
| current_only_integrated_legacy | 0 | 0.0: 8, 0.25: 1 |
| current_only_integrated_legacy | 1 | 0.0: 8, 0.25: 1 |
| current_only_integrated_legacy | 3 | 0.25: 6, 0.5: 3 |
| current_only_integrated_legacy | 5 | 0.0: 2, 0.25: 2, 0.5: 5 |
| current_only_integrated_refreshed | 0 | 0.0: 8, 0.25: 1 |
| current_only_integrated_refreshed | 1 | 0.0: 8, 0.25: 1 |
| current_only_integrated_refreshed | 3 | 0.25: 6, 0.5: 3 |
| current_only_integrated_refreshed | 5 | 0.0: 1, 0.25: 2, 0.5: 6 |
| full_history_integrated_constant | 0 | 0.0: 6, 0.25: 3 |
| full_history_integrated_constant | 1 | 0.0: 6, 0.25: 3 |
| full_history_integrated_constant | 3 | 0.0: 1, 0.25: 5, 0.5: 3 |
| full_history_integrated_constant | 5 | 0.0: 1, 0.25: 4, 0.5: 4 |
| full_history_integrated_legacy | 0 | 0.0: 6, 0.25: 3 |
| full_history_integrated_legacy | 1 | 0.0: 6, 0.25: 3 |
| full_history_integrated_legacy | 3 | 0.25: 6, 0.5: 3 |
| full_history_integrated_legacy | 5 | 0.0: 1, 0.25: 4, 0.5: 4 |
| full_history_integrated_refreshed | 0 | 0.0: 6, 0.25: 3 |
| full_history_integrated_refreshed | 1 | 0.0: 6, 0.25: 3 |
| full_history_integrated_refreshed | 3 | 0.0: 1, 0.25: 5, 0.5: 3 |
| full_history_integrated_refreshed | 5 | 0.0: 1, 0.25: 4, 0.5: 4 |

Partition/seed directions and station gain/harm concentration are included in the companion tables.
A small or sign-changing estimate is not an equivalence test. No new ranking or calibration rule is fitted by this analysis.
