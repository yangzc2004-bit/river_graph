# Source-trained readout for the fixed DOC support representation

The off daily-hydrology expert, native predictions and ecological memory remain fixed.
Four support representations are reported: constant station offset, legacy episodic basis,
refreshed hidden states with the fixed v4 readout, and those same hidden states with a
source-trained readout. Only source training fits the readout; source validation selects
its checkpoint and calibration coefficients. Target outcomes never choose a readout or route.

The learned-versus-fixed contrast asks whether training the projection improves adaptation
on the selected expert's unchanged hidden states. Learned-versus-legacy asks whether the
complete updated representation improves the existing station-support method. Direct
contrasts include alpha/ridge reselection; integrated contrasts also include positive-K
ecological gamma reselection. Native predictions and the K0 ecological mixture stay fixed.

K0 has no support update. K1 has no shape correction: one centered support point has rank
zero and the adapter explicitly enables the shape term only for K>1. These structural
negative controls are reported without interpreting exact zeros as low statistical power.
The loader verifies 198 copied-prediction or invariance checks.

All eight planned contrasts use 5,000 paired whole-station bootstrap draws. Repeated
station identities are jointly resampled across partitions; seed means average within
partitions, then partitions receive equal weight. Q90 is the source-training threshold
including ties. Pointwise intervals are reported on reused development partitions.

## Complete K curves

| Model | K0 MAE | K1 MAE | K3 MAE | K5 MAE | K5 RMSE | K5 R² |
|---|---:|---:|---:|---:|---:|---:|
| off_constant_direct | 1.806996 | 1.758826 | 1.641713 | 1.591869 | 3.611671 | 0.563919 |
| off_legacy_direct | 1.806996 | 1.758826 | 1.631317 | 1.589387 | 3.604470 | 0.565514 |
| off_refreshed_fixed_direct | 1.806996 | 1.758826 | 1.607645 | 1.590976 | 3.606256 | 0.565288 |
| off_refreshed_learned_direct | 1.806996 | 1.758826 | 1.613335 | 1.589982 | 3.606283 | 0.565281 |
| off_constant_integrated | 1.803538 | 1.755368 | 1.632934 | 1.574914 | 3.572022 | 0.571864 |
| off_legacy_integrated | 1.803538 | 1.755368 | 1.620090 | 1.564984 | 3.552260 | 0.575857 |
| off_refreshed_fixed_integrated | 1.803538 | 1.755368 | 1.599087 | 1.572464 | 3.568902 | 0.572656 |
| off_refreshed_learned_integrated | 1.803538 | 1.755368 | 1.604441 | 1.570974 | 3.568761 | 0.572693 |

## Direct readout contrasts

| Comparison | ΔMAE [95% CI] | ΔQ90 MAE [95% CI] | Δordinary MAE [95% CI] | Δrecall [95% CI], pp | Δfalse-Q90 [95% CI], pp |
|---|---:|---:|---:|---:|---:|
| off_refreshed_learned_vs_refreshed_fixed_direct_k3 | +0.005690 [-0.003904, +0.015861] | +0.006630 [-0.010677, +0.024377] | +0.005572 [-0.005214, +0.017485] | +0.051953 [-0.289106, +0.362499] | -0.007533 [-0.029538, +0.014661] |
| off_refreshed_learned_vs_refreshed_fixed_direct_k5 | -0.000994 [-0.003334, +0.001160] | -0.002556 [-0.009849, +0.006421] | -0.000794 [-0.003102, +0.001282] | +0.030593 [-0.092431, +0.173418] | -0.005022 [-0.013999, +0.000000] |
| off_refreshed_learned_vs_legacy_direct_k3 | -0.017982 [-0.033791, -0.001869] | +0.026077 [-0.019861, +0.069748] | -0.023217 [-0.041027, -0.006689] | -0.750810 [-1.918552, +0.148983] | -0.023285 [-0.109971, +0.044706] |
| off_refreshed_learned_vs_legacy_direct_k5 | +0.000596 [-0.004223, +0.006357] | +0.019134 [-0.015366, +0.050697] | -0.001583 [-0.005707, +0.002457] | -0.087909 [-0.595584, +0.464406] | +0.030105 [-0.011820, +0.080616] |

## Integrated readout contrasts

| Comparison | ΔMAE [95% CI] | ΔQ90 MAE [95% CI] | Δordinary MAE [95% CI] | Δrecall [95% CI], pp | Δfalse-Q90 [95% CI], pp |
|---|---:|---:|---:|---:|---:|
| off_refreshed_learned_vs_refreshed_fixed_integrated_k3 | +0.005354 [-0.004141, +0.015427] | +0.007066 [-0.009428, +0.024183] | +0.005140 [-0.005410, +0.016743] | -0.074309 [-0.357008, +0.152959] | -0.004328 [-0.025853, +0.016800] |
| off_refreshed_learned_vs_refreshed_fixed_integrated_k5 | -0.001490 [-0.003791, +0.000481] | -0.002893 [-0.009916, +0.006044] | -0.001311 [-0.003534, +0.000551] | -0.045165 [-0.186654, +0.099858] | -0.008227 [-0.027292, +0.005559] |
| off_refreshed_learned_vs_legacy_integrated_k3 | -0.015649 [-0.031047, +0.000022] | +0.029432 [-0.017346, +0.073998] | -0.021038 [-0.038154, -0.005146] | -0.506046 [-1.488240, +0.330107] | -0.005164 [-0.051564, +0.035353] |
| off_refreshed_learned_vs_legacy_integrated_k5 | +0.005990 [-0.001234, +0.014262] | +0.047388 [-0.002223, +0.094747] | +0.001410 [-0.004130, +0.007182] | -0.029144 [-0.400968, +0.360277] | -0.000142 [-0.039945, +0.036978] |

## Tail and ordinary errors, with detection tradeoffs

| Model | K | Q90 MAE | Q90 bias | Ordinary MAE | Ordinary bias | Recall | Precision | False-Q90 rate |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| off_constant_direct | 3 | 6.871709 | -5.304180 | 1.038219 | +0.162074 | 62.409% | 78.634% | 1.963% |
| off_constant_direct | 5 | 6.659850 | -5.189831 | 1.008181 | +0.163216 | 65.301% | 77.775% | 2.186% |
| off_legacy_direct | 3 | 6.851466 | -5.287768 | 1.028975 | +0.159399 | 62.588% | 78.673% | 1.965% |
| off_legacy_direct | 5 | 6.642277 | -5.175765 | 1.007548 | +0.166518 | 65.545% | 77.741% | 2.203% |
| off_refreshed_fixed_direct | 3 | 6.870913 | -5.324191 | 1.000186 | +0.158514 | 61.785% | 78.626% | 1.949% |
| off_refreshed_fixed_direct | 5 | 6.663967 | -5.195390 | 1.006759 | +0.170366 | 65.427% | 77.470% | 2.239% |
| off_refreshed_learned_direct | 3 | 6.877543 | -5.329717 | 1.005758 | +0.159171 | 61.837% | 78.697% | 1.941% |
| off_refreshed_learned_direct | 5 | 6.661411 | -5.193859 | 1.005965 | +0.167686 | 65.457% | 77.517% | 2.233% |
| off_constant_integrated | 3 | 6.915094 | -5.440819 | 1.023848 | +0.170009 | 61.811% | 79.362% | 1.866% |
| off_constant_integrated | 5 | 6.625268 | -5.040134 | 0.992639 | +0.176215 | 64.928% | 78.023% | 2.154% |
| off_legacy_integrated | 3 | 6.900415 | -5.446349 | 1.011142 | +0.163021 | 61.775% | 79.305% | 1.872% |
| off_legacy_integrated | 5 | 6.586994 | -4.965678 | 0.985640 | +0.171136 | 65.134% | 77.807% | 2.188% |
| off_refreshed_fixed_integrated | 3 | 6.922781 | -5.462978 | 0.984964 | +0.155453 | 61.343% | 79.222% | 1.871% |
| off_refreshed_fixed_integrated | 5 | 6.637274 | -5.046084 | 0.988360 | +0.173087 | 65.150% | 77.788% | 2.196% |
| off_refreshed_learned_integrated | 3 | 6.929847 | -5.469464 | 0.990104 | +0.157294 | 61.269% | 79.236% | 1.866% |
| off_refreshed_learned_integrated | 5 | 6.634382 | -5.045860 | 0.987050 | +0.171046 | 65.105% | 77.837% | 2.187% |

## Source-validation readout checkpoint selection

The saved checkpoint choices below describe source-validation optimization, not held-out
target performance. Epoch zero is retained as a fixed-readout fallback.
The 128-parameter projection is trained with equal-station native MAE averaged over
K={3,5} and ridge={1,10}, at alpha=1; checkpoint validation pools fixed query cells
over the same four conditions. Final calibration then uses the unchanged alpha/ridge grid.
The source baseline combines forest OOF predictions with a source-trained neural correction;
it is not a fully OOF neural expert. Source episode loss is a supervised training diagnostic.

| Partition | Seed | Selected epoch | Epochs run | Initial validation loss | Selected validation loss |
|---|---:|---:|---:|---:|---:|
| 142 | 42 | 4 | 9 | 1.911799 | 1.910431 |
| 142 | 43 | 3 | 8 | 1.914988 | 1.914214 |
| 142 | 44 | 0 | 5 | 1.932961 | 1.932961 |
| 143 | 42 | 5 | 10 | 1.454603 | 1.448217 |
| 143 | 43 | 0 | 5 | 1.433732 | 1.433732 |
| 143 | 44 | 0 | 5 | 1.447521 | 1.447521 |
| 144 | 42 | 1 | 6 | 1.830422 | 1.830408 |
| 144 | 43 | 5 | 10 | 1.840996 | 1.834801 |
| 144 | 44 | 0 | 5 | 1.817865 | 1.817865 |

Selected support alpha/ridge values, all direct candidate scores, ecological gamma
choices and all gamma scores are preserved in the CSVs. Full K curves, partition/seed
directions, transformed errors and station gain/harm concentration are also retained.
No target-based K switch or model selection is produced by this analyzer.
