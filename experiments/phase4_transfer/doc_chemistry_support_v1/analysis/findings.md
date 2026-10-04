# Chemical-state coordinates for station support adaptation

The selected nonlinear chemistry expert and chemistry-tree native predictions are frozen.
Three calibration representations are compared: the legacy two-dimensional GRU basis,
that basis plus two source-fitted mask-state principal components, and that basis plus two
source-fitted measured-chemistry-state components. A fourth curve chooses the representation
using source-validation final-pipeline MAE at each K, with deterministic legacy/masks/chemistry ties.
No neural network or forest is retrained; no target score chooses a representation.

K0 is required to preserve the old native predictions. One-support centered shape correction
has rank zero; K1 invariance is a useful algebraic check, not a low-power significance result.
Augmented coordinates use a fixed source-global center and row-local chemical measurements.
Inactive chemistry retains the previous final prediction. The PCA fit uses only active months
at source stations, without DOC response values or target-fitted normalization.

All thirteen curves are reported; the twenty-two fixed K3/K5 contrasts use 5,000 paired whole-station
bootstrap draws, joint station multiplicities across overlapping partitions, seed means within
partition and equal partition weights. These are reused development station partitions.
Repeated training seeds do not add ecological samples. Q90 includes threshold ties.

The loader verifies 306 copied-reference or K-invariance comparisons.

## Complete K curves

| Model | K0 MAE | K1 MAE | K3 MAE | K5 MAE | K5 RMSE | K5 R² |
|---|---:|---:|---:|---:|---:|---:|
| neural_chemistry_legacy | 1.773213 | 1.741419 | 1.621943 | 1.579294 | 3.563277 | 0.574750 |
| neural_chemistry_masks_aug | 1.773213 | 1.741419 | 1.619626 | 1.580424 | 3.563159 | 0.574745 |
| neural_chemistry_chemistry_aug | 1.773213 | 1.741419 | 1.604624 | 1.566684 | 3.513138 | 0.585222 |
| neural_chemistry_selected | 1.773213 | 1.741419 | 1.603418 | 1.569843 | 3.519886 | 0.583524 |
| neural_chemistry_integrated_legacy | 1.764947 | 1.738600 | 1.614240 | 1.568820 | 3.562414 | 0.574431 |
| neural_chemistry_integrated_masks_aug | 1.764947 | 1.738600 | 1.613222 | 1.569553 | 3.562005 | 0.574475 |
| neural_chemistry_integrated_chemistry_aug | 1.764947 | 1.738600 | 1.595816 | 1.556278 | 3.512037 | 0.584878 |
| neural_chemistry_integrated_selected | 1.764947 | 1.738600 | 1.595851 | 1.556939 | 3.517433 | 0.583510 |
| tree_chemistry_legacy | 1.802261 | 1.768522 | 1.565090 | 1.537905 | 3.518279 | 0.584432 |
| tree_chemistry_masks_aug | 1.802261 | 1.768522 | 1.565169 | 1.539911 | 3.518534 | 0.584347 |
| tree_chemistry_chemistry_aug | 1.802261 | 1.768522 | 1.560229 | 1.535715 | 3.498198 | 0.588710 |
| tree_chemistry_selected | 1.802261 | 1.768522 | 1.560343 | 1.535534 | 3.500971 | 0.588013 |
| point_integrated_legacy | 1.803538 | 1.755368 | 1.620090 | 1.564984 | 3.552260 | 0.575857 |

## All twenty-two fixed contrasts

Negative error deltas favor the candidate; classification deltas are percentage points.

| Contrast | ΔMAE [95% CI] | ΔQ90 MAE [95% CI] | Δordinary MAE [95% CI] | Δrecall [95% CI], pp | Δfalse-Q90 [95% CI], pp |
|---|---:|---:|---:|---:|---:|
| neural_chemistry_chemistry_aug_vs_legacy_k3 | -0.017319 [-0.029367, -0.006163] | -0.019823 [-0.053554, +0.015810] | -0.017392 [-0.030607, -0.005677] | -0.022334 [-0.438307, +0.391128] | -0.019631 [-0.077785, +0.024062] |
| neural_chemistry_chemistry_aug_vs_masks_aug_k3 | -0.015001 [-0.026953, -0.003758] | -0.010333 [-0.044401, +0.030956] | -0.015949 [-0.029206, -0.003721] | -0.053899 [-0.524768, +0.379566] | -0.019355 [-0.075856, +0.021015] |
| neural_chemistry_selected_vs_legacy_k3 | -0.018525 [-0.029415, -0.008439] | -0.033739 [-0.060476, -0.008317] | -0.016986 [-0.029314, -0.006314] | +0.154434 [-0.121698, +0.483484] | +0.009212 [-0.020293, +0.040845] |
| neural_chemistry_chemistry_aug_vs_legacy_k5 | -0.012610 [-0.027640, +0.001004] | -0.075315 [-0.159385, +0.005122] | -0.006392 [-0.017643, +0.007023] | -0.232613 [-0.772969, +0.338260] | +0.053118 [-0.018899, +0.156990] |
| neural_chemistry_chemistry_aug_vs_masks_aug_k5 | -0.013740 [-0.028601, -0.000359] | -0.077530 [-0.163292, +0.002738] | -0.007345 [-0.018327, +0.005651] | -0.181135 [-0.720544, +0.376082] | +0.045585 [-0.025215, +0.148200] |
| neural_chemistry_selected_vs_legacy_k5 | -0.009452 [-0.023899, +0.002963] | -0.070077 [-0.150291, +0.004010] | -0.003496 [-0.013645, +0.008314] | -0.151515 [-0.528266, +0.345064] | +0.049913 [-0.002928, +0.139196] |
| neural_chemistry_integrated_chemistry_aug_vs_legacy_k3 | -0.018423 [-0.029765, -0.008136] | -0.035347 [-0.068051, -0.002455] | -0.016689 [-0.028913, -0.005704] | -0.002421 [-0.445788, +0.360271] | +0.023281 [-0.006441, +0.057622] |
| neural_chemistry_integrated_chemistry_aug_vs_masks_aug_k3 | -0.017406 [-0.028567, -0.007286] | -0.031470 [-0.062285, +0.000640] | -0.016004 [-0.028277, -0.005070] | -0.002421 [-0.445788, +0.360271] | +0.023281 [-0.006441, +0.057622] |
| neural_chemistry_integrated_selected_vs_legacy_k3 | -0.018389 [-0.029182, -0.008617] | -0.036652 [-0.065764, -0.008071] | -0.016485 [-0.028377, -0.006114] | +0.048084 [-0.206802, +0.328640] | +0.016871 [-0.003890, +0.042850] |
| neural_chemistry_integrated_chemistry_aug_vs_legacy_k5 | -0.012542 [-0.029491, +0.001753] | -0.073742 [-0.177652, +0.029448] | -0.006431 [-0.017607, +0.006214] | -0.308370 [-0.824929, +0.173562] | +0.008089 [-0.048444, +0.070417] |
| neural_chemistry_integrated_chemistry_aug_vs_masks_aug_k5 | -0.013275 [-0.030088, +0.000469] | -0.075260 [-0.181183, +0.026669] | -0.007007 [-0.017093, +0.004898] | -0.256892 [-0.786901, +0.215647] | -0.002511 [-0.057256, +0.055035] |
| neural_chemistry_integrated_selected_vs_legacy_k5 | -0.011881 [-0.028914, +0.002077] | -0.068798 [-0.172721, +0.031617] | -0.006313 [-0.017319, +0.005773] | -0.232613 [-0.743503, +0.241701] | +0.008089 [-0.042071, +0.065222] |
| tree_chemistry_chemistry_aug_vs_legacy_k3 | -0.004861 [-0.013775, +0.002826] | -0.046445 [-0.093679, +0.003102] | -0.000674 [-0.007077, +0.005476] | +0.180660 [-0.077453, +0.473025] | -0.006689 [-0.029777, +0.015931] |
| tree_chemistry_chemistry_aug_vs_masks_aug_k3 | -0.004940 [-0.013759, +0.002806] | -0.045876 [-0.093084, +0.003397] | -0.000853 [-0.007137, +0.005223] | +0.180660 [-0.077453, +0.473025] | -0.006689 [-0.029777, +0.015931] |
| tree_chemistry_selected_vs_legacy_k3 | -0.004747 [-0.012791, +0.001445] | -0.044017 [-0.087180, -0.001333] | -0.000853 [-0.005606, +0.004103] | +0.205912 [-0.036430, +0.494101] | -0.012405 [-0.038232, +0.004426] |
| tree_chemistry_chemistry_aug_vs_legacy_k5 | -0.002190 [-0.012041, +0.007428] | -0.030220 [-0.090467, +0.042652] | +0.000571 [-0.006652, +0.008996] | -0.699309 [-1.444747, -0.158353] | -0.038048 [-0.088760, -0.002611] |
| tree_chemistry_chemistry_aug_vs_masks_aug_k5 | -0.004196 [-0.013635, +0.004912] | -0.033274 [-0.093193, +0.035100] | -0.001275 [-0.007795, +0.006598] | -0.639572 [-1.384026, -0.106340] | -0.028847 [-0.078352, +0.008900] |
| tree_chemistry_selected_vs_legacy_k5 | -0.002371 [-0.011774, +0.006778] | -0.029695 [-0.087778, +0.041104] | +0.000300 [-0.006008, +0.007833] | -0.446784 [-1.054630, +0.005746] | -0.022024 [-0.068339, +0.008233] |
| selected_neural_integrated_vs_point_integrated_legacy_k3 | -0.024239 [-0.044780, -0.007424] | -0.107144 [-0.200464, -0.018690] | -0.015302 [-0.033658, +0.000366] | +0.502652 [-0.051181, +1.142225] | -0.005708 [-0.069736, +0.048913] |
| selected_neural_integrated_vs_tree_chemistry_selected_k3 | +0.035508 [+0.007203, +0.067219] | +0.096695 [-0.097504, +0.291195] | +0.029545 [+0.004098, +0.054971] | +1.245180 [-0.058178, +2.841948] | +0.093282 [-0.026498, +0.249090] |
| selected_neural_integrated_vs_point_integrated_legacy_k5 | -0.008045 [-0.023271, +0.003925] | -0.068611 [-0.154643, +0.026831] | -0.001594 [-0.011321, +0.007654] | -0.139839 [-0.695201, +0.371949] | -0.000809 [-0.067227, +0.071547] |
| selected_neural_integrated_vs_tree_chemistry_selected_k5 | +0.021405 [-0.004193, +0.049284] | +0.038186 [-0.041334, +0.119920] | +0.018941 [-0.009794, +0.050321] | +0.316154 [-0.796615, +1.685378] | -0.003209 [-0.146835, +0.112264] |

## High and ordinary DOC at primary K values

| Model | K | Q90 MAE | Q90 bias | Ordinary MAE | Ordinary bias | Recall | Precision | False-Q90 rate |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| neural_chemistry_legacy | 3 | 6.736355 | -5.146483 | 1.031525 | +0.159080 | 63.175% | 79.035% | 1.951% |
| neural_chemistry_legacy | 5 | 6.560347 | -5.081866 | 1.005234 | +0.157009 | 65.493% | 77.968% | 2.184% |
| neural_chemistry_masks_aug | 3 | 6.726865 | -5.137506 | 1.030082 | +0.165595 | 63.206% | 79.030% | 1.951% |
| neural_chemistry_masks_aug | 5 | 6.562562 | -5.077872 | 1.006187 | +0.160581 | 65.441% | 77.898% | 2.192% |
| neural_chemistry_chemistry_aug | 3 | 6.716532 | -5.150993 | 1.014133 | +0.153911 | 63.152% | 79.174% | 1.932% |
| neural_chemistry_chemistry_aug | 5 | 6.485032 | -4.864068 | 0.998842 | +0.170068 | 65.260% | 77.403% | 2.237% |
| neural_chemistry_selected | 3 | 6.702616 | -5.120075 | 1.014540 | +0.153718 | 63.329% | 78.991% | 1.960% |
| neural_chemistry_selected | 5 | 6.490270 | -4.867981 | 1.001737 | +0.168229 | 65.341% | 77.444% | 2.234% |
| neural_chemistry_integrated_legacy | 3 | 6.829922 | -5.369385 | 1.012325 | +0.162441 | 62.230% | 79.671% | 1.849% |
| neural_chemistry_integrated_legacy | 5 | 6.587181 | -5.099975 | 0.990359 | +0.166579 | 65.227% | 77.923% | 2.179% |
| neural_chemistry_integrated_masks_aug | 3 | 6.826045 | -5.365508 | 1.011639 | +0.166240 | 62.230% | 79.671% | 1.849% |
| neural_chemistry_integrated_masks_aug | 5 | 6.588699 | -5.092625 | 0.990936 | +0.169731 | 65.175% | 77.822% | 2.189% |
| neural_chemistry_integrated_chemistry_aug | 3 | 6.794575 | -5.352742 | 0.995636 | +0.152226 | 62.227% | 79.471% | 1.872% |
| neural_chemistry_integrated_chemistry_aug | 5 | 6.513439 | -4.906394 | 0.983929 | +0.169808 | 64.918% | 77.772% | 2.187% |
| neural_chemistry_integrated_selected | 3 | 6.793270 | -5.350625 | 0.995840 | +0.153507 | 62.278% | 79.534% | 1.866% |
| neural_chemistry_integrated_selected | 5 | 6.518383 | -4.904412 | 0.984046 | +0.171739 | 64.994% | 77.791% | 2.187% |
| tree_chemistry_legacy | 3 | 6.740592 | -5.454982 | 0.967148 | +0.234328 | 60.827% | 79.902% | 1.785% |
| tree_chemistry_legacy | 5 | 6.509893 | -5.120486 | 0.964805 | +0.231099 | 65.125% | 77.682% | 2.212% |
| tree_chemistry_masks_aug | 3 | 6.740022 | -5.454412 | 0.967327 | +0.236561 | 60.827% | 79.902% | 1.785% |
| tree_chemistry_masks_aug | 5 | 6.512947 | -5.117214 | 0.966651 | +0.232280 | 65.065% | 77.761% | 2.203% |
| tree_chemistry_chemistry_aug | 3 | 6.694147 | -5.406000 | 0.966474 | +0.227527 | 61.007% | 80.033% | 1.778% |
| tree_chemistry_chemistry_aug | 5 | 6.479673 | -5.068970 | 0.965376 | +0.223983 | 64.425% | 77.847% | 2.174% |
| tree_chemistry_selected | 3 | 6.696575 | -5.398501 | 0.966295 | +0.230228 | 61.033% | 80.086% | 1.773% |
| tree_chemistry_selected | 5 | 6.480197 | -5.065946 | 0.965105 | +0.227316 | 64.678% | 77.802% | 2.190% |
| point_integrated_legacy | 3 | 6.900415 | -5.446349 | 1.011142 | +0.163021 | 61.775% | 79.305% | 1.872% |
| point_integrated_legacy | 5 | 6.586994 | -4.965678 | 0.985640 | +0.171136 | 65.134% | 77.807% | 2.188% |

## Availability and interpretation

Observed-DOC queries and the genuinely missing-DOC grid have different chemistry availability.
The new coordinates cannot imply validated improvement where neither auxiliary indicator is observed.

| Full-grid cohort | Auxiliary availability | Cells | Denominator | Fraction |
|---|---|---:|---:|---:|
| all_station_months | both | 53815 | 233478 | 23.049% |
| all_station_months | ph_only | 815 | 233478 | 0.349% |
| all_station_months | ec_only | 9194 | 233478 | 3.938% |
| all_station_months | neither | 169654 | 233478 | 72.664% |
| doc_observed | both | 21902 | 22571 | 97.036% |
| doc_observed | ph_only | 168 | 22571 | 0.744% |
| doc_observed | ec_only | 131 | 22571 | 0.580% |
| doc_observed | neither | 370 | 22571 | 1.639% |
| doc_genuinely_missing | both | 31913 | 210907 | 15.131% |
| doc_genuinely_missing | ph_only | 647 | 210907 | 0.307% |
| doc_genuinely_missing | ec_only | 9063 | 210907 | 4.297% |
| doc_genuinely_missing | neither | 169284 | 210907 | 80.265% |

| Evaluation availability | Unique query cells | Unique stations | Split-cell occurrences |
|---|---:|---:|---:|
| both | 10216 | 170 | 12271 |
| ph_only | 55 | 7 | 55 |
| ec_only | 84 | 22 | 182 |
| neither | 165 | 9 | 357 |

## Source-validation representation selection

Representation selection is part of the evaluated method. Every raw-basis result is still shown,
so an apparent selected-curve gain cannot hide an unfavorable chemistry-coordinate result.
The full candidate source-validation scores and chosen representations are saved separately.
No target-driven K splice is added. Intervals crossing zero do not establish equivalence.

| Pipeline | K | Legacy / masks / chemistry selected | Mean selected validation MAE |
|---|---:|---:|---:|
| neural_chemistry | 0 | 9/0/0 | 1.769764 |
| neural_chemistry | 1 | 9/0/0 | 1.745523 |
| neural_chemistry | 3 | 1/1/7 | 1.658708 |
| neural_chemistry | 5 | 1/2/6 | 1.603627 |
| neural_chemistry_integrated | 0 | 9/0/0 | 1.769265 |
| neural_chemistry_integrated | 1 | 9/0/0 | 1.745263 |
| neural_chemistry_integrated | 3 | 0/1/8 | 1.651356 |
| neural_chemistry_integrated | 5 | 2/1/6 | 1.591066 |
| tree_chemistry | 0 | 9/0/0 | 1.910344 |
| tree_chemistry | 1 | 9/0/0 | 1.857964 |
| tree_chemistry | 3 | 3/0/6 | 1.673123 |
| tree_chemistry | 5 | 3/0/6 | 1.603398 |
