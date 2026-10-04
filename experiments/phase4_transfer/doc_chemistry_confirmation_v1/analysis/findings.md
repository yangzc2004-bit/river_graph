# Fresh station assignments: chemical-state DOC reconstruction

Nine complete production packages refit the accepted DOC recipe on station-partition seeds242–244
and training seeds42–44. These are fresh role assignments on the same ST357 Mississippi cohort,
not an independent external basin or a new set of field measurements. Historical fitted predictors
are not reused. All twenty fixed curves and all K values are retained; target results select none.

The model combines local/environmental prediction, a temporal residual expert, ecological transfer
and a nonlinear current-month chemistry decoder. The retained spatial self path has no river messages.
Source forests are station-blocked OOF; the complete source neural pipeline is not OOF.
Checkpoint, mixing and support-coordinate choices use source-validation data. Station support is
retrospective and can postdate a query. Known pH/conductance at a DOC-held station is a different
information setting from a completely unmonitored station.

Metrics average three training seeds within each partition, then weight the three partitions equally.
R² and RMSE are averaged run metrics, not metrics computed after pooling all predictions.
The 30 fixed contrasts use 5,000 paired whole-station bootstrap draws.
A global station's sampled multiplicity is shared across partitions. Repeated seeds and overlapping
role assignments do not count as additional ecological observations. Intervals are descriptive
paired intervals without multiplicity adjustment; a crossing-zero interval does not establish equivalence.

## Whole K curves

| Model | K0 MAE | K1 MAE | K3 MAE | K5 MAE | K5 RMSE | K5 R² | K5 log MAE |
|---|---:|---:|---:|---:|---:|---:|---:|
| neural_chemistry_legacy | 1.867004 | 1.810422 | 1.721673 | 1.652849 | 5.960916 | 0.398625 | 0.226193 |
| neural_chemistry_masks_aug | 1.867004 | 1.810422 | 1.716440 | 1.654237 | 5.961218 | 0.398525 | 0.226230 |
| neural_chemistry_chemistry_aug | 1.867004 | 1.810422 | 1.710550 | 1.651855 | 5.943450 | 0.402761 | 0.225224 |
| neural_chemistry_selected | 1.867004 | 1.810422 | 1.710550 | 1.651635 | 5.943470 | 0.402757 | 0.225238 |
| neural_chemistry_integrated_legacy | 1.868348 | 1.810420 | 1.713435 | 1.648792 | 5.965760 | 0.397422 | 0.223912 |
| neural_chemistry_integrated_masks_aug | 1.868348 | 1.810420 | 1.710047 | 1.652247 | 5.967555 | 0.397036 | 0.224277 |
| neural_chemistry_integrated_chemistry_aug | 1.868348 | 1.810420 | 1.701410 | 1.670409 | 5.996957 | 0.388163 | 0.225549 |
| neural_chemistry_integrated_selected | 1.868348 | 1.810420 | 1.701410 | 1.670120 | 5.996872 | 0.388177 | 0.225550 |
| tree_chemistry_legacy | 1.904915 | 1.880329 | 1.715720 | 1.656840 | 5.953162 | 0.399733 | 0.224528 |
| tree_chemistry_masks_aug | 1.904915 | 1.880329 | 1.715013 | 1.662044 | 5.957325 | 0.398733 | 0.224978 |
| tree_chemistry_chemistry_aug | 1.904915 | 1.880329 | 1.712390 | 1.659622 | 5.943518 | 0.401608 | 0.224631 |
| tree_chemistry_selected | 1.904915 | 1.880329 | 1.712390 | 1.658794 | 5.942972 | 0.401696 | 0.224578 |
| point_integrated_legacy | 1.913371 | 1.867621 | 1.730884 | 1.666368 | 5.988791 | 0.391998 | 0.225997 |
| context_legacy | 1.994307 | 1.963719 | 1.764613 | 1.702101 | 5.983649 | 0.391576 | 0.231962 |
| point_legacy | 1.916161 | 1.869710 | 1.728709 | 1.667795 | 5.988094 | 0.392230 | 0.226666 |
| tree_prior_legacy | 1.947658 | 1.923068 | 1.745816 | 1.681565 | 5.992674 | 0.390786 | 0.228993 |
| neural_no_aux_legacy | 1.883168 | 1.819151 | 1.730321 | 1.663650 | 5.977760 | 0.394729 | 0.226959 |
| neural_no_aux_integrated_legacy | 1.880371 | 1.819150 | 1.719455 | 1.658512 | 5.978116 | 0.394561 | 0.224681 |
| neural_masks_legacy | 1.887837 | 1.820413 | 1.729437 | 1.661653 | 5.977594 | 0.394769 | 0.226938 |
| neural_masks_integrated_legacy | 1.884254 | 1.820582 | 1.718977 | 1.658345 | 5.978788 | 0.394442 | 0.224727 |

## Fixed paired comparisons

Negative error deltas favor the candidate. Positive bias means overprediction.

| Contrast | ΔMAE [95% CI] | Relative gain % [95% CI] | ΔQ90 MAE [95% CI] | Δordinary MAE [95% CI] | Better partitions / packages |
|---|---:|---:|---:|---:|---:|
| accepted_vs_point_integrated_legacy_k0 | -0.045023 [-0.089347, -0.001410] | +2.353 [+0.077, +4.575] | -0.179378 [-0.265537, -0.098747] | -0.027554 [-0.075250, +0.017826] | 3/3; 8/9 |
| accepted_vs_tree_chemistry_selected_k0 | -0.036567 [-0.097960, +0.025072] | +1.920 [-1.354, +5.075] | -0.178774 [-0.288462, -0.072935] | -0.020990 [-0.088997, +0.045565] | 2/3; 6/9 |
| accepted_vs_point_integrated_legacy_k3 | -0.029474 [-0.047099, -0.012511] | +1.703 [+0.741, +2.655] | -0.058031 [-0.122464, +0.009721] | -0.025390 [-0.041735, -0.009737] | 3/3; 9/9 |
| accepted_vs_tree_chemistry_selected_k3 | -0.010980 [-0.034666, +0.013037] | +0.641 [-0.764, +2.000] | -0.019987 [-0.085049, +0.048456] | -0.010406 [-0.036757, +0.015609] | 2/3; 6/9 |
| accepted_vs_point_integrated_legacy_k5 | +0.003751 [-0.018596, +0.028861] | -0.225 [-1.719, +1.090] | -0.018971 [-0.117142, +0.084724] | +0.004378 [-0.015663, +0.028825] | 1/3; 5/9 |
| accepted_vs_tree_chemistry_selected_k5 | +0.011325 [-0.013397, +0.036962] | -0.683 [-2.198, +0.772] | +0.010737 [-0.067935, +0.087103] | +0.011097 [-0.015696, +0.040844] | 2/3; 6/9 |
| neural_chemistry_selected_vs_legacy_k3 | -0.011123 [-0.025367, +0.002374] | +0.646 [-0.138, +1.443] | -0.013728 [-0.096052, +0.063571] | -0.009453 [-0.019101, -0.000873] | 3/3; 7/9 |
| neural_chemistry_chemistry_aug_vs_masks_aug_k3 | -0.005890 [-0.017012, +0.005689] | +0.343 [-0.339, +0.963] | +0.012150 [-0.055046, +0.083685] | -0.006492 [-0.014482, +0.001353] | 2/3; 7/9 |
| neural_chemistry_selected_vs_legacy_k5 | -0.001213 [-0.011349, +0.009535] | +0.073 [-0.572, +0.673] | -0.044138 [-0.104460, +0.010472] | +0.003952 [-0.005239, +0.013987] | 2/3; 5/9 |
| neural_chemistry_chemistry_aug_vs_masks_aug_k5 | -0.002381 [-0.011669, +0.007627] | +0.144 [-0.459, +0.688] | -0.014756 [-0.079439, +0.041604] | +0.000232 [-0.007860, +0.008907] | 2/3; 6/9 |
| neural_chemistry_integrated_selected_vs_legacy_k3 | -0.012025 [-0.026487, +0.002235] | +0.702 [-0.134, +1.518] | -0.013380 [-0.087125, +0.059336] | -0.010753 [-0.022256, +0.001146] | 3/3; 8/9 |
| neural_chemistry_integrated_chemistry_aug_vs_masks_aug_k3 | -0.008637 [-0.022073, +0.005208] | +0.505 [-0.305, +1.245] | +0.005486 [-0.060592, +0.072890] | -0.008972 [-0.020291, +0.002823] | 3/3; 7/9 |
| neural_chemistry_integrated_selected_vs_legacy_k5 | +0.021328 [+0.001566, +0.044964] | -1.294 [-2.636, -0.094] | +0.020733 [-0.057400, +0.103025] | +0.019805 [+0.001396, +0.044360] | 0/3; 3/9 |
| neural_chemistry_integrated_chemistry_aug_vs_masks_aug_k5 | +0.018161 [-0.001398, +0.041985] | -1.099 [-2.460, +0.085] | +0.041472 [-0.042418, +0.128109] | +0.014676 [-0.003163, +0.039219] | 1/3; 6/9 |
| tree_chemistry_selected_vs_legacy_k3 | -0.003330 [-0.014975, +0.009149] | +0.194 [-0.536, +0.832] | +0.022961 [-0.042130, +0.093487] | -0.005843 [-0.015768, +0.003765] | 3/3; 8/9 |
| tree_chemistry_chemistry_aug_vs_masks_aug_k3 | -0.002624 [-0.014218, +0.009357] | +0.153 [-0.559, +0.786] | +0.024131 [-0.040513, +0.094692] | -0.005092 [-0.014767, +0.004144] | 2/3; 8/9 |
| tree_chemistry_selected_vs_legacy_k5 | +0.001955 [-0.009698, +0.014189] | -0.118 [-0.867, +0.561] | -0.004961 [-0.060922, +0.057165] | +0.001421 [-0.008162, +0.011892] | 1/3; 2/9 |
| tree_chemistry_chemistry_aug_vs_masks_aug_k5 | -0.002422 [-0.012791, +0.008205] | +0.146 [-0.502, +0.735] | +0.013238 [-0.046511, +0.075044] | -0.004540 [-0.012359, +0.003488] | 2/3; 6/9 |
| chemistry_vs_no_aux_k0 | -0.016164 [-0.044283, +0.010079] | +0.858 [-0.558, +2.271] | -0.041763 [-0.110209, +0.026530] | -0.012213 [-0.042015, +0.016262] | 3/3; 7/9 |
| chemistry_vs_no_aux_k3 | -0.008648 [-0.022778, +0.006169] | +0.500 [-0.361, +1.315] | -0.016451 [-0.076479, +0.073015] | -0.007461 [-0.020172, +0.004988] | 3/3; 6/9 |
| chemistry_vs_no_aux_k5 | -0.010802 [-0.021972, +0.000303] | +0.649 [-0.019, +1.309] | -0.025695 [-0.082572, +0.038092] | -0.008882 [-0.018696, +0.001267] | 3/3; 8/9 |
| chemistry_vs_masks_k0 | -0.020833 [-0.049625, +0.005810] | +1.104 [-0.308, +2.541] | -0.054188 [-0.114608, +0.005660] | -0.016199 [-0.047312, +0.011748] | 3/3; 8/9 |
| chemistry_vs_masks_k3 | -0.007765 [-0.020660, +0.005048] | +0.449 [-0.294, +1.184] | -0.031645 [-0.085412, +0.044167] | -0.004909 [-0.016023, +0.005763] | 3/3; 5/9 |
| chemistry_vs_masks_k5 | -0.008805 [-0.017303, -0.000797] | +0.530 [+0.051, +1.050] | -0.041512 [-0.081286, +0.003244] | -0.004678 [-0.012161, +0.002713] | 3/3; 8/9 |
| chemistry_vs_no_aux_integrated_k0 | -0.012023 [-0.041183, +0.016053] | +0.639 [-0.857, +2.120] | -0.061033 [-0.121581, -0.002863] | -0.005416 [-0.038392, +0.024609] | 3/3; 7/9 |
| chemistry_vs_no_aux_integrated_k3 | -0.006020 [-0.017857, +0.006332] | +0.350 [-0.377, +1.026] | -0.027542 [-0.065718, +0.028457] | -0.003965 [-0.015092, +0.007214] | 3/3; 7/9 |
| chemistry_vs_no_aux_integrated_k5 | -0.009720 [-0.018971, -0.000471] | +0.586 [+0.029, +1.099] | -0.027201 [-0.070778, +0.025102] | -0.008240 [-0.016039, -0.000601] | 3/3; 9/9 |
| chemistry_vs_masks_integrated_k0 | -0.015906 [-0.047350, +0.013826] | +0.844 [-0.762, +2.419] | -0.085210 [-0.141731, -0.031240] | -0.007077 [-0.041847, +0.024446] | 3/3; 8/9 |
| chemistry_vs_masks_integrated_k3 | -0.005542 [-0.016952, +0.005759] | +0.322 [-0.352, +0.971] | -0.033011 [-0.068366, +0.015582] | -0.002722 [-0.013568, +0.007668] | 2/3; 7/9 |
| chemistry_vs_masks_integrated_k5 | -0.009554 [-0.018561, -0.000977] | +0.576 [+0.062, +1.079] | -0.031333 [-0.073907, +0.018833] | -0.007468 [-0.014744, -0.000412] | 3/3; 9/9 |

## Tail and ordinary DOC

Q90 thresholds come from source-training DOC within each partition, including threshold ties.
Counts below20 are marked unstable. Bias is prediction minus observation; negative tail bias
indicates underprediction. See classification tables for recall, precision and false-high rates.

| Model | K | Q90 MAE | Q90 bias | Ordinary MAE | Ordinary bias |
|---|---:|---:|---:|---:|---:|
| point_integrated_legacy | 0 | 7.996624 | -6.849474 | 1.277837 | +0.225305 |
| point_integrated_legacy | 3 | 7.804913 | -6.376003 | 1.104417 | +0.172516 |
| point_integrated_legacy | 5 | 7.620865 | -6.271609 | 1.052432 | +0.154482 |
| neural_chemistry_integrated_legacy | 0 | 7.817246 | -6.447541 | 1.250283 | +0.218403 |
| neural_chemistry_integrated_legacy | 3 | 7.760261 | -6.339546 | 1.089780 | +0.147078 |
| neural_chemistry_integrated_legacy | 5 | 7.581160 | -6.287518 | 1.037005 | +0.137326 |
| neural_chemistry_integrated_selected | 0 | 7.817246 | -6.447541 | 1.250283 | +0.218403 |
| neural_chemistry_integrated_selected | 3 | 7.746882 | -6.362704 | 1.079028 | +0.133746 |
| neural_chemistry_integrated_selected | 5 | 7.601893 | -6.228826 | 1.056810 | +0.141787 |
| tree_chemistry_selected | 0 | 7.996020 | -7.029048 | 1.271273 | +0.382675 |
| tree_chemistry_selected | 3 | 7.766868 | -6.384690 | 1.089433 | +0.206836 |
| tree_chemistry_selected | 5 | 7.591156 | -6.313947 | 1.045713 | +0.201532 |

## Chemistry availability

The availability appendix reports both/one/neither auxiliary measurement, including empty strata.
Observed-DOC test availability and genuinely missing-DOC grid availability are shown separately.
This distinguishes measured chemistry as an added input from availability-only information.

| Cohort | Availability | Cells | Fraction |
|---|---|---:|---:|
| all_station_months | both | 53815 | 23.049% |
| all_station_months | ph_only | 815 | 0.349% |
| all_station_months | ec_only | 9194 | 3.938% |
| all_station_months | neither | 169654 | 72.664% |
| doc_observed | both | 21902 | 97.036% |
| doc_observed | ph_only | 168 | 0.744% |
| doc_observed | ec_only | 131 | 0.580% |
| doc_observed | neither | 370 | 1.639% |
| doc_genuinely_missing | both | 31913 | 15.131% |
| doc_genuinely_missing | ph_only | 647 | 0.307% |
| doc_genuinely_missing | ec_only | 9063 | 4.297% |
| doc_genuinely_missing | neither | 169284 | 80.265% |

## Source-validation selections

Representation choice is part of the method, not a target-selected splice across K values.
Raw legacy, availability-coordinate and chemistry-coordinate curves stay in every table.

| Pipeline | K | Legacy / masks / chemistry choices | Mean selected active-validation MAE |
|---|---:|---:|---:|
| neural_chemistry | 0 | 9/0/0 | 2.168746 |
| neural_chemistry | 1 | 9/0/0 | 2.107126 |
| neural_chemistry | 3 | 0/0/9 | 1.922788 |
| neural_chemistry | 5 | 1/0/8 | 1.862168 |
| neural_chemistry_integrated | 0 | 9/0/0 | 2.167059 |
| neural_chemistry_integrated | 1 | 9/0/0 | 2.107126 |
| neural_chemistry_integrated | 3 | 0/0/9 | 1.914776 |
| neural_chemistry_integrated | 5 | 1/0/8 | 1.852320 |
| tree_chemistry | 0 | 9/0/0 | 2.230191 |
| tree_chemistry | 1 | 9/0/0 | 2.196815 |
| tree_chemistry | 3 | 0/0/9 | 1.913626 |
| tree_chemistry | 5 | 1/0/8 | 1.855782 |

Station responses, positive-gain concentration, all seed/partition directions and the full
native/log metrics are saved as adjacent CSV files. No automatic model-promotion rule is applied.
