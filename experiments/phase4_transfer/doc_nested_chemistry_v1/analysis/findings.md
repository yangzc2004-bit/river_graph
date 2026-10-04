# Nested chemical station calibration: development findings

Only source-validation predictions are analyzed. No outer target query labels or scores are read.
The backbone, chemistry decoder, legacy support calibration and ecological mixture stay fixed.
The added linear chemical2 increment uses a separately selected ridge and strength shared by K3/K5.

## Interpretation and weighting

Full-validation scores use the same observations that select the chemical increment and are tuning evidence.
Conditional station CV selects the increment on other validation stations. The parent has already used
validation for checkpoint/calibration selection, so this is not a fully OOF or independent confirmation
of the complete model. No model promotion or new station-transfer claim is made by this report.

Reported reconstruction MAE pools query cells within each fitted seed, averages seeds within partition,
then weights partitions equally. Adapter selection instead weights K3/K5 equally and stations equally;
both objectives are exposed explicitly. Seeds repeat ecological observations and do not add sample size.

## Main development result

At K3, the nested increment changes legacy MAE by -0.012344 mg/L (+0.635% relative gain), with gains in 2/3 partitions and 8/9 seed fits. Against the existing joint-coordinate procedure, its MAE difference is +0.016442 mg/L.
At K5, the nested increment changes legacy MAE by -0.023222 mg/L (+1.234% relative gain), with gains in 3/3 partitions and 9/9 seed fits. Against the existing joint-coordinate procedure, its MAE difference is +0.005726 mg/L.

These are descriptive conditional-CV results, without an independent-confirmation claim or
confidence-interval threshold. The existing joint procedure remains an explicit reference.

## Conditional station CV

| Model | K | MAE | Q90 MAE | Q90 bias | Station-equal MAE |
|---|---:|---:|---:|---:|---:|
| neural_chemistry_integrated_legacy | 0 | 2.162621 | 10.034560 | -8.791158 | 2.685950 |
| neural_chemistry_integrated_legacy | 1 | 2.103839 | 9.515359 | -8.396287 | 2.618451 |
| neural_chemistry_integrated_legacy | 3 | 1.943208 | 8.927340 | -7.324083 | 2.454998 |
| neural_chemistry_integrated_legacy | 5 | 1.881774 | 8.677965 | -6.515534 | 2.373826 |
| neural_chemistry_integrated_nested | 0 | 2.162621 | 10.034560 | -8.791158 | 2.685950 |
| neural_chemistry_integrated_nested | 1 | 2.103839 | 9.515359 | -8.396287 | 2.618451 |
| neural_chemistry_integrated_nested | 3 | 1.930864 | 8.819446 | -7.135578 | 2.440336 |
| neural_chemistry_integrated_nested | 5 | 1.858551 | 8.551952 | -6.426092 | 2.347015 |
| neural_chemistry_integrated_nested_masks | 0 | 2.162621 | 10.034560 | -8.791158 | 2.685950 |
| neural_chemistry_integrated_nested_masks | 1 | 2.103839 | 9.515359 | -8.396287 | 2.618451 |
| neural_chemistry_integrated_nested_masks | 3 | 1.943208 | 8.927340 | -7.324083 | 2.454998 |
| neural_chemistry_integrated_nested_masks | 5 | 1.882955 | 8.677965 | -6.515534 | 2.374491 |
| neural_chemistry_integrated_selected | 0 | 2.162621 | 10.034560 | -8.791158 | 2.685950 |
| neural_chemistry_integrated_selected | 1 | 2.103839 | 9.515359 | -8.396287 | 2.618451 |
| neural_chemistry_integrated_selected | 3 | 1.914422 | 8.724726 | -7.056081 | 2.420043 |
| neural_chemistry_integrated_selected | 5 | 1.852825 | 8.509790 | -6.406716 | 2.345746 |
| point_integrated_legacy | 0 | 2.234167 | 10.460878 | -9.453840 | 2.764741 |
| point_integrated_legacy | 1 | 2.189420 | 9.844267 | -8.860442 | 2.689993 |
| point_integrated_legacy | 3 | 1.963042 | 9.087967 | -7.534326 | 2.478317 |
| point_integrated_legacy | 5 | 1.906368 | 8.816334 | -6.668606 | 2.392618 |
| tree_chemistry_selected | 0 | 2.222895 | 10.336269 | -9.518612 | 2.751733 |
| tree_chemistry_selected | 1 | 2.189610 | 9.647900 | -8.832839 | 2.682843 |
| tree_chemistry_selected | 3 | 1.912583 | 8.670283 | -7.054565 | 2.420725 |
| tree_chemistry_selected | 5 | 1.854941 | 8.526233 | -6.564835 | 2.350150 |

| Nested comparison | K | ΔMAE | Relative gain | Improved partitions | Improved seed fits |
|---|---:|---:|---:|---:|---:|
| availability_only | 3 | -0.012344 | +0.635% | 2/3 | 8/9 |
| availability_only | 5 | -0.024404 | +1.296% | 3/3 | 9/9 |
| general | 3 | -0.032178 | +1.639% | 3/3 | 9/9 |
| general | 5 | -0.047817 | +2.508% | 3/3 | 9/9 |
| joint | 3 | +0.016442 | -0.859% | 0/3 | 2/9 |
| joint | 5 | +0.005726 | -0.309% | 0/3 | 4/9 |
| legacy | 3 | -0.012344 | +0.635% | 2/3 | 8/9 |
| legacy | 5 | -0.023222 | +1.234% | 3/3 | 9/9 |

Population: 8,641 unique validation station-months at 141 stations;
10,054 partition-cell occurrences. Negative ΔMAE and positive relative gain favor the nested adapter.

## Full-validation tuning

| Model | K | MAE | Q90 MAE | Q90 bias | Station-equal MAE |
|---|---:|---:|---:|---:|---:|
| neural_chemistry_integrated_legacy | 0 | 2.162621 | 10.034560 | -8.791158 | 2.685950 |
| neural_chemistry_integrated_legacy | 1 | 2.103839 | 9.515359 | -8.396287 | 2.618451 |
| neural_chemistry_integrated_legacy | 3 | 1.943208 | 8.927340 | -7.324083 | 2.454998 |
| neural_chemistry_integrated_legacy | 5 | 1.881774 | 8.677965 | -6.515534 | 2.373826 |
| neural_chemistry_integrated_nested | 0 | 2.162621 | 10.034560 | -8.791158 | 2.685950 |
| neural_chemistry_integrated_nested | 1 | 2.103839 | 9.515359 | -8.396287 | 2.618451 |
| neural_chemistry_integrated_nested | 3 | 1.919682 | 8.755261 | -6.989097 | 2.426392 |
| neural_chemistry_integrated_nested | 5 | 1.851727 | 8.499515 | -6.329310 | 2.339412 |
| neural_chemistry_integrated_nested_masks | 0 | 2.162621 | 10.034560 | -8.791158 | 2.685950 |
| neural_chemistry_integrated_nested_masks | 1 | 2.103839 | 9.515359 | -8.396287 | 2.618451 |
| neural_chemistry_integrated_nested_masks | 3 | 1.943166 | 8.927274 | -7.324016 | 2.454855 |
| neural_chemistry_integrated_nested_masks | 5 | 1.881831 | 8.677609 | -6.515178 | 2.373710 |
| neural_chemistry_integrated_selected | 0 | 2.162621 | 10.034560 | -8.791158 | 2.685950 |
| neural_chemistry_integrated_selected | 1 | 2.103839 | 9.515359 | -8.396287 | 2.618451 |
| neural_chemistry_integrated_selected | 3 | 1.914422 | 8.724726 | -7.056081 | 2.420043 |
| neural_chemistry_integrated_selected | 5 | 1.852825 | 8.509790 | -6.406716 | 2.345746 |
| point_integrated_legacy | 0 | 2.234167 | 10.460878 | -9.453840 | 2.764741 |
| point_integrated_legacy | 1 | 2.189420 | 9.844267 | -8.860442 | 2.689993 |
| point_integrated_legacy | 3 | 1.963042 | 9.087967 | -7.534326 | 2.478317 |
| point_integrated_legacy | 5 | 1.906368 | 8.816334 | -6.668606 | 2.392618 |
| tree_chemistry_selected | 0 | 2.222895 | 10.336269 | -9.518612 | 2.751733 |
| tree_chemistry_selected | 1 | 2.189610 | 9.647900 | -8.832839 | 2.682843 |
| tree_chemistry_selected | 3 | 1.912583 | 8.670283 | -7.054565 | 2.420725 |
| tree_chemistry_selected | 5 | 1.854941 | 8.526233 | -6.564835 | 2.350150 |

| Nested comparison | K | ΔMAE | Relative gain | Improved partitions | Improved seed fits |
|---|---:|---:|---:|---:|---:|
| availability_only | 3 | -0.023485 | +1.209% | 3/3 | 9/9 |
| availability_only | 5 | -0.030103 | +1.600% | 3/3 | 9/9 |
| general | 3 | -0.043361 | +2.209% | 3/3 | 9/9 |
| general | 5 | -0.054641 | +2.866% | 3/3 | 9/9 |
| joint | 3 | +0.005260 | -0.275% | 0/3 | 5/9 |
| joint | 5 | -0.001098 | +0.059% | 2/3 | 6/9 |
| legacy | 3 | -0.023527 | +1.211% | 3/3 | 9/9 |
| legacy | 5 | -0.030046 | +1.597% | 3/3 | 9/9 |

Population: 8,641 unique validation station-months at 141 stations;
10,054 partition-cell occurrences. Negative ΔMAE and positive relative gain favor the nested adapter.

## Source-selected strength

| Mode | Zero-strength packages | Nonzero-strength packages |
|---|---:|---:|
| chemistry | 0 | 9 |
| masks | 0 | 9 |

K0/K1 are exact legacy-parent controls. Inactive query chemistry, fewer than two active support rows
and zero chemical increments preserve the complete parent prediction. Availability-only increments
control the added coordinate information; the shared parent itself already uses measured chemistry.

`fold_choices.csv` retains station-held choices and their station-equal K3/K5 scores. Correction sizes,
Q90 bias and station gain/harm concentration remain visible even when overall gains are small.
Tail counts below20 are flagged; absent strata are not silently averaged over fewer partitions.

## Research use

These development results inform whether a separately regularized chemical increment merits a fixed
recipe replication under fresh ST357 station roles 342/343/344, with training seeds 42/43/44. That
replication must refit the complete source package: source/target roles change, so earlier forest, neural,
ecological and chemical-projection states cannot stand in for newly fitted models. The current 242–244
target results already motivated this design and remain development information for the next recipe.
No target-selected legacy/joint/nested splice is created. An external-basin claim requires separate data.
