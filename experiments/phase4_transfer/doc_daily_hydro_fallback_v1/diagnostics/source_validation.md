# Preserve monthly predictions when daily hydrology is absent

The fixed router uses the existing daily expert whenever at least one numeric daily descriptor is valid, and retains the monthly expert otherwise. No neural model was retrained. This diagnostic reads only fixed source-validation queries and their saved calibration choices; validation was already used for fitting those choices.

## Native base and ecological integration

| Stage | Daily information | Monthly MAE | Daily MAE | Routed MAE | Routed−daily MAE | Monthly bias | Daily bias | Routed bias |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| direct | All queries | 1.870033 | 1.804428 | 1.803458 | -0.000970 | -0.561785 | -0.644631 | -0.638799 |
| direct | No valid numeric descriptor | 1.365484 | 1.373486 | 1.365484 | -0.008002 | -0.497625 | -0.550335 | -0.497625 |
| direct | At least one valid descriptor | 2.012129 | 1.931638 | 1.931638 | +0.000000 | -0.591843 | -0.679372 | -0.679372 |
| integrated | All queries | 1.866130 | 1.804067 | 1.803249 | -0.000818 | -0.611256 | -0.660622 | -0.654476 |
| integrated | No valid numeric descriptor | 1.365848 | 1.374141 | 1.366586 | -0.007555 | -0.557096 | -0.567566 | -0.513936 |
| integrated | At least one valid descriptor | 2.006874 | 1.930753 | 1.930753 | +0.000000 | -0.638697 | -0.694723 | -0.694723 |

MAE and signed bias are in native DOC units (mg/L); positive bias is overprediction. Seeds are averaged within each partition, then partitions receive equal weight. Group summaries have their own denominators and are not additive.

## Where routing applies

| Partition | Fixed queries | No valid numeric descriptor | At least one valid descriptor | Routed−daily direct MAE | Routed−daily integrated MAE |
| --- | ---: | ---: | ---: | ---: | ---: |
| 142 | 2013 | 259 | 1754 | -0.006829 | -0.006829 |
| 143 | 2798 | 410 | 2388 | +0.004518 | +0.004518 |
| 144 | 3236 | 1103 | 2133 | -0.000599 | -0.000143 |

On all nine validation products, routed native predictions are **bitwise equal to monthly predictions where all validity flags are zero**, and to daily predictions where any flag is one. The only partial-descriptor validation case is one station-month in partition 143; it takes the daily route. Adequately observed zero discharge retains valid flags and also takes the daily route.

This exact equality applies before ecological mixing and K-shot adaptation. Those downstream operations select their own coefficients and can change predictions in either route; the final adapted model has no exact-monthly preservation guarantee.

## Selected support adaptation

| Model | K=0 validation MAE | K=1 | K=3 | K=5 |
| --- | ---: | ---: | ---: | ---: |
| monthly_gru_tuned_anchor | 1.870033 | 1.822957 | 1.679693 | 1.620223 |
| daily_gru_tuned_anchor | 1.804428 | 1.774421 | 1.675127 | 1.616446 |
| hybrid_gru_tuned_anchor | 1.803458 | 1.773834 | 1.671854 | 1.615121 |
| monthly_integrated_gru_tuned_anchor | 1.866130 | 1.820441 | 1.675484 | 1.617370 |
| daily_integrated_gru_tuned_anchor | 1.804067 | 1.774060 | 1.663327 | 1.604602 |
| hybrid_integrated_gru_tuned_anchor | 1.803249 | 1.773626 | 1.661740 | 1.604714 |

## Coefficient choices

| Model | K | Ecological gamma counts | Support alpha counts | Ridge counts |
| --- | ---: | --- | --- | --- |
| monthly_integrated_gru_tuned_anchor | 0 | 0.0: 5; 0.25: 3; 1.0: 1 | 0.0: 9 | infinity: 9 |
| monthly_integrated_gru_tuned_anchor | 5 | 0.0: 3; 0.25: 5; 0.5: 1 | 0.5: 4; 0.75: 2; 1.0: 3 | 1.0: 8; 10.0: 1 |
| daily_integrated_gru_tuned_anchor | 0 | 0.0: 7; 0.25: 2 | 0.0: 9 | infinity: 9 |
| daily_integrated_gru_tuned_anchor | 5 | 0.0: 1; 0.25: 4; 0.5: 4 | 0.5: 3; 0.75: 3; 1.0: 3 | 1.0: 9 |
| hybrid_integrated_gru_tuned_anchor | 0 | 0.0: 7; 0.25: 2 | 0.0: 9 | infinity: 9 |
| hybrid_integrated_gru_tuned_anchor | 5 | 0.0: 1; 0.25: 5; 0.5: 3 | 0.5: 4; 0.75: 2; 1.0: 3 | 1.0: 8; 10.0: 1 |

The coefficient table describes selections on the existing validation episodes, not new information about target-station performance. A smaller MAE after adaptation can reflect both the routed base and different ecological/support coefficients. The saved CSV retains both constant and temporal support adapters at every K.

## Scientific reading

The route has a narrow purpose: use the extra hydrologic detail where it exists while retaining the monthly expert elsewhere. Its validation effect is small and varies across the three partitions. It should therefore be judged as a compact integration of two existing experts, rather than as evidence for a new learned transport mechanism. The separate spatial evaluation determines whether this preservation rule improves generalization.
