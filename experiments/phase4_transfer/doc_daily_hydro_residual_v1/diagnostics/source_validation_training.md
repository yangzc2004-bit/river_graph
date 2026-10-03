# Source-validation training of daily hydrologic descriptors

All nine packages and 27 neural fits completed. This diagnostic reads only source-validation model summaries/traces, run configuration and package timing. It does not read held-out DOC values, predictions or performance tables. Validation is the reused checkpoint/scale selection set; these results describe optimization and input response, not independent spatial generalization.

All arms retain the same last-self/ecology encoder, GRU, source OOF context base, native DOC objective with tail weight two, 120-epoch ceiling, patience five and pooled validation MAE selection. The matched 38-column head has the same nominal parameters: the monthly control zeros all eight daily channels; availability retains daily coverage and validity flags; daily additionally retains width, flashiness and rising fraction. Daily numeric hidden interactions remain inactive in the two controls.

## Selected checkpoints

| Input arm | Validation MAE (mg/L) | Context MAE | Selected epochs | Epochs run | Cap-limited | Scale counts |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| monthly | 1.870033 | 1.990984 | 5–47 | 10–52 | 0/9 | 0.5: 2; 1: 7 |
| availability | 1.869337 | 1.990984 | 5–47 | 10–52 | 0/9 | 0.5: 2; 1: 7 |
| daily | 1.804428 | 1.990984 | 27–72 | 32–77 | 0/9 | 1: 9 |

MAE is seed-averaged within partition, then equally averaged across three partitions. Each run uses pooled fixed source-validation query cells. Counts do not turn reused stations or seeds into independent ecological observations.

## Matched source-validation effects

Negative delta favors the candidate. Daily versus availability isolates the added descriptor values within this training design; availability versus monthly measures the contribution of daily coverage information.

| Candidate − reference | Partition | Delta MAE (mg/L) | Relative reduction | Improved seeds |
| --- | ---: | ---: | ---: | ---: |
| availability − monthly | 142 | -0.002313 | +0.119% | 3/3 |
| availability − monthly | 143 | -0.000201 | +0.011% | 2/3 |
| availability − monthly | 144 | +0.000425 | -0.023% | 1/3 |
| daily − monthly | 142 | -0.031798 | +1.638% | 3/3 |
| daily − monthly | 143 | -0.141352 | +7.627% | 3/3 |
| daily − monthly | 144 | -0.023666 | +1.303% | 3/3 |
| daily − availability | 142 | -0.029485 | +1.521% | 3/3 |
| daily − availability | 143 | -0.141151 | +7.617% | 3/3 |
| daily − availability | 144 | -0.024090 | +1.326% | 3/3 |

## Optimization status

- 0/27 fits exhausted the epoch ceiling before ordinary patience stopping; 0/27 selected epoch 120.
- All recorded training losses after epoch zero and all validation scores are finite; 0/27 fits select the zero-residual context fallback.
- Recorded package duration totals 375.5 seconds; package range 26.9–59.9 seconds. These include three fits, feature extraction, inference and product assembly; per-arm wall times were not recorded.
- Every arm has 31,559 nominal trainable parameters, including 5,536 selected spatial/ecology parameters. Inactive zero-input columns mean equal nominal size is not equal effective input information.

| Arm | Selected temporal parameter displacement | Selected spatial displacement | Selected ecology displacement |
| --- | ---: | ---: | ---: |
| monthly | 0.9830–6.6578 | 0.0606–0.4318 | 0.0333–0.2673 |
| availability | 1.0032–6.6974 | 0.0614–0.4433 | 0.0339–0.2770 |
| daily | 4.7748–7.7477 | 0.2852–0.5246 | 0.1600–0.3251 |

Displacements are Euclidean distances from the original expert at the selected checkpoint; they describe learned changes, not causal or physical coefficients. The CSV preserves all 27 selections, losses, scale choices and parameter changes. No training setting, feature threshold or model is changed by this diagnostic.

## Reading the result

availability versus monthly: selected validation MAE changes by -0.000696 mg/L (+0.037% relative reduction), with improved-seed counts [3, 2, 1] for partitions 142, 143 and 144.
daily versus monthly: selected validation MAE changes by -0.065605 mg/L (+3.508% relative reduction), with improved-seed counts [3, 3, 3] for partitions 142, 143 and 144.
daily versus availability: selected validation MAE changes by -0.064909 mg/L (+3.472% relative reduction), with improved-seed counts [3, 3, 3] for partitions 142, 143 and 144.

The source-validation contrast determines whether the new hydrologic summaries affect the fitted solution consistently. High-DOC recovery, ordinary-value overprediction and transfer to the fixed target stations remain separate questions for the planned evaluation. Longer training is not proposed simply because an input block has changed.
