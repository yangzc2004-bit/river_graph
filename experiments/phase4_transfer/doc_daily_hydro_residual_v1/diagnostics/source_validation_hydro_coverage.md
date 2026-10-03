# Daily-flow descriptors: source-validation coverage diagnostic

This diagnostic compares the existing monthly control, availability-only control and daily-descriptor model on the fixed source-validation query cells at K=0. It reads no outer spatial-query DOC values or outcomes. Both the checkpoints and ecological mixture weights were selected using this validation set; these are descriptive selection-set effects. All models and thresholds remain unchanged.

“Available” means that all three frozen numeric descriptor-validity flags equal one. The complement includes missing daily records and short/gappy calendars. A separate count distinguishes complete absence of numeric daily information from the rare partial-descriptor case. Errors are native DOC errors in mg/L; positive signed bias means overprediction.

## Counts and native prediction errors

| Partition | Daily descriptors | Query cells | Context MAE | Monthly MAE | Daily MAE | Daily−monthly MAE | Monthly bias | Daily bias |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 142 | All valid | 1754 | 2.2481 | 2.0201 | 1.9758 | -0.0443 | -0.7353 | -0.7847 |
| 142 | Not all valid | 259 | 1.5484 | 1.4047 | 1.4578 | +0.0531 | -0.2946 | -0.2695 |
| 143 | All valid | 2387 | 1.9857 | 1.9325 | 1.7720 | -0.1605 | +0.0214 | -0.2230 |
| 143 | Not all valid | 411 | 1.4477 | 1.3930 | 1.3626 | -0.0304 | -0.4289 | -0.6434 |
| 144 | All valid | 2133 | 2.2027 | 2.0844 | 2.0476 | -0.0368 | -1.0613 | -1.0302 |
| 144 | Not all valid | 1103 | 1.3388 | 1.2969 | 1.2987 | +0.0018 | -0.7699 | -0.7384 |

## Selected ecological integration

At K=0 the constant and temporal support adapters coincide. The integrated columns use each arm’s already selected K=0 ecological-mixture coefficient, not a newly selected coefficient.

| Partition | Daily descriptors | Monthly integrated MAE | Daily integrated MAE | Daily−monthly MAE | Monthly bias | Daily bias | Improved seeds |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 142 | All valid | 2.0201 | 1.9758 | -0.0443 | -0.7353 | -0.7847 | 3/3 |
| 142 | Not all valid | 1.4047 | 1.4578 | +0.0531 | -0.2946 | -0.2695 | 0/3 |
| 143 | All valid | 1.9216 | 1.7720 | -0.1496 | -0.0822 | -0.2230 | 3/3 |
| 143 | Not all valid | 1.3903 | 1.3626 | -0.0278 | -0.5576 | -0.6434 | 2/3 |
| 144 | All valid | 2.0795 | 2.0449 | -0.0346 | -1.0984 | -1.0762 | 3/3 |
| 144 | Not all valid | 1.3002 | 1.3006 | +0.0004 | -0.8188 | -0.7901 | 1/3 |

## Equal-partition comparison

| Comparison | Daily descriptors | Candidate MAE | Reference MAE | Delta MAE | Improved seed runs |
| --- | --- | ---: | ---: | ---: | ---: |
| direct: daily−monthly | All valid | 1.931779 | 2.012310 | -0.080531 | 9/9 |
| direct: daily−monthly | Not all valid | 1.373003 | 1.364866 | +0.008136 | 3/9 |
| direct: daily−availability | All valid | 1.931779 | 2.011665 | -0.079887 | 9/9 |
| direct: daily−availability | Not all valid | 1.373003 | 1.363196 | +0.009807 | 4/9 |
| integrated: daily−monthly | All valid | 1.930893 | 2.007081 | -0.076188 | 9/9 |
| integrated: daily−monthly | Not all valid | 1.373658 | 1.365079 | +0.008579 | 3/9 |
| integrated: daily−availability | All valid | 1.930893 | 2.006534 | -0.075641 | 9/9 |
| integrated: daily−availability | Not all valid | 1.373658 | 1.363440 | +0.010218 | 4/9 |

Each partition first averages the three training seeds. Partition means are then weighted equally. Cell counts are unchanged across seeds; pooled query counts across partitions would include repeated stations/months. No bootstrap inference or new route selection is performed on these reused validation values.

## Missingness detail

| Partition | All numeric descriptors absent | No daily records | Short/gappy records, no valid numeric descriptor | Partial numeric descriptor |
| --- | ---: | ---: | ---: | ---: |
| 142 | 259 | 253 | 6 | 0 |
| 143 | 410 | 408 | 2 | 1 |
| 144 | 1103 | 1099 | 4 | 0 |

The source-validation role changes between partitions, so this table describes the availability regimes in the existing experiment rather than a new cohort. The saved CSV retains the availability-only control, every partition/seed, signed bias and ecological-mixture choices.


## Recommendation for the next focused comparison

First test a deterministic missing-information fallback using the two saved experts: retain the matched monthly expert when **none of the three daily numeric descriptors is valid**, and otherwise use the daily-descriptor expert. This is an observed-data availability rule, with the existing 80% QC definitions unchanged; it does not introduce a fitted reliability gate or a new DOC threshold. If evaluated with ecological mixing and K-shot adaptation, apply the fixed routing to the base first and select those existing downstream coefficients on source-validation only.

The evidence motivating this comparison is specific: where all descriptors are valid, daily inputs reduce integrated validation MAE by 0.076188 mg/L, with improvement in all nine runs. Where they are not all valid, the mean difference is +0.008579 mg/L and only three runs improve. That second result is heterogeneous—partition 143 improves, partition 142 worsens, and 144 is nearly unchanged—so the fallback is a candidate comparison, not an established universal advantage. Apart from one partially valid cell in partition 143, this complement contains no usable numeric daily descriptor.

This direct fallback is the next useful step before injecting daily states into the GRU. The present model already changes both its temporal and selected spatial/ecological parameters during training; consequently, predictions can shift even where its new numeric inputs are absent. A fixed fallback would isolate that prediction change without requiring another architecture or more training. Daily-state recurrence can be studied later if the remaining errors suggest that the sequence of hydrologic conditions matters beyond these current-month summaries.

The native bias remains negative on average in both availability groups. Lower MAE therefore does not by itself establish that high-DOC underprediction has been resolved. The separate tail and ordinary-value evaluation remains necessary.
