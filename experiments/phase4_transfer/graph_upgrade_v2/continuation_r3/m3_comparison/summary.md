# M3 multi-scale temporal pilot

The refactored M3 pilot is complete: three analytes, two holdout families and
three seeds. The model uses causal short and seasonal convolutions together
with a full causal trend state. The paired comparison uses the same hidden
station-month cells as M1, H2X-T and temporal RF.

| analyte | holdout | M3 MAE | M1 MAE | M3 improvement vs M1 |
|---|---|---:|---:|---:|
| DOC | temporal | 1.313 | 1.143 | -14.9% |
| DOC | spatial | 3.300 | 3.490 | +5.4% |
| pH | temporal | 0.265 | 0.272 | +2.4% |
| pH | spatial | 0.275 | 0.280 | +1.8% |
| specific conductance | temporal | 206.46 | 190.97 | -8.1% |
| specific conductance | spatial | 487.91 | 489.47 | +0.3% |

M3 improves the spatial DOC holdout, but does not provide a consistent
all-analyte or all-missingness gain. It remains substantially behind the
temporal RF baseline for DOC and conductance. The result supports using
multi-scale time as a targeted spatial-transfer component rather than as a
replacement for the observation-aware model in every scenario.
