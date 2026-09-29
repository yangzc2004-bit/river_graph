# R7/R8 RF--GNN complementarity result

R7 fits the observation-aware M1 model with river messages and a Temporal RF
on the same DOC query cells. R8 repeats the fits with the GNN message path
removed while retaining the same observation-age and river-support features.
Both use validation-only alpha selection on `{0, 0.05, ..., 1}` and three
training seeds. The test set is evaluated once after alpha is frozen.

| holdout | RF MAE | river-M1 MAE | river blend MAE | no-message M1 MAE | no-message blend MAE |
| --- | ---: | ---: | ---: | ---: | ---: |
| temporal (`e2a_strict`) | 1.089 | 1.140 | **1.006** | 1.440 | 1.091 |
| spatial (`e3_spatial_seed42`) | 2.683 | 3.494 | 2.683 | 3.776 | 2.683 |

On temporal extrapolation the river blend is 7.64% below RF, while the
no-message blend is 0.16% above RF. The paired station bootstrap interval for
RF minus river blend is `[-0.020, 0.178]`; the positive direction is
consistent across all three seeds but the interval includes zero. For the
no-message blend the interval is `[-0.150, 0.108]`. The matched control
therefore removes the observed complementarity, although the current sample
does not establish a confirmatory effect size.

On spatial extrapolation validation selects pure RF for all six fits, so both
blends equal RF and the graph contributes no test improvement. This is a
missingness-dependent result: river messages can provide a complementary
temporal signal in the current data, while the same model does not transfer to
the held-out spatial component.

These runs support a focused next experiment: learn a graph residual or gate
only for temporal extrapolation, with the RF-like branch fixed as the local
baseline. They do not justify adding more GNN depth or claiming general
spatial superiority.
