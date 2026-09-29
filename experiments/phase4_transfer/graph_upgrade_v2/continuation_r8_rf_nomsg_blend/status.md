# R8 matched-input no-message blend

R8 repeats the R7 RF/M1 blend with `edge_set=empty` in the spatial encoder.
The M1 observation features still use the river edge table through
`feature_edge_set=river`, so the only changed information source is directed
graph message passing.

| holdout | RF MAE | no-message M1 MAE | no-message blend MAE | relative change vs RF |
| --- | ---: | ---: | ---: | ---: |
| temporal (`e2a_strict`) | 1.089 | 1.440 | 1.091 | -0.16% |
| spatial (`e3_spatial_seed42`) | 2.683 | 3.776 | 2.683 | 0.00% |

For temporal extrapolation, the corresponding river-message blend is 1.006
MAE (7.64% below RF). Removing messages raises the blend to 1.091 MAE and
removes the gain. The river-message and no-message paired summaries are in
`../continuation_r7_rf_m1_blend/r7_vs_r8_summary.csv`.

For spatial extrapolation validation selects pure RF for all seeds in both
experiments. The graph model therefore contributes no spatial test gain in
this comparison.

The saved support strata were regenerated after fitting using the river
feature edge table; this changed diagnostic labels only, not predictions,
validation selection, or metrics. `strata_repair.json` records the repair.
