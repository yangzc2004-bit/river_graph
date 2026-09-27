# U3 temporal random-forest verdict

The temporal random forest uses the same target masks and query cells as the
EcoHydroGraph pilot. Features include current hydroclimate, ecological and
static descriptors, current visible-network summaries, upstream/downstream
coverage, and target lags at 1, 3, 6, and 12 months. It was run for DOC and
specific conductance, strict temporal extrapolation and unmonitored stations,
with seeds 42--44 and 200 trees.

| analyte | scenario | Temporal RF MAE | EcoHydroGraph 30-epoch MAE |
|---|---|---:|---:|
| DOC | strict temporal extrapolation | 1.089 | 1.223 |
| DOC | unmonitored stations | 2.683 | 3.032 |
| Specific conductance | strict temporal extrapolation | 163.680 | 221.760 |
| Specific conductance | unmonitored stations | 311.378 | 448.622 |

The tabular temporal model is lower in all four tested families. This closes
the case for further blind architecture expansion: the current graph model's
weakness is feature-to-prediction efficiency and optimization, not simply the
absence of a more complex temporal block. The result is a model-boundary
finding, not a failure of the broader reconstruction problem.
