# Source geomorphology descriptor experiment (K0)

This independent E3 spatial-transfer experiment compares the existing label-free station descriptor (`base`) against the same descriptor augmented with source-fitted geomorphology fields (`base_geomorph`). Added fields are station elevation (`alt_va`, with missingness indicator), log1p reach length (`lengthkm`), and log1p reach area (`areasqkm`). Existing regime fields (stream order, drainage area, slope, watershed elevation and environmental aggregates) were not duplicated; NLDI `measure` was explicitly excluded because it is a within-reach location coordinate, not stream order.

## Protocol

Three seeds (42, 43, 44), ExtraTrees 120 trees, leaf 4. Internal station-heldout validation selected K from 20/40/80/160 separately for each feature set. Median imputation and standardization were fit on source stations only in each split. The frozen outer E3 test was evaluated only at the selected K.

## Results

| feature set | selected K (internal) | outer MAE mean ± SD | outer RMSE mean ± SD | outer log-MAE mean ± SD |
|---|---:|---:|---:|---:|
| base | 40 | 2.447 ± 0.010 | 5.312 ± 0.009 | 0.260 ± 0.002 |
| base + geomorph | 80 | 2.489 ± 0.014 | 5.360 ± 0.020 | 0.266 ± 0.002 |

The added fields change the selected K and worsen outer MAE by 1.7% relative to the source-fitted base descriptor (2.447 vs 2.489 mg/L mean). This low-cost addition therefore does not improve the current K0 spatial transfer model and should not be carried into the next model iteration. The fields remain available for targeted station-diagnostic analysis; no retraining of the main model is justified by this result.
