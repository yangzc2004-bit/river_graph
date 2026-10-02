# Support-residual method development verdict

This is a development extension after the earlier outer result for the mean
support-residual calibrator had been observed. The base prediction cache is
independent and contains no target labels. Candidate selection used the nested
internal E3 validation split only. The outer E3 split was scored only for the
validation-selected method at each K and the previously selected mean-residual
control; no outer candidate table was used for selection.

## Candidate set

The fixed candidate set was:

* mean, median and Huber residual locations with alpha in
  {0.25, 0.50, 0.75, 1.00};
* strongly regularized affine residuals with lambda in {1, 10, 100};
* strongly regularized seasonal residuals with lambda in {1, 10, 100}.

Affine and seasonal coefficients were fit independently at each target station
from support residuals only, with a zero coefficient ridge prior. The seasonal
variant uses sine and cosine month terms. K=0 remains the raw source expert.

## Selection and outer score

| K | validation-selected method | outer selected MAE | mean-residual control MAE |
| ---: | --- | ---: | ---: |
| 0 | raw | 2.456 | — |
| 1 | Huber, alpha 0.25 | 2.314 | 2.314 |
| 3 | mean, alpha 0.50 | 2.159 | 2.159 |
| 5 | seasonal, lambda 1 | 2.108 | **2.023** |

The Huber and mean methods are identical at K=1 because each target station
has one support cell. At K=3 the validation-selected mean is the existing
control. At K=5, the seasonal method wins the internal validation score
(1.751 versus 1.780 for the mean residual), but is worse on the frozen outer
E3 score (2.108 versus 2.023 for the mean control). This is a useful warning
that station-level seasonal residuals are not yet stable across the held-out
stations.

The operational conclusion from this extension is to retain the simpler mean
support-residual correction. The affine and seasonal candidates do not provide
a new performance claim. The K=5 mean-residual result remains the strongest
current spatial reconstruction result.

## Reproducible files

* Cache runner: scripts/cache_spatial_residual_methods.py
* Selection and scoring: scripts/analyze_spatial_residual_methods.py
* Candidate functions: src/river_graph/experiments/spatial_residual_methods.py
* Label-free base cache: base_predictions.parquet
* Validation candidates: analysis/validation_candidates.csv
* Outer selected/control scores: analysis/test_selected_and_control.csv
* Figure: analysis/residual_method_curve.png and .pdf
* Cache and selection manifests: cache_manifest.json and
  analysis/analysis_manifest.json
