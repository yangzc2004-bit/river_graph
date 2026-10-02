# Paired few-shot spatial transfer with support residual calibration

## Question and protocol

This experiment asks whether a small number of target-station DOC observations
can calibrate a source-station regional expert. It uses the same outer E3
stations and the same fixed 2,316-cell query set for every K. Five support
cells are reserved per target station and K is evaluated at 0, 1, 3 and 5
using nested prefixes of the value-blind schedule.

The source regional ExtraTrees model is fitted once with the target stations
hidden. For each support cell, the model produces a prediction while all
target support labels remain hidden. The target station correction is the mean
support residual in log1p space:

    log1p(y_support) - log1p(base_support_prediction)

multiplied by an alpha selected from the nested internal station-held-out
validation split. The correction is then applied to the fixed query cells.
No query labels enter fitting, alpha selection or prediction.

## Results

| target support K | pooled query MAE | reduction vs K=0 |
| ---: | ---: | ---: |
| 0 | 2.456 | 0.0% |
| 1 | 2.314 | 5.8% |
| 3 | 2.159 | 12.1% |
| 5 | **2.023** | **17.6%** |

Nested validation selected alpha = 0.25, 0.50 and 0.75 for K = 1, 3 and 5.
The K=5 station-clustered paired reduction is 16.6 percentage points, with a
95% bootstrap interval of 5.3--33.4 percentage points. K=1 is positive in the
pooled result but its station-clustered interval crosses zero; K=3 and K=5
show the clearest evidence.

The corresponding raw source-model MAE is 2.456 at every K because the source
model is fitted once without target support. The gain therefore comes from
support residual calibration rather than refitting or adding support cells to
the source model.

## Support specificity

Shuffling the station-level support residuals gives mean MAE of 2.504, 2.539
and 2.727 at K = 1, 3 and 5, compared with 2.314, 2.159 and 2.023 using the
correct station residuals. The improvement is tied to the observed target
station values.

## Interpretation

The result identifies a useful spatial transfer decomposition. The source
regional expert transfers the temporal pattern learned from similar stations;
the support residual estimates the target station's local DOC level. Direct
residual calibration improves on the previous level correction, which used
support mean minus query-period base mean and could mix the station offset with
seasonal mismatch.

This is retrospective sparse reconstruction over the historical record. It is
not a prospective warm-start forecast. The result motivates a next experiment
with a strongly regularized seasonal residual term after the intercept
calibration, while retaining the fixed-query protocol.

## Reproducible outputs

* Runner: scripts/run_spatial_fewshot_residual.py
* Analysis: scripts/analyze_spatial_fewshot_residual.py
* Query and selection manifest: query_manifest.json
* Full metrics: validation.csv, test_all_alphas.csv, test_selected.csv
* Query predictions: test_predictions.parquet
* Support predictions and residuals: support_predictions.parquet
* Summary tables and figure: analysis/curve_summary.csv,
  analysis/paired_improvement.csv,
  analysis/support_residual_shuffle.csv,
  analysis/spatial_fewshot_residual.png and .pdf
