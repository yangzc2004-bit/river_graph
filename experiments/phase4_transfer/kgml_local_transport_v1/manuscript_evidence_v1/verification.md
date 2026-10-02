# DOC hybrid manuscript evidence

All results were recomputed from saved per-seed predictions; no training occurred. Query membership and truth were checked against the dataset and original masks. Context predictions match the validation-selected source model exactly; RF uses the fixed `rf_default` from the same feature/split/seed run. Source sidecars match the actual dataset, mask, and prediction hashes. Temporal residual rows match the saved no-message residual expert, and the affine predictions are independently reconstructed from the saved coefficients. Main MAEs reproduce the existing five-seed affine report.

## Estimands and interpretation

- MAE, RMSE and R2 are means of the five seed-specific metrics, not metrics of an ensemble mean or median prediction. Cell and station counts are unique within each scenario.
- Paired 95% intervals resample whole stations after averaging cell losses across seeds, recomputing cell-weighted MAE on each draw. They are descriptive intervals for one repeatedly used development mask per family, not independent external confirmation.
- Random gaps and unmonitored stations use the context route exactly, so zero hybrid improvement there is a routing identity, not a measured neural increment.
- The temporal residual expert is no-message. These gains establish value of the selected temporal residual combination, not an independent river-message contribution.
- The hybrid combines complete context and residual-expert predictions in log-space. It is not numerically identical to adding a raw temporal correction to the context forest.
- Final test inference uses train+val+context observations; fitting and validation inference use train+context. This matches the existing protocol.
- Q90 is based only on train labels. Temporal tail samples are small; see the unique-cell counts and unstable flags rather than treating five seed repetitions as additional ecological samples.
- `component_metrics.csv` compares the matched local ExtraTrees base (`local_pred` in the residual source file, in raw DOC units), context expert, complete temporal residual expert, and selected affine hybrid on identical query cells for each temporal family and seed, plus a mean-seed row. Local-to-residual improvement measures the increment of the full deep residual branch; it is not an attention or graph ablation.
- `pooled_metrics.csv` weights family-cell occurrences equally after averaging seed losses. The 11,046 occurrences overlap across scenarios and are not 11,046 independent or distinct observations; distinct counts are reported separately.

## Reproduction

`uv run --no-sync python scripts/analyze_doc_hybrid_manuscript.py`

Figure: `doc_hybrid_performance.pdf` (vector) and `.png` (300dpi). Panel a error bars are seed SD; panel b intervals are station bootstrap CIs. `doc_spatial_support_publication` and `doc_station_response_publication` are 6.5-inch publication-sized copies of verified spatial evidence, with 8–9pt type at full-width inclusion; historical spatial figures and numbers are unchanged.
