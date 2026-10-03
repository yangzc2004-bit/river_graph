# DOC context residual shift: source OOF and source-validation

This diagnostic compares the actual station-blocked OOF environmental predictions used
for residual training with full-source environmental predictions on fixed validation
query cells. It reads no outer-test query labels and fits no model. Validation has already
been used for checkpoint and adapter selection, so these are selection-set diagnostics.

Residual = observed DOC − context prediction, in mg/L; positive values mean
underprediction. Q90 is the original source-training threshold. The source and validation
populations contain different stations and concentration distributions. OOF forests also
use fewer source stations than the full-source validation predictor. These differences
cannot isolate training-set size, station composition or missing process information as
the cause. No in-sample source residual comparator is used.

Seeds are averaged within partitions and partitions are weighted equally. Displayed
means and quantiles are averages of per-run summaries, not pooled-sample quantiles.
Exact per-run cell/station counts and partition summaries are saved separately.

## Overall and concentration-specific distributions

| Role | Region | Mean cells per partition | Mean DOC | Mean base | Mean residual | Median residual | Residual Q90 | MAE |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| source_oof | nontail | 13445 | 3.895 | 4.275 | -0.380 | -0.452 | 1.589 | 1.247 |
| source_oof | overall | 14975 | 5.268 | 4.959 | 0.308 | -0.352 | 2.831 | 1.871 |
| source_oof | q90 | 1530 | 17.312 | 10.954 | 6.357 | 4.525 | 14.526 | 7.361 |
| source_validation | nontail | 2449 | 3.896 | 4.258 | -0.362 | -0.391 | 1.656 | 1.291 |
| source_validation | overall | 2682 | 5.226 | 4.770 | 0.456 | -0.252 | 3.007 | 1.991 |
| source_validation | q90 | 234 | 18.840 | 9.377 | 9.463 | 5.297 | 16.535 | 9.985 |

## Partition-level Q90 residuals

| Partition | Role | Cells | Mean DOC | Mean base | Mean residual | MAE |
|---|---|---:|---:|---:|---:|---:|
| 142 | source_oof | 1596 | 16.982 | 10.401 | 6.581 | 7.323 |
| 142 | source_validation | 305 | 16.729 | 10.214 | 6.514 | 6.720 |
| 143 | source_oof | 1541 | 17.882 | 11.043 | 6.838 | 7.912 |
| 143 | source_validation | 213 | 16.408 | 10.061 | 6.347 | 7.309 |
| 144 | source_oof | 1453 | 17.071 | 11.418 | 5.653 | 6.847 |
| 144 | source_validation | 183 | 23.383 | 7.855 | 15.528 | 15.926 |

## Fixed information strata: Q90

| Stratum | Role | Mean cells | Mean DOC | Mean residual | MAE | Small group flag |
|---|---|---:|---:|---:|---:|---|
| flow_observed | source_oof | 1338.0 | 17.534 | 6.418 | 7.453 | False |
| flow_unobserved | source_oof | 192.0 | 15.826 | 5.975 | 6.765 | False |
| no_upstream_support | source_oof | 1332.7 | 17.681 | 6.558 | 7.622 | False |
| upstream_support | source_oof | 197.3 | 14.775 | 5.013 | 5.593 | False |
| flow_observed | source_validation | 203.7 | 20.090 | 10.341 | 10.935 | False |
| flow_unobserved | source_validation | 30.0 | 16.866 | 9.135 | 9.250 | True |
| no_upstream_support | source_validation | 196.0 | 19.639 | 10.097 | 10.716 | False |
| upstream_support | source_validation | 37.7 | 14.761 | 6.307 | 6.383 | False |

Flow strata use the current-month discharge observation flag. Upstream support
means at least one currently visible DOC observation on a true upstream edge; zero
also includes stations with no upstream neighbor. Source support uses each station's
saved OOF fold with that entire fold hidden. Validation support uses training observations.
The two stratification axes are marginal, not a new cross-classified experiment matrix.
A group with fewer than 20 cells in any run is flagged; absent groups are not silently pooled.

## Context training size

- Partition 142: OOF context fits use 185–186 stations and 11,819–13,617 labels; full-source context uses 232 stations and 15,958 labels.
- Partition 143: OOF context fits use 185–186 stations and 11,090–12,978 labels; full-source context uses 232 stations and 15,241 labels.
- Partition 144: OOF context fits use 185–186 stations and 10,490–11,485 labels; full-source context uses 232 stations and 13,727 labels.

## Scientific reading

Both populations' high-DOC distributions are shown explicitly: source Q90 mean DOC 17.312 and mean residual 6.357; validation Q90 mean DOC 18.840 and mean residual 9.463 mg/L.

The partition pattern matters:
- Partition 142: validation minus source mean tail residual = -0.066 mg/L; mean observed DOC changes by -0.253, while mean context prediction changes by -0.187 mg/L. These are an arithmetic decomposition of the residual shift, not independent causal explanations.
- Partition 143: validation minus source mean tail residual = -0.492 mg/L; mean observed DOC changes by -1.474, while mean context prediction changes by -0.982 mg/L. These are an arithmetic decomposition of the residual shift, not independent causal explanations.
- Partition 144: validation minus source mean tail residual = +9.875 mg/L; mean observed DOC changes by +6.312, while mean context prediction changes by -3.563 mg/L. These are an arithmetic decomposition of the residual shift, not independent causal explanations.

A large positive tail residual establishes underprediction by the context base,
but does not establish that a larger tail weight will improve transfer. Mean residual
differences accompany different target distributions and fitted context populations.
The observed validation early-stopping behavior and the fixed tail/non-tail comparisons
should be read alongside these distributions before changing the objective. This
diagnostic makes no model, weight, threshold or station selection.
