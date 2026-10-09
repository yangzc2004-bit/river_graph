# Wider source pool: research decision

## Decision

Close the 60-candidate mechanism without geographical training. Retain the
preceding 20-candidate current-availability model. More source observations
became usable, but widening the ecological pool did not improve its overall
prediction error. Do not run a candidate-count sweep or report its gains over
older models as the incremental benefit of widening.

All nine fixed source packages (142/143/144 × 42/43/44) completed: 18 neural
fits, 90 reused double-held references, no new forest. Candidate, source-role,
initial-state, prediction, fusion and diagnostic replay passed exactly. The
analysis uses 5,000 paired station draws with seed means within partition and
equal partition weights. The final PNG was inspected. Validation: 1,005 tests
passed, two skipped; Ruff and the historical artifact audit passed.

## Matched result

| Complete procedure | MAE, mg/L | Q90 MAE, mg/L |
|---|---:|---:|
| Retained complete | 1.769053 | 9.075420 |
| Current source, 20 candidates | **1.723002** | 8.895024 |
| Current source, 60 candidates | 1.726834 | **8.884919** |
| Seasonal values only, 60 candidates | 1.746777 | 8.962595 |
| Strong station-hidden trees | 1.866362 | 9.394458 |

The isolated complete-model gain versus the preceding 20-candidate model is
**-0.222% [-0.766%, 0.302%]**, four of nine packages improving, two of three
partition means and one of three seed means positive. Q90 gain is 0.114%
[-0.362%, 0.580%], unresolved. Native-only MAE is 1.731087 versus 1.726375:
-0.273% [-0.976%, 0.377%]; native Q90 gain is 0.072%, unresolved. Thus the
wider pool has no established incremental overall or tail benefit.

Actual current values still outperform seasonal-only wide values by
1.142% [0.680%, 1.624%]; Q90 improves 0.867% [0.423%, 1.600%]. Those
comparisons establish source-value utility within the wide pool, not utility
of enlarging it. Accumulated MAE gain versus retained complete is
2.386% [1.563%, 3.318%] and versus strong trees is 7.476% [4.757%, 10.534%],
including changes from preceding source studies. Station-equal MAE is
2.181899, compared with 2.175422 for 20 current candidates. Signed bias is
-0.602898 mg/L; improved bias alone does not change the primary decision.

## Interpretation and next mechanism

The original nearest 20 remain the prefix of the 60-candidate pool, with the
same nearest-five distance scale, exponential ecological prior, zero prior,
keys, receiving query, 37,900 parameters, matched aggregate readout and budget.
Usable current donors rise from 5.017/4.007/3.477 to
11.460/9.714/8.520, and supported-cell fractions rise to approximately
98.6/98.7/99.4%. Additional donor access is therefore real. It does not by
itself supply better donor selection.

Return to the 20-candidate current-availability model and study whether source
DOC state should inform the attention keys. At present ecology and daily
hydrology choose the donor weights, while the contemporaneous DOC residual
enters only the values. A source-state key can let the same receiving query
distinguish currently informative observations from source-state deviations
that are not relevant to that receiver. Compare current-state, seasonal-only
and zero additional keys, preserving actual donor values in all three arms.
Source-only normalization and double-held references must preserve query-fold
exclusion. This is a new information-selection mechanism, not another pool
size or layer scan. Use 142/143/144 source validation to decide its value;
do not select it from geographical or external labels.

Reproduce with `scripts/analyze_doc_wide_source_attention_v1.py
--bootstrap-draws 5000` and `scripts/plot_doc_wide_source_attention_v1.py`.
Large fitting caches remain local, with related small results recorded in the
existing pending-submission whitelist. Geographical, external and portable
products remain unchanged.
