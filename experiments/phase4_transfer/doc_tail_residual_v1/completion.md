# DOC native residual: completed iteration

## Research result

The existing observation-aware GRU can improve DOC prediction at stations with
no target observations when its scalar residual is trained directly for
concentration-space error. The gain is measurable before station adaptation;
it largely disappears after five target observations are supplied.

All nine partition/seed packages and eighteen neural fits completed. Both
objectives improve K0 MAE in all three station partitions and all nine fitted
packages. These are development results on the same previously studied ST357
station partitions, not a new external validation.

## What changed

The selected environmental ExtraTrees context predictor and spatial encoder
were fixed. Copies of the original GRU and observation-decay weights were
trained with a zero-initialized signed scalar head: **25,537 parameters**.
The prediction is a nonnegative concentration-space correction of the context
base. One arm minimizes ordinary MAE; the other doubles weights for source
observations at or above the source-training Q90. Source station-blocked OOF
forest predictions provide the training bases. Validation overall MAE selects
the checkpoint and residual scale, including an exact context-only candidate.

The existing v4 support representation is frozen and identical for all arms.
Each arm recomputes support residuals against its own corrected base and fits
the same source-validation adapter grid. This separates zero-observation
prediction from adaptation to local observations.

## Matched performance

MAE in mg/L, averaging seeds within each station partition and then weighting
the three partitions equally. K is the number of target observations per station.

| Prediction model, with the same GRU support adapter | K = 0 | K = 5 |
|---|---:|---:|
| Environmental context base | 1.902823 | 1.596001 |
| Context + ordinary-MAE neural residual | 1.862470 | 1.599917 |
| Context + tail-weighted neural residual | 1.854491 | 1.595924 |
| Previous v4 fusion + updated GRU adapter | 1.913634 | 1.595209 |

Against the matched context base at K0:

- Ordinary native MAE reduces error by **2.1207%**, with a paired whole-station
  bootstrap 95% interval of **0.5763% to 3.7871%**.
- Tail-weighted MAE reduces error by **2.5400%**, with an interval of
  **1.1696% to 3.8874%**.
- Both improve 96 of the 172 distinct target stations; 76 worsen. The top five
  stations account for 25.1% and 34.2% of positive gain, respectively. Aggregate
  improvement is not improvement at every station.

The two objectives are not clearly separated on overall MAE. Tail weighting
changes the error trade-off: ordinary MAE worsens Q90 error from 7.5586 to
7.7375 mg/L, while tail weighting gives 7.4446 mg/L. The tail-weighted improvement
over context at Q90 has an interval crossing zero, so the overall improvement
does not establish that high-DOC reconstruction is solved. Relative to ordinary
training, weighting improves Q90 error but sacrifices some non-tail accuracy
and raises the false high-DOC rate by 0.255 percentage points.

At K5, neither new residual establishes an improvement over the matched context
adapter or the previous fusion adapter. The existing K5 model remains the
performance reference. The new scalar residual is a promising K0 component;
this iteration does not replace the default model across all observation counts.

## Source-record and water-flow diagnosis

The 20 largest unique source-validation errors were traced to local provider
records. All 20 monthly values reproduce correctly; all 14 raw files match the
build-input manifest. No processing defect was established and no label was
removed. Laboratory-method metadata are absent for several historical records,
so archival agreement does not independently verify their physical accuracy.

Only five of 7,197 unique validation station-months exceed 100 mg/L; they account
for 10.8% of absolute error and 85.7% of squared error under unique-cell weighting.
The audit also retains the experiment's equal-partition summaries separately.
This concentration is why MAE, tail error and RMSE must be read together.

High-DOC errors are not uniformly associated with high monthly flow. Current
hydro coverage is generally available; modest flow-change associations suggest
that short-term relative flow anomalies are a plausible next input experiment,
not a demonstrated cause of the remaining errors.

## Execution and reproduction

There were 384 executed epochs across the eighteen fits (30 maximum epochs,
patience 5). Every fit selected a nonzero residual scale. Recorded training and
product-generation time totals 113.5 seconds, using cached spatial encodings;
this excludes previous expert training, analysis and replay.

- All nine packages replayed from saved weights. Native residuals, full-grid
  predictions and adapted queries are bitwise identical. Independent forest
  recomputation is checked separately within floating-point tolerance.
- Full suite: **577 passed, 2 skipped**. Ruff passes. Historical artifact audit
  exits 0; it does not upgrade historical missing-sidecar evidence.
- Old context predictions and prior experiment files remain unchanged.

```bash
uv run python scripts/run_ladder.py --experiment doc-tail-residual-v1
uv run python scripts/verify_doc_tail_residual_v1.py
uv run python scripts/analyze_doc_tail_residual_v1.py
uv run python scripts/audit_doc_tail_source_v1.py
uv run python scripts/diagnose_doc_tail_hydro_v1.py
```

## Next model question

Preserve the demonstrated K0 residual improvement. The next useful experiment
is whether short-term flow anomalies and 1/3-month changes add time-varying
information beyond the current concentration residual, using the same model
and a matched input ablation. Assess K0 and K5 separately: a correction selected
for K0 overall MAE need not add value after station calibration. Further gains
should target variation that remains after local support has corrected the
station-level offset.

Detailed comparisons: [analysis/findings.md](analysis/findings.md).
Source trace: [data_audit/audit.md](data_audit/audit.md).
Hydro diagnostics: [hydro_diagnostics/diagnostic.md](hydro_diagnostics/diagnostic.md).
Replay results: [verification/replay_checks.json](verification/replay_checks.json).
