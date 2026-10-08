# Validation record

Date: 2026-10-08

## Calculations and reproduction

- Nine focused tests cover independently calculated source-pair correlation,
  the receiving fluctuation reference, excursion denominators, scale and order
  invariance, shared calendar projection, constant-series handling,
  receiver-centered periods, whole-year resampling and native DOC/date checks.
- Full repository suite: 1,319 passed, two skipped and eight warnings. This run
  preceded the final annual-count metadata rename and date-tick clarification;
  neither changed the numeric analysis. Final full replays verify those scripts.
- Ruff passed after the final edits. The historical artifact audit exited zero;
  its existing absent-dataset and 85 historical no-sidecar caveats remain.
- Both primary and shortest-route full verification commands exited zero.
  Saved inputs, source snapshots, analysis products and English/Chinese figure
  receipts agree. Complete table and summary replays reproduce the saved outputs.
- Independent reconstruction checks all 32 monthly receiving networks on both
  routes: shared calendar projection, source covariance, receiving reference and
  excursion counts. Maximum source-coordination discrepancy is 1.93e-14.
- Independent whole-catchment-system resampling reproduces the primary
  excursion-risk difference and its 95% interval. The primary 1,333 and
  shortest-route 1,330 receiving-month records reconcile with source labels and
  the area-weighted source mixtures.

## Figures and interpretation

- All eight final PNG products were visually inspected. The primary Chinese
  overview was also rechecked after completion. Labels, point counts, legends,
  interval units and date ticks agree with the saved tables.
- PDFs use the same Matplotlib canvas; they were not separately rendered for
  inspection. English and Chinese figure receipts bind each plotted input.
- Dense-case year 2004 is chosen by observation availability, not DOC peaks.
  The first and last displayed dates are explicit. Annual counts refer to years,
  not replicated receiving networks.
- Whole-system intervals preserve overlapping catchments. The dense-case
  intervals instead resample its calendar years. Receiver-equal conditional
  probabilities are distinguished from pooled observation-date fractions.
- The excursion threshold is the series-specific upper quartile of DOC anomalies,
  with Q90 kept separately. Same-calendar-day observations are not simultaneous;
  the sampling-clock audit is recorded in `research_decision.md`.
- Original form classes and source frontiers are retained. Negative receiver
  differences, inconclusive form contrasts, sparse within-network periods and
  the less decisive one-day sensitivity remain in the results.

No new training, lag fitting or replacement of historical results was performed.
