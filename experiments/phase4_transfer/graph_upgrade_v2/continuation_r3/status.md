# R3 status

- New code snapshot: committed after the causal M3 and provenance changes.
- M3 smoke: DOC, temporal holdout, seed 42, two epochs; passed in about four
  seconds and produced a finite full-grid product.
- Running: M2 static same-month control, 18 configurations, one CPU thread.
- Pending: M2 no-message and fixed-lag controls, then M3 matched pilot.

The earlier learned/fixed batches under `continuation_r2/` remain exploratory
because they used the previous runner snapshot and do not have the R3
checkpoint/trace fields.
