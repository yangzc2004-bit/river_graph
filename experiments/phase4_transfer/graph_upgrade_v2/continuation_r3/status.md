# R3 status

- New code snapshot: committed after the causal M3 and provenance changes.
- M3 smoke: DOC, temporal holdout, seed 42, two epochs; passed in about four
  seconds and produced a finite full-grid product.
- M2 static same-month control: 18/18 complete.
- M2 no-message control: 18/18 complete.
- M2 learned/fixed controls: complete in the preceding continuation and
  compared with the R3 static/no-message products.
- M3 matched pilot: 18/18 complete.
- Pending: final paired analysis and the decision whether a combined
  observation-aware + multi-scale model is worth a small follow-up run.

The earlier learned/fixed batches under `continuation_r2/` remain exploratory
because they used the previous runner snapshot and do not have the R3
checkpoint/trace fields.
