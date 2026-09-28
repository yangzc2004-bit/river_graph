# Continuation R1 status

- Implementation: M3 causal long-history refactor, M2 lag-mode separation,
  aligned support/age diagnostics.
- Tests: graph-upgrade contract tests pass after the refactor.
- Smoke: M3 DOC, e2a strict, seed 42, two epochs completed with finite
  predictions and a complete sidecar.
- Running: matched-budget M2 learned-lag arm, all three analytes, two holdout
  families and seeds 42--44.
- Pending: fixed-lag, same-month static and no-message M2 controls, followed
  by an updated comparison table.

The original mechanism pilot remains exploratory and is not overwritten.
