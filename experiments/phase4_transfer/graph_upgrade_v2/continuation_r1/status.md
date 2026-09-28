# Continuation R1 status

- Implementation: M3 causal long-history refactor, M2 lag-mode separation,
  aligned support/age diagnostics.
- Tests: graph-upgrade contract tests pass after the refactor.
- Smoke: M3 DOC, e2a strict, seed 42, two epochs completed with finite
  predictions and a complete sidecar.
- The first R1 M2 launch was interrupted after the code changed during the
  process; its partial files are retained as an interrupted exploratory
  attempt and are not included in a comparison table.
- Pending: restart the four lag modes from a stable code snapshot, then run
  fixed-lag, same-month static and no-message controls before updating the
  comparison table.

The original mechanism pilot remains exploratory and is not overwritten.
