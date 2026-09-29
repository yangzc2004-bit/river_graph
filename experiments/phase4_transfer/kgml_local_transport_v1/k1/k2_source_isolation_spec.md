# K2 source-isolation pilot

## Question

Does the KGML residual branch use directed upstream information, or does it
mainly learn a global correction to the RF-local prediction?

## Design

Use DOC, `e2a_strict` and `e3_spatial_seed42`, and seeds 42--44. Keep the RF
features, OOF residual target, optimizer, and 30-epoch budget fixed from K1.

Compare two structurally matched residual models:

1. **message-only residual**: the residual head receives an explicit upstream
   message representation and its no-message input is exactly zero;
2. **message-only null**: the same head and parameter count, with the message
   representation zeroed for every station-month.

Also retain the K1 residual and RF-context products as fixed references. The
primary quantity is the paired MAE difference between message-only and its null
control, using the same query cells and station bootstrap.

## Interpretation

- A spatially varying gain in the temporal mask supports a genuine upstream
  contribution and justifies a small fixed-lag follow-up.
- A message-only gain that disappears after removing a scalar OOF bias supports
  the interpretation that K1 learned calibration rather than transport.
- A null result closes the lagged-message branch for this dataset and moves the
  next model iteration toward process-state or simulation-pretrained features.

This pilot is deliberately small: 2 arms × 2 masks × 3 seeds = 12 runs.
No multi-lag or multi-indicator expansion starts before this source is
identified.
