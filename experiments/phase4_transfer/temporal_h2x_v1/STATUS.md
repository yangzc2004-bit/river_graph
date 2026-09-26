# H2X-T temporal extension status

## Engineering gates

- **T0: passed.** Spatial encoder/head split, causal rolling windows, analyte
  transforms, visibility isolation, and V3 provenance contracts are covered
  by `tests/test_temporal_h2x.py` and the provenance/cache tests.
- **T1: passed.** The DOC/E2a smoke path produced a finite nonconstant full
  station-month product with a recomputable V3 sidecar.  An E3 one-epoch
  smoke was also used to exercise the held-out-site metadata path.
- The initial DOC-frozen mask was audited across targets.  pH and specific
  conductance have additional missing labels, so the runner now materializes
  deterministic target-specific intersections under `target_masks/` before
  training; this is part of the T2 input contract.

## T2 pilot

The complete 72-run pilot uses the frozen 3-analyte × 4-mask × 3-seed ×
{H2X, H2X-T} design. Both arms use the explicit `matched_full_grid` training
protocol (one optimizer update after the full monthly sequence per epoch),
with a pilot budget of 10 epochs and patience 3.

`evaluate_temporal_h2x.py --stage pilot` audits all 72 products, including
dataset/mask/prediction hashes, station-month alignment, role visibility,
metric recomputation, and paired protocol identity. The audit is **72/72
pass**. The feasibility gate is also **pass**: all three analytes improve on
average in E2a/E2b and none shows an E1 mean degradation. Detailed numbers are
in `pilot_summary.csv`; the machine-readable gate is in `pilot_verdict.json`.

This is a pilot result, not authorization to start T3. Before formal training,
review the matched-budget comparison and decide whether to retain the current
10-epoch feasibility budget or increase the formal H2X-T budget.

## T3 formal confirmation

T3 has been authorized after the pilot review and is running under
`t3_formal/`. The initial 50-epoch trial was stopped before any product was
written because the measured runtime was impractical. The frozen confirmation
budget is now 20 epochs with patience 5, applied identically to all five seeds;
this is a runtime decision recorded in `t3_formal/run_plan.json`, not a change
to the model or endpoints.
