# H2X-T temporal extension status

## Engineering gates

- **T0: passed.** Spatial encoder/head split, causal rolling windows, analyte
  transforms, visibility isolation, and V3 provenance contracts are covered
  by `tests/test_temporal_h2x.py` and the provenance/cache tests.
- **T1: passed.** The DOC/E2a smoke path produced a finite nonconstant full
  station-month product with a recomputable V3 sidecar.  An E3 one-epoch
  smoke was also used to exercise the held-out-site metadata path.

## T2 pilot

The 72-run pilot was started with the frozen 3-analyte × 4-mask × 3-seed ×
{H2X, H2X-T} plan.  It was stopped after 11 pilot units had complete,
identity-valid products because the sequential 50-epoch temporal runs were
using several minutes per unit.  The runner is resumable: rerunning
`scripts/run_temporal_h2x.py --stage pilot` reuses only complete matching
sidecars and continues missing units.  T2 has **not** been accepted and T3 is
not authorized by this status file.

The partial products are diagnostic only until all expected units pass
`scripts/evaluate_temporal_h2x.py --stage pilot`.
