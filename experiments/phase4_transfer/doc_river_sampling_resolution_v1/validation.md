# Validation record

Completed 2026-10-07 in the repository's uv-managed environment.

- `uv run pytest -q`: **1,144 passed, 2 skipped**, eight existing warnings.
  The new eight tests cover activity reconciliation, UTC conversion, missing
  times, concentration-independent selection, upstream sampling order, tie
  determinism, measured-flow gaps and first-series/conflict handling.
- `uv run ruff check .`: all checks passed.
- `uv run python scripts/verify_doc_river_sampling_resolution_v1.py`: passed.
  Re-extracts raw DOC, reconciles monthly truth, replays every triplet and daily
  flow join, perturbs DOC without changing sampling selection, recomputes all
  matched populations and 5,000-draw system-bootstrap summaries, and checks
  the full monthly baseline against the previous signal-mechanism study.
- `uv run python scripts/audit_artifacts.py --verify`: exit 0. Historical audit
  reports 83 parquet-only verified predictions, the existing kriging
  zero-coverage case and the already excluded G0 conflict. Historical
  no-sidecar status remains explicitly reported by that audit.
- Three English and three Chinese figures, each PNG/PDF, were generated and
  visually inspected. Repairs moved overlapping legend/annotation text and
  labelled absent daily-flow series. DOC points remain discrete observations;
  daily-flow lines have missing-day breaks. Real maps reuse measured station
  positions and actual cached NHDPlus channel geometries.
- Raw monthly means reproduce the frozen DOC values within 0.0000031 mg/L.
  All 3,026 connection-months, 59 pairs, 22 receivers and 11 systems remain
  accounted for. Every within-cut comparison uses identical pairs and months;
  single-system sampling cuts receive no population confidence interval.

The study does not refit a prediction model or revise earlier endpoints.
