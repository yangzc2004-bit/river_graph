# Validation record

Date: 2026-10-07. All commands used the existing uv-managed `.venv` with
offline resolution and single-threaded BLAS. No neural training or calibrator
refitting was started.

## Analysis replay

- Analysis completed with 5,000 monitoring-system bootstrap draws.
- Verification replay passed: 59 connections, 22 receivers, 11 systems,
  3,026 connection-months and 1,092 distinct receiver-months.
- All 177 pulse scenarios conserve anomaly mass and steady concentration;
  removing the common trunk preserves peak and width exactly.
- All 12,104 fixed predictions and 17 CSV tables reproduce. The complete saved
  reference reproduces within 1e-10 mg/L. Six structural associations are kept.
- Hidden receiver DOC perturbations and later source-input changes do not
  alter the corresponding earlier predictions.
- Real map geometries are complete for all four selected corridors, with zero
  endpoint gaps. Both language versions were visually inspected; the compass
  and pulse annotations were repositioned to avoid overlap with station labels
  or data curves.

## Tests and historical artifacts

- `uv run pytest -q`: **1,136 passed, 2 skipped, 8 existing warnings**.
- `uv run pytest -q tests/test_river_monitored_footprint.py`: **7 passed**.
- `uv run ruff check .`: **all checks passed**.
- `uv run python scripts/audit_artifacts.py --verify`: **exit 0**. The historical
  audit retains 83 parquet-only verified records, the already documented G0
  conflict and the existing zero-coverage kriging record. It does not upgrade
  the identity status of old predictions that lack sidecars.

The initial geometry check exposed missing leading zeros in the official node
table's HUC column. Eight-digit zero padding restores HUC4 comparison against
the prior inventory; HUCs are read from `huc_cd`, never station prefixes. The
fix changes no cohort, river route or scientific endpoint.

Old results and freezes were preserved. This version adds only matched-footprint
analysis code, tests, figures and replayable study products. Unrelated working
tree changes and large historical caches are excluded from the commit.
