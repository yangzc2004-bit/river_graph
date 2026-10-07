# Executed validation

- Source-placement verifier: passed, with 371 source-file identities,
  1,087,269 landscape reaches and independent recomputation of 357 station
  source summaries.
- The allowed source-cell union is 21,459. Perturbing every other DOC cell
  leaves all response summaries unchanged.
- All four diagnostic models score the same stations. Each station and HUC4
  appears in exactly one held-out fold for each population.
- Unit coverage includes directed midpoint routing, secondary links,
  deduplicated edges, source placement at fixed amount, missing land cover,
  undefined zero-source position, riparian population matching and invalid
  input handling.
- Full repository pytest: **1,071 passed, 2 skipped**, eight existing warnings.
- Ruff: all checks passed.
- Historical `scripts/audit_artifacts.py --verify`: exit 0. Its existing
  no-sidecar and missing historical dataset limitations remain unchanged.
- English and Chinese plots were actually viewed. A row-header overlap in the
  association plot was fixed; source-only coefficient bounds are now shared
  across periods within each response column. Real maps retain scale bars,
  source-percentage colour bars and outlet markers.

Catchment percentages must have at least 95% NLCD completeness. The EPA API
returned a server-side error for riparian completeness arguments; the successful
plain riparian request has valid area and class percentages but no raster
completeness field. This distinction remains recorded in the acquisition receipt.
Independent station-specific split-catchment basin requests failed and were
not used to claim completed mapping repairs.

No new training, historical prediction replacement or old graph modification
was performed. Only study-specific scripts, tests, scientific outputs and
execution snapshots belong to this commit.
