# Validation record

- Analysis and 5,000-draw bootstrap replay: passed. All source tables, whole-network
  routing, tributary routing, spatial-resolution sensitivity, representative
  traces and summary tables reproduce.
- 305 execution input files checked. Cohort: 297 receiving stations, 295 distinct
  receiving COMIDs; repeats retain the fixed station-weighted cohort and fall in
  the same HUC4 bootstrap blocks.
- Primary cases: 891 whole-network routes and 10,890 two-tributary scenarios.
  Additional cases: 297 reach-distribution sensitivities and 121 arrival-alignment
  diagnostics.
- Conservative integrated anomaly load, unit flow and constant concentration:
  preserved. Mean delay is preserved by path contraction and within-reach
  redistribution. Uniform-speed, uniform-rate junction partition invariants
  reproduce to tolerance 1e−12.
- Maximum paired-network peak difference at half time step: 0.0002618. Maximum
  arrival-alignment peak error: 0.0002628. Both are far smaller than the main
  elongated–broad peak contrast of 0.0496.
- Tests: **1,099 passed, 2 skipped**, 8 existing warnings; full run 44.45 s.
- `ruff check .`: passed.
- Historical `audit_artifacts.py --verify`: exit 0. This remains a historical
  parquet-only check: 83 metric artifacts verified, the known corrupt G0 conflict
  retained/excluded under the existing policy, and partial kriging coverage
  reported. It does not upgrade the old missing sidecars or replace this study's
  replay verifier.
- All six English/Chinese PNG figures inspected. Legends were moved away from
  pulse curves, then figures and their source manifests regenerated. PDF versions
  are exports of the same Matplotlib figures; the visual inspection used PNGs.
- No model training, hidden-test model tuning, old result replacement or raw
  cache modification in this study. Imposed velocities and processing rates are
  documented scenarios, not fitted physical estimates.

The full replay is `uv run python scripts/verify_doc_river_routing_mechanisms_v1.py`.
The final figure-manifest check was repeated after the layout-only revision;
analysis arrays and bootstrap tables were unchanged.
