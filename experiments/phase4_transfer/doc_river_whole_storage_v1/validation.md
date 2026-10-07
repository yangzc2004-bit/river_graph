# Validation — 2026-10-07

- Focused physical tests: **13 passed**. They cover contiguous waterbody
  grouping, segmentation invariance, distinct serial elements, noncontiguous
  repeated IDs, analytic Gaussian/exponential response, exact no-storage
  baseline, within-reach source positions, unit gain, causal budget, shortest
  secondary links, primary-link tie preference and invalid inputs.
- Complete replay: **297/297 instances**, 4,158 scenario rows recomputed from
  VAA and saved real routes, without the derived numerical cache. Descriptor
  identities, serial moments, unit gain, fixed path means, all class summaries,
  original matched-pair summaries, five-point sensitivity and figure receipts
  pass. No new DOC model was trained.
- Half-grid replay: three fixed geometry representatives plus the largest
  geometric serial-storage variance case; maximum peak difference 1.33e-5
  and central-duration difference 3.05e-6.
- Full pytest: **1,171 passed, 2 skipped, 8 existing warnings**. Warnings are
  the existing torch deprecation, extreme-interval overflow, empty-edge std,
  and degenerate Shapely envelope notices; no new test failure.
- `ruff check .`: **All checks passed**.
- `scripts/audit_artifacts.py --verify`: **exit 0**. Historical qualifications
  remain unchanged: 83 parquet-only verifications, one recorded G0 overwrite
  conflict excluded by the audit, one zero-coverage artifact, and 85 historical
  predictions without sidecars. These are not upgraded to full mask/identity
  validation by this follow-up.
- Both English and Chinese PNG figure sets were opened and inspected.
  Response panels and class-effect panels use common axes; map panels retain
  truthful native geometry with independent physical scale bars. Negative
  peak reductions are explicitly identified as peak increases.
- Prior classification, DOC predictions, endpoints and model results were
  preserved. New source/product receipts and code snapshots accompany this
  study; unrelated pending model-development changes are outside its commit.
