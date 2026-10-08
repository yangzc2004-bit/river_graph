# Validation record

Date: 2026-10-08

## Reproduction

- Primary and shortest-route analyses completed with 5,000 whole-system draws.
- Both `--full` verifiers replayed all 297 inventories, source frontiers, raw
  monthly reconciliation, dated selections, observed signals, class contrasts,
  bootstrap summaries and weekly case products. Exit 0 for each.
- Sources, generated tables and English/Chinese figure receipts agree.
- All twelve authored PNG figures were visually inspected. Dense channel layers
  are rasterized inside PDFs; text, gauges and remaining figure elements retain
  vector output. This reduces real-map PDFs from about 99 MB to about 1.5 MB
  without changing analyzed geometry or displayed channel locations.

## Tests

- New focused tests: 14 passed. They cover disjoint incremental catchments,
  metadata-only upstream/date selection, array-order consistency, overlap-system
  membership, multi-source covariance identities, hidden-cell exclusion,
  same-day activity selection and common-system form contrasts.
- Full pytest suite: 1,291 passed, two skipped, eight warnings.
- `ruff check .`: exit 0, all checks passed.
- Historical `audit_artifacts.py --verify`: exit 0. Its established historical
  caveats remain, including 85 legacy files with no sidecars and unavailable
  historical datasets. This audit does not provide identity evidence for the
  new observational tables; their source-ledger and full-replay checks do.

## Research scope

No model was trained. No hidden geographic/external query prediction was used
for selecting a mechanism. Historical model results, fixed river forms and
earlier routing/observed analyses remain intact. The existing unrelated tracked
workspace edits are excluded from this research commit.
