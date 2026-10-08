# Validation record

Date: 2026-10-08

## Scientific calculations

- The mainstem and shortest-route analyses each retain 32 receiving networks
  and all 6,912 controlled scenarios, with 89 and 88 cropped source paths,
  respectively. The fixed
  high-coverage subset contains 12 receivers in five overlapping-catchment
  systems; source paths are not treated as independent field replicates.
- Both final verifiers replay the source geometry, scenario tables, constructed
  waves, summaries and 5,000-draw complete-system bootstrap. Source ledgers,
  configuration, execution snapshots and English/Chinese figure receipts agree.
- Independent frequency-domain integration checks all 32 receiving networks at
  the frozen working point. Maximum absolute discrepancy is 6.48e-13 for both
  route definitions. The covariance tests also cover equal and nearly equal
  source-correlation and memory times.
- Independent concatenation and resampling of entire catchment systems
  reproduces the broad-minus-elongated difference and its interval: mainstem
  -0.02245429 [-0.06900100, +0.02553018]; shortest route -0.02080935
  [-0.06627433, +0.02653373]. These are controlled arithmetic SD differences,
  distinct from the preceding observed geometric-mean form contrast.
- The path/common-memory log decomposition closes to within 2.22e-16.
- The passive alignment example uses actual path lengths but constructed source
  phases. Its variance/amplitude reference is the same instantaneous source
  mixture. No travel speed, residence time or DOC loss is estimated from field
  observations.

## Tests and repository checks

- All 11 new tests passed, including causal memory allocation, source
  independence, time-unit and common-translation invariance, short/slow limits,
  spectral checks, passive phase alignment and shared-path geometry.
- Full repository suite: 1,310 passed, 2 skipped, 8 warnings, 43.69 seconds.
- Ruff: all checks passed.
- Historical artifact audit: exit 0. Its existing missing-sidecar and absent
  historical dataset limitations remain; this analysis does not repair or
  relabel those historical predictions.
- The full suite preceded the final figure-layout and CSV identifier-type
  repairs. Both complete final analysis verifiers subsequently passed. No
  scientific formula or result was changed by those repairs.

## Repairs and visual review

- The first full replay failed on CSV metadata types for the two illustrative
  station identifiers. Reading those fields explicitly as strings repaired the
  round trip without changing numerical results. Both failed verification logs
  are retained alongside the successful final logs.
- Initial illustration legends covered parts of the waves. Legends were moved
  below the panels, then unnecessary whitespace was reduced. All eight final
  PNGs were actually inspected: two figures, two languages, two route versions.
  Their PDF counterparts use the same Matplotlib canvas and were not separately
  inspected.
- English and Chinese source receipts identify the source tables, geometry
  sample, controlled inputs, reference denominator and bootstrap unit.

## Scope

This version performs mechanism analysis only. It launches no model training,
adds no field measurements and fits no transport parameters to observed DOC.
Previous observational results and protocols are preserved. The next research
step is to measure source synchronization from actual tributary records while
keeping river form and path arrangement central.
