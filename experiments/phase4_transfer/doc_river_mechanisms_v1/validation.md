# Validation record

- Full uv-managed pytest suite: **1,066 passed, 2 skipped**, 40.73 seconds.
  Eight pre-existing numerical/Shapely warnings remain; no failed tests.
- `ruff check .`: passed.
- Historical `audit_artifacts.py --verify`: exit 0; historical scope unchanged.
- Mechanism verifier: passed, 786 source-file identities, 21,459 allowed source
  cells, 253 case summaries, 9,736 paired record rows and 189 pathway-edge
  concentration changes independently checked. These record rows overlap across
  combinations/weightings and are not independent ecological samples.
- Raw submonthly observations: 29,285 unique station-days, all inside allowed
  source station-months. Same-day matching is exact; no interpolation/date widening.
- Hidden DOC perturbation: allowed responses unchanged.
- Mixing arithmetic: native concentration and positive weights; float64 arithmetic
  used consistently, including originally float32 monthly flow inputs.
- Official metadata: all 357 station records returned, 342 have reported areas.
  The incorrect Hull Hollow receiver candidate is excluded from screened results.
  The original NHD-candidate analysis is retained separately.
- Bootstrap means: same receiver-equal estimand as point summaries; shared-source
  station components and HUC4 sensitivities alongside receiver resampling.
  One-unit intervals stay missing.
- Simulation: equal source flow and reach budget; all post-warmup outlet flows
  equal input flow; conservative and first-order-removal cases kept separate.
- Figures: five families, English and Chinese PNG/PDF exports. Every PNG family
  was visually inspected for readability, actual geometry, correct axes/units,
  missing intervals and simulation labels. No observed figure clipping or overlap.
- No neural models retrained; historical graph, classes and model products unchanged.
