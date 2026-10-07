# Validation

## Calculation and input checks

The dedicated verifier passed. It verifies source content, reconstructs the
same-HUC4, non-nested matches, and recomputes all paired DOC differences and
5,000-draw intervals. Changing DOC outcomes and path/sinuosity descriptors
does not change selected pairs. Stations are not reused within each contrast.
The primary contrast contains 22 pairs in 12 HUC4s; its maximum area ratio is
1.780. One-HUC4 intervals are stored as unavailable.

All catchment-distance profiles sum to one. The controlled routing scenario
preserves the integrated input anomaly for every network, and every pulse
value is finite. Coverage of routed area is effectively 100%. Different shape
classes receive the same forcing, and no land-cover-specific source weights
enter that calculation. Seven new unit tests cover response-independent
matching, geography/calipers, nesting exclusions, paired difference direction,
single-region intervals, conservative routing, scale invariance and explicit
unreachable-area handling.

The eight information-block models predict the same 297 stations, with each
station appearing in one held-out fold per arm and each HUC4 in only one fold.
Both native and log1p score estimates and intervals were independently
recomputed from saved prediction errors. This verifies calculations, not a
claim of independent ecological confirmation or physical causal identification.

## Repository checks

- Full pytest: **1,083 passed, 2 skipped, 8 warnings**, 42.19 seconds.
- Ruff: passed.
- Historical `audit_artifacts.py --verify`: exit 0. The existing G0 conflict
  is reported/excluded under the original policy; 85 historical no-sidecar
  records remain explicitly identified. No historical record was repaired.

## Visual checks

All three English and three Chinese PNGs were opened and inspected. One
percentage label overlapping the full-morphology interval was moved into a
separate label column. Class-mean routing curves and individual-network peaks
are labelled as different statistics. Labels, units, legends, intervals and
simulation notices are visible in the final figures. PNG/PDF companions use
the same plotting calls; English and Chinese views preserve the same data.

## Preserved scope

Only this new morphology study, its scripts, library and tests were added.
No neural training was performed. Old experiments, graphs, model predictions,
paper endpoints and unrelated local edits were preserved. The observed
comparison uses the source-training union 142/143/144; existing geographic
confirmation and independent external model-test outputs were not read.

Implementation corrections and the added matched geometry companion are
recorded in `README.md`. Neither changes the selected scientific question or
replaces a primary response after viewing results.
