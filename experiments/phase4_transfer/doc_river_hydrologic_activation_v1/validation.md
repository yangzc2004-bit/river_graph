# Validation record

## Scientific calculations

The dedicated verifier passed. It reconstructed the 21,459 permitted source
station-month cells from the union of the split142/143/144 source-training
roles, rebuilt the monthly panel, and confirmed identical matched station
cohorts across geographic diagnostic arms.

Changing every non-source DOC value and every hidden hydro payload did not
change the analysis input panel. Missing discharge remained missing; the two
negative/reverse-flow source observations were explicitly retained and flagged
in the panel and excluded from the nonnegative log1p flow analysis.

Station responses were checked against independently fitted OLS with the full
nuisance-and-flow design, rather than only repeating the residualization code.
Maximum absolute coefficient differences were:

| Population | Stations | Monthly pairs | Maximum difference |
|---|---:|---:|---:|
| All source months | 210 | 13,890 | 5.09e-15 |
| Observed-temperature sensitivity | 209 | 13,571 | 6.33e-15 |
| Source months since 2009 | 49 | 4,939 | 9.23e-16 |

The five new unit tests cover visibility and source-cell isolation, negative
flow handling, coefficient recovery after nuisance removal, missing-flow and
sample-size exclusions, equal-station weighting under replicated month
series, and temperature confounding. Bootstrap uses 5,000 whole-HUC4 draws
for response moderation and direct class contrasts. Geographic descriptor
comparisons retain both station and HUC4 resampling results.

## Repository checks

- Full pytest: **1,076 passed, 2 skipped, 8 warnings**, 41.74 seconds.
- `ruff check .`: passed.
- Historical `scripts/audit_artifacts.py --verify`: exit 0. The known G0
  conflict was reported and excluded under the existing audit policy. The
  historical 85 predictions without sidecars remain explicitly identified;
  this analysis does not repair or reinterpret those historical records.

## Figure inspection

All three English and three Chinese PNGs were opened and visually checked:
planform flow responses, source-flow moderation, and geographic response
increment. Labels, intervals, legends and population notes are visible without
overlap or clipping. PDF companions were generated from the same plotting
calls. Recent-period moderation has an explicitly separate axis range, and
the geographic panel explicitly identifies its response-descriptor target.

## Interpretation and changes during execution

These are retrospective monthly source-station relationships, not new K0 DOC
predictions, external validation, instantaneous transport estimates or a
neural-model upgrade. Monthly discharge is aggregated from NWIS daily means
in cfs. The analysis distinguishes concentration response from DOC load.

A nullable metadata serialization repair and direct paired class contrasts
were added during execution, as recorded in `README.md`. Neither changed the
fitted station responses or selected the analysis cohort from its results.
All earlier experiment directories and neural training code were preserved.
