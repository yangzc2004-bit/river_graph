# Daily-hydrology fallback: completed spatial reconstruction comparison

Nine packages are complete using three station partitions and three seeds.
The monthly and daily neural experts are unchanged. A fixed availability router
uses daily predictions when at least one numerical daily-discharge descriptor
is valid and monthly predictions otherwise. Direct support adapters and
ecological integration are fitted on the existing source-validation episodes.

## Performance

MAE in mg/L; partition means have equal weight after seed averaging.

| Integrated model, fixed GRU support basis | K0 | K1 | K3 | K5 |
| --- | ---: | ---: | ---: | ---: |
| Matched monthly | 1.809588 | 1.777090 | 1.626636 | 1.588987 |
| Daily hydrology | 1.803538 | 1.755368 | 1.620090 | 1.564984 |
| Availability-preserving hybrid | 1.794386 | 1.748926 | 1.619420 | 1.571403 |
| Earlier ecological-affine reference | 1.809217 | 1.791696 | 1.622073 | 1.579905 |

The hybrid lowers K0 error by **0.51%** versus daily, with paired MAE difference
**−0.009152 mg/L [−0.020062, +0.000582]**. Its three partition means improve,
but the overall interval includes zero. Ordinary-DOC K0 error improves;
high-DOC Q90 error is essentially unchanged relative to daily.

At K5, hybrid error increases **0.41%** relative to daily:
**+0.006419 mg/L [+0.000291, +0.014868]**. Ecological/support calibration can
offset the benefit of protecting the native prediction. This also appeared on
source validation: integrated K5 MAE was 1.604602 for daily and 1.604714 for
hybrid. There is no new unified winner and no test-selected K-dependent route.
The daily integrated model remains the main K5 performance candidate.

## What the fallback fixes

There are 2,379 unique target station-months with no valid numerical daily
descriptor, across 57 stations. On these cells, hybrid native and direct K0
predictions equal the monthly expert exactly. Integrated K0 MAE changes from
**1.205614 to 1.164704** versus daily, a **3.39%** descriptive group reduction.
It remains above the monthly integrated group's 1.155796 because the hybrid's
global validation-selected ecological mixture can change its routed bases.

On the 8,141 all-valid target cells, native predictions retain the daily expert.
There are no partially valid cells among the fixed target queries; the
any-valid branch has contract coverage and one source-validation example.
Valid constant or zero discharge uses daily rather than triggering fallback.

The scientific lesson is that input preservation and final adapted performance
are different properties. A physically sensible absence rule improves the
unadapted missing-information behavior, but its interaction with station support
must also be measured. The source-validation comparison and all negative
comparisons are retained.

## Reproduction

```bash
uv run python scripts/run_ladder.py --experiment doc-daily-hydro-fallback-v1
uv run python scripts/verify_doc_daily_hydro_fallback_v1.py
uv run python scripts/analyze_doc_daily_hydro_fallback_v1.py
```

The independent replay verifies exact routing, all 720 model×K query panels,
72 direct adapter refits, 54 integrated mixer refits and unchanged parent
controls/references. No new neural or forest fit was needed. The full suite
passes 706 tests with two explicit skips; lint and historical artifact audit
also pass.

See `PRODUCTS.md`, `diagnostics/source_validation.md`, `analysis/findings.md`,
`analysis/availability_appendix.md` and `verification/replay_checks.json`.
These are development comparisons on previously examined station partitions;
K-positive support remains retrospective.
