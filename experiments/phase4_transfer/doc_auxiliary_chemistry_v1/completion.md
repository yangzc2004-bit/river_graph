# Auxiliary-chemistry DOC experiment: completed

Twenty-seven native neural corrections and twenty-seven matched ExtraTrees
probes completed across three station partitions and three seeds. The retained
DOC backbone, ecological memory and support representation were fixed.
No-auxiliary, mask-only and measured-chemistry modes share the same actual
pH-or-conductance availability gate and exact parent fallback.

## Results

MAE is mg/L, seed-averaged within partition, then equally averaged across
partitions. Positive K uses the unchanged retrospective support protocol.

| Model | K0 MAE | K5 MAE | K5 Q90 MAE |
|---|---:|---:|---:|
| Retained integrated model | 1.803538 | 1.564984 | 6.586994 |
| Neural no auxiliary, integrated | 1.802007 | 1.562256 | 6.583055 |
| Neural masks, integrated | 1.799529 | 1.564502 | 6.597386 |
| Neural chemistry, integrated | 1.792258 | 1.571309 | 6.609833 |
| Tree no auxiliary | 1.847481 | 1.577596 | 6.648477 |
| Tree masks | 1.851629 | 1.579903 | 6.666255 |
| Tree chemistry | 1.802261 | 1.537905 | 6.509893 |

Measured chemistry lowers tree K0/K5 MAE by2.448%/2.516% against the matched
no-auxiliary control. Paired95% station-bootstrap intervals for the absolute
changes are[-.084230,-.010683] and[-.056866,-.024888]. All three partitions and
all nine packages improve;116 of172 unique stations improve at K5. K5 Q90
error falls by.138584, interval[-.225612,-.045865]. Availability flags alone
do not reproduce the chemistry-tree gain.

The current linear neural branch has a different result. Integrated K0 MAE
falls0.625% against the retained model, but its interval crosses zero. K5 MAE
is0.404% worse, also unresolved. K0 Q90 error falls.074084,
interval[-.127043,-.019321], with recall+.639 percentage points and false-high
rate+.0765 points. K5 does not preserve the tail gain. The integrated chemical
neural model is worse than the chemical tree at K5 by.033405,
interval[.005082,.064418].

## Decision and next experiment

Retain the preceding integrated DOC model as the main model. Preserve the
chemical tree as a stronger reference for reconstruction when auxiliary
chemistry is known. Chemical values provide useful information, while the
tested linear neural correction is not an improved replacement.

The next focused experiment learns a tiny nonlinear pH/conductance embedding
and its interaction with the existing hidden state. It uses the same current
model and matched no-auxiliary/mask controls; no additional tree tuning or
larger training budget is indicated by these results.

## Scope and verification

Either auxiliary measurement is available for10355/10520 unique observed-DOC
query cells, but only41623/210907=19.735% of genuinely DOC-missing cells.
Monthly chemistry means may come from different sample instants. The result
therefore concerns reconstruction with known auxiliary observations.

All54 fits completed in308.09 summed seconds. Independent replay verified27
neural heads,27 trees,81 adapters and36 mixers; all468 query panels,
2006940 rows and144 retained reference panels match exactly. All2101302
full-grid rows and absent-chemistry fallback fields passed. Forest regeneration
has at most1.07e-13 roundoff; the declared numerical tolerance is1e-12.
Neural and saved final-query replay differences are zero.

The full suite has798 passed and2 skipped, with three existing warnings; ruff
passes. See analysis/findings.md, analysis/interpretation.md and
verification/replay_checks.json. Historical experiments and endpoints remain
unchanged. These station partitions remain development data.
