# Bounded chemical residual interpolation: completed source-only pilot

Nine packages retain the accepted chemical-coordinate model and add a normalized
RBF interpolation of its remaining support residual. The neural backbone,
chemical decoder, forests, linear calibrators and ecological mixture are fixed.
The interpolation has an exactzero control and a matched availability-only
coordinate control. It evaluates source-validation stations only.

## Result and model decision

The extra interpolator does not improve transfer of calibration settings across
validation stations. Retain the completed chemical-coordinate parent as the
model candidate; omit this added kernel from later confirmation.

| Pipeline | K | Parent validation MAE | Kernel conditional-CV MAE | Error increase |
|---|---:|---:|---:|---:|
| Direct neural | 3 | 1.658708 | 1.662812 | 0.247% |
| Direct neural | 5 | 1.603627 | 1.604917 | 0.080% |
| Neural plus ecology | 3 | 1.651356 | 1.656726 | 0.325% |
| Neural plus ecology | 5 | 1.591066 | 1.592138 | 0.067% |
| Chemical tree | 3 | 1.673123 | 1.678272 | 0.308% |
| Chemical tree | 5 | 1.603398 | 1.607420 | 0.251% |

These are validation-set scores, not the preceding study's outer-station
scores. Five station folds choose eta/bandwidth on other-fold active queries
and score held-fold queries. The parent itself was already selected on all
validation stations; this evaluates the new settings conditional on that fixed
parent and does not make the complete model OOF.

The integrated neural kernel worsens all three partitions at K3. At K5 only
one partition and one of nine seed fits improve. Chemistry-kernel conditional
CV is also worse than the availability-only added kernel. In contrast,
parameters tuned and evaluated on all validation queries produce apparently
better MAE1.645588/1.584281 at K3/K5. That contrast shows the extra operator's
limited station-to-station stability. The full-validation scores are tuning
diagnostics, not evidence to promote the kernel.

This negative pilot does not reverse the preceding chemical-coordinate study's
K3 gain. The accepted source-selected neural candidate retains outer-station
development K0/1/3/5 MAE1.764947/1.738600/1.595851/1.556939. Its next comparison
should retrain the fixed candidate and references on new station assignments.

## Execution and reproduction

No neural network, forest or linear adapter was fitted. All9 source-validation
packages completed. Independent replay verifies18 source-distance inventories,
108 preclip all-K linear remainder cases,1404 candidate-score rows,69,984
station-candidate rows,216 selected kernel choices and324 validation panels.
No target labels were extracted or target outcomes evaluated. K0/K1 and
zero-correction/inactive rows retain the parent exactly.

The separate reproducible CV audit verifies540 parameter choices,5832 held
station records,108 run summaries and36 overall/active contrasts. Held station
records match exactly; choice/contrast differences are below1e-15. Its report
is verification/conditional_cv_checks.json; the numerical replay is preserved.

Source-validation queries contain7197 unique station-months at140 unique
stations, with8047 partition-cell occurrences. Source-distance units are
feature-only and finite; active donor counts are effectively3/5 at K3/K5,
so the operator has enough support to act. The full suite has822 passed and
2 skipped; targeted final kernel/basis tests and Ruff pass. The historical
artifact audit and preceding independent prediction replay also pass.

See analysis/interpretation.md for all contrasts, station gains/harms,
correction magnitudes and the tuning-versus-CV figure. Products and commands
are described in PRODUCTS.md. This completed experiment has no new target
prediction product and no new full-grid prediction.
