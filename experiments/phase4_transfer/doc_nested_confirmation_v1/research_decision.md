# Separate chemical station calibration: confirmation decision

Date: 2026-10-04. All nine fresh fitted packages and saved-state replays are complete.

## Decision

The separate chemical increment has a small, reproducible benefit over its
legacy-calibrated parent at K5, but this experiment does not establish an
improvement over the complete current joint procedure. Keep the existing joint
procedure as the current reference and retain the separate increment as a
research candidate. Do not assemble a K3 joint / K5 separate winner from these
target results.

This result narrows the next modeling problem: chemical information adds value,
but estimating a station-specific chemical response from a few observations is
still too weak to produce a meaningful overall upgrade.

## Complete reconstruction curves

DOC MAE in mg/L; seeds are averaged within each partition, then partitions
receive equal weight.

| Procedure | K0 | K1 | K3 | K5 |
|---|---:|---:|---:|---:|
| General model | 1.798880 | 1.765514 | 1.624407 | 1.591951 |
| Current joint chemical procedure | 1.779435 | 1.743581 | 1.612311 | 1.584968 |
| Chemical decoder with legacy calibration | 1.779435 | 1.743581 | 1.613529 | 1.586328 |
| Chemistry-aware trees | 1.800454 | 1.784413 | 1.616055 | 1.578438 |
| Separate chemical increment | 1.779435 | 1.743581 | 1.615206 | 1.582420 |
| Separate availability-only increment | 1.779435 | 1.743581 | 1.614418 | 1.587272 |

K0/K1 are exactly unchanged relative to the legacy chemistry parent by design.
The separate procedure's K5 RMSE is 3.716457 mg/L and mean partition R² is
0.502420. These are means of run metrics, not metrics recomputed after pooling
all predictions. Native MAE decreases with increasing support for every curve.

## Primary comparison and incremental evidence

Positive relative gain means lower MAE for the separate increment. Intervals
are unadjusted 95% percentile intervals from 5,000 paired whole-station
bootstrap draws; station multiplicities are shared across repeated partitions.

| Separate increment versus | K | Relative MAE gain [95% CI], % | Improved partitions / fitted packages |
|---|---:|---:|---:|
| Current joint procedure | 5 | +0.161 [−0.013, +0.332] | 3/3; 7/9 |
| Current joint procedure | 3 | −0.180 [−0.671, +0.272] | 1/3; 5/9 |
| Legacy chemistry calibration | 5 | +0.246 [+0.080, +0.415] | 3/3; 8/9 |
| Availability-only increment | 5 | +0.306 [+0.110, +0.527] | 3/3; 7/9 |
| General model | 5 | +0.599 [+0.198, +1.005] | 3/3; 8/9 |
| Chemistry-aware trees | 5 | −0.252 [−1.352, +0.889] | 1/3; 6/9 |

The primary K5 difference is −0.002548 mg/L [−0.005219, +0.000216]. Its positive
direction in all three partitions is useful, but the small effect and
crossing-zero interval leave the full-model improvement unresolved. At K3,
MAE is higher by 0.002895 mg/L on average, with two partitions favoring the
joint procedure. The earlier conditional-validation gains of 0.635%/1.234%
over legacy calibration did not reproduce in magnitude: fresh held-station
gains are −0.104%/+0.246% at K3/K5.

Chemistry still carries incremental predictive information at K5: the separate
value-based correction improves on both its unchanged parent and the matched
availability-only correction. That specific comparison cannot be expanded to
a claim that the new complete procedure beats the current joint procedure or
chemistry trees. All ten fixed comparisons remain in `analysis/paired_effects.csv`.

## High DOC and station heterogeneity

At K5 the separate procedure's Q90 MAE is 6.545656 mg/L, compared with 6.567971
for the joint procedure, 6.579637 for legacy chemistry calibration, 6.623093
for the general model and 6.598785 for chemistry trees. The paired tail error
reduction is supported against legacy calibration and the general model;
against joint calibration and chemistry trees, its interval crosses zero.
Q90 thresholds come from each partition's source-training DOC.

High-DOC underprediction remains substantial: Q90 signed bias is −5.511197
mg/L. Q90 recall is 0.622457 versus 0.617410 for the joint procedure, while
false-positive rate rises from 0.020996 to 0.021512. The modest recall change
does not resolve the tail reconstruction problem.

Across the 179 unique evaluated stations, K5 improves on the joint procedure
at 113 stations and worsens at 66. The five largest contributors account for
27.96% of positive station gain and 47.18% of station harm. At K3, a larger
harm contribution outweighs improvements at a majority of individual stations.
Station-level counts and cell-weighted mean effects therefore answer different
questions; both are retained.

## Chemistry availability and evaluation scope

The final queries contain 10,233 unique station-months across 179 stations,
with 11,886 partition-specific cell occurrences. Of these unique queries,
10,221 (99.88%) have at least one auxiliary chemical observation and 10,131
have both. Groups with only pH, only conductance or neither contain just
55, 35 and 12 unique cells, respectively. Their accuracy estimates are sparse;
availability-specific tail flags remain visible in the tables.

The full 233,478-cell grid has 210,907 genuinely missing DOC cells. Only
41,623 (19.74%) of those cells have at least one auxiliary chemical observation;
169,284 have neither. The existing absent-chemistry fallback remains necessary.
Accuracy on chemistry-rich observed queries does not establish accuracy on
the large missing-DOC population without chemistry.

This is a complete fresh source refit under ST357 station-role assignments
342/343/344, using training seeds 42/43/44. It is a same-cohort replication,
not an external-basin validation. Support is retrospective and may postdate
queries. The retained spatial encoder uses its empty-edge self path. Forest
source predictions are station-blocked OOF; the complete neural source pipeline
is not OOF. Conditional adapter validation is kept separate from final target
evaluation.

## Next research priority

Develop population-guided chemical shrinkage within the existing architecture.
Estimate a transferable chemical response from source-station episodes, then
let a target station's few support observations make a limited update. Study
how that update should depend on the number and diversity of chemically active
supports, including partial-chemistry cases. This directly addresses the weak
K3 response and heterogeneous station harm without adding another unconstrained
support coefficient block.

In parallel, make the persistent high-DOC negative bias the main reconstruction
diagnostic. The present tail errors are much larger than the overall errors,
so another hundredth-percent calibration improvement is unlikely to be the
most useful performance target. Any new tail or population-prior method should
first be developed using source roles, with the completed target comparison
preserved as evidence for this fixed recipe.

This confirmation is closed. No model was retuned from its target outcomes;
no new training is launched as part of closeout. Pause its completion monitor
after notifying the user.

## Reproduction and checks

```bash
uv run python scripts/verify_doc_nested_confirmation_v1.py
uv run python scripts/analyze_doc_nested_confirmation_v1.py --bootstrap-draws 5000
uv run python scripts/plot_doc_nested_confirmation_v1.py
uv run pytest
uv run ruff check .
uv run python scripts/audit_artifacts.py --verify
```

All nine saved-state replays passed. Direct parquet calculations independently
reproduced all 24 MAE and Q90-MAE curve points. The generated figure was
visually inspected; its fixed curves and paired intervals agree with the
tables. Final checks: 889 tests passed, 2 skipped; Ruff passed; historical
artifact audit exited 0 with its previously documented historical limitations.
Training took 63.10 minutes in total.

See `analysis/findings.md`, `verification/`, and `figures/` for complete
results. Small fitted states, source snapshots, final six-curve predictions
and sidecars are retained with this study. Large fitted forests, intermediate
features and full-grid caches remain local; full saved-state replay requires
those local artifacts or their regeneration.
