# Conditional expert route pilot

The route pilot compares the explicit RF-context spatial expert with the
local causal residual expert under four missingness families. The first two
rows use the new matched-budget three-seed runs; the latter two reuse the
completed three-seed K1 pilot under the same endpoint.

| missingness family | RF-context MAE | local residual MAE | preferred expert | relative gain |
|---|---:|---:|---|---:|
| Random point (`e1_r20_seed42`) | 1.4518 | 1.4935 | RF-context | 2.87% |
| Partial time (`e2b_partial`) | 1.0610 | 0.8094 | local residual | 23.71% |
| Strict time (`e2a_strict`) | 1.0890 | 0.9816 | local residual | 9.86% |
| Spatial station holdout (`e3_spatial_seed42`) | 2.6838 | 2.9583 | RF-context | 10.23% |

The preferred expert changes with the observation design. The temporal expert
is useful when the target station has a usable history but the current target
month is hidden. The RF-context expert is stronger when the missingness is
spatial or consists of scattered points, where current-month network context
is more informative than a learned local residual.

This table is a model-selection pilot, not a pooled test-set claim. The next
run should evaluate a fixed router using the missingness family known before
prediction, with all four families evaluated by the same script and endpoint.

## Validation-selected route

The route analyzer selected the expert from validation cells only:

| family | selected expert | test MAE |
|---|---|---:|
| Random point | RF-context | 1.45185 |
| Partial time | local residual | 0.80936 |
| Strict time | local residual | 0.98164 |
| Spatial station holdout | RF-context | 2.68376 |

Across the four query collections, the routed system had pooled MAE 1.53607,
compared with 1.59818 for always using RF-context and 1.61601 for always using
the local residual expert. The pooled number is descriptive because the four
missingness families contain different query cells; the family-specific values
are the primary evidence.
