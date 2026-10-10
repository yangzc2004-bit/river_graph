# Verification record

Executed on 2026-10-10 using the uv-managed local environment.

| Check | Result |
|---|---|
| Full training | 9 packages, 45 river fits, 0 new forests |
| Consumed retained source identities | Matched original completion records |
| Full bank/query reconstruction | Exact for all nine packages |
| Saved prediction replay | Bitwise for all 45 branches |
| Fixed reference products | Original prediction rows retained exactly |
| No observed river support | Exact complete-base prediction |
| Silenced river values | Exact complete-base prediction |
| Query/donor-fold exclusion | Verified from source records and perturbation tests |
| Causal hydro/observation indexing | Covered by unit tests |
| Matched source support | Real/control validity identical |
| Primary estimand / interval | Independent recalculation agrees |
| Bootstrap | 5,000 paired whole-station draws |
| Figure inspection | Both figures viewed; comparison re-rendered and viewed after label fix |
| Ruff | All checks passed |
| Historical audit | Exit 0; known G0 conflict and old no-sidecar limitations retained |
| Initial full pytest | 1 failed, 1,437 passed, 2 skipped; missing archived geography forest |
| Full pytest after local-asset skip guard | 1,437 passed, 3 skipped, 8 existing warnings |

The additional skip is an explicit missing-local-model skip under the repository
convention. It does not waive a failing numerical test or alter a model. No
external archive asset was fetched. Historical audit generated its usual dated
inventory/verification files; those two new files are part of this run's audit
evidence. Unrelated historical NEON acquisition/analysis logs are not included.
