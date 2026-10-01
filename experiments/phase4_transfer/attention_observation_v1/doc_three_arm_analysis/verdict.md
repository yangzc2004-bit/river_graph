# Three-arm attention comparison

Analyte: doc  
Mask: e2a_strict  
Seeds: 42, 43, 44

The comparison uses the same query mask and seed identities for all three
arms. The compact estimate is the unweighted mean of the three seed metrics;
three_arm_paired_deltas.csv retains the per-seed paired differences.

| arm | MAE mean | RMSE mean | R2 mean | Q90 MAE mean |
| --- | ---: | ---: | ---: | ---: |
| same-month upstream | 0.986783 | 1.61691 | 0.296006 | 6.15751 |
| same-month plus river lag | 0.984607 | 1.61419 | 0.298337 | 6.1658 |
| no-message control | 0.986764 | 1.61752 | 0.295525 | 6.15467 |

Mean MAE gain of river lag versus same-month upstream: **0.220%**.

Mean MAE gain of river lag versus no-message: **0.224%**.

Mean MAE gain of same-month upstream versus no-message: **0.004%**.

Positive gain means lower MAE. These are paired seed summaries; inspect seed
directions, query counts, and the run provenance before assigning a transport
interpretation. Q90 values with fewer than 20 test cells are descriptive only.
