# Phase-3B evaluation (empirical validation calibration — not conformal)

Generated 2026-09-23T17:03:43.750246+00:00 by `scripts/run3b_fullgrid.py` (one-command recompute with --eval-only).

## Coverage on hidden test cells (station-clustered bootstrap)

| tool      | mask              | family   | subset   |   coverage |   ci95_lo |   ci95_hi |   n_test |   tail_mae |   small_sample_flag |
|:----------|:------------------|:---------|:---------|-----------:|----------:|----------:|---------:|-----------:|--------------------:|
| H2X       | e1_r20_seed42     | E1       | test     |      0.823 |     0.803 |     0.841 |     4514 |        nan |                 nan |
| H2X_nomsg | e1_r20_seed42     | E1       | test     |      0.766 |     0.740 |     0.791 |     4514 |        nan |                 nan |
| eco_RF    | e1_r20_seed42     | E1       | test     |      0.595 |     0.578 |     0.611 |     4514 |        nan |                 nan |
| H2X       | e1_r20_seed43     | E1       | test     |      0.854 |     0.833 |     0.873 |     4514 |        nan |                 nan |
| H2X_nomsg | e1_r20_seed43     | E1       | test     |      0.785 |     0.760 |     0.808 |     4514 |        nan |                 nan |
| eco_RF    | e1_r20_seed43     | E1       | test     |      0.583 |     0.564 |     0.602 |     4514 |        nan |                 nan |
| H2X       | e1_r20_seed44     | E1       | test     |      0.791 |     0.768 |     0.813 |     4514 |        nan |                 nan |
| H2X_nomsg | e1_r20_seed44     | E1       | test     |      0.773 |     0.750 |     0.795 |     4514 |        nan |                 nan |
| eco_RF    | e1_r20_seed44     | E1       | test     |      0.573 |     0.555 |     0.590 |     4514 |        nan |                 nan |
| H2X       | e2a_strict        | E2a      | test     |      0.923 |     0.888 |     0.951 |     2223 |        nan |                 nan |
| H2X_nomsg | e2a_strict        | E2a      | test     |      0.853 |     0.807 |     0.891 |     2223 |        nan |                 nan |
| eco_RF    | e2a_strict        | E2a      | test     |      0.666 |     0.606 |     0.722 |     2223 |        nan |                 nan |
| H2X       | e2b_partial       | E2b      | test     |      0.925 |     0.891 |     0.952 |     1778 |        nan |                 nan |
| H2X_nomsg | e2b_partial       | E2b      | test     |      0.854 |     0.810 |     0.892 |     1778 |        nan |                 nan |
| eco_RF    | e2b_partial       | E2b      | test     |      0.635 |     0.584 |     0.686 |     1778 |        nan |                 nan |
| H2X       | e3_spatial_seed42 | E3       | test     |      0.830 |     0.770 |     0.874 |     2531 |        nan |                 nan |
| H2X_nomsg | e3_spatial_seed42 | E3       | test     |      0.762 |     0.664 |     0.831 |     2531 |        nan |                 nan |
| eco_RF    | e3_spatial_seed42 | E3       | test     |      0.534 |     0.425 |     0.616 |     2531 |        nan |                 nan |
| H2X       | e3_spatial_seed43 | E3       | test     |      0.805 |     0.734 |     0.884 |     2353 |        nan |                 nan |
| H2X_nomsg | e3_spatial_seed43 | E3       | test     |      0.770 |     0.711 |     0.818 |     2353 |        nan |                 nan |
| eco_RF    | e3_spatial_seed43 | E3       | test     |      0.597 |     0.522 |     0.664 |     2353 |        nan |                 nan |
| H2X       | e3_spatial_seed44 | E3       | test     |      0.793 |     0.721 |     0.851 |     2649 |        nan |                 nan |
| H2X_nomsg | e3_spatial_seed44 | E3       | test     |      0.770 |     0.692 |     0.828 |     2649 |        nan |                 nan |
| eco_RF    | e3_spatial_seed44 | E3       | test     |      0.665 |     0.583 |     0.729 |     2649 |        nan |                 nan |

## High-DOC tail (3C rules: Q90 primary; n<20 = unstable)

| tool      | mask              | family   | subset   |   coverage |   ci95_lo |   ci95_hi |   n_test |   tail_mae | small_sample_flag   |
|:----------|:------------------|:---------|:---------|-----------:|----------:|----------:|---------:|-----------:|:--------------------|
| H2X       | e1_r20_seed42     | E1       | top5     |      0.658 |       nan |       nan |      234 |     10.534 | False               |
| H2X       | e1_r20_seed42     | E1       | top10    |      0.659 |       nan |       nan |      431 |      7.398 | False               |
| H2X_nomsg | e1_r20_seed42     | E1       | top5     |      0.513 |       nan |       nan |      234 |     10.892 | False               |
| H2X_nomsg | e1_r20_seed42     | E1       | top10    |      0.522 |       nan |       nan |      431 |      7.784 | False               |
| eco_RF    | e1_r20_seed42     | E1       | top5     |      0.402 |       nan |       nan |      234 |      9.573 | False               |
| eco_RF    | e1_r20_seed42     | E1       | top10    |      0.445 |       nan |       nan |      431 |      6.634 | False               |
| H2X       | e1_r20_seed43     | E1       | top5     |      0.574 |       nan |       nan |      235 |      8.904 | False               |
| H2X       | e1_r20_seed43     | E1       | top10    |      0.655 |       nan |       nan |      449 |      6.268 | False               |
| H2X_nomsg | e1_r20_seed43     | E1       | top5     |      0.515 |       nan |       nan |      235 |      9.017 | False               |
| H2X_nomsg | e1_r20_seed43     | E1       | top10    |      0.595 |       nan |       nan |      449 |      6.353 | False               |
| eco_RF    | e1_r20_seed43     | E1       | top5     |      0.374 |       nan |       nan |      235 |      7.744 | False               |
| eco_RF    | e1_r20_seed43     | E1       | top10    |      0.452 |       nan |       nan |      449 |      5.435 | False               |
| H2X       | e1_r20_seed44     | E1       | top5     |      0.507 |       nan |       nan |      223 |     10.451 | False               |
| H2X       | e1_r20_seed44     | E1       | top10    |      0.576 |       nan |       nan |      434 |      7.167 | False               |
| H2X_nomsg | e1_r20_seed44     | E1       | top5     |      0.462 |       nan |       nan |      223 |     11.053 | False               |
| H2X_nomsg | e1_r20_seed44     | E1       | top10    |      0.535 |       nan |       nan |      434 |      7.433 | False               |
| eco_RF    | e1_r20_seed44     | E1       | top5     |      0.381 |       nan |       nan |      223 |      9.547 | False               |
| eco_RF    | e1_r20_seed44     | E1       | top10    |      0.410 |       nan |       nan |      434 |      6.545 | False               |
| H2X       | e2a_strict        | E2a      | top5     |      0.000 |       nan |       nan |        2 |     15.146 | True                |
| H2X       | e2a_strict        | E2a      | top10    |      0.333 |       nan |       nan |       12 |      6.936 | True                |
| H2X_nomsg | e2a_strict        | E2a      | top5     |      0.500 |       nan |       nan |        2 |     14.203 | True                |
| H2X_nomsg | e2a_strict        | E2a      | top10    |      0.333 |       nan |       nan |       12 |      7.174 | True                |
| eco_RF    | e2a_strict        | E2a      | top5     |      0.000 |       nan |       nan |        2 |     16.422 | True                |
| eco_RF    | e2a_strict        | E2a      | top10    |      0.417 |       nan |       nan |       12 |      6.622 | True                |
| H2X       | e2b_partial       | E2b      | top5     |      0.000 |       nan |       nan |        2 |     15.121 | True                |
| H2X       | e2b_partial       | E2b      | top10    |      0.286 |       nan |       nan |        7 |      7.708 | True                |
| H2X_nomsg | e2b_partial       | E2b      | top5     |      0.500 |       nan |       nan |        2 |     14.203 | True                |
| H2X_nomsg | e2b_partial       | E2b      | top10    |      0.429 |       nan |       nan |        7 |      7.729 | True                |
| eco_RF    | e2b_partial       | E2b      | top5     |      0.000 |       nan |       nan |        2 |     13.536 | True                |
| eco_RF    | e2b_partial       | E2b      | top10    |      0.429 |       nan |       nan |        7 |      6.358 | True                |
| H2X       | e3_spatial_seed42 | E3       | top5     |      0.663 |       nan |       nan |      347 |      8.511 | False               |
| H2X       | e3_spatial_seed42 | E3       | top10    |      0.706 |       nan |       nan |      510 |      6.807 | False               |
| H2X_nomsg | e3_spatial_seed42 | E3       | top5     |      0.588 |       nan |       nan |      347 |      9.111 | False               |
| H2X_nomsg | e3_spatial_seed42 | E3       | top10    |      0.635 |       nan |       nan |      510 |      7.154 | False               |
| eco_RF    | e3_spatial_seed42 | E3       | top5     |      0.380 |       nan |       nan |      347 |      9.567 | False               |
| eco_RF    | e3_spatial_seed42 | E3       | top10    |      0.425 |       nan |       nan |      510 |      7.573 | False               |
| H2X       | e3_spatial_seed43 | E3       | top5     |      0.750 |       nan |       nan |      228 |      6.813 | False               |
| H2X       | e3_spatial_seed43 | E3       | top10    |      0.688 |       nan |       nan |      378 |      6.013 | False               |
| H2X_nomsg | e3_spatial_seed43 | E3       | top5     |      0.618 |       nan |       nan |      228 |      7.343 | False               |
| H2X_nomsg | e3_spatial_seed43 | E3       | top10    |      0.608 |       nan |       nan |      378 |      6.182 | False               |
| eco_RF    | e3_spatial_seed43 | E3       | top5     |      0.390 |       nan |       nan |      228 |      7.587 | False               |
| eco_RF    | e3_spatial_seed43 | E3       | top10    |      0.397 |       nan |       nan |      378 |      5.983 | False               |
| H2X       | e3_spatial_seed44 | E3       | top5     |      0.707 |       nan |       nan |      328 |      6.884 | False               |
| H2X       | e3_spatial_seed44 | E3       | top10    |      0.681 |       nan |       nan |      496 |      6.115 | False               |
| H2X_nomsg | e3_spatial_seed44 | E3       | top5     |      0.637 |       nan |       nan |      328 |      7.051 | False               |
| H2X_nomsg | e3_spatial_seed44 | E3       | top10    |      0.623 |       nan |       nan |      496 |      6.210 | False               |
| eco_RF    | e3_spatial_seed44 | E3       | top5     |      0.457 |       nan |       nan |      328 |      7.004 | False               |
| eco_RF    | e3_spatial_seed44 | E3       | top10    |      0.500 |       nan |       nan |      496 |      5.796 | False               |

## Monotonicity (pre-frozen rule, train-derived tertiles)

| tool      | mask              | family   | covariate          | ok    | reason                      |   direction_correct |   strongly_reversed |   bottom_mean |   top_mean | edges                                    |
|:----------|:------------------|:---------|:-------------------|:------|:----------------------------|--------------------:|--------------------:|--------------:|-----------:|:-----------------------------------------|
| H2X       | e1_r20_seed42     | E1       | network_distance   |       | empty tertile on test cells |             nan     |             nan     |       nan     |    nan     | nan                                      |
| H2X       | e1_r20_seed42     | E1       | ecological_novelty | False | nan                         |               0.000 |               0.000 |         0.360 |      0.334 | [0.7781589812986134, 1.1829817937497789] |
| H2X       | e1_r20_seed42     | E1       | support_count      | True  | nan                         |               1.000 |               0.000 |         0.397 |      0.294 | [164.0, 408.66666666666663]              |
| H2X_nomsg | e1_r20_seed42     | E1       | network_distance   |       | empty tertile on test cells |             nan     |             nan     |       nan     |    nan     | nan                                      |
| H2X_nomsg | e1_r20_seed42     | E1       | ecological_novelty | True  | nan                         |               1.000 |               0.000 |         0.302 |      0.332 | [0.7781589812986134, 1.1829817937497789] |
| H2X_nomsg | e1_r20_seed42     | E1       | support_count      | True  | nan                         |               1.000 |               0.000 |         0.337 |      0.287 | [164.0, 408.66666666666663]              |
| eco_RF    | e1_r20_seed42     | E1       | network_distance   |       | empty tertile on test cells |             nan     |             nan     |       nan     |    nan     | nan                                      |
| eco_RF    | e1_r20_seed42     | E1       | ecological_novelty | False | nan                         |               0.000 |               1.000 |         0.205 |      0.173 | [0.7781589812986134, 1.1829817937497789] |
| eco_RF    | e1_r20_seed42     | E1       | support_count      | True  | nan                         |               1.000 |               0.000 |         0.214 |      0.172 | [164.0, 408.66666666666663]              |
| H2X       | e1_r20_seed43     | E1       | network_distance   |       | empty tertile on test cells |             nan     |             nan     |       nan     |    nan     | nan                                      |
| H2X       | e1_r20_seed43     | E1       | ecological_novelty | False | nan                         |               0.000 |               0.000 |         0.384 |      0.382 | [0.7781589812986134, 1.1829817937497789] |
| H2X       | e1_r20_seed43     | E1       | support_count      | True  | nan                         |               1.000 |               0.000 |         0.425 |      0.337 | [162.66666666666666, 408.3333333333333]  |
| H2X_nomsg | e1_r20_seed43     | E1       | network_distance   |       | empty tertile on test cells |             nan     |             nan     |       nan     |    nan     | nan                                      |
| H2X_nomsg | e1_r20_seed43     | E1       | ecological_novelty | True  | nan                         |               1.000 |               0.000 |         0.323 |      0.329 | [0.7781589812986134, 1.1829817937497789] |
| H2X_nomsg | e1_r20_seed43     | E1       | support_count      | True  | nan                         |               1.000 |               0.000 |         0.360 |      0.289 | [162.66666666666666, 408.3333333333333]  |
| eco_RF    | e1_r20_seed43     | E1       | network_distance   |       | empty tertile on test cells |             nan     |             nan     |       nan     |    nan     | nan                                      |
| eco_RF    | e1_r20_seed43     | E1       | ecological_novelty | False | nan                         |               0.000 |               1.000 |         0.204 |      0.174 | [0.7781589812986134, 1.1829817937497789] |
| eco_RF    | e1_r20_seed43     | E1       | support_count      | True  | nan                         |               1.000 |               0.000 |         0.210 |      0.171 | [162.66666666666666, 408.3333333333333]  |
| H2X       | e1_r20_seed44     | E1       | network_distance   |       | empty tertile on test cells |             nan     |             nan     |       nan     |    nan     | nan                                      |
| H2X       | e1_r20_seed44     | E1       | ecological_novelty | True  | nan                         |               1.000 |               0.000 |         0.317 |      0.334 | [0.7781589812986134, 1.1829817937497789] |
| H2X       | e1_r20_seed44     | E1       | support_count      | True  | nan                         |               1.000 |               0.000 |         0.352 |      0.285 | [160.66666666666666, 410.3333333333333]  |
| H2X_nomsg | e1_r20_seed44     | E1       | network_distance   |       | empty tertile on test cells |             nan     |             nan     |       nan     |    nan     | nan                                      |
| H2X_nomsg | e1_r20_seed44     | E1       | ecological_novelty | True  | nan                         |               1.000 |               0.000 |         0.313 |      0.322 | [0.7781589812986134, 1.1829817937497789] |
| H2X_nomsg | e1_r20_seed44     | E1       | support_count      | True  | nan                         |               1.000 |               0.000 |         0.372 |      0.287 | [160.66666666666666, 410.3333333333333]  |
| eco_RF    | e1_r20_seed44     | E1       | network_distance   |       | empty tertile on test cells |             nan     |             nan     |       nan     |    nan     | nan                                      |
| eco_RF    | e1_r20_seed44     | E1       | ecological_novelty | False | nan                         |               0.000 |               0.000 |         0.197 |      0.179 | [0.7781589812986134, 1.1829817937497789] |
| eco_RF    | e1_r20_seed44     | E1       | support_count      | True  | nan                         |               1.000 |               0.000 |         0.214 |      0.170 | [160.66666666666666, 410.3333333333333]  |
| H2X       | e2a_strict        | E2a      | network_distance   | False | nan                         |               0.000 |               0.000 |         0.413 |      0.380 | [0.0, 0.0]                               |
| H2X       | e2a_strict        | E2a      | ecological_novelty | True  | nan                         |               1.000 |               0.000 |         0.409 |      0.448 | [0.7813802066888693, 1.2375341118418304] |
| H2X       | e2a_strict        | E2a      | support_count      | True  | nan                         |               1.000 |               0.000 |         0.419 |      0.391 | [194.66666666666666, 438.66666666666663] |
| H2X_nomsg | e2a_strict        | E2a      | network_distance   | False | nan                         |               0.000 |               1.000 |         0.310 |      0.272 | [0.0, 0.0]                               |
| H2X_nomsg | e2a_strict        | E2a      | ecological_novelty | False | nan                         |               0.000 |               1.000 |         0.332 |      0.298 | [0.7813802066888693, 1.2375341118418304] |
| H2X_nomsg | e2a_strict        | E2a      | support_count      | True  | nan                         |               1.000 |               0.000 |         0.333 |      0.279 | [194.66666666666666, 438.66666666666663] |
| eco_RF    | e2a_strict        | E2a      | network_distance   | True  | nan                         |               1.000 |               0.000 |         0.274 |      0.398 | [0.0, 0.0]                               |
| eco_RF    | e2a_strict        | E2a      | ecological_novelty | False | nan                         |               0.000 |               1.000 |         0.306 |      0.251 | [0.7813802066888693, 1.2375341118418304] |
| eco_RF    | e2a_strict        | E2a      | support_count      | True  | nan                         |               1.000 |               0.000 |         0.301 |      0.282 | [194.66666666666666, 438.66666666666663] |
| H2X       | e2b_partial       | E2b      | network_distance   | True  | nan                         |               1.000 |               0.000 |         0.409 |      0.507 | [0.0, 0.0]                               |
| H2X       | e2b_partial       | E2b      | ecological_novelty | True  | nan                         |               1.000 |               0.000 |         0.408 |      0.447 | [0.7782525675216565, 1.191115532370611]  |
| H2X       | e2b_partial       | E2b      | support_count      | True  | nan                         |               1.000 |               0.000 |         0.408 |      0.390 | [199.0, 440.3333333333333]               |
| H2X_nomsg | e2b_partial       | E2b      | network_distance   | True  | nan                         |               1.000 |               0.000 |         0.307 |      0.516 | [0.0, 0.0]                               |
| H2X_nomsg | e2b_partial       | E2b      | ecological_novelty | True  | nan                         |               1.000 |               0.000 |         0.291 |      0.295 | [0.7782525675216565, 1.191115532370611]  |
| H2X_nomsg | e2b_partial       | E2b      | support_count      | True  | nan                         |               1.000 |               0.000 |         0.330 |      0.278 | [199.0, 440.3333333333333]               |
| eco_RF    | e2b_partial       | E2b      | network_distance   | True  | nan                         |               1.000 |               0.000 |         0.282 |      0.358 | [0.0, 0.0]                               |
| eco_RF    | e2b_partial       | E2b      | ecological_novelty | False | nan                         |               0.000 |               1.000 |         0.296 |      0.266 | [0.7782525675216565, 1.191115532370611]  |
| eco_RF    | e2b_partial       | E2b      | support_count      | False | nan                         |               0.000 |               0.000 |         0.281 |      0.286 | [199.0, 440.3333333333333]               |
| H2X       | e3_spatial_seed42 | E3       | network_distance   |       | empty tertile on test cells |             nan     |             nan     |       nan     |    nan     | nan                                      |
| H2X       | e3_spatial_seed42 | E3       | ecological_novelty | False | nan                         |               0.000 |               0.000 |         0.529 |      0.483 | [0.8219615077226107, 1.3027692275736678] |
| H2X       | e3_spatial_seed42 | E3       | support_count      | False | nan                         |               0.000 |               0.000 |         0.494 |      0.527 | [189.33333333333331, 447.33333333333326] |
| H2X_nomsg | e3_spatial_seed42 | E3       | network_distance   |       | empty tertile on test cells |             nan     |             nan     |       nan     |    nan     | nan                                      |
| H2X_nomsg | e3_spatial_seed42 | E3       | ecological_novelty | False | nan                         |               0.000 |               0.000 |         0.404 |      0.382 | [0.8219615077226107, 1.3027692275736678] |
| H2X_nomsg | e3_spatial_seed42 | E3       | support_count      | True  | nan                         |               1.000 |               0.000 |         0.450 |      0.359 | [189.33333333333331, 447.33333333333326] |
| eco_RF    | e3_spatial_seed42 | E3       | network_distance   |       | empty tertile on test cells |             nan     |             nan     |       nan     |    nan     | nan                                      |
| eco_RF    | e3_spatial_seed42 | E3       | ecological_novelty | False | nan                         |               0.000 |               0.000 |         0.275 |      0.265 | [0.8219615077226107, 1.3027692275736678] |
| eco_RF    | e3_spatial_seed42 | E3       | support_count      | False | nan                         |               0.000 |               1.000 |         0.258 |      0.285 | [189.33333333333331, 447.33333333333326] |
| H2X       | e3_spatial_seed43 | E3       | network_distance   |       | empty tertile on test cells |             nan     |             nan     |       nan     |    nan     | nan                                      |
| H2X       | e3_spatial_seed43 | E3       | ecological_novelty | False | nan                         |               0.000 |               1.000 |         0.509 |      0.459 | [0.8134358020501052, 1.2810420542932268] |
| H2X       | e3_spatial_seed43 | E3       | support_count      | True  | nan                         |               1.000 |               0.000 |         0.573 |      0.420 | [179.0, 441.0]                           |
| H2X_nomsg | e3_spatial_seed43 | E3       | network_distance   |       | empty tertile on test cells |             nan     |             nan     |       nan     |    nan     | nan                                      |
| H2X_nomsg | e3_spatial_seed43 | E3       | ecological_novelty | True  | nan                         |               1.000 |               0.000 |         0.349 |      0.440 | [0.8134358020501052, 1.2810420542932268] |
| H2X_nomsg | e3_spatial_seed43 | E3       | support_count      | True  | nan                         |               1.000 |               0.000 |         0.535 |      0.311 | [179.0, 441.0]                           |
| eco_RF    | e3_spatial_seed43 | E3       | network_distance   |       | empty tertile on test cells |             nan     |             nan     |       nan     |    nan     | nan                                      |
| eco_RF    | e3_spatial_seed43 | E3       | ecological_novelty | False | nan                         |               0.000 |               1.000 |         0.258 |      0.227 | [0.8134358020501052, 1.2810420542932268] |
| eco_RF    | e3_spatial_seed43 | E3       | support_count      | True  | nan                         |               1.000 |               0.000 |         0.247 |      0.243 | [179.0, 441.0]                           |
| H2X       | e3_spatial_seed44 | E3       | network_distance   |       | empty tertile on test cells |             nan     |             nan     |       nan     |    nan     | nan                                      |
| H2X       | e3_spatial_seed44 | E3       | ecological_novelty | True  | nan                         |               1.000 |               0.000 |         0.405 |      0.764 | [0.8730898886360192, 1.3980465567287663] |
| H2X       | e3_spatial_seed44 | E3       | support_count      | True  | nan                         |               1.000 |               0.000 |         0.491 |      0.373 | [167.66666666666666, 406.0]              |
| H2X_nomsg | e3_spatial_seed44 | E3       | network_distance   |       | empty tertile on test cells |             nan     |             nan     |       nan     |    nan     | nan                                      |
| H2X_nomsg | e3_spatial_seed44 | E3       | ecological_novelty | True  | nan                         |               1.000 |               0.000 |         0.368 |      0.567 | [0.8730898886360192, 1.3980465567287663] |
| H2X_nomsg | e3_spatial_seed44 | E3       | support_count      | True  | nan                         |               1.000 |               0.000 |         0.419 |      0.336 | [167.66666666666666, 406.0]              |
| eco_RF    | e3_spatial_seed44 | E3       | network_distance   |       | empty tertile on test cells |             nan     |             nan     |       nan     |    nan     | nan                                      |
| eco_RF    | e3_spatial_seed44 | E3       | ecological_novelty | True  | nan                         |               1.000 |               0.000 |         0.256 |      0.327 | [0.8730898886360192, 1.3980465567287663] |
| eco_RF    | e3_spatial_seed44 | E3       | support_count      | True  | nan                         |               1.000 |               0.000 |         0.268 |      0.263 | [167.66666666666666, 406.0]              |
