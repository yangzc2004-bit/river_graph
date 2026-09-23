# 3B-R1 val-only report (empirical validation calibration — not conformal)

Method revision made after v1 results were seen (docs/paper/phase3_uncertainty_spec_v2.md). v1 numbers unchanged and kept side by side.

## Coverage and width trade-off (per tool, equal-mask means)

| tool      |   coverage_v1 |   coverage_v2 | gate_080_095_v2   |   width_med_v1 |   width_med_v2 |   width_ratio_v2_v1 |
|:----------|--------------:|--------------:|:------------------|---------------:|---------------:|--------------------:|
| H2X       |         0.843 |         0.898 | True              |          4.050 |          5.090 |               1.260 |
| H2X_nomsg |         0.792 |         0.908 | True              |          3.370 |          5.260 |               1.560 |
| eco_RF    |         0.606 |         0.895 | True              |          2.400 |          5.530 |               2.310 |

Tools reaching the coverage gate: **3** → calibration-level improvement only; Phase 4/5 stay locked

## Per tool x family

| tool      | family   |   coverage_v1 |   coverage_v2 |   width_med_v1 |   width_med_v2 |
|:----------|:---------|--------------:|--------------:|---------------:|---------------:|
| H2X       | E1       |         0.823 |         0.895 |          2.880 |          3.801 |
| H2X       | E2a      |         0.923 |         0.925 |          3.678 |          3.720 |
| H2X       | E2b      |         0.925 |         0.928 |          3.663 |          3.716 |
| H2X       | E3       |         0.809 |         0.881 |          5.476 |          7.295 |
| H2X_nomsg | E1       |         0.775 |         0.899 |          2.693 |          4.228 |
| H2X_nomsg | E2a      |         0.853 |         0.930 |          2.692 |          3.521 |
| H2X_nomsg | E2b      |         0.854 |         0.931 |          2.689 |          3.527 |
| H2X_nomsg | E3       |         0.767 |         0.902 |          4.497 |          7.444 |
| eco_RF    | E1       |         0.584 |         0.894 |          1.506 |          3.484 |
| eco_RF    | E2a      |         0.666 |         0.925 |          2.652 |          5.567 |
| eco_RF    | E2b      |         0.635 |         0.907 |          2.769 |          5.643 |
| eco_RF    | E3       |         0.599 |         0.881 |          3.078 |          7.539 |

## Standardized residual distributions (Q90 / Q95)

| tool      |   score_q90_train |   score_q90_val |   score_q90_test |   score_q95_train |   score_q95_val |   score_q95_test |
|:----------|------------------:|----------------:|-----------------:|------------------:|----------------:|-----------------:|
| H2X       |             5.549 |           6.864 |            6.940 |             7.228 |           9.253 |            9.159 |
| H2X_nomsg |             5.060 |           8.063 |            7.875 |             6.876 |          10.880 |           10.435 |
| eco_RF    |            10.071 |          25.681 |           26.401 |            12.366 |          33.983 |           34.889 |

## Tails (3C rules; Q95 n<20 unstable)

| tool      | subset   |   coverage_v2 |   width_med_v2 |       n |
|:----------|:---------|--------------:|---------------:|--------:|
| H2X       | Q90      |         0.631 |         15.267 | 339.625 |
| H2X       | Q95      |         0.541 |         18.189 | 199.875 |
| H2X_nomsg | Q90      |         0.636 |         14.545 | 339.625 |
| H2X_nomsg | Q95      |         0.642 |         20.596 | 199.875 |
| eco_RF    | Q90      |         0.736 |         15.014 | 339.625 |
| eco_RF    | Q95      |         0.711 |         17.426 | 199.875 |

Monotonicity is NOT recomputed here (spec v2 §4): val-only changes only the uniform scale and cannot change rankings or directions.
