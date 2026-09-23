# Phase-2B freeze report (directional pilot) — corrected 2B-R1

Frozen at 2026-09-23T11:28:14.193963+00:00 by `scripts/analyze_phase2b.py` (one-command recompute). **2B is a pilot: these numbers are not paper claims** (spec §7).

## Corrections applied (pilot already seen)

- Comparison sets are separated: `tabular` (H2X vs eco_RF/eco_MLP) supports only "better than the current tabular implementations"; `message_ablation` (H2X vs H2X_nomsg, matched inputs) is the evidence about edge messages. The no-message control stays in the gate comparator set — renaming cannot remove it.
- Main table and bootstrap now estimate the same quantity (equal-mask within seed, then mean over seeds).
- Seed direction counts use training seeds (k of N), not seed×mask.
- CI covering 0 = insufficient evidence, never equivalence. `mae_sd_seed` is training-seed spread; `mae_sd_mask` is mask spread.
- High-DOC rows carry denominators; small samples say "undetected, estimate unstable". Seeds re-predict the same cells.

## Mean MAE (equal-mask within seed, then mean over seeds)

| arm       | family   |   n_seed |   n_mask |   mae_mean |   mae_sd_seed |   mae_sd_mask |   r2_mean |   n_test_mean |
|:----------|:---------|---------:|---------:|-----------:|--------------:|--------------:|----------:|--------------:|
| H2        | E1       |        3 |        3 |      1.526 |         0.003 |         0.048 |     0.409 |      4514.000 |
| H2        | E2a      |        3 |        1 |      0.859 |         0.014 |         0.000 |     0.555 |      2223.000 |
| H2        | E2b      |        3 |        1 |      0.850 |         0.010 |         0.000 |     0.556 |      1778.000 |
| H2        | E3       |        3 |        3 |      2.505 |         0.018 |         0.201 |     0.446 |      2511.000 |
| H2E       | E1       |        3 |        3 |      1.519 |         0.009 |         0.044 |     0.398 |      4514.000 |
| H2E       | E2a      |        3 |        1 |      0.861 |         0.062 |         0.000 |     0.542 |      2223.000 |
| H2E       | E2b      |        3 |        1 |      0.846 |         0.061 |         0.000 |     0.549 |      1778.000 |
| H2E       | E3       |        3 |        3 |      2.382 |         0.017 |         0.148 |     0.457 |      2511.000 |
| H2X       | E1       |        3 |        3 |      1.514 |         0.011 |         0.055 |     0.400 |      4514.000 |
| H2X       | E2a      |        3 |        1 |      0.834 |         0.046 |         0.000 |     0.549 |      2223.000 |
| H2X       | E2b      |        3 |        1 |      0.822 |         0.038 |         0.000 |     0.552 |      1778.000 |
| H2X       | E3       |        3 |        3 |      2.351 |         0.018 |         0.196 |     0.469 |      2511.000 |
| H2X_nomsg | E1       |        3 |        3 |      1.588 |         0.009 |         0.052 |     0.378 |      4514.000 |
| H2X_nomsg | E2a      |        3 |        1 |      0.819 |         0.025 |         0.000 |     0.568 |      2223.000 |
| H2X_nomsg | E2b      |        3 |        1 |      0.818 |         0.024 |         0.000 |     0.568 |      1778.000 |
| H2X_nomsg | E3       |        3 |        3 |      2.269 |         0.016 |         0.181 |     0.469 |      2511.000 |
| eco_RF    | E1       |        3 |        3 |      1.426 |         0.001 |         0.060 |     0.497 |      4514.000 |
| eco_RF    | E2a      |        3 |        1 |      1.188 |         0.013 |         0.000 |     0.147 |      2223.000 |
| eco_RF    | E2b      |        3 |        1 |      1.304 |         0.024 |         0.000 |     0.047 |      1778.000 |
| eco_RF    | E3       |        3 |        3 |      2.298 |         0.007 |         0.389 |     0.472 |      2511.000 |
| eco_MLP   | E1       |        3 |        3 |      1.545 |         0.011 |         0.031 |     0.425 |      4514.000 |
| eco_MLP   | E2a      |        3 |        1 |      1.490 |         0.086 |         0.000 |    -0.043 |      2223.000 |
| eco_MLP   | E2b      |        3 |        1 |      1.359 |         0.118 |         0.000 |     0.144 |      1778.000 |
| eco_MLP   | E3       |        3 |        3 |      4.686 |         1.500 |         3.409 |   -16.113 |      2511.000 |

## Paired comparisons (same estimand; station-clustered bootstrap, 2000 draws)

| set              | family   | arm_a   | arm_b     |   mae_a |   mae_b |   rel_reduction_pct |   paired_d_mae |   ci95_lo |   ci95_hi | ci_excludes_0   | evidence              |   seeds_a_better |   n_seeds |
|:-----------------|:---------|:--------|:----------|--------:|--------:|--------------------:|---------------:|----------:|----------:|:----------------|:----------------------|-----------------:|----------:|
| tabular          | E1       | H2X     | eco_RF    |  1.5138 |  1.4262 |             -6.1432 |         0.0876 |    0.0555 |    0.1221 | True            | B better              |                0 |         3 |
| tabular          | E1       | H2X     | eco_MLP   |  1.5138 |  1.5445 |              1.9904 |        -0.0307 |   -0.0615 |   -0.0011 | True            | A better              |                3 |         3 |
| message_ablation | E1       | H2X     | H2X_nomsg |  1.5138 |  1.5879 |              4.6681 |        -0.0741 |   -0.1011 |   -0.0510 | True            | A better              |                3 |         3 |
| encoder          | E1       | H2X     | H2E       |  1.5138 |  1.5185 |              0.3114 |        -0.0047 |   -0.0167 |    0.0069 | False           | insufficient evidence |                2 |         3 |
| tabular          | E2a      | H2X     | eco_RF    |  0.8339 |  1.1877 |             29.7921 |        -0.3538 |   -0.5514 |   -0.1796 | True            | A better              |                3 |         3 |
| tabular          | E2a      | H2X     | eco_MLP   |  0.8339 |  1.4905 |             44.0535 |        -0.6566 |   -0.8045 |   -0.5085 | True            | A better              |                3 |         3 |
| message_ablation | E2a      | H2X     | H2X_nomsg |  0.8339 |  0.8194 |             -1.7613 |         0.0144 |   -0.0175 |    0.0483 | False           | insufficient evidence |                2 |         3 |
| encoder          | E2a      | H2X     | H2E       |  0.8339 |  0.8612 |              3.1779 |        -0.0274 |   -0.0564 |    0.0061 | False           | insufficient evidence |                2 |         3 |
| tabular          | E2b      | H2X     | eco_RF    |  0.8216 |  1.3041 |             37.0001 |        -0.4825 |   -0.6344 |   -0.3293 | True            | A better              |                3 |         3 |
| tabular          | E2b      | H2X     | eco_MLP   |  0.8216 |  1.3588 |             39.5356 |        -0.5372 |   -0.6387 |   -0.4437 | True            | A better              |                3 |         3 |
| message_ablation | E2b      | H2X     | H2X_nomsg |  0.8216 |  0.8176 |             -0.4828 |         0.0039 |   -0.0283 |    0.0369 | False           | insufficient evidence |                2 |         3 |
| encoder          | E2b      | H2X     | H2E       |  0.8216 |  0.8456 |              2.8388 |        -0.0240 |   -0.0521 |    0.0065 | False           | insufficient evidence |                2 |         3 |
| tabular          | E3       | H2X     | eco_RF    |  2.3507 |  2.2984 |             -2.2750 |         0.0523 |   -0.0729 |    0.1656 | False           | insufficient evidence |                0 |         3 |
| tabular          | E3       | H2X     | eco_MLP   |  2.3507 |  4.6862 |             49.8383 |        -2.3355 |   -7.7564 |    0.1183 | False           | insufficient evidence |                3 |         3 |
| message_ablation | E3       | H2X     | H2X_nomsg |  2.3507 |  2.2695 |             -3.5779 |         0.0812 |   -0.0329 |    0.1894 | False           | insufficient evidence |                0 |         3 |
| encoder          | E3       | H2X     | H2E       |  2.3507 |  2.3819 |              1.3109 |        -0.0312 |   -0.1304 |    0.0863 | False           | insufficient evidence |                3 |         3 |

## High-DOC (pooled detection with unique-cell denominators)

| arm       | family   |     q |   n_true_high_unique_cells |   n_seed_repeats |   recall_pooled |   precision_pooled |   tail_mae_mean |   tail_sqerr_share_mean | small_sample_flag   |
|:----------|:---------|------:|---------------------------:|-----------------:|----------------:|-------------------:|----------------:|------------------------:|:--------------------|
| H2        | E1       | 0.900 |                       1314 |                3 |           0.601 |              0.776 |           7.046 |                   0.899 | False               |
| H2        | E1       | 0.950 |                        692 |                3 |           0.447 |              0.691 |          10.146 |                   0.850 | False               |
| H2        | E2a      | 0.900 |                         12 |                3 |           0.222 |              0.615 |           6.757 |                   0.218 | True                |
| H2        | E2a      | 0.950 |                          2 |                3 |           0.167 |              0.200 |          13.115 |                   0.111 | True                |
| H2        | E2b      | 0.900 |                          7 |                3 |           0.381 |              0.727 |           7.845 |                   0.217 | True                |
| H2        | E2b      | 0.950 |                          2 |                3 |           0.167 |              0.500 |          13.031 |                   0.141 | True                |
| H2        | E3       | 0.900 |                       1384 |                3 |           0.684 |              0.815 |           6.769 |                   0.804 | False               |
| H2        | E3       | 0.950 |                        903 |                3 |           0.595 |              0.668 |           7.939 |                   0.713 | False               |
| H2E       | E1       | 0.900 |                       1314 |                3 |           0.610 |              0.767 |           7.027 |                   0.898 | False               |
| H2E       | E1       | 0.950 |                        692 |                3 |           0.488 |              0.671 |           9.971 |                   0.847 | False               |
| H2E       | E2a      | 0.900 |                         12 |                3 |           0.333 |              0.667 |           6.769 |                   0.213 | True                |
| H2E       | E2a      | 0.950 |                          2 |                3 |           0.000 |              0.000 |          13.807 |                   0.116 | True                |
| H2E       | E2b      | 0.900 |                          7 |                3 |           0.429 |              0.750 |           7.209 |                   0.204 | True                |
| H2E       | E2b      | 0.950 |                          2 |                3 |           0.000 |              0.000 |          13.768 |                   0.150 | True                |
| H2E       | E3       | 0.900 |                       1384 |                3 |           0.729 |              0.828 |           6.510 |                   0.806 | False               |
| H2E       | E3       | 0.950 |                        903 |                3 |           0.663 |              0.664 |           7.509 |                   0.707 | False               |
| H2X       | E1       | 0.900 |                       1314 |                3 |           0.610 |              0.772 |           7.038 |                   0.902 | False               |
| H2X       | E1       | 0.950 |                        692 |                3 |           0.482 |              0.667 |          10.019 |                   0.850 | False               |
| H2X       | E2a      | 0.900 |                         12 |                3 |           0.278 |              0.667 |           7.004 |                   0.234 | True                |
| H2X       | E2a      | 0.950 |                          2 |                3 |           0.000 |            nan     |          15.387 |                   0.136 | True                |
| H2X       | E2b      | 0.900 |                          7 |                3 |           0.381 |              0.727 |           7.771 |                   0.231 | True                |
| H2X       | E2b      | 0.950 |                          2 |                3 |           0.000 |            nan     |          15.378 |                   0.173 | True                |
| H2X       | E3       | 0.900 |                       1384 |                3 |           0.726 |              0.821 |           6.446 |                   0.820 | False               |
| H2X       | E3       | 0.950 |                        903 |                3 |           0.648 |              0.676 |           7.542 |                   0.729 | False               |
| H2X_nomsg | E1       | 0.900 |                       1314 |                3 |           0.615 |              0.752 |           7.258 |                   0.894 | False               |
| H2X_nomsg | E1       | 0.950 |                        692 |                3 |           0.473 |              0.628 |          10.295 |                   0.842 | False               |
| H2X_nomsg | E2a      | 0.900 |                         12 |                3 |           0.333 |              0.667 |           7.060 |                   0.233 | True                |
| H2X_nomsg | E2a      | 0.950 |                          2 |                3 |           0.000 |              0.000 |          14.321 |                   0.128 | True                |
| H2X_nomsg | E2b      | 0.900 |                          7 |                3 |           0.429 |              0.750 |           7.705 |                   0.223 | True                |
| H2X_nomsg | E2b      | 0.950 |                          2 |                3 |           0.000 |              0.000 |          14.321 |                   0.163 | True                |
| H2X_nomsg | E3       | 0.900 |                       1384 |                3 |           0.738 |              0.822 |           6.537 |                   0.851 | False               |
| H2X_nomsg | E3       | 0.950 |                        903 |                3 |           0.628 |              0.680 |           7.821 |                   0.771 | False               |
| eco_MLP   | E1       | 0.900 |                       1314 |                3 |           0.650 |              0.745 |           6.718 |                   0.869 | False               |
| eco_MLP   | E1       | 0.950 |                        692 |                3 |           0.533 |              0.651 |           9.483 |                   0.814 | False               |
| eco_MLP   | E2a      | 0.900 |                         12 |                3 |           0.250 |              0.196 |           7.337 |                   0.108 | True                |
| eco_MLP   | E2a      | 0.950 |                          2 |                3 |           0.167 |              0.200 |          15.428 |                   0.061 | True                |
| eco_MLP   | E2b      | 0.900 |                          7 |                3 |           0.238 |              0.263 |           7.555 |                   0.097 | True                |
| eco_MLP   | E2b      | 0.950 |                          2 |                3 |           0.000 |              0.000 |          13.541 |                   0.068 | True                |
| eco_MLP   | E3       | 0.900 |                       1384 |                3 |           0.753 |              0.747 |           7.044 |                   0.623 | False               |
| eco_MLP   | E3       | 0.950 |                        903 |                3 |           0.569 |              0.589 |           8.326 |                   0.564 | False               |
| eco_RF    | E1       | 0.900 |                       1314 |                3 |           0.643 |              0.763 |           6.225 |                   0.866 | False               |
| eco_RF    | E1       | 0.950 |                        692 |                3 |           0.553 |              0.691 |           8.897 |                   0.817 | False               |
| eco_RF    | E2a      | 0.900 |                         12 |                3 |           0.250 |              0.173 |           6.619 |                   0.120 | True                |
| eco_RF    | E2a      | 0.950 |                          2 |                3 |           0.000 |              0.000 |          16.240 |                   0.079 | True                |
| eco_RF    | E2b      | 0.900 |                          7 |                3 |           0.286 |              0.167 |           6.323 |                   0.075 | True                |
| eco_RF    | E2b      | 0.950 |                          2 |                3 |           0.000 |              0.000 |          13.517 |                   0.060 | True                |
| eco_RF    | E3       | 0.900 |                       1384 |                3 |           0.742 |              0.823 |           6.495 |                   0.845 | False               |
| eco_RF    | E3       | 0.950 |                        903 |                3 |           0.518 |              0.668 |           8.056 |                   0.780 | False               |

## Input-visibility shift (tabular features, fit stage vs test stage)

| mask              | family   | stage   |   train_rows_median_visdoc |   train_rows_mean_visdoc |   test_rows_median_visdoc |   test_rows_mean_visdoc |
|:------------------|:---------|:--------|---------------------------:|-------------------------:|--------------------------:|------------------------:|
| e1_r20_seed42     | E1       | fit     |                      31.00 |                    31.75 |                     31.00 |                   31.49 |
| e1_r20_seed42     | E1       | test    |                      35.00 |                    35.31 |                     34.00 |                   35.01 |
| e1_r20_seed43     | E1       | fit     |                      31.00 |                    31.73 |                     31.00 |                   31.77 |
| e1_r20_seed43     | E1       | test    |                      34.00 |                    35.23 |                     34.00 |                   35.32 |
| e1_r20_seed44     | E1       | fit     |                      31.00 |                    31.75 |                     31.00 |                   31.71 |
| e1_r20_seed44     | E1       | test    |                      35.00 |                    35.29 |                     34.00 |                   35.18 |
| e1_r40_seed42     | E1       | fit     |                      23.00 |                    23.82 |                     23.00 |                   23.79 |
| e1_r40_seed42     | E1       | test    |                      26.00 |                    26.45 |                     25.50 |                   26.35 |
| e1_r40_seed43     | E1       | fit     |                      23.00 |                    23.82 |                     24.00 |                   23.81 |
| e1_r40_seed43     | E1       | test    |                      26.00 |                    26.47 |                     26.00 |                   26.44 |
| e1_r40_seed44     | E1       | fit     |                      24.00 |                    23.86 |                     23.00 |                   23.81 |
| e1_r40_seed44     | E1       | test    |                      26.00 |                    26.49 |                     26.00 |                   26.46 |
| e1_r60_seed42     | E1       | fit     |                      15.00 |                    15.86 |                     15.00 |                   15.82 |
| e1_r60_seed42     | E1       | test    |                      17.00 |                    17.64 |                     17.00 |                   17.61 |
| e1_r60_seed43     | E1       | fit     |                      16.00 |                    15.90 |                     15.00 |                   15.87 |
| e1_r60_seed43     | E1       | test    |                      17.00 |                    17.69 |                     18.00 |                   17.63 |
| e1_r60_seed44     | E1       | fit     |                      16.00 |                    15.83 |                     16.00 |                   15.86 |
| e1_r60_seed44     | E1       | test    |                      17.00 |                    17.59 |                     17.00 |                   17.61 |
| e2a_strict        | E2a      | fit     |                      43.00 |                    44.99 |                      0.00 |                    0.00 |
| e2a_strict        | E2a      | test    |                      43.00 |                    44.99 |                      0.00 |                    0.00 |
| e2b_partial       | E2b      | fit     |                      43.00 |                    44.99 |                      7.00 |                    6.99 |
| e2b_partial       | E2b      | test    |                      43.00 |                    44.99 |                      7.00 |                    6.99 |
| e3_spatial_seed42 | E3       | fit     |                      35.00 |                    35.18 |                     36.00 |                   35.30 |
| e3_spatial_seed42 | E3       | test    |                      39.00 |                    39.09 |                     40.00 |                   39.16 |
| e3_spatial_seed43 | E3       | fit     |                      35.00 |                    35.34 |                     35.00 |                   35.91 |
| e3_spatial_seed43 | E3       | test    |                      38.00 |                    39.32 |                     39.00 |                   39.98 |
| e3_spatial_seed44 | E3       | fit     |                      34.00 |                    34.67 |                     36.00 |                   36.13 |
| e3_spatial_seed44 | E3       | test    |                      38.00 |                    38.54 |                     40.00 |                   40.10 |

## What each comparison set can support

- tabular set: "H2X predicts better than these RF/MLP implementations" — it says nothing about topology.
- message_ablation set: matched-inputs evidence about edge messages. If no-message matches H2X, edge messages are not shown necessary.
- encoder set (H2X vs H2E): incremental value of the encoder beyond raw ecological context.
