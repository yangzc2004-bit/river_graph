# Phase-2B freeze report (directional pilot)

Frozen at 2026-09-23T10:49:39.796888+00:00 by `scripts/analyze_phase2b.py` (recomputable). **2B is a pilot: these numbers are not paper claims** (docs/paper/phase2_ablation_spec.md §7).

Policy bindings verified by `scripts/run2b_executor.py` before every unit: `configs/phase2_2c_policy.json` (h2x_policy_sha256, endpoints_sha256, dataset/mask hashes), the runtime snapshot pin, and run identity (`config_hash` + `run_identity_sha256` + `runtime_code_snapshot_sha256`) on all 144 sidecars. Audit: 144 `identity ok`, zero retrained duplicates, zero suspicious names.

2B pilot -> frozen reconciliation: the discarded pilot criterion (15%, two-of-three) is non-operative; operative gates remain endpoints-v1 (>=10% union over {E2a, E2b, E3} + per-scenario no-harm margins). E2a and E2b are separate estimands and are never pooled.

## Mean MAE (mg/L) by arm x scenario family (3 seeds x key masks)

| arm       | family   |   n_cells |   mae_mean |   mae_sd |   r2_mean |   n_test_mean |
|:----------|:---------|----------:|-----------:|---------:|----------:|--------------:|
| H2        | E1       |         9 |      1.526 |    0.049 |     0.409 |      4514.000 |
| H2        | E2a      |         3 |      0.859 |    0.014 |     0.555 |      2223.000 |
| H2        | E2b      |         3 |      0.850 |    0.010 |     0.556 |      1778.000 |
| H2        | E3       |         9 |      2.505 |    0.213 |     0.446 |      2511.000 |
| H2E       | E1       |         9 |      1.519 |    0.049 |     0.398 |      4514.000 |
| H2E       | E2a      |         3 |      0.861 |    0.062 |     0.542 |      2223.000 |
| H2E       | E2b      |         3 |      0.846 |    0.061 |     0.549 |      1778.000 |
| H2E       | E3       |         9 |      2.382 |    0.166 |     0.457 |      2511.000 |
| H2X       | E1       |         9 |      1.514 |    0.057 |     0.400 |      4514.000 |
| H2X       | E2a      |         3 |      0.834 |    0.046 |     0.549 |      2223.000 |
| H2X       | E2b      |         3 |      0.822 |    0.038 |     0.552 |      1778.000 |
| H2X       | E3       |         9 |      2.351 |    0.200 |     0.469 |      2511.000 |
| H2X_nomsg | E1       |         9 |      1.588 |    0.053 |     0.378 |      4514.000 |
| H2X_nomsg | E2a      |         3 |      0.819 |    0.025 |     0.568 |      2223.000 |
| H2X_nomsg | E2b      |         3 |      0.818 |    0.024 |     0.568 |      1778.000 |
| H2X_nomsg | E3       |         9 |      2.269 |    0.187 |     0.469 |      2511.000 |
| eco_RF    | E1       |         9 |      1.399 |    0.065 |     0.494 |      4514.000 |
| eco_RF    | E2a      |         3 |      1.188 |    0.013 |     0.147 |      2223.000 |
| eco_RF    | E2b      |         3 |      1.304 |    0.024 |     0.047 |      1778.000 |
| eco_RF    | E3       |         9 |      2.269 |    0.397 |     0.479 |      2511.000 |
| eco_MLP   | E1       |         9 |      1.528 |    0.040 |     0.422 |      4514.000 |
| eco_MLP   | E2a      |         3 |      2.062 |    0.691 |    -1.181 |      2223.000 |
| eco_MLP   | E2b      |         3 |      1.252 |    0.040 |     0.239 |      1778.000 |
| eco_MLP   | E3       |         9 |      2.964 |    1.355 |    -1.144 |      2511.000 |

## Paired comparisons (station-clustered bootstrap, 2000 draws)

| family   | arm_a   | arm_b     |   mae_a |   mae_b |   rel_reduction_pct |   paired_d_mae |   ci95_lo |   ci95_hi |   seed_signs_a_better |   seed_signs_total |   n_rows |
|:---------|:--------|:----------|--------:|--------:|--------------------:|---------------:|----------:|----------:|----------------------:|-------------------:|---------:|
| E1       | H2X     | H2E       |   1.514 |   1.519 |               0.311 |         -0.005 |    -0.017 |     0.007 |                     6 |                  9 |    40626 |
| E1       | H2X     | H2X_nomsg |   1.514 |   1.588 |               4.668 |         -0.074 |    -0.101 |    -0.051 |                     9 |                  9 |    40626 |
| E1       | H2X     | eco_RF    |   1.514 |   1.399 |              -8.233 |          0.115 |     0.082 |     0.151 |                     0 |                  9 |    40626 |
| E1       | H2X     | eco_MLP   |   1.514 |   1.528 |               0.929 |         -0.014 |    -0.044 |     0.013 |                     6 |                  9 |    40626 |
| E2a      | H2X     | H2E       |   0.834 |   0.861 |               3.178 |         -0.027 |    -0.056 |     0.006 |                     2 |                  3 |     6669 |
| E2a      | H2X     | H2X_nomsg |   0.834 |   0.819 |              -1.761 |          0.014 |    -0.018 |     0.048 |                     2 |                  3 |     6669 |
| E2a      | H2X     | eco_RF    |   0.834 |   1.188 |              29.792 |         -0.354 |    -0.551 |    -0.180 |                     3 |                  3 |     6669 |
| E2a      | H2X     | eco_MLP   |   0.834 |   2.062 |              59.556 |         -1.228 |    -1.527 |    -0.963 |                     3 |                  3 |     6669 |
| E2b      | H2X     | H2E       |   0.822 |   0.846 |               2.839 |         -0.024 |    -0.052 |     0.006 |                     2 |                  3 |     5334 |
| E2b      | H2X     | H2X_nomsg |   0.822 |   0.818 |              -0.483 |          0.004 |    -0.028 |     0.037 |                     2 |                  3 |     5334 |
| E2b      | H2X     | eco_RF    |   0.822 |   1.304 |              37.000 |         -0.483 |    -0.634 |    -0.329 |                     3 |                  3 |     5334 |
| E2b      | H2X     | eco_MLP   |   0.822 |   1.252 |              34.382 |         -0.430 |    -0.529 |    -0.343 |                     3 |                  3 |     5334 |
| E3       | H2X     | H2E       |   2.351 |   2.382 |               1.311 |         -0.032 |    -0.129 |     0.079 |                     6 |                  9 |    22599 |
| E3       | H2X     | H2X_nomsg |   2.351 |   2.269 |              -3.578 |          0.080 |    -0.031 |     0.187 |                     1 |                  9 |    22599 |
| E3       | H2X     | eco_RF    |   2.351 |   2.269 |              -3.586 |          0.080 |    -0.065 |     0.211 |                     3 |                  9 |    22599 |
| E3       | H2X     | eco_MLP   |   2.351 |   2.964 |              20.686 |         -0.656 |    -2.343 |     0.156 |                     3 |                  9 |    22599 |

## High-DOC identification (mean over cells, threshold = train quantile)

| arm       | family   |     q |   precision |   recall |   tail_mae |   tail_sqerr_share |
|:----------|:---------|------:|------------:|---------:|-----------:|-------------------:|
| H2        | E1       | 0.900 |       0.776 |    0.601 |      7.046 |              0.899 |
| H2        | E1       | 0.950 |       0.693 |    0.447 |     10.146 |              0.850 |
| H2        | E2a      | 0.900 |       0.611 |    0.222 |      6.757 |              0.218 |
| H2        | E2a      | 0.950 |       0.111 |    0.167 |     13.115 |              0.111 |
| H2        | E2b      | 0.900 |       0.750 |    0.381 |      7.845 |              0.217 |
| H2        | E2b      | 0.950 |       0.500 |    0.167 |     13.031 |              0.141 |
| H2        | E3       | 0.900 |       0.819 |    0.682 |      6.769 |              0.804 |
| H2        | E3       | 0.950 |       0.673 |    0.594 |      7.939 |              0.713 |
| H2E       | E1       | 0.900 |       0.768 |    0.610 |      7.027 |              0.898 |
| H2E       | E1       | 0.950 |       0.678 |    0.487 |      9.971 |              0.847 |
| H2E       | E2a      | 0.900 |       0.667 |    0.333 |      6.769 |              0.213 |
| H2E       | E2a      | 0.950 |       0.000 |    0.000 |     13.807 |              0.116 |
| H2E       | E2b      | 0.900 |       0.750 |    0.429 |      7.209 |              0.204 |
| H2E       | E2b      | 0.950 |       0.000 |    0.000 |     13.768 |              0.150 |
| H2E       | E3       | 0.900 |       0.831 |    0.720 |      6.510 |              0.806 |
| H2E       | E3       | 0.950 |       0.667 |    0.649 |      7.509 |              0.707 |
| H2X       | E1       | 0.900 |       0.773 |    0.610 |      7.038 |              0.902 |
| H2X       | E1       | 0.950 |       0.670 |    0.481 |     10.019 |              0.850 |
| H2X       | E2a      | 0.900 |       0.672 |    0.278 |      7.004 |              0.234 |
| H2X       | E2a      | 0.950 |     nan     |    0.000 |     15.387 |              0.136 |
| H2X       | E2b      | 0.900 |       0.722 |    0.381 |      7.771 |              0.231 |
| H2X       | E2b      | 0.950 |     nan     |    0.000 |     15.378 |              0.173 |
| H2X       | E3       | 0.900 |       0.827 |    0.719 |      6.446 |              0.820 |
| H2X       | E3       | 0.950 |       0.681 |    0.647 |      7.542 |              0.729 |
| H2X_nomsg | E1       | 0.900 |       0.753 |    0.615 |      7.258 |              0.894 |
| H2X_nomsg | E1       | 0.950 |       0.639 |    0.471 |     10.295 |              0.842 |
| H2X_nomsg | E2a      | 0.900 |       0.667 |    0.333 |      7.060 |              0.233 |
| H2X_nomsg | E2a      | 0.950 |       0.000 |    0.000 |     14.321 |              0.128 |
| H2X_nomsg | E2b      | 0.900 |       0.750 |    0.429 |      7.705 |              0.223 |
| H2X_nomsg | E2b      | 0.950 |       0.000 |    0.000 |     14.321 |              0.163 |
| H2X_nomsg | E3       | 0.900 |       0.825 |    0.733 |      6.537 |              0.851 |
| H2X_nomsg | E3       | 0.950 |       0.686 |    0.624 |      7.821 |              0.771 |
| eco_MLP   | E1       | 0.900 |       0.752 |    0.644 |      6.692 |              0.876 |
| eco_MLP   | E1       | 0.950 |       0.666 |    0.517 |      9.506 |              0.824 |
| eco_MLP   | E2a      | 0.900 |       0.193 |    0.389 |      7.466 |              0.074 |
| eco_MLP   | E2a      | 0.950 |       0.073 |    0.333 |     13.283 |              0.036 |
| eco_MLP   | E2b      | 0.900 |       0.379 |    0.381 |      6.588 |              0.089 |
| eco_MLP   | E2b      | 0.950 |       0.000 |    0.000 |     12.287 |              0.064 |
| eco_MLP   | E3       | 0.900 |       0.788 |    0.741 |      6.550 |              0.679 |
| eco_MLP   | E3       | 0.950 |       0.675 |    0.562 |      7.884 |              0.612 |
| eco_RF    | E1       | 0.900 |       0.776 |    0.633 |      6.180 |              0.876 |
| eco_RF    | E1       | 0.950 |       0.725 |    0.540 |      8.942 |              0.832 |
| eco_RF    | E2a      | 0.900 |       0.182 |    0.250 |      6.619 |              0.120 |
| eco_RF    | E2a      | 0.950 |       0.000 |    0.000 |     16.240 |              0.079 |
| eco_RF    | E2b      | 0.900 |       0.174 |    0.286 |      6.323 |              0.075 |
| eco_RF    | E2b      | 0.950 |       0.000 |    0.000 |     13.517 |              0.060 |
| eco_RF    | E3       | 0.900 |       0.834 |    0.723 |      6.450 |              0.848 |
| eco_RF    | E3       | 0.950 |       0.686 |    0.504 |      8.040 |              0.785 |

## Decision-table reading (pilot only)

- H2X vs H2E (encoder gain?): E2a 0.834 vs 0.861, E2b 0.822 vs 0.846, E3 2.351 vs 2.382 — encoder adds value beyond raw context (pilot)
- H2X vs no-message (topology necessary?): nomsg best-E2/nomsg-E3 min 0.818 vs H2X min 0.822 — no-message matches or beats H2X: river topology NOT shown necessary; do not sell topology as the core contribution
- H2X vs best no-graph (primary gate preview): best tabular MAE 1.188 vs H2X best-scenario values above — the >=10% gate is decided in 2C on seeds 42-46 with paired bootstrap, not here.
- All numbers above are pilot screening (3 seeds). They cannot appear as paper claims and cannot move a primary endpoint.
