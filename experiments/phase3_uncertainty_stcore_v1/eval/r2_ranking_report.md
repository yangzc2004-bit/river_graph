# 3B-R2 ranking report

Selection score = v2 uncertainty only (spec frozen before the run). Test DOC joined at final evaluation only.

## Main operating point (top 10% ranking diagnostic)

| tool      | family   |   enrichment_log1p |   enrichment_mg |   n_selected |
|:----------|:---------|-------------------:|----------------:|-------------:|
| H2X       | E1       |              0.482 |           1.179 |      452.000 |
| H2X       | E2a      |              0.173 |           0.146 |      223.000 |
| H2X       | E2b      |              0.213 |           0.231 |      178.000 |
| H2X       | E3       |              0.432 |           0.391 |      251.667 |
| H2X_nomsg | E1       |              0.276 |           0.510 |      452.000 |
| H2X_nomsg | E2a      |              0.196 |           0.140 |      223.000 |
| H2X_nomsg | E2b      |              0.165 |           0.132 |      178.000 |
| H2X_nomsg | E3       |              0.447 |           0.344 |      251.667 |
| eco_RF    | E1       |              0.735 |           1.282 |      452.000 |
| eco_RF    | E2a      |              0.854 |           1.203 |      223.000 |
| eco_RF    | E2b      |              0.925 |           1.245 |      178.000 |
| eco_RF    | E3       |              0.257 |           0.315 |      251.667 |

## Judgment

|           |   families_direction_positive |   families_ci_qualified | family_detail                                                                                                                                                                                                                                                                                                                     | highdoc_q90_direction_positive   | ranking_value_at_top10   |
|:----------|------------------------------:|------------------------:|:----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|:---------------------------------|:-------------------------|
| H2X       |                             4 |                       1 | {'E1': {'enrichment_mean': 0.482, 'masks_ci_excl_0': '3/3', 'qualifies': True}, 'E2a': {'enrichment_mean': 0.173, 'masks_ci_excl_0': '0/1', 'qualifies': False}, 'E2b': {'enrichment_mean': 0.213, 'masks_ci_excl_0': '0/1', 'qualifies': False}, 'E3': {'enrichment_mean': 0.432, 'masks_ci_excl_0': '0/3', 'qualifies': False}} | True                             | False                    |
| H2X_nomsg |                             4 |                       1 | {'E1': {'enrichment_mean': 0.276, 'masks_ci_excl_0': '3/3', 'qualifies': True}, 'E2a': {'enrichment_mean': 0.196, 'masks_ci_excl_0': '0/1', 'qualifies': False}, 'E2b': {'enrichment_mean': 0.165, 'masks_ci_excl_0': '0/1', 'qualifies': False}, 'E3': {'enrichment_mean': 0.447, 'masks_ci_excl_0': '1/3', 'qualifies': False}} | True                             | False                    |
| eco_RF    |                             4 |                       4 | {'E1': {'enrichment_mean': 0.735, 'masks_ci_excl_0': '3/3', 'qualifies': True}, 'E2a': {'enrichment_mean': 0.854, 'masks_ci_excl_0': '1/1', 'qualifies': True}, 'E2b': {'enrichment_mean': 0.925, 'masks_ci_excl_0': '1/1', 'qualifies': True}, 'E3': {'enrichment_mean': 0.257, 'masks_ci_excl_0': '3/3', 'qualifies': True}}    | True                             | True                     |

Tools with ranking value: **1** → R2 failed: close stable-blind-spot and active-sampling claims; keep reconstruction, calibration, coverage-width trade-off and the negative ranking result

## Rank association (Spearman, diagnostic)

| tool      | family   |   spearman_unc_abslogerr |
|:----------|:---------|-------------------------:|
| H2X       | E1       |                    0.197 |
| H2X       | E2a      |                    0.150 |
| H2X       | E2b      |                    0.135 |
| H2X       | E3       |                    0.134 |
| H2X_nomsg | E1       |                    0.164 |
| H2X_nomsg | E2a      |                    0.194 |
| H2X_nomsg | E2b      |                    0.182 |
| H2X_nomsg | E3       |                    0.154 |
| eco_RF    | E1       |                    0.319 |
| eco_RF    | E2a      |                    0.231 |
| eco_RF    | E2b      |                    0.346 |
| eco_RF    | E3       |                    0.120 |
