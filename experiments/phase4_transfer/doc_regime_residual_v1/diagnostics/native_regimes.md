# Source/validation DOC residual regimes

Positive native residual is observed DOC minus the context prediction (underprediction).
Only source OOF cells and fixed source-validation queries are used. No outer-test query
labels or outcomes are inspected. Source and validation differ in station composition and
OOF versus full-source context fitting; their contrasts do not isolate a cause.

All nine existing ecology fields, context prediction and valid causal relative-flow anomaly
are divided at source-defined tertiles. Ecology uses unique source stations; dynamic cuts
use equal source-station weights. Missing values remain separate. Equal-station summaries
give each station with observations in the reported group the same total weight; pooled
summaries weight every cell equally. Seeds are averaged within partition, then partitions
are weighted equally. Counts are mean per-partition counts, not independent repeated samples.

## Overall concentration regimes

| Role | Region | Weighting | Mean DOC | Mean base | Residual | MAE | Stations |
|---|---|---|---:|---:|---:|---:|---:|
| source_oof | nontail | equal_stations | 4.55 | 5.42 | -0.87 | 1.78 | 228 |
| source_oof | nontail | pooled_cells | 3.90 | 4.27 | -0.38 | 1.25 | 228 |
| source_oof | overall | equal_stations | 6.40 | 5.71 | +0.69 | 2.36 | 232 |
| source_oof | overall | pooled_cells | 5.27 | 4.96 | +0.31 | 1.87 | 232 |
| source_oof | q90 | equal_stations | 16.16 | 7.40 | +8.76 | 9.12 | 146 |
| source_oof | q90 | pooled_cells | 17.31 | 10.95 | +6.36 | 7.36 | 146 |
| source_validation | nontail | equal_stations | 4.27 | 4.81 | -0.54 | 1.65 | 53 |
| source_validation | nontail | pooled_cells | 3.90 | 4.26 | -0.36 | 1.29 | 53 |
| source_validation | overall | equal_stations | 6.05 | 5.06 | +1.00 | 2.48 | 54 |
| source_validation | overall | pooled_cells | 5.23 | 4.77 | +0.46 | 1.99 | 54 |
| source_validation | q90 | equal_stations | 17.61 | 6.65 | +10.96 | 11.11 | 32 |
| source_validation | q90 | pooled_cells | 18.84 | 9.38 | +9.46 | 9.98 | 32 |

## High-DOC residual strata

| Variable | Level | Source: pooled / station-balanced residual | Validation: pooled / station-balanced residual | Validation tail cells / stations | Small group |
|---|---|---:|---:|---:|---|
| agriculture | high | +5.28 / +7.18 | +6.77 / +8.84 | 82 / 13 | False |
| agriculture | low | +6.31 / +8.17 | +7.01 / +7.68 | 68 / 10 | False |
| agriculture | middle | +8.53 / +11.66 | +14.01 / +18.27 | 83 / 10 | False |
| base_pred | high | +5.68 / +7.01 | +9.45 / +9.66 | 179 / 20 | False |
| base_pred | low | +12.83 / +12.94 | +11.74 / +11.70 | 15 / 7 | True |
| base_pred | middle | +9.53 / +9.73 | +10.45 / +11.70 | 40 / 13 | False |
| baseflow_index | high | +9.41 / +10.22 | +11.83 / +13.73 | 96 / 14 | False |
| baseflow_index | low | +4.58 / +7.35 | +5.76 / +8.94 | 103 / 8 | False |
| baseflow_index | middle | +7.58 / +8.66 | +9.18 / +9.79 | 34 / 10 | False |
| elevation | high | +7.89 / +9.11 | +13.76 / +14.49 | 115 / 13 | False |
| elevation | low | +8.23 / +10.35 | +9.07 / +10.03 | 34 / 11 | True |
| elevation | middle | +4.41 / +7.00 | +5.49 / +7.20 | 85 / 8 | False |
| flow_anomaly | high | +6.89 / +7.54 | +10.90 / +12.52 | 62 / 22 | False |
| flow_anomaly | low | +5.45 / +8.33 | +7.45 / +10.23 | 78 / 15 | False |
| flow_anomaly | middle | +7.86 / +8.76 | +13.09 / +16.31 | 60 / 18 | False |
| flow_anomaly | missing | +5.56 / +5.86 | +8.60 / +9.08 | 34 / 7 | True |
| flow_observed | missing | +5.98 / +7.16 | +9.13 / +10.30 | 30 / 6 | True |
| flow_observed | observed | +6.42 / +8.94 | +10.34 / +11.29 | 204 / 27 | False |
| forest | high | +7.79 / +10.08 | +8.83 / +9.01 | 39 / 7 | True |
| forest | low | +5.27 / +7.31 | +6.58 / +9.14 | 125 / 12 | False |
| forest | middle | +9.03 / +9.81 | +13.75 / +12.64 | 70 / 13 | False |
| log_temperature_normal | high | +8.68 / +10.79 | +9.64 / +10.31 | 21 / 8 | True |
| log_temperature_normal | low | +5.18 / +6.20 | +6.45 / +7.23 | 154 / 12 | False |
| log_temperature_normal | middle | +8.66 / +10.16 | +15.53 / +14.26 | 58 / 13 | False |
| precipitation | high | +11.33 / +13.01 | +11.72 / +11.51 | 16 / 7 | True |
| precipitation | low | +5.74 / +7.42 | +10.98 / +13.77 | 163 / 14 | False |
| precipitation | middle | +7.39 / +8.12 | +7.71 / +8.63 | 55 / 11 | True |
| predicted_q90 | above | +4.26 / +4.21 | +5.36 / +5.64 | 104 / 7 | True |
| predicted_q90 | below | +9.22 / +9.94 | +11.42 / +11.44 | 130 / 28 | False |
| soil_organic_matter | high | +6.12 / +7.29 | +8.31 / +9.21 | 82 / 11 | False |
| soil_organic_matter | low | +6.74 / +8.07 | +7.51 / +8.24 | 66 / 11 | False |
| soil_organic_matter | middle | +6.38 / +11.31 | +20.10 / +16.96 | 86 / 9 | False |
| upstream_support | absent | +6.56 / +8.95 | +10.10 / +11.39 | 196 / 27 | False |
| upstream_support | present | +5.01 / +7.35 | +6.31 / +8.63 | 38 / 9 | False |
| urban | high | +9.63 / +12.26 | +9.46 / +11.11 | 30 / 8 | False |
| urban | low | +5.91 / +7.88 | +12.02 / +15.28 | 113 / 12 | False |
| urban | middle | +6.08 / +7.32 | +6.28 / +7.92 | 91 / 12 | False |
| wetland | high | +6.62 / +8.36 | +7.41 / +8.61 | 101 / 13 | False |
| wetland | low | +5.92 / +9.85 | +7.16 / +10.13 | 50 / 8 | False |
| wetland | middle | +6.44 / +8.41 | +16.69 / +15.24 | 83 / 11 | False |

Small groups have fewer than 20 cells or five stations in at least one run. Groups
missing any partition are retained in the per-run tables and not silently averaged.
Tertile labels may describe unequal groups when source values tie; the saved cut table
identifies degenerate boundaries. Strata are marginal descriptions, not adjusted effects
or evidence that one covariate causes the error.

## Partition-level validation tail

| Partition | Weighting | Mean DOC | Mean base | Residual | MAE |
|---|---|---:|---:|---:|---:|
| 142 | equal_stations | 14.64 | 7.22 | +7.43 | 7.51 |
| 142 | pooled_cells | 16.73 | 10.21 | +6.51 | 6.72 |
| 143 | equal_stations | 15.71 | 6.63 | +9.09 | 9.38 |
| 143 | pooled_cells | 16.41 | 10.06 | +6.35 | 7.31 |
| 144 | equal_stations | 22.46 | 6.10 | +16.37 | 16.44 |
| 144 | pooled_cells | 23.38 | 7.86 | +15.53 | 15.93 |

## Implication for the residual head

Tail underprediction persists after equal-station weighting. It is not explained by
missing discharge alone: large residuals also occur when discharge is observed, and
the relative-flow strata do not show a consistent monotone trend. Static ecological
strata show heterogeneous residual levels; several relations are nonmonotone and
differ between source and validation. This supports testing joint regime conditioning,
rather than choosing one ecological variable as a physical rule.

The context base often predicts below its source Q90 threshold on true high-DOC cells.
Those cells have larger positive residuals than tail cells already predicted above Q90.
A correction activated only by a high context prediction would therefore miss an
important part of the underprediction. Context concentration should be a continuous
conditioning input, not an oracle tail indicator or a hard high-concentration gate.

The minimal next test is a source-normalized regime-conditioned scalar residual head:
retain the GRU and flow-interaction branch, expose the existing nine ecology variables
and context prediction directly, and compare additive inputs, hidden-by-concentration
interaction, and hidden-by-ecology interaction. Source training uses OOF context
predictions; validation uses its full-source context predictor. Availability flags stay
explicit. The tail loss weight need not increase for this test. These controls ask
whether conditioning the correction helps beyond merely exposing the same features.

The tables describe where residuals concentrate. They do not fit an alternative
predictor, choose a threshold from validation, or estimate an error floor.
