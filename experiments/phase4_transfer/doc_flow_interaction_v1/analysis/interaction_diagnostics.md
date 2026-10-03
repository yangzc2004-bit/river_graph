# Source-validation flow interaction diagnostics

These diagnostics describe how the fitted readout uses its recurrent state and explicit flow inputs. They do not fit/select models, inspect target-query labels, or introduce a performance endpoint.

For numerical feature j, the conditional readout coefficient is `beta_j + h @ W_j`. The effective coefficient also applies the selected residual scale and nonnegative output floor. Coefficients are in mg/L per unit of the bounded feature, holding the recurrent state and other derived features fixed. Correlated flow features and flow-dependent recurrent states mean these are not causal effects, physical transport parameters, or total derivatives with respect to discharge.

## Magnitudes across all validation queries

Each value below is the equal mean of the nine run-level means; repeated seeds reuse validation cells. Signed components can cancel, so their absolute magnitudes are not attribution percentages.

| Arm | Numeric additive mean absolute contribution | Numeric interaction mean absolute contribution | Realized correction mean absolute |
|---|---:|---:|---:|
| interaction_frozen | 0.0197 | 0.1095 | 0.2169 |
| interaction_tuned | 0.0154 | 0.1326 | 0.3362 |

## State-dependent signs on feature-valid query cells

Fractions are equal run means and describe the effective readout coefficient after the selected scale/floor. `interaction_diagnostics.csv` retains each run’s quantiles, validity fractions and both valid-only and all-query summaries.

| Arm | Feature | Positive fraction | Negative fraction | Zero fraction |
|---|---|---:|---:|---:|
| interaction_frozen | flow_relative_anomaly | 0.704 | 0.296 | 0.000 |
| interaction_frozen | flow_change_1 | 1.000 | 0.000 | 0.000 |
| interaction_frozen | flow_change_3 | 1.000 | 0.000 | 0.000 |
| interaction_tuned | flow_relative_anomaly | 0.742 | 0.258 | 0.000 |
| interaction_tuned | flow_change_1 | 0.999 | 0.001 | 0.000 |
| interaction_tuned | flow_change_3 | 0.980 | 0.020 | 0.000 |

These summaries describe state-dependent readout use. Whether these adjustments improve held-out prediction remains a question for the matched performance comparisons.
