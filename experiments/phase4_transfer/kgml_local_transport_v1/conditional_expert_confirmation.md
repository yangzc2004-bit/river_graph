# Conditional expert route confirmation

Two additional seeds (45 and 46) were run for the expert selected by the
validation pilot in each missingness family.

| family | selected expert | pilot seeds 42--44 | confirmation seeds 45--46 |
|---|---|---:|---:|
| Random point | RF-context | 1.4518 | 1.4561 |
| Partial time | local residual | 0.8094 | 0.8088 |
| Strict time | local residual | 0.9816 | 0.9890 |
| Spatial station holdout | RF-context | 2.6838 | 2.6956 |

The expert ordering is unchanged in both confirmation seeds. The routed model
therefore has a stable first version:

- random point and spatial station holdout → RF-context;
- partial and strict time holdout → local residual expert.

The route is determined by the missingness family known before prediction. It
does not inspect query labels or select an expert from the terminal test error.
