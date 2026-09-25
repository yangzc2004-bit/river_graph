# Stage-2 support-integrity audit report

Status: **completed bounded audit, 2026-09-25**.

This report is an addendum to the frozen Stage-2 route. It does not replace
`stage2_route_decision.md`, change an endpoint, or unlock Stage 3.

## Execution and audit

- 45/45 H2X units completed: three analytes × five target HUC6 tasks × three
  seeds.
- 46,116 product rows passed the pre-score label-free audit.
- Every unit has matching task, shuffle, spec, model-config, and runtime
  identities; no forbidden label column, query-label flag, duplicate key, or
  non-finite identifiable prediction was found.
- K=5 value-shuffle coverage is 100% of query rows. Site-shuffle coverage is
  1,542/3,843 K=5 query rows (40.1%); the remainder is explicitly
  `not_identifiable_by_design` because the frozen target month has too few
  alternative observed target-basin stations.

## K=5 readout

The reported delta is `MAE(true support) - MAE(shuffle)`, so negative values
favor the declared support. Cell estimates are equal-weighted in the pooled
summary; intervals are calendar-month-clustered within HUC6.

Native-unit deltas are summarized within analyte; DOC, pH, and conductance
are never combined into one numeric mean.

| Control | Analyte | Stable cells | True better | Cell CIs excluding zero | Mean cell delta |
|---|---|---:|---:|---:|---:|
| Value-shuffle | DOC | 5 | 4 | 4 | -0.099 mg/L |
| Value-shuffle | pH | 5 | 5 | 3 | -0.0065 pH units |
| Value-shuffle | Specific conductance | 5 | 4 | 1 | -7.36 uS/cm |
| Site-shuffle | DOC | 2 | 2 | 1 | -0.469 mg/L |
| Site-shuffle | pH | 2 | 2 | 1 | -0.0091 pH units |
| Site-shuffle | Specific conductance | 2 | 2 | 2 | -36.26 uS/cm |

The site-shuffle summary excludes two cells with fewer than 20 identifiable
task-months, which remain in `shuffle_metrics.csv` with an `unstable_n_lt20`
flag. Site-shuffle cells are not representative of all five HUC6 tasks because
identifiability is determined by monitoring density.

DOC value-shuffle results are directionally supportive in four of five basins,
but HUC6 `101900` favors the shuffled control. The existing DOC no-harm failure
at HUC6 `102701` remains unchanged; this audit does not remove it.

## Decision

The audit supports a bounded statement that H2X predictions can respond to the
declared target support cells under this frozen task. It does not establish a
general few-shot transfer claim, erase basin-level harm, or justify a new
architecture, confirmation matrix, external replication, blind-spot claim, or
active-sampling claim. `stage2_unlocked=false` and `stage3_unlocked=false`
remain in force.
