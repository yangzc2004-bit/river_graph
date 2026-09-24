# Stage 2A same-analyte support diagnostic

This product is an internal diagnostic, not held-out-analyte transfer. The
target analyte is visible in source HUC6 rows when constructing its
same-analyte climatology. The purpose is only to test whether the frozen
support/query tasks contain local signal before a shared cross-analyte model is
considered.

The runner used the frozen largest HUC6 component rows, source-only Q90
thresholds, and three task seeds. The evaluator averaged task seeds within
basin-month before cluster bootstrap and kept analyte units separate.

The apparent support direction is positive for DOC, pH, and conductance, but
`verdict.json` intentionally keeps `stage2_gate_pass=false` and
`stage3_unlocked=false`: the complete Stage-2 controls (EcoRF, single-analyte
H2X, matched no-graph/no-ecology controls, and information decomposition) and
the held-out-analyte isolation protocol are not yet implemented.
