# EcoHydroGraph evidence audit (v1)

Date: 2026-09-27

## Scope

This review checks the LaTeX draft against the frozen T8 matched-budget synthesis, the T8 paired bootstrap table, and the T9 five-seed ensemble summary. It checks numerical transcription, claim strength, and whether each main result has an uncertainty statement.

## Checks passed

- The 12 analyte-by-scenario MAEs and relative reductions in Table 1 match `t8_synthesis/main_summary.csv` after rounding.
- The DOC and specific-conductance ranges in the abstract and Results match the frozen reductions (DOC 37.47--47.24%; conductance 38.84--57.19%).
- The pH range and seed-direction counts match the frozen table (0.51--3.77%; 3/5 to 4/5 seeds).
- The history-window values match the three-seed diagnostic: DOC 1.420 to 1.415, pH 0.362 to 0.301, and conductance 279.12 to 245.75.
- The current-only comparisons (pH +20.9%, conductance +13.8%, DOC +0.5%) match the T5 summary.
- The new station-clustered paired intervals match `t8_synthesis/paired_cluster_ci.csv`; pH under unmonitored stations is the only main family whose station-clustered interval includes zero.
- The manuscript does not claim calibrated intervals, stable blind spots, active sampling, causal effects, or universal transfer.

## Changes made in this review

1. Added a station-clustered 95% paired-bootstrap interval table beside the main MAE table.
2. Changed the abstract conclusion from establishing temporal memory to showing that temporal context can improve reconstruction.
3. Stated that the fixed spatial trunk and shared masks make the comparison a temporal-extension test, not a ranking of all spatial models.
4. Clarified that the history ablations identify the value of additional context but do not isolate chronological order or target-analyte history.
5. Kept the four missingness scenarios descriptive throughout the manuscript and figures.

## Evidence still absent from the main draft

- The main manuscript compares EcoHydroGraph with its matched snapshot graph baseline. EcoRF and message-ablation results remain useful contextual evidence but are not included as same-protocol three-analyte baselines in this draft.
- The formal runs use the frozen 10-epoch/patience-3 budget. The paper should describe this as the matched study budget, not as evidence of full optimization or convergence.
- The cohort is ST357 only. No external-basin replication is included.
- The uncertainty product and R2 ranking result are supporting context, not a new monitoring-design claim in this manuscript.

## Editorial recommendation

The central result is ready for a full draft: adding a temporal wrapper to the fixed ecology-aware river graph produces large, reproducible gains for DOC and specific conductance, while pH remains near parity under station holdout. The next manuscript pass should add literature on water-quality gap filling and spatiotemporal graph learning, then decide whether the existing EcoRF results can be included as a strictly matched supplementary baseline. No new large training matrix is required for this review pass.
