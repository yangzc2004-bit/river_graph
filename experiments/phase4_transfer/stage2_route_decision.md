# Stage 2 route decision

Status: **frozen after the Stage-2C v1.1 pilot and metrics-only gate report,
2026-09-25**.

## Evidence reviewed

- Stage-2B deterministic baselines and the three-analyte, five-basin task
  manifest are unchanged.
- Stage-2C v1.1 completed all 180 planned units under the reviewed 50-epoch
  bounded pilot budget. Prediction products passed the exact task inventory,
  identity, finite-value, and label-free product audit.
- H2X K=5 versus the source-selected simple baseline has positive pooled
  reductions for DOC (32.9%), pH (16.9%), and specific conductance (31.6%);
  the paired bootstrap intervals exclude zero for all three pooled summaries.
- The K=5 direction is present in all five target HUC6 tasks for each analyte,
  and the pooled H2X K curve passes the K5-not-worse-than-K0/K1 check.
- The no-harm condition fails: DOC at target HUC6 `102701` is 17.6% worse than
  the selected simple baseline.
- Support value-shuffle and support site-shuffle controls were not run. The
  information decomposition is metrics-only and covers the one frozen
  same-month spatial HUC6 family; no secondary missingness family is available.

## Decision

`stage2_unlocked=false` and `stage3_unlocked=false` remain in force. The
pilot establishes a useful transfer signal and a clear analyte/basin boundary,
but it does not authorize a new shared task-conditioned architecture,
confirmatory five-seed expansion, or external-basin replication.

The retained scientific route is:

1. Use the pilot as bounded evidence for ecology-time and local-support value
   that depends on analyte and basin.
2. Keep the existing DOC reconstruction and empirical uncertainty work as the
   manuscript fallback/mainline.
3. Treat support shuffles as an optional, separately versioned integrity audit.
   Running them would not erase the observed DOC no-harm failure; they cannot
   be used to select a new model or endpoint.
4. Do not make topology, blind-spot, active-sampling, external-transfer, or
   universal few-shot claims from this route.

All quantitative products and claim statuses are recorded in
`stage2_gate_v1/`, `stage2_information_decomposition_v1/`, and
`docs/paper/claim_evidence_matrix_v4.csv`.
