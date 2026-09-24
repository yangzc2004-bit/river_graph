# Stage 1 review (2026-09-24)

## Decision

**Local ST357 availability and QC checks: passed. External validation:
pending and unverified. The two screened HUC8 candidates did not pass. The
authorized internal fallback permits bounded ST357 baseline diagnostics;
the overall Stage 2 gate has not been evaluated.**

## Evidence

The data-only audit is reproducible with:

```bash
.venv/bin/python scripts/audit_transfer_data_gate.py --allow-external-pending
```

The current report is in `experiments/phase4_transfer/data_gate/`.

| target | graph rows | active stations | months | observed station-months |
|---|---:|---:|---:|---:|
| DOC | 357 | 357 | 654 | 22,571 |
| pH | 357 | 356 | 654 | 54,630 |
| specific conductance | 357 | 355 | 654 | 63,009 |

All three targets share station order, monthly grid, graph, hydro inputs,
static features and regime features. The current graph has 324 directed edges;
the largest active component contains about 60.8–61.1% of stations. The 80%
component threshold is an external candidate rule and is not applied to the
fixed ST357 cohort.

The retrospective replay of the 370 raw cached station files reproduces all
three local datasets exactly:

- DOC: 22,571 cells, zero mask mismatches, zero value difference;
- pH: 54,630 cells, zero mask mismatches, zero value difference;
- specific conductance: 63,009 cells, zero mask mismatches, zero value difference.

This replay is labelled retrospective; it does not upgrade old artifacts to
contemporaneous provenance.

## Task availability finding

The five existing HUC6 task IDs remain valid after graph membership checks.
The legacy task ID `510020` corresponds to canonical HUC6 `051002` in the
authoritative `huc_cd` metadata and is retained as an alias for compatibility.

All 15 target-analyte/HUC6 combinations have at least one K=5 same-month task.
However, all have **zero post-2020 task-months**. Therefore K-shot task
availability cannot be treated as evidence for future-month adaptation. The
existing E2a/E2b temporal masks and same-month K-shot tasks remain separate
experiments until a new time-aware support protocol is frozen.

## External probe

The bounded WQP station metadata probe received DOC and pH station inventories
and hit the 20 MB safety cap for specific conductance. These are screening
records only; they do not establish concentration counts, graph connectivity,
or eligibility. No external basin has been selected.

The external gate was screened using raw WQP availability evidence. No model
prediction was made for either external candidate. The two candidate screens are recorded in
`data_gate/candidate_02040104_raw_inventory.json` and
`data_gate/candidate_02030103_raw_inventory.json`:

| HUC8 | valid Stream stations in all three inventories | DOC station-months | result evidence |
|---|---:|---:|---|
| 02040104 | 114 | 7,064 | DOC complete; pH/EC bulk responses incomplete |
| 02030103 | 171 | 3,242 | DOC complete; pH/EC bulk responses incomplete |

Both candidates fail the frozen 10,000 station-month requirement for DOC
under the analyte-specific active-mask rule on the union of valid Stream
stations. The table's intersection counts are separate diagnostics, not the
denominators used to restrict DOC counts. The incomplete pH/EC files are not
counted as zero; their availability remains unknown. No graph was built for
either candidate. These two failed screens do not establish that all external
basins are ineligible. A complete independent external case remains pending.

## Route consequence

The Stage-1 route decision is in `stage1_route_decision.md`. The internal
ST357 multi-HUC6 fallback can proceed without claiming external replication.
Internal held-out-HUC6 results, if valid, must remain clearly distinguished
from validation in an independent external basin.

The immediate work is the limited Stage 2A same-analyte support diagnostic.
It uses privileged target-analyte source labels outside the whole target
HUC6, with support/query tasks only on its frozen largest component. It is
not a leave-one-analyte transfer result. Exploratory query results have
already been seen and the first two implementations require the documented
withdrawals/corrections in `stage2a_baseline_correction.md`.

The overall Stage 2 gate remains **not evaluated** until all required
baselines, controls, missingness regimes, and information comparisons are
complete. The target-unseen K=0 output scale and source-only selection
protocol also remain unresolved. Stage 3 transfer training is not authorized.
