# Stage 1 review (2026-09-24)

## Decision

**Local ST357 data gate: passed. External case gate: pending. No training authorized.**

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

The external gate remains pending until a candidate manifest contains three
QC-frozen datasets, node metadata and edge metadata satisfying the frozen
availability thresholds. The candidate must be selected by those availability
facts before any prediction result is inspected.

## Route consequence

Do not launch Stage 2 baselines or any new training yet. The next authorized
work is to obtain or construct an external candidate manifest and to decide
whether the transfer protocol should add an explicitly time-aware support
condition. If no external case can pass, the manuscript falls back to the
ST357 DOC reconstruction and empirical uncertainty route.
