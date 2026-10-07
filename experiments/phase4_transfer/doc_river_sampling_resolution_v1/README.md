# Sampling resolution of DOC river-structure mechanisms

This follow-up reconstructs sampling dates on the preceding 59 monitored
tributary pairs. It compares monthly and dated DOC signals on matched months,
joins measured daily flow, and establishes the event-resolution available for
studying common-trunk storage. No prediction model is retrained.

Read [the scientific decision](research_decision.md) and [study design](study_plan.md).

## Reproduce

From the repository root, with cached WQP/NWIS data and the ST357 dataset:

```bash
uv run python scripts/analyze_doc_river_sampling_resolution_v1.py --bootstrap-draws 5000
uv run python scripts/plot_doc_river_sampling_resolution_v1.py
uv run python scripts/plot_doc_river_sampling_resolution_v1.py --chinese
uv run python scripts/verify_doc_river_sampling_resolution_v1.py
```

## Evidence

- `analysis/doc_activities.parquet`: original dates/times, activity identifiers
  and DOC; repeated result counts retained.
- `analysis/monthly_reconciliation.parquet`: accepted row-weighted monthly
  means versus frozen DOC and allowed source cells.
- `analysis/sample_triplets.parquet`: every fixed pair-month, the selected
  activities, dates, concentration and measured-flow fields.
- `analysis/station_cadence.csv`, `sampling_intervals.csv`: actual sampling days.
- `analysis/date_cut_ledger.csv`: all date thresholds and sample denominators.
- `analysis/mixing_connections.csv`, `mixing_receivers.csv`, `signal_summary.csv`:
  matched-calendar variance comparisons and system-bootstrap intervals.
- `analysis/hydro_date_summary.csv`: unique source-date flow comparisons,
  availability and measured change, without gap filling.
- `analysis/event_opportunity.csv`: dense-sampling opportunity with/without
  mapped common-trunk storage.
- `analysis/daily_flow_context.parquet`: measured days within archived DOC
  months for plotting; not a synthetic concentration series.
- `figures/`: English and Chinese PNG/PDF figures with source records.

Prior frozen results are preserved. This is a within-ST357 follow-up and not
external validation. The next experiment isolates branch dispersion and
common-trunk storage under identical forcing.
