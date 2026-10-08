# Complete river form, monitored tributary overlap and receiving DOC

## Research advance

Connect the **complete mapped river network** to one fixed, non-overlapping set
of observed upstream inputs. This extends the earlier two-source connections to
multi-gauge frontiers without counting nested catchments twice. It retains the
original elongated, mainstem-dominated and broad form classes.

The primary analysis contains 32 distinct receiving networks, 15 with at least
three independent upstream inputs, in seven overlapping-catchment systems.
Median represented catchment area is 71.9%. Original geometric-trunk routing is
primary; `shortest_route_sensitivity/` repeats the analysis on saved shortest
routes. These are observational ST357 source-role analyses, not external
validation or new model training.

`study_plan.md` records the selection and signal definitions. A later section
records the weekly-resolution follow-up prompted by the coverage inventory.
`research_decision.md` explains the findings and next scientific question.

## Products

- `analysis/network_inventory.csv`: all 297 station-network instances, including
  aliases on the 295 distinct receiving COMIDs and reasons for non-inclusion.
- `analysis/candidate_gauges.csv`: candidate frontier, selected gauges, fixed
  area shares and gauge-to-receiver distances.
- `analysis/receiver_signals.csv`, `signal_summary.csv`: exact multi-source
  covariance decomposition, observed receiving relationships and 5,000-draw
  whole-system bootstrap summaries.
- `analysis/form_contrasts.csv`, `geometry_signal_associations.csv`: descriptive
  broad-minus-elongated contrasts and whole/covered path associations, retaining
  uncertain results.
- `analysis/monthly_series.parquet`, `selected_activities.parquet`: actual
  monthly records and metadata-selected activities; no synthetic DOC waves.
- `analysis/raw_monthly_reconciliation.csv`, `sampling_ledger.csv`: accepted
  raw results reconciled to the frozen dataset, and actual date-alignment counts.
- `analysis/weekly_case_*.csv/parquet`: one sampling-density-selected Loch Vale
  case, exactly common daily samples and within-month fluctuations. Same-day
  activity choice is independent of DOC; daily means are a replicate sensitivity.
- `figures/`: real-network maps, monthly summaries and actual weekly DOC in
  English/Chinese PNG/PDF, with input/output receipts.

The dated cuts have different eligible populations. Mixing variance reduction
is calculated from concentration signals and fixed catchment-area shares; it is
not DOC removal, measured flow-weighted mixing or downstream peak attenuation.
The weekly case includes a lake outlet and does not add independent rivers to
the form comparison.

## Reproduce

Use the existing uv environment and local raw/geometric caches. Run sequentially
within each analysis root; English and Chinese plots share metadata tables.

```bash
uv run python scripts/analyze_doc_river_monitored_arrivals_v1.py --bootstrap-draws 5000
uv run python scripts/plot_doc_river_monitored_arrivals_v1.py
uv run python scripts/plot_doc_river_monitored_arrivals_v1.py --chinese
uv run python scripts/verify_doc_river_monitored_arrivals_v1.py --full

uv run python scripts/analyze_doc_river_monitored_arrivals_v1.py --bootstrap-draws 5000 --shortest-routes
uv run python scripts/plot_doc_river_monitored_arrivals_v1.py --shortest-routes
uv run python scripts/plot_doc_river_monitored_arrivals_v1.py --shortest-routes --chinese
uv run python scripts/verify_doc_river_monitored_arrivals_v1.py --shortest-routes --full
```

Run source-role and geometric analyses from the saved inputs before plotting.
The ledgers bind the three source masks, raw WQP archives, routing files,
complete morphology panel and executing source snapshot. Large geographic and
raw observation caches stay in their existing local locations.
