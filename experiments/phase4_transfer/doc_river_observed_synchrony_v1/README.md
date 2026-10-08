# Actual tributary coordination and receiving DOC

This follow-up connects the controlled arrival-overlap mechanism to accepted
DOC observations on the same fixed monitored river networks. Complete-network
form, cropped paths and shared corridors are retained. See `study_plan.md` and
`research_decision.md` for the scientific question and interpretation.

## Reproduce

From the repository root, using the uv-managed environment:

```bash
uv run python scripts/analyze_doc_river_observed_synchrony_v1.py --bootstrap-draws 5000
uv run python scripts/plot_doc_river_observed_synchrony_v1.py
uv run python scripts/plot_doc_river_observed_synchrony_v1.py --chinese
uv run python scripts/verify_doc_river_observed_synchrony_v1.py --full
```

Add `--shortest-routes` to each command for the independently retained routing
sensitivity. No new model training, physical lag fitting or historical updates
are performed.

## Products

- `network_metrics.csv` / `observed_series.parquet`: common-calendar source and
  receiving anomalies, coordination, fluctuation references and excursion counts.
- `sampling_availability.csv`: every strict/relaxed sample-span subset, including
  those that do not support stable comparisons.
- `signal_summary.csv` / `form_contrasts.csv`: receiver-equal summaries and the
  original broad-minus-elongated comparison with whole-system intervals.
- `geometry_synchrony_associations.csv`: observed coordination and mapped path
  context against independently measured receiving signals.
- `same_date_averaging_contrasts.csv`: activity and monthly values compared on
  identical receiver dates and source sets.
- `period_inventory.csv` / `network_periods.csv` / `within_network_periods.csv`:
  fixed five-year periods, availability and receiver-centered changes.
- `dense_case_*`: the previously metadata-selected Loch Vale case, including
  annual periods, same-day traces, activity/daily-mean sensitivity and year-block
  intervals. Annual sample sizes count years, not independent river networks.
- `figures/`: English/Chinese PNG and PDF figures; receipts bind plotted inputs.

Excursions are concentration anomalies above each series' empirical upper
quartile, with a separately retained 90th-percentile sensitivity. They are not
regulatory exceedances or estimates of DOC mass transport.
