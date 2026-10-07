# Observed river signal mechanisms

Follow-up to real morphology, geometric routing, observed transport and flow
responses. Read `research_decision.md` for the scientific interpretation and
`study_plan.md` for the fixed diagnostic definitions and later scale sensitivity.
This version performs no model training or parameter selection.

## Reproduce

Run from the repository root using its uv-managed environment:

```bash
uv run python scripts/analyze_doc_river_signal_mechanisms_v1.py --bootstrap-draws 5000
uv run python scripts/plot_doc_river_signal_mechanisms_v1.py
uv run python scripts/plot_doc_river_signal_mechanisms_v1.py --chinese
uv run python scripts/verify_doc_river_signal_mechanisms_v1.py
uv run pytest
uv run ruff check .
uv run python scripts/audit_artifacts.py --verify
```

## Inspect

- `analysis/mixing_connections.csv`: exact native variance budget per connection.
- `analysis/mixing_receivers.csv`: receiver-equal structural/signal summaries.
- `analysis/mixing_summary.csv`: system intervals and HUC4 sensitivity.
- `analysis/calendar_anomalies.parquet`, `calendar_fits.json`: same-date signals
  and common calendar fits; these are observed diagnostics, not new imputation.
- `analysis/arrival_opportunity.parquet`: exact branch-minus-mean-path input gap
  and sensitivity using the same saved model coefficients.
- `analysis/arrival_connections.csv`, `arrival_receivers.csv`, `arrival_summary.csv`.
- `analysis/monthly_flow_ledger.parquet`: positive measured flows, area/water
  availability screens and apparent monthly concentration departure.
- `analysis/budget_availability.csv`: includes unique receiver-month counts.
- `analysis/apparent_departure_summary.csv`: concentration differences, not loads
  or measured retention rates.
- `analysis/omitted_system_sensitivity.csv`: every leave-one-system estimate.
- `figures/tributary_integration_and_arrival[_cn].{png,pdf}`: structural mechanism.
- `figures/variability_and_channel_budget[_cn].{png,pdf}`: variability and budgets.

The 59 connections share receivers and monitoring systems. Do not treat 3,026
connection-months as independent ecological observations, interpret modeled
mixing variance reduction as concentration removal, or assign one sparse
receiver's value to an entire morphology class. The source records and saved
mean-delay coefficients belong to earlier exploratory work; this analysis
preserves that lineage and is not external validation.
