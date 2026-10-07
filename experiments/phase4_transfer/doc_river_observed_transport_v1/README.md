# Observed DOC and measured river-path arrival

The question, comparison choices and sampling rules were fixed in
`study_plan.md` before the outer evaluations. See `research_decision.md` for
results, interpretation and the next morphology-centered experiment.

## Reproduce

From the repository root, using its uv-managed environment:

```bash
uv run python scripts/analyze_doc_river_observed_transport_v1.py --bootstrap-draws 5000
uv run python scripts/plot_doc_river_observed_transport_v1.py
uv run python scripts/plot_doc_river_observed_transport_v1.py --chinese
uv run python scripts/verify_doc_river_observed_transport_v1.py
```

Required local inputs are the ST357 dataset, the permitted source cells and
screened tributary inventory from `doc_river_mechanisms_v1`, and the fixed
morphology context from `doc_river_routing_mechanisms_v1`. Dataset contents are
not committed in this directory. No neural training is invoked.

## Product definitions

- `analysis/input_records.parquet`: one eligible connection/calendar-month
  record, with observed current/previous upstream DOC and scoring-only receiver
  DOC. Source weights are incremental-area proxies, not measured flow.
- `analysis/connection_predictions.parquet`: one record per connection,
  observed receiver month and operator. Shared receivers repeat across
  connections; they are not additional ecological replicates.
- `analysis/fitted_states.json`, `selection_trials.csv`: source-only
  imputation, normalization, coefficients and nested lag choices for every
  held monitoring system.
- `connection_metrics.csv` and `receiver_metrics.csv`: dates first, then
  connections within receivers, then equal receiver averages.
- `receiver_coverage.csv`: **unique** receiver-month and high-DOC counts.
- `operator_summary.csv` and `paired_gains.csv`: connected-system bootstrap
  intervals, HUC4 sensitivity, and retained class summaries. Groups 1/2/3 mean
  elongated-tributary-rich, mainstem-sparse, and broad-tributary-rich.
- `system_sensitivity.csv`: effects after omitting each connected system.
- `window_availability.csv`: availability-only audit of longer calendar windows.
- `figures/`: scientific PNG/PDF in English and Chinese, with figure sources.

This study uses observed upstream concentration. It is not an unmonitored K0
evaluation or an external-basin experiment. Time weights are monthly empirical
arrival proxies; this study does not estimate physical travel velocities.
