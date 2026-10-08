# DOC river-pathway context

Exploratory connection of whole river form, cropped monitored tributary paths,
lake/reservoir context and observed receiving DOC variability. See
`study_plan.md` for the design and `research_decision.md` for interpretation.

## Reproduce

Use the project's uv-managed environment from the repository root:

```bash
uv run python scripts/analyze_doc_river_pathway_context_v1.py --bootstrap-draws 5000
uv run python scripts/plot_doc_river_pathway_context_v1.py
uv run python scripts/plot_doc_river_pathway_context_v1.py --chinese
uv run python scripts/verify_doc_river_pathway_context_v1.py --full
```

Add `--shortest-routes` to each command for the independent routing sensitivity.
No model training or historical experiment updates are performed.

## Main products

- `receiver_pathway_panel.csv`: one row per physical receiving network.
- `source_corridor_reaches.csv`: gauge-to-receiver reaches with cropped measures;
  a shared reach appears in each source path but is not independently counted.
- `receiving_variance_budgets.csv` / `variance_series.parquet`: exact receiving,
  upstream-mixture and discrepancy variance identities.
- `coverage_opportunities.csv` / `same_region_form_pairs.csv`: sampling coverage
  and all same-HUC4 elongated/broad candidate comparisons.
- `flow_availability.csv` / `flow_paired_contrasts.csv`: same-date flow-share
  sensitivity, retaining unavailable and incomplete cases.
- `form_contrasts.csv` / `geometry_associations.csv` / `signal_summary.csv`:
  receiver-equal estimates with whole-catchment-system bootstrap intervals.
- `figures/`: English/Chinese PNG and PDF figures generated from the tables.

The 80%-coverage form contrast is a positive descriptive lead; the absence of
adequate observed same-region, area-comparable pairs prevents isolated shape
attribution. DOC variance changes are not concentration reductions or mass loss.
