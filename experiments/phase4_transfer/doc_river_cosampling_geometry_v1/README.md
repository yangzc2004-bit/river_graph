# Co-sampling coverage for river-form DOC mechanisms

Assess accepted activity dates on all 297 mapped station networks, preserve
actual paths and original form classes, and identify where elongated/broad
timing comparisons need additional observations. See `study_plan.md`,
`disjoint_sampling_plan.md` and `research_decision.md`.

## Reproduce

Use the repository's uv-managed environment:

```bash
uv run python scripts/analyze_doc_river_cosampling_geometry_v1.py --bootstrap-draws 5000
uv run python scripts/plot_doc_river_cosampling_geometry_v1.py
uv run python scripts/plot_doc_river_cosampling_geometry_v1.py --chinese
uv run python scripts/verify_doc_river_cosampling_geometry_v1.py --full
uv run python scripts/audit_doc_river_disjoint_cosampling_v1.py
uv run python scripts/verify_doc_river_disjoint_cosampling_v1.py
```

The first command reconciles raw DOC to the saved source-role dataset. After
that pass, `--from-archive` reuses the recorded accepted activity table while
checking the raw archive hashes. It does not open excluded target-role labels.

## Products

- `network_inventory.csv`: all included/excluded receiver instances, candidate
  availability and old monthly membership.
- `doc_activities.parquet`, `raw_monthly_reconciliation.csv`: accepted activity
  archive and independent station-month reconciliation.
- `candidate_gauges.csv`, `candidate_pair_inventory.csv`: original routed
  frontier, metadata-only source selection and short-record pairs.
- `same_day_activities.parquet`, `clock_ledger.csv`, `sampling_months.csv`:
  complete local-date activity sets, clock uncertainty and actual sampling gaps.
- `network_geometry.csv`, `source_corridor_reaches.csv`: cropped source paths,
  entry positions, shared corridor and mapped storage context.
- `network_signals.csv`, `signal_series.parquet`, `signal_summary.csv`:
  within-month primary signals and retained calendar/raw sensitivities.
- `all_disjoint_pair_inventory.csv`, `all_disjoint_sampling_summary.csv`:
  sampling availability for every non-nested candidate gauge pair.
- `all_disjoint_form_pair_priorities.csv`, `non_nested_form_pair_priorities.csv`:
  all 105 same-region comparable-area opportunities and the 91 non-nested
  receiving-pair shortlist. These counts are not independent replicates.
- `figures/`: bilingual scientific PNG/PDF figures with plotted-input receipts.

No new model training, lag fitting, data fabrication or historical-result
replacement is performed. Same calendar day does not imply simultaneous
sampling or resolve event travel time. The figures show the selected sampling
population, including absent form combinations rather than zero DOC effects.
