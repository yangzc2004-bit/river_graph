# Observed DOC response at four actual confluence arrangements

The full-archive and joint-calendar laboratory comparison is summarized in
`research_decision.md`. Figures use actual observations and mapped path lengths.
They are separate from the preceding four-procedure predictive-model comparison.

## Reproduce

From the repository root with the uv-managed GNN/development environment:

```bash
uv run python scripts/analyze_doc_river_joint_campaigns_v1.py --bootstrap-draws 5000
uv run python scripts/plot_doc_river_joint_campaigns_v1.py
uv run python scripts/plot_doc_river_joint_campaigns_v1.py --chinese
uv run python scripts/verify_doc_river_joint_campaigns_v1.py
```

The analysis reads committed source tables in `doc_river_event_observations_v1`
and `doc_river_flow_mixing_v1`; their saved retrieval manifests and raw-source
records supply data lineage. Downloading a new SITES object version is not
required for this replay. No prediction model, imputed DOC or constructed pulse
enters these comparisons.

## Products

- `analysis/campaign_ledger.parquet`: every matched campaign, including exclusions.
- `analysis/adjusted_campaigns.parquet`: observed concentrations, partial mixture,
  daily flow, fitted-pattern residuals and flow state for both calendars.
- `analysis/configuration_comparison.csv`: point estimates and year intervals.
- `analysis/paired_configuration_contrasts.csv`: joint-year paired differences.
- `analysis/flow_state_comparison.csv` and `flow_state_contrasts.csv`: within-site
  comparisons, retaining the original full-population seasonal projection.
- `analysis/leave_year_out.csv`: refitted year-omission sensitivity, added after
  inspecting the first estimates to check whether one year supplies the result.
- `analysis/water_coverage_sensitivity.csv`: available and unavailable flow-screen
  comparisons, with actual counts.
- `figures/joint_calendar_structure_and_doc.png` / `_cn.png`: arrangement,
  variability, seasonal sensitivity and water coverage.
- `figures/joint_calendar_actual_doc.png` / `_cn.png`: all observed joint-date
  concentration points; matching PDF versions are included.

## Interpretation

There are four local configurations in one nested research catchment. Shared
receiving observations and calendar years are preserved together in the main
resampling. The 561 configuration-campaigns are not independent rivers; the
47-date joint calendar is not an external replication of ST357 morphology.
Intervals measure temporal repeatability, not a population geometry effect.

The concentration mixture is a partial measured input. Its mismatch with the
outlet does not identify removal, production or a travel time. Original
whole-network classes, model predictions and endpoints remain unchanged.

SITES data are attributed under the parent's recorded licence and object URLs.
This study has been made possible by data provided by the Swedish Infrastructure
for Ecosystem Science (SITES).
