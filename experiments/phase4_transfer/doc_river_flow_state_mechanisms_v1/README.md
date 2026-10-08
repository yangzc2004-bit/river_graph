# Observed water-state mechanisms at actual confluences

This follow-up separates changes in incoming DOC fluctuations, branch water
contribution, and the remaining outlet departure on the unchanged joint-calendar
and all-calendar populations. It refines the meaning of the preceding observed
outlet/mixture ratio without interpreting that ratio as channel retention.

## Reproduce

From the repository root, using the project uv environment:

```bash
uv run python scripts/analyze_doc_river_flow_state_mechanisms_v1.py --bootstrap-draws 5000
uv run python scripts/verify_doc_river_flow_state_mechanisms_v1.py
uv run python scripts/plot_doc_river_flow_state_mechanisms_v1.py
uv run python scripts/plot_doc_river_flow_state_mechanisms_v1.py --chinese
```

No model is fitted. Source observations and the hydro-only state definitions
come from the preceding `doc_river_joint_campaigns_v1` study. Its results remain
unchanged. The new calculation refits its annual-harmonic plus linear-time
projection within each complete-year bootstrap sample.

## Outputs

- `analysis/state_summary.csv`: every population, confluence and eligible flow
  state, with absolute DOC SD, correlation, mean branch flow fraction, fixed-
  weight mixing potential, departure identities, and repeatability intervals.
- `analysis/high_minus_low.csv`: exact log-ratio contributions and all-order
  attribution of the mixing-summary change, with paired year-bootstrap intervals.
- `analysis/analytic_substitutions.csv`: all eight low/high summary substitutions
  at each configuration. These are mathematical constructions, not interventions.
- `analysis/water_budget_ledger.parquet`: every original compared date, including
  negative implied unmonitored concentrations and excluded water budgets.
- `analysis/water_budget_counts.csv`: mutually exclusive budget categories.
- `figures/`: English and Chinese standalone PNG/PDF figures.
- `research_decision.md`: findings and their role in the river-arrangement story.

SD is expressed in mg C/L. Mixing potential is a percentage reduction in
fixed-weight mixture variance relative to flow-share-weighted branch variances,
not a percentage of DOC removed. The denominator of a no-processing solution
is unmonitored water, not an observed tributary. These four monitored arrangements
are nested within one Krycklan catchment, and are not four independent network
morphology classes.

This study has been made possible by data provided by the Swedish Infrastructure
for Ecosystem Science (SITES).
