# DOC flow responses in fixed real river forms

Question: does DOC change differently from low to high flow in elongated,
mainstem-sparse and broad tributary-rich river networks?

Read `study_plan.md` for the fixed design and `research_decision.md` for results.
This extends the source-role morphology analysis using within-river changes
and unchanged environmentally matched pairs.

## Reproduce

From the repository root and the uv-managed environment:

```bash
uv run python scripts/analyze_doc_river_form_flow_response_v1.py --bootstrap-draws 5000
uv run python scripts/plot_doc_river_form_flow_response_v1.py
uv run python scripts/plot_doc_river_form_flow_response_v1.py --chinese
uv run python scripts/verify_doc_river_form_flow_response_v1.py
```

Required local inputs are ST357, the permitted source-cell array, and the fixed
morphology panel and original pair table. Paths are in `analysis_sources.json`.

## Products

- `flow_reference_months.parquet`, `flow_reference_thresholds.csv`: measured
  flow-only references including months with no DOC, local terciles and status.
- `observed_flow_months.parquet`: permitted DOC rows and independently defined
  flow states; includes unavailable hydro rather than silently dropping it.
- `flow_response_inclusion.csv`: eligibility of every fixed station under both
  flow and observed-temperature populations.
- `station_flow_responses.csv`, `station_state_summaries.csv`,
  `class_flow_responses.csv`: individual responses, state counts/values and
  station-equal regional summaries.
- `shared_pair_months.parquet`, `pair_inclusion.csv`, `pair_state_summaries.csv`,
  `pair_response_evidence.csv`, `pair_response_contrasts.csv`: identical-calendar
  comparisons and equal-pair differences in response; all original pairs retained.
- `joint_state_months.parquet`, `joint_state_inclusion.csv`,
  `joint_state_evidence.csv`, `joint_state_contrasts.csv`,
  `joint_calendar_counts.csv`: simultaneous local-low/local-high month diagnostics.
- `omitted_region_influence.csv`, `joint_omitted_region_influence.csv`: regional
  sensitivity of paired findings.
- `structure_response_associations.csv`, `structure_covariate_inclusion.csv`:
  separately fitted morphology blocks and explicit missing-covariate exclusions.
- `fitted_states.json`: individual response coefficients, ranks and fixed
  explanatory-model coefficients/scales.
- `figures/`: three scientific figures as PNG/PDF in English and Chinese.

Primary response is the season/year-adjusted **log1p high-minus-low DOC**.
Native-scale results remain separate. Flow states use local terciles; high DOC
uses the fixed 10 mg/L threshold. Bootstrap resamples entire HUC4 groups 5,000
times; raw months are not independent bootstrap replicates. This is a measured
monthly association study, with no retraining of the DOC reconstruction model.
