# River morphology, source placement and hydrologic activation

This is a source-only, monthly DOC-flow mechanism study. Read
`research_decision.md` for results and the next scientific question.
`study_plan.md` records the questions and analysis design before response fitting.

## Reproduce from the repository root

```bash
uv run python scripts/analyze_doc_river_hydrologic_activation_v1.py --bootstrap-draws 5000
uv run python scripts/verify_doc_river_hydrologic_activation_v1.py
uv run python scripts/plot_doc_river_hydrologic_activation_v1.py
uv run python scripts/plot_doc_river_hydrologic_activation_v1.py --chinese
```

Requires the local ST357 dataset and the earlier mechanism/source-placement
products. Raw landscape acquisition is not repeated. Chinese figure rendering
uses the macOS Arial Unicode font; English figures do not need it.

## Inspectable evidence

- `monthly_source_panel.parquet`: all permitted DOC station-months, hydro
  visibility, negative/reverse-flow exclusions and fixed landscape metadata.
- `within_station_residuals.parquet`: response and flow terms after removing
  each station's nuisance variation, for all three populations.
- `station_response_fits.csv`: included/excluded station fits, sample counts,
  C-Q slopes, curvature, interquartile contrast and condition diagnostics.
- `hydrologic_moderation.csv`: complete pooled flow-interaction coefficients,
  including unidentified terms explicitly flagged rather than interpreted.
- `adjusted_class_responses.csv` and `adjusted_class_contrasts.csv`: common-
  environment responses and direct class differences, with paired intervals.
- `class_response_distributions.csv`: empirical class medians and positive
  response fractions, with HUC4-cluster bootstrap.
- `huc4_response_predictions.csv` and `huc4_response_gains.csv`: matched Ridge
  response-descriptor diagnostics, with station and HUC4 bootstrap.

These products are not monthly neural predictions. Geographic response-
descriptor error must not be reported as current K0 DOC-model performance.
Station-specific nuisance removal is for retrospective response estimation;
it is not a forward prediction algorithm for an unmonitored station.

`code_snapshot/` retains executing scripts, the library, tests and imported
scientific helpers. The prior untracked `river_doc_structure.py` dependency is
saved as a snapshot without committing unrelated earlier local work. A fresh
checkout missing that helper can restore the exact snapshot to its `src/`
location. Review any different local implementation before replacing it.

## Implementation notes

The numeric controls are fixed in `AMOUNT_CONTROLS`. Source amount and placement
refer to the same represented catchments as the preceding reconciled study.
Pooled fitting gives each station total weight one. Bootstrap draws repeat
whole HUC4 groups, preserving every station's month series. Recent-period
covariates use their own standard deviations; figure notes identify this.

During verification, a Parquet NaN/None conversion in unmatched metadata was
fixed by explicit nullable string/boolean columns. This changed serialization,
not the response coefficients. Direct paired class contrasts were added to
the planned class analysis so comparisons do not rely on overlapping marginal
intervals. No feature selection or response-driven cohort change was made.
