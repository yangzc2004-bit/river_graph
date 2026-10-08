# River form event observations

Public high-frequency DOC source suitability, original storm-interval coverage,
timestamp conflicts and unexplained repeated cross-year sequences. The research
question and decision are in `study_plan.md` and `research_decision.md`.

From the repository root, using the project's uv environment:

```bash
uv run python scripts/fetch_doc_river_external_event_catalog_v1.py --source arctic
uv run python scripts/fetch_doc_river_external_event_catalog_v1.py --source kervidy
uv run python scripts/analyze_doc_river_external_event_catalog_v1.py
uv run python scripts/plot_doc_river_external_event_catalog_v1.py
uv run python scripts/plot_doc_river_external_event_catalog_v1.py --chinese
uv run pytest tests/test_river_external_events.py
```

Fetch commands refuse to replace existing manifests. Reuse the recorded local
objects when reproducing this release; later source versions need a separate
retrieval record. The Kervidy command records/checks previously cached files.

## Products

- `retrieval_manifest.json`: Arctic EML attributes, clocks, units and objects.
- `kervidy_retrieval_manifest.json`: repository version, description and objects.
- `analysis/arctic_doc_record_audit.parquet`: canonical timestamps; conflicting
  DOC is missing, never averaged.
- `analysis/duplicate_timestamps.csv`: source alternatives and row indices.
- `analysis/cross_year_sequences.csv`: exact ordered-value diagnostics.
- `analysis/author_storm_audit.csv`: all 69 unchanged author intervals.
- `analysis/arctic_station_years.csv`, `author_storm_years.csv`,
  `kervidy_years.csv`: source resolution, coordinates and coverage summaries.
- `analysis/source_suitability.csv`: all four source families and spatial scope.
- `figures/`: English and Chinese observed-record and source-diagnostic figures.

DOC-only record coverage does not satisfy the joint DOC-flow requirement in
the study plan. All acquired chemistry tables lack continuous discharge. No
new river morphology class or independent form effect is inferred here.
