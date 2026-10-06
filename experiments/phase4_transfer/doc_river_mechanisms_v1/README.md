# River morphology and DOC mechanisms

First observational study of tributary integration, terrestrial source contrasts
and longitudinal processing, with an illustrative fixed-budget routing experiment.

- `study_plan.md`: study population, comparisons and documented review changes.
- `research_decision.md`: results and next research.
- `metadata/`: independent official USGS gauge drainage inventory.
- `analysis/`: screened main tables and observed source records.
- `figures/`: English/Chinese PNG and PDF scientific figures.
- `preliminary_nhd_candidates/`: pre-location-screen exploratory results retained
  for traceability; **not site-confirmed confluences**.
- `code_snapshot/`: analysis dependencies as executed, including pre-existing
  untracked dependencies without adding unrelated historical files to this version.

Reproduce from the repository root:

```bash
uv run python scripts/fetch_doc_river_gauge_metadata_v1.py
uv run python scripts/analyze_doc_river_mechanisms_v1.py --bootstrap-draws 5000
uv run python scripts/verify_doc_river_mechanisms_v1.py
uv run python scripts/plot_doc_river_mechanisms_v1.py
uv run python scripts/plot_doc_river_mechanisms_v1.py --chinese
```

The active research workspace has two historical analysis dependencies that
were not tracked before this study. Exact copies are included rather than
committing unrelated historical development. In a fresh checkout, restore them
only when absent, before the analysis commands:

```bash
cp -n experiments/phase4_transfer/doc_river_mechanisms_v1/code_snapshot/src/river_graph/analysis/river_doc_structure.py src/river_graph/analysis/
cp -n experiments/phase4_transfer/doc_river_mechanisms_v1/code_snapshot/src/river_graph/topology/river_structure.py src/river_graph/topology/
```

The study uses source-training union 142/143/144, never the geographic/external
confirmation outcomes. No model fitting or original graph changes occur here.
Paired sample records are observations, not full-grid prediction products.
