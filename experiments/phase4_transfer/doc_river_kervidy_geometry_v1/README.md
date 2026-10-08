# Kervidy river geometry and DOC/flow observations

Single-catchment, geographically anchored field case for the DOC river-form
study. Start with `research_decision.md` and the real-network figure. No model
training and no independent narrow-versus-broad form ranking.

## Reproduce

```bash
uv run python scripts/fetch_doc_river_kervidy_geometry_v1.py
uv run python scripts/analyze_doc_river_kervidy_geometry_v1.py
uv run python scripts/plot_doc_river_kervidy_geometry_v1.py
uv run python scripts/plot_doc_river_kervidy_geometry_v1.py --chinese
```

The chemistry raw file comes from the separate
`fetch_doc_river_external_event_catalog_v1.py --source kervidy` retrieval.
An existing retrieval manifest is preserved; only analysis/plots are rerun.
Source raw objects and the partial initial pagination attempt are gitignored.

- `analysis/corrected_doc_flow_matches.parquet`: corrected DOC observations with
  original flow timestamps and exact/nearest match labels; no interpolation.
- `analysis/flow_selected_windows.csv`: all four flow-selected annual windows,
  including the incomplete and flow-censored cases.
- `analysis/mapped_geometry.json`: actual mapped rivers, official basin and
  monitoring gauge, EPSG:2154.
- `analysis/year_inventory.csv`: missingness, non-flowing records and source
  censoring by observation year.
- `figures/`: PNG and editable SVG, English and Chinese versions.

Mandatory map credit: Source : UMR 1069 SAS INRA - Agrocampus Ouest.
