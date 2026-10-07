# DOC source placement within real river networks

Read `research_decision.md` for the scientific result. This extends the real
planform and river-mechanism studies without fitting a new neural predictor.

## Reproduction

Run from the repository root in the uv-managed environment. The earlier
planform typology, source response and physical membership products are
required, together with ST357, `cache/nldplus_vaa.parquet`, full upstream
membership files and cached geometry. Raw landscape requests and routing
arrays are local re-derivable caches under
`data/raw/river_source_placement_v1/` and are not committed.

```bash
uv run python scripts/review_doc_river_station_locations_v1.py
uv run python scripts/fetch_doc_river_source_landscape_v1.py
uv run python scripts/build_doc_river_source_placement_v1.py
uv run python scripts/analyze_doc_river_source_placement_v1.py --bootstrap-draws 5000
uv run python scripts/verify_doc_river_source_placement_v1.py
uv run python scripts/plot_doc_river_source_placement_v1.py
uv run python scripts/plot_doc_river_source_placement_v1.py --chinese
```

The optional Chinese rendering uses the macOS Arial Unicode font. English
rendering does not require it. The source receipt records files used for each
stage; large cached inputs must be present for full recomputation. The final
analysis uses `study_plan_v1_1.md`; original comparisons are preserved in
`initial_watershed_controls/`.

`code_snapshot/` retains the executing code and its imported scientific
helpers. In particular, `src/river_graph/analysis/river_doc_structure.py` was a
pre-existing local, untracked dependency. Its snapshot is saved here rather
than folding unrelated earlier work into this commit. A checkout without the
helper can copy that snapshot to its corresponding `src/` location before
reproduction. Do not overwrite a different local implementation without
reviewing it. Snapshot file hashes are listed in `code_snapshot/manifest.json`.

## Tables

- `station_location_review.csv`: all 15 mapping flags and independent proposals.
- `station_source_placement.csv`: all 357 gauges, coverage and exclusion status.
- `station_doc_source_panel.csv`: classified source stations, old aggregate
  cover and reconciled catchment cover, DOC summaries and controls.
- `recent_station_doc_source_panel.csv`: source-month sensitivity since 2009.
- `huc4_blocked_predictions.csv`: common station/fold population for four arms.
- `huc4_blocked_gains.csv`: native/log1p gains and station/HUC4 bootstrap.
- `source_placement_associations.csv`: exploratory shape and source-position
  coefficients for median DOC, CV and seasonal amplitude, both periods.
- `class_source_placement.csv`: descriptive class medians and valid counts.

## Figures

Each is provided as PNG/PDF, with `_cn` variants:

1. `real_network_source_maps`: actual flowlines coloured by their own local
   catchment wetland/forest fraction; colour is not DOC concentration.
2. `source_placement_distribution`: source amount versus position by shape class.
3. `source_placement_prediction_increment`: geographically blocked diagnostic.
4. `source_placement_dynamic_associations`: all fixed position terms, both periods.

The primary diagnostic is station median DOC, not full-grid prediction or K0
model performance. Preserve that distinction when reusing the plots.
