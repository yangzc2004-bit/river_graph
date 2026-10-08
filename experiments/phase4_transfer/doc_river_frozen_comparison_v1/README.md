# Four procedure river information experiment

The completed whole-region study and supplemental nested station study ask
whether actual upstream relations and measured form improve the complete DOC
predictor. Start with `research_decision.md` and `figures/`.

Recompute from the saved local packages using the project uv environment:

```bash
uv run python scripts/verify_doc_river_frozen_comparison_v1.py
uv run python scripts/analyze_doc_river_frozen_comparison_v1.py --bootstrap-draws 5000
uv run python scripts/analyze_doc_river_connected_comparison_v1.py --bootstrap-draws 5000
uv run python scripts/plot_doc_river_four_procedures_v1.py
```

Training uses `run_ladder.py` with experiments `doc-river-frozen-comparison-v1`
and `doc-river-connected-comparison-v1`. Saved runtime snapshots identify the
execution versions; complete packages must not be overwritten with changed
code. Source station geometry, prior fitted DOC packages and the ST357 dataset
remain local prerequisites. Large full-grid and feature caches remain local;
the small predictions, calibration states, configuration and analyzed tables
are committed. This archive supports local replay, not a data-free fresh clone.

`analysis/primary_summary.csv` uses all observed geographical K0 cells.
`analysis/curves_summary.csv` uses a separate fixed query for K0/1/3/5.
The supplementary experiment is in `../doc_river_connected_comparison_v1/`.
Do not pool these two populations or call the station development study an
external or independent geographical validation.
