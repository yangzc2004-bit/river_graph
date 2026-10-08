# River form and DOC fluctuation timescale

This version asks how real source-path differences and the common downstream
corridor respond to fast versus slow input fluctuations. All forcing and shared
memory are controlled scenarios. Actual monthly DOC is retained as explicitly
separate observational context; no residence time is fitted to those outcomes.

## Reproduce

```bash
uv run python scripts/analyze_doc_river_signal_timescale_v1.py --bootstrap-draws 5000
uv run python scripts/plot_doc_river_signal_timescale_v1.py
uv run python scripts/plot_doc_river_signal_timescale_v1.py --chinese
uv run python scripts/verify_doc_river_signal_timescale_v1.py --full-replay

uv run python scripts/analyze_doc_river_signal_timescale_v1.py --bootstrap-draws 5000 --shortest-routes
uv run python scripts/plot_doc_river_signal_timescale_v1.py --shortest-routes
uv run python scripts/plot_doc_river_signal_timescale_v1.py --shortest-routes --chinese
uv run python scripts/verify_doc_river_signal_timescale_v1.py --shortest-routes --full-replay
```

## Products

- `source_paths.csv`: cropped real source paths, their shared suffix and weights.
- `scenario_metrics.parquet`: all 6,912 controlled scenarios for 32 receivers.
- `scenario_summary.csv` / `form_contrasts.csv`: complete timescale/coherence/
  memory grid with complete-system bootstrap intervals; all receivers and the
  fixed >=80% represented-area subset are retained.
- `observed_and_controlled_context.csv`: deliberately separates copied observed
  fluctuation ratios from the imposed-process working point.
- `illustration_metrics.csv` / `illustration_waves.parquet`: constructed source
  phases on two actual Loch Vale paths; these are not measured DOC waves and do
  not enter population intervals.
- Two English/Chinese scientific figures, their source receipts, and a separate
  complete shortest-route sensitivity.

The mean-path time is dimensionless under an imposed uniform-speed proxy. Source
weights are drainage-area shares. Common-memory strength describes an allocated
spread within the original corridor travel budget, not lake volume, measured
flow routing, DOC reaction or a fitted physical transit time.
