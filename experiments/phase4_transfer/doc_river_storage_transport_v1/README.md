# Conservative DOC branch and storage experiment

The [research decision](research_decision.md) explains how branch-arrival
dispersion differs from common-trunk storage broadening at the same mean
arrival. All input signals and mixture shares are fixed. The 59 footprints
are measured geometry; the response curves are controlled simulations.

## Reproduction

Run from the repository root with its uv-managed environment:

```bash
uv run python scripts/analyze_doc_river_storage_transport_v1.py
uv run python scripts/plot_doc_river_storage_transport_v1.py
uv run python scripts/plot_doc_river_storage_transport_v1.py --chinese
uv run python scripts/verify_doc_river_storage_transport_v1.py
uv run pytest
uv run ruff check .
uv run python scripts/audit_artifacts.py --verify
```

For this host, uv is available at
`/tmp/river-graph-authoring-tools/bin/uv`; commands may use `--no-sync` with the
already synchronized environment. Source data are the completed monitored
footprint analysis and cached NHDPlus flowlines. No network request or training
is required.

## Products

- `analysis/scenario_metrics.csv`: all 1,416 cases, integrated/analytical timing,
  peak, duration and anomaly-area checks, plus original corridor geometry.
- `analysis/structural_contrasts.csv`: all 708 matched branch/storage contrasts.
- `analysis/receiver_contrasts.csv`, `cohort_summary.csv`: receiver-equal summaries
  and descriptive connection distributions; scenario ranges are not confidence
  intervals.
- `analysis/outline_context.csv`: the unchanged three whole-network classes,
  with actual monitored receiver/system counts.
- `analysis/mapped_storage_opportunity.csv`: all five mapped common-storage cases.
- `analysis/variance_dominance.csv`: geometry-only conditional variance comparison.
- `analysis/representative_responses.parquet`: five geometry-selected examples.
- `analysis/numerical_convergence.csv`: complete finer-grid comparison.
- `figures/matched_branch_storage_responses*`: real corridors and controlled
  responses at the same mean arrival.
- `figures/structural_time_scale_effects*`: branch/storage effects across forcing
  durations and exact variance decomposition.

Both figures are available in English and Chinese as PNG/PDF. Previous studies,
classifications and prediction products are retained.
