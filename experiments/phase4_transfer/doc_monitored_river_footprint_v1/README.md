# Monitored river footprint and DOC transmission

The matched-scale follow-up to the internal river-structure study. River paths
are partitioned into independent branches and a shared downstream trunk.
The scientific interpretation and next study are in `research_decision.md`.

## Reproduce

From the repository root, using the project's uv-managed environment:

```bash
uv run python scripts/analyze_doc_monitored_river_footprint_v1.py --bootstrap-draws 5000
uv run python scripts/plot_doc_monitored_river_footprint_v1.py
uv run python scripts/plot_doc_monitored_river_footprint_v1.py --chinese
uv run python scripts/verify_doc_monitored_river_footprint_v1.py
uv run pytest tests/test_river_monitored_footprint.py
```

Local dependencies: graph node table with NHD station measures, cached VAA
topology and real channel geometry, and previous observed-transport, signal and
internal-structure analysis products. The SQLite cache is read-only during
plotting. Geography uses actual EPSG:5070 channels, cropped at station positions.

## Products

- `analysis/footprints.csv`: all 59 measured branch/trunk configurations.
- `analysis/corridor_reaches.csv`: unique reaches per pair, including partial
  lengths, storage status and station linear measures.
- `analysis/pulse_scenarios.csv`: 177 matched routing scenarios.
- `analysis/observed_receivers.csv` and `structural_associations.csv`: the same
  22 receivers; six adjusted associations and system-bootstrap intervals.
- `analysis/timing_inputs.parquet`: exact source-interpolation decomposition.
- `analysis/fixed_predictions.parquet` with sidecar: known-upstream prediction
  sensitivity under saved held-system coefficients, without refitting.
- `analysis/paired_error_comparisons.csv`: MAE, log-MAE, MSE, bias magnitude and
  Q90 differences against the complete saved mean-delay reference.
- `figures/real_monitored_footprints*`: four real geometry-selected examples.
- `figures/branch_and_common_trunk_signals*`: dispersion versus translation.
- `figures/matched_structure_and_doc*`: measured DOC and fixed-model comparison.
- `verification.json`: geometry, numerical replay and visibility checks.

Four display quadrants do not replace the three complete-network outline
classes. Controlled relative time is not measured travel days, and pulse
damping is not a removal estimate. This is a within-ST357 mechanism study,
not independent external validation or a new K0 model benchmark.
