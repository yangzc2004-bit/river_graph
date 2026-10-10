# Protected dynamic river DOC comparison

This experiment extends the retained complete DOC predictor with real upstream
attention and causal 0/1/3-month observation slots. The fixed study_plan.md,
research_decision.md and analysis/ describe the comparison and its results.

## Reproduce

Run from the repository root in its uv environment:

```bash
uv run python scripts/run_ladder.py --experiment doc-dynamic-river-v1
uv run python scripts/verify_doc_dynamic_river_v1.py --rebuild-inputs
uv run python scripts/analyze_doc_dynamic_river_v1.py --bootstrap-draws 5000
uv run python scripts/check_doc_dynamic_river_calculations_v1.py
uv run python scripts/plot_doc_dynamic_river_v1.py
uv run pytest -q
uv run ruff check .
uv run python scripts/audit_artifacts.py --verify
```

The execution used .venv/bin/uv with --no-sync, UV_OFFLINE=1 and
UV_CACHE_DIR=/private/tmp/river-graph-uv-cache. Torch used two threads;
OMP_NUM_THREADS and OPENBLAS_NUM_THREADS were one. All source roles, data,
retained models and consumed caches are recorded in run configs.

The runner resumes a completed stage only under its saved configuration and
runtime. It does not overwrite a changed execution. Full model initialization
and source preprocessing come from the bound older complete predictor. Archived
forest joblib files are not needed: the environmental OOF prediction arrays and
retained neural states are the consumed inputs. Missing required retained files
produce an explicit error; they are not silently replaced by a refit.

## Products

- runs/: configs, completion records, source roles, availability matching,
  branch summaries, prediction parquet/sidecars and diagnostic arrays.
- runs/*/fitting_inputs.npz and *.pt: local caches and neural states, gitignored.
- analysis/: per-run and per-partition metrics, paired effects, station-equal
  error, predefined subgroups and query attention diagnostics.
- verification/: exact input/prediction replay and independent arithmetic checks.
- figures/: standalone scientific comparison and conditional-gain figures.
- runtime_snapshot.json and code_snapshot/: executed fitting version.

These are source-validation development results, not independent geographical
or external-basin validation. The fixed complete model remains the released
prediction procedure; the new branch is retained as a candidate.
