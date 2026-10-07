# Complete river-form and storage experiment

See [study_plan.md](study_plan.md) and [research_decision.md](research_decision.md).
This analysis uses the existing real-network cohort, labels, representatives
and covariate-selected pairs; no neural model is trained.

From the repository root, using the existing uv-managed environment:

```bash
uv run python scripts/analyze_doc_river_whole_storage_v1.py --bootstrap-draws 5000
uv run python scripts/plot_doc_river_whole_storage_v1.py
uv run python scripts/plot_doc_river_whole_storage_v1.py --chinese
uv run python scripts/verify_doc_river_whole_storage_v1.py --full-replay
uv run pytest
uv run ruff check .
uv run python scripts/audit_artifacts.py --verify
```

Local dependencies are the cached NHDPlus VAA, saved shortest directed routing
arrays, mapped basin/flowline geometry and existing morphology panel. Derived
per-outlet numerical caches remain in the gitignored
`data/processed/river_whole_storage_v1/`. Cache identity includes the analysis
engine, VAA, route array, basin scale and numerical design. Cached results are
never substituted after an identity change. Verification recomputes without
using them; `--full-replay` checks all 297 instances.

`analysis_sources.json` binds input and product contents; `code_snapshot/`
retains the execution implementation. Figure receipts bind their tables and
actual map sources. The outputs are identical-input process scenarios,
not observed DOC peak attenuation or calibrated residence times.
