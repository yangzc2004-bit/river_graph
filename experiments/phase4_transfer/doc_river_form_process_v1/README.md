# River form and observed DOC transmission

This study follows `doc_river_morphology_effect_v1` by connecting real morphology
to tributary integration and along-channel DOC fluctuation transmission.

Read `research_decision.md` for the findings and the next channel-organization
experiment. The focus is river form; landscape differences are background.

## Reproduce

```bash
uv run python scripts/analyze_doc_river_form_process_v1.py --bootstrap-draws 5000
uv run python scripts/plot_doc_river_form_process_v1.py
uv run python scripts/plot_doc_river_form_process_v1.py --chinese
uv run python scripts/verify_doc_river_form_process_v1.py
```

The scripts require the local ST357 dataset and source-only earlier mechanism
products. No model fitting or network download is needed.

## Products

- `analysis/*_receivers.csv`: receiver-weighted observations and morphology.
- `analysis/mixing_connections.csv`, `path_connections.csv`: connection evidence.
- `analysis/inclusion_ledger.csv`: included/unmatched monitoring stations.
- `analysis/class_descriptors.csv`, `class_contrasts.csv`: fixed shape categories.
- `analysis/morphology_associations.csv`: every focal term/outcome and adjustment.
- `analysis/leave_one_block_out.csv`: shared-system and HUC4 sensitivity.
- `figures/observed_form_transmission*`: empirical signals by actual river form.
- `figures/form_process_associations*`: branch/path/footprint comparison.
- `study_plan.md`, `analysis_sources.json`, `code_snapshot/`, `verification.json`:
  study definitions and reproducibility evidence.

The hydro-complete calendar-only and hydro-adjusted panels use identical monthly
observations, so changing populations is not confused with removing local hydro.
The primary uncertainty unit joins monitoring systems that share source/receiver
stations. Pointwise intervals remain exploratory; class descriptions are not
causal or physical travel-time estimates. No DOC reconstruction model is retrained.
