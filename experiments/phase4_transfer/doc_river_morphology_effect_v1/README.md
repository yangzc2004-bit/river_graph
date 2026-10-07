# Morphology-centred river DOC study

Read `research_decision.md` for the results. The question is how river form,
branch organization and channel paths relate to DOC. Environmental sources are
comparison covariates rather than the main outcome of this study.

## Reproduce

```bash
uv run python scripts/analyze_doc_river_morphology_effect_v1.py --bootstrap-draws 5000
uv run python scripts/verify_doc_river_morphology_effect_v1.py
uv run python scripts/plot_doc_river_morphology_effect_v1.py
uv run python scripts/plot_doc_river_morphology_effect_v1.py --chinese
```

Requires the preceding source-response and location-screened landscape panel,
hydrologic response fits, cached NHDPlus VAA and upstream routing files. Large
raw files stay local. There is no neural training or external model-test read.

## Evidence

- `station_morphology_doc_panel.csv`:297 source-response stations and geometric
  route descriptors, with environment and fixed three-class membership.
- `covariate_selected_pairs.csv`, `matching_balance.csv`: fixed covariate-only,
  same-HUC4, non-nested matches and all comparison-covariate imbalances.
- `paired_doc_contrasts.csv`, `paired_doc_evidence.csv`: all response differences,
  sample counts and paired regional intervals; one-block intervals remain NA.
- `paired_geometry_contrasts.csv`, `paired_geometry_evidence.csv`: the same
  comparisons for vegetation-independent paths and controlled routing.
- `station_path_profiles.csv`, `class_routing_descriptors.csv`: area-weighted
  distance profiles and group summaries.
- `identical_input_routing.parquet`: artificial uniform concentration-anomaly
  responses and no-delay input, explicitly separate from observed DOC.
- `morphology_block_predictions.csv`, `morphology_block_gains.csv`: matched
  source-station median-DOC diagnostics and every geometric-block comparison.

Three EN/CN figure families compare matched forms, visualize paths and identical-
input routing, and decompose geometric information. Simulations are labelled;
station-median error improvements are not monthly model improvements.

## Execution notes

Before final verification, the area caliper was made exact on physical area
(rather than area+1 from its stored log1p covariate), without changing the pair
list or conclusions. One-HUC4 bootstrap intervals were made unavailable rather
than displaying a degenerate point interval. Matched geometry contrasts use the
same outcome-independent pairs and the path descriptors planned in advance.
These repairs did not select a new class, covariate, population or endpoint.

`code_snapshot/` retains the execution scripts, new library and tests. The
environmental baseline columns are frozen in the new library, identical to
the preceding morphology study. Its controlling source file is hashed as
method lineage; executing this study does not import its unrelated, previously
untracked helper. No original local helper was added to the commit.
